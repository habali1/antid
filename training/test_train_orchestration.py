#!/usr/bin/env python3
"""test_train_orchestration.py — REAL ORCHESTRATION / INTEGRATION tests.

Unlike test_train_harness.py (pure helper/unit tests on isolated functions),
this file exercises train.py's ACTUAL run_epochs() orchestration function --
the same code the real main() calls -- against a tiny synthetic model and a
tiny deterministic dataset, never a reimplementation of the resume logic.

The centerpiece is TestInterruptedVsUninterrupted: it proves mathematical
resume equivalence by running two epochs straight through, then separately
running epoch 1, pausing, tearing down every in-memory object, and resuming
into a brand-new model/optimizer/DataLoader exactly the way a real
interrupted-and-restarted process would -- then asserts the two runs are
identical in every deterministic respect (sample order, augmentation-draw-
affected losses, final model parameters, optimizer state, selected best
epoch/metrics, and history rows).

Never initializes or trains the real B4 model and never downloads
pretrained weights. Run directly: python test_train_orchestration.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkpoint as ckpt_mod  # noqa: E402
import numerics  # noqa: E402
import run_manifest_schema  # noqa: E402
import train  # noqa: E402

SEED = 20260906
NUM_CLASSES = 3
EMBEDDING_DIM = 6
FEATURE_DIM = 4
N_TRAIN = 9
N_VAL = 6


class TinyEmbedModel(nn.Module):
    """Mirrors AntIDModel's embed()/forward() interface (embed() returns the
    pre-head embedding used for prototypes/cosine ranking; forward() returns
    classifier logits for cross-entropy training) at a tiny, CPU-fast scale.
    A random draw INSIDE forward() (only in training mode) stands in for
    real augmentation, consuming the same global torch RNG stream that
    numerics.capture_rng_state()/restore_rng_state() manage -- so getting
    resume ordering wrong is actually observable here, not just theoretical.
    """

    def __init__(self, num_classes: int, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(FEATURE_DIM, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x + torch.randn_like(x) * 0.05  # synthetic "augmentation"
        return self.classifier(self.dropout(self.embed(x)))


class TinyDataset(Dataset):
    """Fixed, non-random synthetic "images": deterministic feature vectors
    keyed by index, so any difference in observed order/loss between two
    runs can only come from RNG/shuffle/augmentation state, never from the
    underlying data itself."""

    def __init__(self, n: int, num_classes: int, offset: int = 0):
        g = torch.Generator().manual_seed(1000 + offset)
        self.features = torch.randn(n, FEATURE_DIM, generator=g)
        self.labels = (torch.arange(n) + offset) % num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def _build_fresh(seed: int):
    train_gen = numerics.seed_everything(seed)
    model = TinyEmbedModel(NUM_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    return train_gen, model, opt


def _build_loaders(train_gen):
    train_ds = TinyDataset(N_TRAIN, NUM_CLASSES, offset=0)
    proto_ds = TinyDataset(N_TRAIN, NUM_CLASSES, offset=0)
    val_ds = TinyDataset(N_VAL, NUM_CLASSES, offset=100)
    train_dl = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=0,
                          generator=train_gen, worker_init_fn=numerics.worker_init_fn)
    proto_dl = DataLoader(proto_ds, batch_size=3, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=3, num_workers=0)
    return train_dl, proto_dl, val_dl


def _fixed_taxonomy():
    return {i: {"slug": f"species-{i}"} for i in range(NUM_CLASSES)}


def _fixed_provenance(**overrides) -> dict:
    base = {
        "manifest_sha256": "m" * 64, "taxonomy_sha256": "t" * 64,
        "val_split_sha256": "v" * 64, "resolved_config_sha256": "c" * 64,
        "backbone": "tiny-test-model", "num_classes": NUM_CLASSES,
        "git_commit": "test-commit", "numerical_policy": dict(numerics.NUMERICAL_POLICY),
        "run_kind": "smoke", "limit_batches": None, "wandb_enabled": False,
        "validation_cadence": 1,
    }
    base.update(overrides)
    return base


def _fixed_manifest_record() -> dict:
    return {"path": "/fake/manifest.csv", "sha256": "a" * 64, "rows": N_TRAIN + N_VAL}


def _fixed_taxonomy_record() -> dict:
    return {"path": "/fake/taxonomy.json", "sha256": "b" * 64, "num_classes": NUM_CLASSES}


def _fixed_val_split_record(art: Path) -> dict:
    return {"path": str(art / "val_split.json"), "sha256": "c" * 64,
           "n_train": N_TRAIN, "n_val": N_VAL, "n_total": N_TRAIN + N_VAL}


def _invocation_record(resume: bool = False, pause_after_epoch=None) -> dict:
    return {"timestamp_utc": "2026-01-01T00:00:00Z", "argv": [], "resume": resume,
           "pause_after_epoch": pause_after_epoch}


def _valid_run_manifest(art: Path, validation_cadence: int = 1) -> dict:
    """A schema-valid run_manifest dict at "data_verified" stage, built via
    the REAL functions main() itself calls (bootstrap_run_manifest,
    persist_or_verify_data_provenance) -- not a reimplementation -- so
    run_epochs' internal schema validation (stage="epoch_committed") always
    has the fields it requires."""
    rm = train.bootstrap_run_manifest(
        art / "run_manifest.json", resume=False, invocation_record=_invocation_record(),
        run_kind="smoke", git_commit="test-commit", git_dirty=False,
        validation_cadence=validation_cadence,
    )
    train.persist_or_verify_data_provenance(
        rm, resume=False, manifest_record=_fixed_manifest_record(),
        taxonomy_record=_fixed_taxonomy_record(), val_split_record=_fixed_val_split_record(art),
    )
    return rm


def _strip_wallclock(history: list[dict]) -> list[dict]:
    """Deterministic fields only -- duration_seconds and timestamp_utc are
    real wall-clock measurements and are never expected to match bit-for-bit
    between two separate runs."""
    return [
        {k: v for k, v in row.items() if k not in ("duration_seconds", "timestamp_utc")}
        for row in history
    ]


class TestInterruptedVsUninterrupted(unittest.TestCase):
    """The one test that actually proves resume equivalence -- see the file
    docstring. Helper/unit tests on capture_rng_state()/restore_rng_state()
    alone (in test_train_harness.py) are NOT sufficient on their own; this is
    what makes "resume tested" a true claim."""

    def test_two_epoch_uninterrupted_matches_pause_and_resume(self):
        cfg = {"epochs": 2}
        provenance = _fixed_provenance()

        # ---------------- Run A: straight through, two epochs, no pause ----------------
        with tempfile.TemporaryDirectory() as td_a:
            art_a = Path(td_a)
            train_gen_a, model_a, opt_a = _build_fresh(SEED)
            train_dl_a, proto_dl_a, val_dl_a = _build_loaders(train_gen_a)
            batches_a: list[tuple] = []
            best_a, history_a, outcome_a = train.run_epochs(
                art=art_a, cfg=cfg, start_epoch=0, model=model_a, opt=opt_a,
                train_dl=train_dl_a, proto_dl=proto_dl_a, val_dl=val_dl_a,
                num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, taxonomy=_fixed_taxonomy(),
                device=torch.device("cpu"), provenance=provenance, canonical_history=[],
                best_ref=None, train_gen=train_gen_a, run_manifest=_valid_run_manifest(art_a),
                run_manifest_path=art_a / "run_manifest.json", limit_batches=None,
                pause_after_epoch=None, validation_cadence=1, use_wandb=False,
                tqdm_fn=lambda x, **k: x,
                on_batch=lambda epoch, bi, labels, loss: batches_a.append((epoch, labels, loss)),
            )
            self.assertEqual(outcome_a, "completed")
            final_last_a = ckpt_mod.torch_load_trusted(art_a / "checkpoint_last.pth", map_location="cpu")
            final_best_payload_a = ckpt_mod.verify_referenced_best(art_a, final_last_a)

        # ---------------- Run B: epoch 1, pause, tear down, resume for epoch 2 ----------------
        with tempfile.TemporaryDirectory() as td_b:
            art_b = Path(td_b)
            train_gen_b1, model_b1, opt_b1 = _build_fresh(SEED)  # identical seed to Run A
            train_dl_b1, proto_dl_b1, val_dl_b1 = _build_loaders(train_gen_b1)
            batches_b_epoch1: list[tuple] = []
            best_b, history_b, outcome_b = train.run_epochs(
                art=art_b, cfg=cfg, start_epoch=0, model=model_b1, opt=opt_b1,
                train_dl=train_dl_b1, proto_dl=proto_dl_b1, val_dl=val_dl_b1,
                num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, taxonomy=_fixed_taxonomy(),
                device=torch.device("cpu"), provenance=provenance, canonical_history=[],
                best_ref=None, train_gen=train_gen_b1, run_manifest=_valid_run_manifest(art_b),
                run_manifest_path=art_b / "run_manifest.json", limit_batches=None,
                pause_after_epoch=1, validation_cadence=1, use_wandb=False,
                tqdm_fn=lambda x, **k: x,
                on_batch=lambda epoch, bi, labels, loss: batches_b_epoch1.append((epoch, labels, loss)),
            )
            self.assertEqual(outcome_b, "paused")

            # Tear down every in-memory object from the pre-pause process --
            # nothing below may reference train_gen_b1/model_b1/opt_b1/etc.
            del train_gen_b1, model_b1, opt_b1, train_dl_b1, proto_dl_b1, val_dl_b1

            # ---- Simulate a brand-new process resuming ----
            saved = ckpt_mod.torch_load_trusted(art_b / "checkpoint_last.pth", map_location="cpu")
            mismatches = ckpt_mod.provenance_mismatches(provenance, saved["provenance"])
            self.assertEqual(mismatches, [])
            ckpt_mod.verify_referenced_best(art_b, saved)

            # Corrected ordering under test: model (pretrained=False-equivalent
            # -- fresh init doesn't matter, immediately overwritten) ->
            # optimizer -> load state -> DataLoaders -> RNG restore LAST,
            # immediately before the resumed epoch loop.
            model_b2 = TinyEmbedModel(NUM_CLASSES)
            opt_b2 = torch.optim.AdamW(model_b2.parameters(), lr=0.01)
            model_b2.load_state_dict(saved["model"])
            opt_b2.load_state_dict(saved["optimizer"])
            train_gen_b2 = torch.Generator()  # placeholder
            train_dl_b2, proto_dl_b2, val_dl_b2 = _build_loaders(train_gen_b2)
            numerics.restore_rng_state(saved["rng_state"])
            train_gen_b2.set_state(saved["train_generator_state"])

            # Resume bootstrap through the REAL gate main() uses -- run_manifest.json
            # already exists on disk (written during the pre-pause pause-write), so
            # this must go through resume=True, exactly like a real restarted process.
            resumed_run_manifest = train.bootstrap_run_manifest(
                art_b / "run_manifest.json", resume=True,
                invocation_record=_invocation_record(resume=True),
                run_kind="smoke", git_commit="test-commit", git_dirty=False,
                validation_cadence=1,
            )
            train.persist_or_verify_data_provenance(
                resumed_run_manifest, resume=True, manifest_record=_fixed_manifest_record(),
                taxonomy_record=_fixed_taxonomy_record(),
                val_split_record=_fixed_val_split_record(art_b),
            )

            batches_b_epoch2: list[tuple] = []
            best_b2, history_b2, outcome_b2 = train.run_epochs(
                art=art_b, cfg=cfg, start_epoch=saved["completed_epoch"] + 1,
                model=model_b2, opt=opt_b2, train_dl=train_dl_b2, proto_dl=proto_dl_b2,
                val_dl=val_dl_b2, num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM,
                taxonomy=_fixed_taxonomy(), device=torch.device("cpu"), provenance=provenance,
                canonical_history=saved["history"], best_ref=saved["best"],
                train_gen=train_gen_b2, run_manifest=resumed_run_manifest,
                run_manifest_path=art_b / "run_manifest.json", limit_batches=None,
                pause_after_epoch=None, validation_cadence=1, use_wandb=False,
                tqdm_fn=lambda x, **k: x,
                on_batch=lambda epoch, bi, labels, loss: batches_b_epoch2.append((epoch, labels, loss)),
            )
            self.assertEqual(outcome_b2, "completed")
            final_last_b = ckpt_mod.torch_load_trusted(art_b / "checkpoint_last.pth", map_location="cpu")
            final_best_payload_b = ckpt_mod.verify_referenced_best(art_b, final_last_b)

        # ---------------- Assertions ----------------
        # 1. Second-epoch sample order (labels) AND augmentation/random-draw-
        #    affected loss values, batch for batch.
        batches_a_epoch2 = [b for b in batches_a if b[0] == 1]
        self.assertTrue(batches_a_epoch2, "Run A produced no epoch-2 batches to compare")
        self.assertEqual(batches_a_epoch2, batches_b_epoch2)
        # And epoch 1 (pre-pause) matches too, as a sanity check on the setup itself.
        batches_a_epoch1 = [b for b in batches_a if b[0] == 0]
        self.assertEqual(batches_a_epoch1, batches_b_epoch1)

        # 2. Final model parameters.
        for (na, pa), (nb, pb) in zip(final_best_payload_a["model"].items(),
                                      final_best_payload_b["model"].items()):
            self.assertEqual(na, nb)
            self.assertTrue(torch.equal(pa, pb), f"parameter {na} diverged after resume")

        # 3. Optimizer state (from checkpoint_last -- the resumable one).
        opt_state_a = final_last_a["optimizer"]["state"]
        opt_state_b = final_last_b["optimizer"]["state"]
        self.assertEqual(set(opt_state_a.keys()), set(opt_state_b.keys()))
        for k in opt_state_a:
            for field in ("exp_avg", "exp_avg_sq", "step"):
                va, vb = opt_state_a[k][field], opt_state_b[k][field]
                if torch.is_tensor(va):
                    self.assertTrue(torch.equal(va, vb), f"optimizer state {k}.{field} diverged")
                else:
                    self.assertEqual(va, vb)

        # 4. Selected best epoch/metrics.
        self.assertEqual(final_last_a["best"], final_last_b["best"])

        # 5. History rows (deterministic fields only -- wall-clock timing legitimately differs).
        self.assertEqual(_strip_wallclock(final_last_a["history"]),
                         _strip_wallclock(final_last_b["history"]))


class TestPauseNeverMidEpoch(unittest.TestCase):
    def test_pause_after_epoch_1_commits_epoch_1_fully_before_returning(self):
        cfg = {"epochs": 2}
        provenance = _fixed_provenance()
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            train_gen, model, opt = _build_fresh(SEED)
            train_dl, proto_dl, val_dl = _build_loaders(train_gen)
            best_ref, history, outcome = train.run_epochs(
                art=art, cfg=cfg, start_epoch=0, model=model, opt=opt,
                train_dl=train_dl, proto_dl=proto_dl, val_dl=val_dl,
                num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, taxonomy=_fixed_taxonomy(),
                device=torch.device("cpu"), provenance=provenance, canonical_history=[],
                best_ref=None, train_gen=train_gen, run_manifest=_valid_run_manifest(art),
                run_manifest_path=art / "run_manifest.json", limit_batches=None,
                pause_after_epoch=1, validation_cadence=1, use_wandb=False,
                tqdm_fn=lambda x, **k: x,
            )
            self.assertEqual(outcome, "paused")
            self.assertTrue((art / "checkpoint_last.pth").exists())
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            self.assertEqual(last["completed_epoch"], 0)  # epoch 1 (one-based) == index 0
            self.assertEqual(len(last["history"]), 1)
            ckpt_mod.verify_referenced_best(art, last)  # best file fully committed too
            history_rows = (art / "history.jsonl").read_text().splitlines()
            self.assertEqual(len(history_rows), 1)
            run_manifest_path = art / "run_manifest.json"
            # run_epochs writes run_manifest itself; simulate the caller
            # having passed a real path and check the final status field the
            # function set before returning "paused".
            saved_manifest = json.loads(run_manifest_path.read_text())
            self.assertEqual(saved_manifest["status"], "paused_for_smoke")

    def test_no_epochs_run_when_start_epoch_at_or_past_total(self):
        cfg = {"epochs": 1}
        provenance = _fixed_provenance()
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            train_gen, model, opt = _build_fresh(SEED)
            train_dl, proto_dl, val_dl = _build_loaders(train_gen)
            best_ref, history, outcome = train.run_epochs(
                art=art, cfg=cfg, start_epoch=1, model=model, opt=opt,  # already "done"
                train_dl=train_dl, proto_dl=proto_dl, val_dl=val_dl,
                num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, taxonomy=_fixed_taxonomy(),
                device=torch.device("cpu"), provenance=provenance, canonical_history=[],
                best_ref=None, train_gen=train_gen, run_manifest=_valid_run_manifest(art),
                run_manifest_path=art / "run_manifest.json", limit_batches=None,
                pause_after_epoch=None, validation_cadence=1, use_wandb=False,
                tqdm_fn=lambda x, **k: x,
            )
            self.assertEqual(outcome, "completed")
            self.assertIsNone(best_ref)
            self.assertFalse((art / "checkpoint_last.pth").exists())


class TestValidationCadenceOrchestration(unittest.TestCase):
    """Real run_epochs() exercise of the validation-cadence gating -- not a
    reimplementation of should_validate (see TestValidationCadenceSchedule in
    test_train_harness.py for that), but proof that run_epochs actually skips
    the expensive prototype/validation work, still commits every epoch, and
    only ever moves checkpoint_best on a validated improvement."""

    def _run(self, cfg, validation_cadence, pause_after_epoch=None, start_epoch=0,
             art=None, model=None, opt=None, train_gen=None, run_manifest=None,
             canonical_history=None, best_ref=None):
        train_gen = train_gen or numerics.seed_everything(SEED)
        model = model or TinyEmbedModel(NUM_CLASSES)
        opt = opt or torch.optim.AdamW(model.parameters(), lr=0.01)
        train_dl, proto_dl, val_dl = _build_loaders(train_gen)
        return train.run_epochs(
            art=art, cfg=cfg, start_epoch=start_epoch, model=model, opt=opt,
            train_dl=train_dl, proto_dl=proto_dl, val_dl=val_dl,
            num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, taxonomy=_fixed_taxonomy(),
            device=torch.device("cpu"), provenance=_fixed_provenance(validation_cadence=validation_cadence),
            canonical_history=canonical_history if canonical_history is not None else [],
            best_ref=best_ref, train_gen=train_gen,
            run_manifest=run_manifest if run_manifest is not None else _valid_run_manifest(art, validation_cadence),
            run_manifest_path=art / "run_manifest.json", limit_batches=None,
            pause_after_epoch=pause_after_epoch, validation_cadence=validation_cadence,
            use_wandb=False, tqdm_fn=lambda x, **k: x,
        )

    def test_skipped_epochs_never_call_prototypes_or_validation(self):
        # cadence 3 over 4 epochs validates [1,3,4] -- epoch 2 is skipped.
        cfg = {"epochs": 4}
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            with mock.patch.object(train, "compute_prototypes",
                                   wraps=train.compute_prototypes) as proto_spy, \
                 mock.patch.object(train, "topk_accuracy",
                                   wraps=train.topk_accuracy) as val_spy:
                best_ref, history, outcome = self._run(cfg, validation_cadence=3, art=art)
            self.assertEqual(outcome, "completed")
            self.assertEqual(len(history), 4)
            self.assertEqual(proto_spy.call_count, 3)  # epochs 1, 3, 4 -- never epoch 2
            self.assertEqual(val_spy.call_count, 3)

    def test_skipped_epoch_history_row_has_nulls_not_zeros(self):
        cfg = {"epochs": 4}
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            best_ref, history, outcome = self._run(cfg, validation_cadence=3, art=art)
            skipped_row = history[1]  # epoch_number 2, zero-based index 1
            self.assertEqual(skipped_row["epoch"], 1)
            self.assertFalse(skipped_row["validation_ran"])
            self.assertIsNone(skipped_row["val_top1"])
            self.assertIsNone(skipped_row["val_top3"])
            self.assertIsNone(skipped_row["duration_seconds"]["prototypes"])
            self.assertIsNone(skipped_row["duration_seconds"]["validation"])
            self.assertIsNotNone(skipped_row["duration_seconds"]["train"])
            self.assertFalse(skipped_row["is_best"])
            for i in (0, 2, 3):  # validated epochs must have real numbers, never null
                self.assertTrue(history[i]["validation_ran"])
                self.assertIsNotNone(history[i]["val_top1"])
                self.assertIsNotNone(history[i]["val_top3"])

    def test_checkpoint_last_and_history_and_run_manifest_committed_on_skipped_epoch(self):
        cfg = {"epochs": 4}
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            # Pause right after the skipped epoch (epoch_number 2) to inspect
            # exactly what got committed for it.
            best_ref, history, outcome = self._run(cfg, validation_cadence=3,
                                                    pause_after_epoch=2, art=art)
            self.assertEqual(outcome, "paused")
            self.assertTrue((art / "checkpoint_last.pth").exists())
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            self.assertEqual(last["completed_epoch"], 1)  # epoch_number 2, zero-based 1
            self.assertEqual(len(last["history"]), 2)
            self.assertFalse(last["history"][1]["validation_ran"])
            ckpt_mod.verify_referenced_best(art, last)  # best (from epoch 1) still verifies
            self.assertEqual(last["best"]["epoch"], 0)  # only epoch 1 (index 0) ever validated so far
            history_rows = (art / "history.jsonl").read_text().splitlines()
            self.assertEqual(len(history_rows), 2)
            saved_manifest = json.loads((art / "run_manifest.json").read_text())
            self.assertEqual(saved_manifest["status"], "paused_for_smoke")
            self.assertEqual(saved_manifest["last_completed_epoch"], 1)

    def test_checkpoint_best_never_written_for_a_skipped_epoch(self):
        cfg = {"epochs": 4}
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            best_ref, history, outcome = self._run(cfg, validation_cadence=3, art=art)
            best_files = sorted(p.name for p in art.glob("checkpoint_best_epoch_*.pth"))
            # Only ever epoch indices 0, 2, or 3 (epoch_numbers 1, 3, 4) can
            # appear -- epoch index 1 (the skipped one) must never appear,
            # and every row with is_best=True must have validation_ran=True.
            self.assertNotIn("checkpoint_best_epoch_001.pth", best_files)
            for row in history:
                if row["is_best"]:
                    self.assertTrue(row["validation_ran"], row)

    def test_interruption_after_skipped_epoch_resumes_correctly(self):
        cfg = {"epochs": 4}
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            # ---- Pause right after the skipped epoch (epoch_number 2) ----
            best_ref, history, outcome = self._run(cfg, validation_cadence=3,
                                                    pause_after_epoch=2, art=art)
            self.assertEqual(outcome, "paused")

            # ---- Simulate a brand-new process resuming ----
            saved = ckpt_mod.load_resume_checkpoint(art / "checkpoint_last.pth")
            model2 = TinyEmbedModel(NUM_CLASSES)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
            model2.load_state_dict(saved["model"])
            opt2.load_state_dict(saved["optimizer"])
            train_gen2 = torch.Generator()
            train_dl2, proto_dl2, val_dl2 = _build_loaders(train_gen2)
            numerics.restore_rng_state(saved["rng_state"])
            train_gen2.set_state(saved["train_generator_state"])

            resumed_rm = train.bootstrap_run_manifest(
                art / "run_manifest.json", resume=True,
                invocation_record=_invocation_record(resume=True), run_kind="smoke",
                git_commit="test-commit", git_dirty=False, validation_cadence=3,
            )
            train.persist_or_verify_data_provenance(
                resumed_rm, resume=True, manifest_record=_fixed_manifest_record(),
                taxonomy_record=_fixed_taxonomy_record(),
                val_split_record=_fixed_val_split_record(art),
            )

            best_ref2, history2, outcome2 = train.run_epochs(
                art=art, cfg=cfg, start_epoch=saved["completed_epoch"] + 1,
                model=model2, opt=opt2, train_dl=train_dl2, proto_dl=proto_dl2,
                val_dl=val_dl2, num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM,
                taxonomy=_fixed_taxonomy(), device=torch.device("cpu"),
                provenance=_fixed_provenance(validation_cadence=3),
                canonical_history=saved["history"], best_ref=saved["best"],
                train_gen=train_gen2, run_manifest=resumed_rm,
                run_manifest_path=art / "run_manifest.json", limit_batches=None,
                pause_after_epoch=None, validation_cadence=3, use_wandb=False,
                tqdm_fn=lambda x, **k: x,
            )
            self.assertEqual(outcome2, "completed")
            self.assertEqual(len(history2), 4)
            validation_ran_flags = [row["validation_ran"] for row in history2]
            self.assertEqual(validation_ran_flags, [True, False, True, True])
            last2 = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            self.assertEqual(last2["completed_epoch"], 3)
            ckpt_mod.verify_referenced_best(art, last2)  # final best still verifies


class TestResumeGateStatusTransition(unittest.TestCase):
    """The run_epochs-only interrupted/resumed test above proves epoch-level
    mathematical equivalence, but it does not exercise the CLI/run_manifest
    GATE itself -- the sequence of checks main() runs before it ever gets to
    run_epochs (check_output_dir_safety, bootstrap_run_manifest's schema
    validation and check_resumable_status call, persist_or_verify_data_
    provenance). This class calls those exact functions, in the exact order
    main() does, to prove the real status transition a smoke run relies on:
    initialized/running -> paused_for_smoke -> --resume accepted."""

    def test_paused_for_smoke_resumes_through_the_real_gate(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)

            # ---- Fresh run: exactly main()'s own bootstrap + data_verified sequence ----
            train.check_output_dir_safety(art, resume=False)
            rm = train.bootstrap_run_manifest(
                art / "run_manifest.json", resume=False,
                invocation_record=_invocation_record(), run_kind="smoke",
                git_commit="test-commit", git_dirty=False,
                validation_cadence=1,
            )
            self.assertEqual(rm["status"], "initialized")
            manifest_record = _fixed_manifest_record()
            taxonomy_record = _fixed_taxonomy_record()
            val_split_record = _fixed_val_split_record(art)
            train.persist_or_verify_data_provenance(
                rm, resume=False, manifest_record=manifest_record,
                taxonomy_record=taxonomy_record, val_split_record=val_split_record,
            )
            rm["status"] = "running"
            train._write_json_atomic(art / "run_manifest.json", rm)

            # ---- Commit one real epoch via checkpoint.commit_epoch, then pause
            # (mirrors exactly what run_epochs does at a real pause boundary) ----
            provenance = _fixed_provenance(run_kind="smoke", limit_batches=2)
            best_ref = ckpt_mod.commit_epoch(
                art, epoch=0, is_best=True, model_state={"w": torch.tensor([1.0])},
                optimizer_state={}, provenance=provenance, resolved_config={"c": 1},
                rng_state={}, train_generator_state=torch.Generator().get_state(),
                metrics={"top1": 0.5, "top3": 0.7}, canonical_history=[],
                history_row={"epoch": 0}, previous_best=None,
            )
            rm["status"] = "paused_for_smoke"
            rm["last_completed_epoch"] = 0
            rm["best"] = best_ref
            run_manifest_schema.validate_run_manifest(rm, stage="epoch_committed")
            train._write_json_atomic(art / "run_manifest.json", rm)

            # ---- Simulate a brand-new process resuming: the REAL gate main() runs ----
            train.check_output_dir_safety(art, resume=True)  # must not raise: dir is nonempty
            resumed_rm = train.bootstrap_run_manifest(
                art / "run_manifest.json", resume=True,
                invocation_record=_invocation_record(resume=True), run_kind="smoke",
                git_commit="test-commit", git_dirty=False,
                validation_cadence=1,
            )
            self.assertEqual(resumed_rm["status"], "running")  # flipped back by bootstrap
            self.assertEqual(len(resumed_rm["invocations"]), 2)

            # persist_or_verify_data_provenance must ACCEPT identical re-derived records.
            train.persist_or_verify_data_provenance(
                resumed_rm, resume=True, manifest_record=manifest_record,
                taxonomy_record=taxonomy_record, val_split_record=val_split_record,
            )  # must not raise

    def test_completed_status_still_refuses_resume_through_the_real_gate(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            train.check_output_dir_safety(art, resume=False)
            rm = train.bootstrap_run_manifest(
                art / "run_manifest.json", resume=False,
                invocation_record=_invocation_record(), run_kind="smoke",
                git_commit="test-commit", git_dirty=False,
                validation_cadence=1,
            )
            train.persist_or_verify_data_provenance(
                rm, resume=False, manifest_record=_fixed_manifest_record(),
                taxonomy_record=_fixed_taxonomy_record(),
                val_split_record=_fixed_val_split_record(art),
            )
            rm["status"] = "completed"
            rm["final_artifact_hashes"] = {name: "a" * 64 for name in
                                           run_manifest_schema.FINAL_ARTIFACT_NAMES}
            rm["last_completed_epoch"] = 0
            rm["best"] = {"epoch": 0, "metrics": {"top1": 0.5, "top3": 0.7},
                         "filename": "checkpoint_best_epoch_000.pth", "sha256": "a" * 64}
            run_manifest_schema.validate_run_manifest(rm, stage="completed")
            train._write_json_atomic(art / "run_manifest.json", rm)

            train.check_output_dir_safety(art, resume=True)  # dir nonempty -- allowed
            with self.assertRaises(SystemExit) as cm:
                train.bootstrap_run_manifest(
                    art / "run_manifest.json", resume=True,
                    invocation_record=_invocation_record(resume=True), run_kind="smoke",
                    git_commit="test-commit", git_dirty=False,
                    validation_cadence=1,
                )
            self.assertIn("completed", str(cm.exception))

    def test_resume_rejects_disagreeing_manifest_record(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            train.check_output_dir_safety(art, resume=False)
            rm = train.bootstrap_run_manifest(
                art / "run_manifest.json", resume=False,
                invocation_record=_invocation_record(), run_kind="smoke",
                git_commit="test-commit", git_dirty=False,
                validation_cadence=1,
            )
            train.persist_or_verify_data_provenance(
                rm, resume=False, manifest_record=_fixed_manifest_record(),
                taxonomy_record=_fixed_taxonomy_record(),
                val_split_record=_fixed_val_split_record(art),
            )
            rm["status"] = "paused_for_smoke"
            train._write_json_atomic(art / "run_manifest.json", rm)

            resumed_rm = train.bootstrap_run_manifest(
                art / "run_manifest.json", resume=True,
                invocation_record=_invocation_record(resume=True), run_kind="smoke",
                git_commit="test-commit", git_dirty=False,
                validation_cadence=1,
            )
            tampered_manifest_record = {**_fixed_manifest_record(), "rows": 999}
            with self.assertRaises(SystemExit) as cm:
                train.persist_or_verify_data_provenance(
                    resumed_rm, resume=True, manifest_record=tampered_manifest_record,
                    taxonomy_record=_fixed_taxonomy_record(),
                    val_split_record=_fixed_val_split_record(art),
                )
            self.assertIn("manifest", str(cm.exception))

    def test_schema_v1_run_manifest_cannot_be_resumed(self):
        # A schema-v1 manifest (no validation_cadence field at all, implicit
        # cadence=1) is readable (see test_train_harness.py's evaluate.py-
        # binding tests) but must never be resumable under this harness.
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            v1_manifest = {
                "run_manifest_schema_version": 1,
                "status": "running", "run_kind": "full",
                "git_head": "abc123", "git_dirty": False,
                "invocations": [_invocation_record()],
                "started_at_utc": "2026-01-01T00:00:00Z",
                "updated_at_utc": "2026-01-01T00:00:00Z",
                "finished_at_utc": None, "final_artifact_hashes": None,
                "manifest": _fixed_manifest_record(),
                "taxonomy_source": _fixed_taxonomy_record(),
                "val_split": _fixed_val_split_record(art),
                "last_completed_epoch": 2,
                "best": {"epoch": 1, "metrics": {"top1": 0.5, "top3": 0.7},
                        "filename": "checkpoint_best_epoch_001.pth", "sha256": "a" * 64},
            }
            self.assertNotIn("validation_cadence", v1_manifest)
            run_manifest_schema.validate_run_manifest(v1_manifest, stage="epoch_committed")  # readable
            art.mkdir(parents=True, exist_ok=True)
            train._write_json_atomic(art / "run_manifest.json", v1_manifest)

            with self.assertRaises(SystemExit) as cm:
                train.bootstrap_run_manifest(
                    art / "run_manifest.json", resume=True,
                    invocation_record=_invocation_record(resume=True), run_kind="full",
                    git_commit="abc123", git_dirty=False, validation_cadence=3,
                )
            message = str(cm.exception)
            self.assertIn("run_manifest_schema_version", message)
            self.assertIn("1", message)


if __name__ == "__main__":
    unittest.main()
