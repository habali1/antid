#!/usr/bin/env python3
"""test_parity_check_v2.py — unit + integration tests for parity_check_v2.py.

Never touches the real northeast_v1_b4_dev_v2 run or any frozen evaluation
set. The integration fixture uses a tiny, real (not mocked) exported ONNX
graph and a matching tiny PyTorch model so the CPU-provider/profiling and
ONNX-execution machinery is genuinely exercised, not stubbed out.

Run directly: python test_parity_check_v2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import parity_check_v2 as pcv2  # noqa: E402
from data import Sample  # noqa: E402
from report_dev_metrics import resolve_pinned_val_split  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


# ----------------------------------------------------------------- fixtures
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TinyParityModel(nn.Module):
    """No hidden state beyond a fixed identity-like linear map: channel-mean
    (3,) -> embedding (3,). Deterministic, no seeding needed."""

    def __init__(self, embedding_dim: int = 3):
        super().__init__()
        self.fc = nn.Linear(3, embedding_dim, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(embedding_dim, 3))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class _ExportWrapper(nn.Module):
    def __init__(self, model: TinyParityModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.embed(x)


def _export_tiny_onnx(model: TinyParityModel, out_path: Path, image_size: int) -> None:
    wrapper = _ExportWrapper(model).eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    kwargs = dict(
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(out_path), **kwargs)


def _write_solid_png(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), rgb).save(path, format="PNG")


def _write_manifest_csv(path: Path, rows: list[dict]) -> None:
    import csv
    fields = ["slug", "photo_id", "species", "common_name", "taxon_id", "lat", "lon",
             "split", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})


def build_parity_fixture(tmp_path: Path, *, mismatched_normalize: bool = False):
    """A complete, self-consistent, 2-species COMPLETED run: 2 train + 1 val
    image per species, a real tiny exported ONNX backbone matching a real
    tiny PyTorch model. cfg uses image_size=380 (matching api/inference.py's
    hardcoded contract) and ImageNet normalization unless
    mismatched_normalize=True, which deliberately breaks preprocessing
    parity (different normalization constants, same shape) without needing
    to mock anything."""
    local_data_dir = tmp_path / "clean"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    colors = {"alpha-ant": (255, 0, 0), "beta-ant": (0, 255, 0)}
    photo_ids = {"alpha-ant": ["train1", "train2", "val1"],
                "beta-ant": ["train1", "train2", "val1"]}
    splits = {"train1": "train", "train2": "train", "val1": "val"}

    rows = []
    for slug, color in colors.items():
        for pid in photo_ids[slug]:
            img_path = local_data_dir / slug / f"{pid}.png"
            _write_solid_png(img_path, color)
            rows.append({
                "slug": slug, "photo_id": pid,
                "species": "Alpha ant" if slug == "alpha-ant" else "Beta ant",
                "common_name": "", "taxon_id": "", "lat": "", "lon": "",
                "split": splits[pid], "sha256": _sha256_file(img_path),
            })

    manifest_csv = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest_csv, rows)
    manifest_sha256 = _sha256_file(manifest_csv)

    committed_taxonomy = {
        "0": {"species_name": "Alpha ant", "common_name": None, "taxon_id": None,
             "slug": "alpha-ant", "genus": "Alpha"},
        "1": {"species_name": "Beta ant", "common_name": None, "taxon_id": None,
             "slug": "beta-ant", "genus": "Beta"},
    }
    taxonomy_json = tmp_path / "taxonomy.json"
    taxonomy_json.write_text(json.dumps(committed_taxonomy))
    taxonomy_sha256 = _sha256_file(taxonomy_json)

    val_split_record = {
        "seed": 42, "val_fraction": 0.5, "n_total": 6, "n_train": 4, "n_val": 2,
        "train": ["alpha-ant/train1", "alpha-ant/train2",
                 "beta-ant/train1", "beta-ant/train2"],
        "val": ["alpha-ant/val1", "beta-ant/val1"],
    }
    val_split_bytes = (json.dumps(val_split_record, indent=2, sort_keys=True) + "\n").encode()
    (run_dir / "val_split.json").write_bytes(val_split_bytes)
    val_split_sha256 = _sha256_bytes(val_split_bytes)

    normalize = ({"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]} if mismatched_normalize
                else {"mean": IMAGENET_MEAN, "std": IMAGENET_STD})
    cfg = {
        "model": {"backbone": "tiny-parity-model", "embedding_dim": 3, "pretrained": False},
        "dropout": 0.0, "batch_size": 4, "image_size": 380, "normalize": normalize,
    }

    model = TinyParityModel(embedding_dim=3)
    torch.save({"model": model.state_dict(), "config": cfg,
               "provenance": {
                   "manifest_sha256": manifest_sha256, "taxonomy_sha256": taxonomy_sha256,
                   "val_split_sha256": val_split_sha256, "resolved_config_sha256": "c" * 64,
                   "backbone": "tiny-parity-model", "num_classes": 2, "git_commit": "f" * 40,
                   "numerical_policy": {}, "run_kind": "full", "limit_batches": None,
                   "wandb_enabled": False, "validation_cadence": 1,
               },
               "best_epoch": 0}, run_dir / "model.pth")

    protos = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]], dtype=np.float32)
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    np.save(run_dir / "prototypes.npy", protos)

    _export_tiny_onnx(model, run_dir / "backbone.onnx", image_size=380)

    (run_dir / "taxonomy.json").write_text(json.dumps(committed_taxonomy))
    (run_dir / "geo_index.json").write_text(json.dumps({"cell_size_deg": 1.0, "cells": {}}))
    (run_dir / "eval.json").write_text(json.dumps(
        {"overall": {"n": 2, "top1": 1.0, "top3": 1.0}, "per_species": {}}))

    final_artifact_hashes = {
        name: _sha256_file(run_dir / name)
        for name in ("model.pth", "prototypes.npy", "taxonomy.json", "geo_index.json",
                     "eval.json", "backbone.onnx", "val_split.json")
    }
    run_manifest = {
        "run_manifest_schema_version": 2, "status": "completed", "run_kind": "full",
        "validation_cadence": 1, "git_head": "f" * 40, "git_dirty": False,
        "invocations": [{"timestamp_utc": "2026-01-01T00:00:00Z", "argv": [],
                         "resume": False, "pause_after_epoch": None}],
        "started_at_utc": "2026-01-01T00:00:00Z", "updated_at_utc": "2026-01-01T00:00:00Z",
        "finished_at_utc": "2026-01-01T00:00:00Z",
        "manifest": {"path": str(manifest_csv), "sha256": manifest_sha256, "rows": 6},
        "taxonomy_source": {"path": str(taxonomy_json), "sha256": taxonomy_sha256,
                            "num_classes": 2},
        "val_split": {"path": str(run_dir / "val_split.json"), "sha256": val_split_sha256,
                     "n_train": 4, "n_val": 2, "n_total": 6},
        "last_completed_epoch": 0,
        "best": {"epoch": 0, "metrics": {"top1": 1.0, "top3": 1.0},
                 "filename": "checkpoint_best_epoch_000.pth", "sha256": "a" * 64},
        "final_artifact_hashes": final_artifact_hashes,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True))

    return run_dir, local_data_dir, manifest_csv, taxonomy_json


def _run(run_dir, local_data_dir, manifest_csv, taxonomy_json, **kwargs):
    device = torch.device("cuda")
    kwargs.setdefault("n_per_species", 1)
    kwargs.setdefault("database_url", None)
    with mock.patch.object(
        pcv2, "build_model",
        lambda cfg, num_classes: TinyParityModel(embedding_dim=cfg["model"]["embedding_dim"]),
    ):
        return pcv2.run_parity(
            run_dir=run_dir, local_data_dir=local_data_dir, manifest_csv=manifest_csv,
            taxonomy_json=taxonomy_json, device=device, **kwargs,
        )


# ----------------------------------------------------------------- unit tests
class TestNoFixedClassCountAssumption(unittest.TestCase):
    def test_contiguous_taxonomy_accepts_any_size(self):
        for n in (2, 3, 65, 100):
            taxonomy = {i: {"slug": f"s{i}"} for i in range(n)}
            pcv2.assert_contiguous_taxonomy(taxonomy)  # must not raise

    def test_contiguous_taxonomy_rejects_gap(self):
        taxonomy = {0: {"slug": "a"}, 2: {"slug": "b"}}
        with self.assertRaises(pcv2.ParityError):
            pcv2.assert_contiguous_taxonomy(taxonomy)

    def test_prototype_shape_derived_not_hardcoded(self):
        for n, dim in ((2, 3), (65, 1792), (65, 1280)):
            protos = np.zeros((n, dim), dtype=np.float32)
            pcv2.assert_prototype_shape(protos, n, dim)  # must not raise
            with self.assertRaises(pcv2.ParityError):
                pcv2.assert_prototype_shape(protos, n, dim + 1)


class TestSampleSelection(unittest.TestCase):
    def _taxonomy(self, n=2):
        return {0: {"slug": "alpha-ant"}, 1: {"slug": "beta-ant"}} if n == 2 else \
              {i: {"slug": f"species-{i}"} for i in range(n)}

    def _val_samples(self, n_per_species=2):
        samples = []
        for slug in ("alpha-ant", "beta-ant"):
            for i in range(n_per_species):
                samples.append(Sample(f"{slug}/val{i}.png", 0, slug))
        return samples

    def test_expected_total_derived_from_taxonomy(self):
        taxonomy = self._taxonomy()
        val_samples = self._val_samples(n_per_species=4)
        selected, expected_total = pcv2.select_parity_samples(val_samples, taxonomy, 42, 4)
        self.assertEqual(expected_total, len(taxonomy) * 4)
        self.assertEqual(len(selected), expected_total)
        # Never hardcoded to 200 or 260 -- a different taxonomy size scales cleanly.
        taxonomy3 = {0: {"slug": "a"}, 1: {"slug": "b"}, 2: {"slug": "c"}}
        val3 = [Sample(f"{s}/v{i}.png", 0, s) for s in "abc" for i in range(4)]
        selected3, expected3 = pcv2.select_parity_samples(val3, taxonomy3, 42, 4)
        self.assertEqual(expected3, 12)

    def test_missing_species_sample_rejected(self):
        taxonomy = self._taxonomy()
        val_samples = [Sample("alpha-ant/val0.png", 0, "alpha-ant")]  # beta-ant has none
        with self.assertRaisesRegex(pcv2.ParityError, "insufficient pinned-val images"):
            pcv2.select_parity_samples(val_samples, taxonomy, 42, 1)

    def test_train_key_never_reaches_selection(self):
        """resolve_pinned_val_split (already unit-tested in
        test_report_dev_metrics.py) is what guarantees only 'val' membership
        reaches select_parity_samples -- prove that guarantee end-to-end
        here: a mixed train+val sample list, once filtered through
        resolve_pinned_val_split, yields ONLY val samples for selection."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            record = {"n_val": 2, "val": ["alpha-ant/val0", "beta-ant/val0"]}
            payload = json.dumps(record).encode()
            (run_dir / "val_split.json").write_bytes(payload)
            run_manifest = {"val_split": {"sha256": _sha256_bytes(payload)}}

            mixed_samples = [
                Sample("alpha-ant/train0.png", 0, "alpha-ant"),  # train -- must be excluded
                Sample("alpha-ant/val0.png", 0, "alpha-ant"),
                Sample("beta-ant/val0.png", 1, "beta-ant"),
            ]
            val_samples, _ = resolve_pinned_val_split(run_dir, run_manifest, mixed_samples)
            self.assertEqual({s.slug for s in val_samples}, {"alpha-ant", "beta-ant"})
            self.assertTrue(all("val" in s.storage_path for s in val_samples))

            selected, expected_total = pcv2.select_parity_samples(
                val_samples, self._taxonomy(), 42, 1)
            self.assertEqual(len(selected), expected_total)
            self.assertTrue(all("val" in s.storage_path for s in selected))

    def test_selection_record_has_seed_keys_and_hash(self):
        taxonomy = self._taxonomy()
        val_samples = self._val_samples(n_per_species=1)
        selected, _ = pcv2.select_parity_samples(val_samples, taxonomy, 7, 1)
        record = pcv2.selection_record(selected, 7, 1)
        self.assertEqual(record["seed"], 7)
        self.assertEqual(len(record["selected_keys"]), 2)
        self.assertEqual(len(record["selection_hash_sha256"]), 64)


class TestManifestHashVerification(unittest.TestCase):
    def test_matching_hash_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "alpha-ant" / "val0.png"
            _write_solid_png(img_path, (1, 2, 3))
            manifest_csv = Path(tmp) / "manifest.csv"
            _write_manifest_csv(manifest_csv, [{
                "slug": "alpha-ant", "photo_id": "val0", "species": "Alpha ant",
                "sha256": _sha256_file(img_path),
            }])
            index = pcv2.load_manifest_sha256_index(manifest_csv)
            sample = Sample(str(img_path), 0, "alpha-ant")
            pcv2.verify_selected_image_hashes([sample], index)  # must not raise

    def test_mismatched_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "alpha-ant" / "val0.png"
            _write_solid_png(img_path, (1, 2, 3))
            manifest_csv = Path(tmp) / "manifest.csv"
            _write_manifest_csv(manifest_csv, [{
                "slug": "alpha-ant", "photo_id": "val0", "species": "Alpha ant",
                "sha256": "0" * 64,
            }])
            index = pcv2.load_manifest_sha256_index(manifest_csv)
            sample = Sample(str(img_path), 0, "alpha-ant")
            with self.assertRaisesRegex(pcv2.ParityError, "hash mismatch"):
                pcv2.verify_selected_image_hashes([sample], index)


class TestOldThresholdAbsent(unittest.TestCase):
    def test_source_has_no_060_threshold_constant_or_gate_logic(self):
        """The module docstring legitimately MENTIONS the old script's 0.60
        threshold (to explain why this tool doesn't reuse it) -- what must
        never exist is an actual constant, comparison, or gate function
        using it."""
        source = Path(pcv2.__file__).read_text(encoding="utf-8")
        self.assertNotIn("GATE_THRESHOLD", source)
        self.assertNotIn("gate_reject", source)
        self.assertNotIn("gate_disagree", source)
        self.assertNotIn("= 0.60", source)
        self.assertNotIn("< 0.60", source)

    def test_report_keys_never_mention_gate(self):
        import inspect
        src = inspect.getsource(pcv2.run_parity)
        for forbidden in ("gate", "threshold", "calibration"):
            self.assertNotIn(forbidden, src.lower())


class TestOrtNodePlacementRejection(unittest.TestCase):
    def test_non_cpu_provider_raises(self):
        fake_events = [
            {"cat": "Node", "name": "Conv_0", "args": {"provider": "CPUExecutionProvider"}},
            {"cat": "Node", "name": "Relu_1", "args": {"provider": "CUDAExecutionProvider"}},
        ]

        class FakeSession:
            def get_providers(self):
                return ["CPUExecutionProvider"]

            def get_inputs(self):
                class _In:
                    name = "input"
                return [_In()]

            def run(self, *a, **k):
                return [np.zeros((1, 3), dtype=np.float32)]

            def end_profiling(self):
                p = Path(tempfile.mkstemp(suffix=".json")[1])
                p.write_text(json.dumps(fake_events))
                return str(p)

        with mock.patch("onnxruntime.InferenceSession", return_value=FakeSession()):
            with self.assertRaisesRegex(pcv2.ParityError, "outside CPUExecutionProvider"):
                pcv2.confirm_cpu_only_via_profiling(
                    Path("unused.onnx"), [np.zeros((1, 3, 4, 4), dtype=np.float32)])


class TestPercentilesWithMin(unittest.TestCase):
    def test_includes_min(self):
        result = pcv2.percentiles_with_min([0.1, 0.5, 0.9])
        self.assertIn("min", result)
        self.assertEqual(result["min"], 0.1)
        self.assertEqual(result["max"], 0.9)


class TestSourceCanonicalLfSha256(unittest.TestCase):
    """This repo has no .gitattributes and core.autocrlf=true, so the same
    committed source can check out as LF or CRLF -- the generator binding
    must be line-ending independent."""

    def test_lf_and_crlf_produce_the_same_hash(self):
        content = 'print("hello")\nprint("world")\n'
        with tempfile.TemporaryDirectory() as tmp:
            lf_path = Path(tmp) / "lf.py"
            crlf_path = Path(tmp) / "crlf.py"
            lf_path.write_bytes(content.encode("utf-8"))
            crlf_path.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))
            self.assertNotEqual(_sha256_file(lf_path), _sha256_file(crlf_path),
                               "fixture sanity check: raw bytes must actually differ")
            self.assertEqual(pcv2.source_canonical_lf_sha256(lf_path),
                             pcv2.source_canonical_lf_sha256(crlf_path))

    def test_lone_cr_also_normalized(self):
        content = 'a = 1\nb = 2\n'
        with tempfile.TemporaryDirectory() as tmp:
            lf_path = Path(tmp) / "lf.py"
            cr_path = Path(tmp) / "cr.py"
            lf_path.write_bytes(content.encode("utf-8"))
            cr_path.write_bytes(content.replace("\n", "\r").encode("utf-8"))
            self.assertEqual(pcv2.source_canonical_lf_sha256(lf_path),
                             pcv2.source_canonical_lf_sha256(cr_path))

    def test_actual_content_change_produces_different_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.py"
            p2 = Path(tmp) / "b.py"
            p1.write_text('x = 1\n', encoding="utf-8")
            p2.write_text('x = 2\n', encoding="utf-8")
            self.assertNotEqual(pcv2.source_canonical_lf_sha256(p1),
                                pcv2.source_canonical_lf_sha256(p2))


# ----------------------------------------------------------- integration tests
@unittest.skipUnless(CUDA_AVAILABLE, "requires CUDA for the PyTorch reference path")
class TestRunParityIntegration(unittest.TestCase):
    def test_success_preprocessing_byte_identical_and_no_gate_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            result = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)

            self.assertTrue(result["preprocessing_parity"]["byte_identical"])
            self.assertEqual(result["preprocessing_parity"]["max_abs_diff"], 0.0)
            self.assertIn("model_parity", result)
            self.assertEqual(result["model_parity"]["n_images"], 2)
            self.assertIn("min", result["model_parity"]["max_cosine_abs_divergence"])
            self.assertTrue(
                result["ort_provider_evidence"]["registered_providers"]
                == ["CPUExecutionProvider"])
            self.assertFalse(result["ort_provider_evidence"]["any_node_executed_outside_cpu"])
            self.assertTrue(result["onnx_determinism"]["all_bit_identical"])
            dumped = json.dumps(result).lower()
            self.assertNotIn("gate", dumped)
            self.assertNotIn("threshold", dumped)
            self.assertNotIn("calibration_v1", dumped)

    def test_candidate_training_git_head_comes_from_run_manifest_not_workspace_head(self):
        """The candidate's own recorded training commit ('f'*40 in the
        fixture's run_manifest) must appear verbatim -- NOT this workspace's
        actual current git HEAD, which is a completely different value."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            with mock.patch.object(pcv2, "get_workspace_git_state",
                                   return_value=("deadbeef" * 5, True)):
                result = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)
            self.assertEqual(result["candidate_training_git_head"], "f" * 40)
            self.assertEqual(result["candidate_training_git_dirty"], False)
            self.assertNotEqual(result["candidate_training_git_head"], "deadbeef" * 5)
            self.assertNotIn("workspace_git_head", result)
            self.assertNotIn("workspace_git_dirty", result)
            self.assertNotIn("git_commit", result.get("environment", {}))

    def test_workspace_git_state_does_not_affect_deterministic_content(self):
        """Changing the ambient workspace HEAD/dirty state between two
        otherwise-identical runs must not change a single byte of
        deterministic_content -- only generation_metadata may reflect it."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))

            with mock.patch.object(pcv2, "get_workspace_git_state",
                                   return_value=("1111111111111111111111111111111111111111",
                                                 False)):
                result_a = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)
                report_a = pcv2.build_full_report(result_a)

            with mock.patch.object(pcv2, "get_workspace_git_state",
                                   return_value=("2222222222222222222222222222222222222222",
                                                 True)):
                result_b = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)
                report_b = pcv2.build_full_report(result_b)

            self.assertEqual(pcv2.serialize_deterministic(result_a),
                             pcv2.serialize_deterministic(result_b))
            # But generation_metadata DOES record the (different) ambient state.
            self.assertNotEqual(report_a["generation_metadata"]["workspace_git_head"],
                                report_b["generation_metadata"]["workspace_git_head"])
            self.assertEqual(report_a["generation_metadata"]["workspace_git_head"],
                             "1111111111111111111111111111111111111111")
            self.assertEqual(report_b["generation_metadata"]["workspace_git_dirty"], True)

    def test_generator_hash_change_invalidates_verify(self):
        """A changed generator source hash (i.e. the tool itself changed)
        must change deterministic_content and cause --verify to fail --
        this is the deliberate generator binding inside deterministic_content
        (not just generation_metadata, which --verify ignores)."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            result = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)
            report = pcv2.build_full_report(result)
            existing = Path(tmp) / "committed_parity.json"
            existing.write_text(json.dumps(report, indent=2, sort_keys=True))

            # Sanity: an unmodified regeneration verifies clean.
            pcv2.verify_byte_identical(result, existing)

            tampered = json.loads(json.dumps(result))
            tampered["generator"]["source_canonical_lf_sha256"] = "0" * 64
            with self.assertRaises(pcv2.ParityError):
                pcv2.verify_byte_identical(tampered, existing)

    def test_verify_validates_schema_before_comparing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            result = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)

            malformed = Path(tmp) / "malformed.json"
            malformed.write_text(json.dumps({"report_schema_version": 999}))
            with self.assertRaisesRegex(pcv2.ParityError, "report_schema_version"):
                pcv2.verify_byte_identical(result, malformed)

            missing_content = Path(tmp) / "missing_content.json"
            missing_content.write_text(json.dumps(
                {"report_schema_version": pcv2.REPORT_SCHEMA_VERSION}))
            with self.assertRaisesRegex(pcv2.ParityError, "deterministic_content"):
                pcv2.verify_byte_identical(result, missing_content)

    def test_preprocessing_contract_disagreement_unit(self):
        """Direct unit test of the contract-agreement assertion, independent
        of the full model/ONNX pipeline."""
        matching = {
            "image_size": 380, "normalize": {"mean": [0.485, 0.456, 0.406],
                                             "std": [0.229, 0.224, 0.225]},
            "interpolation_effective": "bilinear",
        }
        pcv2.assert_preprocessing_contract_agreement(matching)  # must not raise

        mismatched = {
            "image_size": 380, "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "interpolation_effective": "bilinear",
        }
        with self.assertRaisesRegex(pcv2.ParityError, "preprocessing-contract disagreement"):
            pcv2.assert_preprocessing_contract_agreement(mismatched)

    def test_preprocessing_mismatch_aborts_before_model_parity(self):
        """A candidate whose recorded normalization disagrees with
        api/inference.py's real constants must be caught by the
        contract-level check -- before even the empirical tensor
        measurement, let alone any model inference."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp), mismatched_normalize=True)
            with self.assertRaisesRegex(
                    pcv2.ParityError, "preprocessing-contract disagreement"):
                _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)

    def test_database_url_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            from data_provenance import DataIntegrityError
            with self.assertRaises(DataIntegrityError):
                _run(run_dir, local_data_dir, manifest_csv, taxonomy_json,
                    database_url="postgres://fake")

    def test_incomplete_run_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            rm_path = run_dir / "run_manifest.json"
            rm = json.loads(rm_path.read_text())
            rm["status"] = "running"
            rm_path.write_text(json.dumps(rm))
            with self.assertRaises(Exception):  # report_dev_metrics.DevReportError
                _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)

    def test_artifact_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            with (run_dir / "eval.json").open("ab") as fh:
                fh.write(b" ")
            with self.assertRaises(Exception):
                _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)

    def test_verify_mode_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_parity_fixture(
                Path(tmp))
            result_1 = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)
            report_1 = pcv2.build_full_report(result_1)
            existing = Path(tmp) / "committed_parity.json"
            existing.write_text(json.dumps(report_1, indent=2, sort_keys=True))

            result_2 = _run(run_dir, local_data_dir, manifest_csv, taxonomy_json)
            pcv2.verify_byte_identical(result_2, existing)  # must not raise

            corrupted = json.loads(existing.read_text())
            corrupted["deterministic_content"]["sample_selection"]["seed"] = 999
            existing.write_text(json.dumps(corrupted, indent=2, sort_keys=True))
            with self.assertRaises(pcv2.ParityError):
                pcv2.verify_byte_identical(result_2, existing)


if __name__ == "__main__":
    unittest.main()
