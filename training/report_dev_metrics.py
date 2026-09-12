#!/usr/bin/env python3
"""report_dev_metrics.py — deterministic, hash-bound development-split report
for a COMPLETED training run.

Reads ONLY the pinned development validation split recorded by the run's own
run_manifest.json / val_split.json. Never reads benchmark_v1, calibration_v1,
unknown_test_v1, northeast_final_test_v1, or any other frozen evaluation set.

Every report this script writes is labeled:
    "pinned development-split analysis; not benchmark or final-test evidence."

Pipeline (fails closed at the first problem, writes nothing on failure):
  0. Require this repo's own git tree to resolve HEAD and be clean (the
     report's generator/git_commit binding must name the exact clean commit
     that produced it) -- and that NORTHEAST_NEW_SPECIES_SLUGS itself still
     names exactly 15 unique slugs.
  1. Load + schema-validate run_manifest.json; require status == "completed".
  2. Verify every run_manifest.final_artifact_hashes entry against the actual
     on-disk file.
  3. Verify the checkpoint's embedded provenance agrees with run_manifest's
     own manifest/taxonomy_source/val_split records.
  4. Resolve the manifest + committed taxonomy explicitly (hash-bound), then
     hash-verify every referenced image's bytes; require every one of the
     15 new-species slugs to actually be present in this run's taxonomy.
  5. Assert every taxonomy entry used has a nonblank genus.
  6. Resolve the pinned val split from val_split.json (hash-verified).
  7. Load the model + prototypes, run raw-cosine (no geo, no gate) inference
     over the pinned val split, and cross-check the result THREE independent
     ways: against evaluate.topk_accuracy() run fresh over the same loader,
     against run_manifest.best.metrics, and against the shipped eval.json.
     Any disagreement aborts before anything is written.
  8. Compute genus-oriented metrics from the same raw top-3 predictions;
     require the new_15/legacy_50/all_65 val-image counts to exactly match
     the frozen 65-species catalog's known shape.
  9. Serialize deterministically (sorted keys, indent=2, LF, no embedded
     absolute machine paths, no timestamp).

(Steps 0's frozen-catalog checks and step 8's group-denominator check are
skippable via assert_frozen_catalog_shape=False / require_clean_git=False,
used only by this module's own tests against small synthetic fixtures.)

Usage:
    python report_dev_metrics.py --run-dir artifacts/northeast_v1_b4_dev_v2 \\
        --local-data-dir ../data/clean \\
        --out reports/northeast_v1_b4_dev_report.json

    python report_dev_metrics.py --run-dir artifacts/northeast_v1_b4_dev_v2 \\
        --local-data-dir ../data/clean --verify \\
        --existing reports/northeast_v1_b4_dev_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import numerics
import run_manifest_schema
from data import AntDataset, resolve_interpolation
from data_provenance import (
    NORTHEAST_NEW_SPECIES_SLUGS,
    load_explicit_manifest_source,
    verify_image_bytes,
)
from evaluate import (
    topk_accuracy,
    verify_run_manifest_matches_checkpoint_provenance,
)
from model import AntIDModel
from train import get_git_state, require_clean_git_state

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

LABEL = "pinned development-split analysis; not benchmark or final-test evidence."

# Bumped whenever the report's FIELD SHAPE changes (not for metric-value
# changes, which are just different data under the same shape).
REPORT_SCHEMA_VERSION = 1
GENERATOR_NAME = "report_dev_metrics.py"
GENERATOR_VERSION = "1.0.0"

FINAL_ARTIFACT_NAMES = (
    "model.pth", "prototypes.npy", "taxonomy.json", "geo_index.json",
    "eval.json", "backbone.onnx", "val_split.json",
)

# Frozen shape of the real 65-species Northeast catalog this report is built
# for (see data_provenance.py's own EXPECTED_* constants, which this
# deliberately mirrors) -- checked only when generate_report's
# assert_frozen_catalog_shape=True (the default; tests pass False for their
# small synthetic fixtures, which are not this catalog).
EXPECTED_NEW_SPECIES_COUNT = 15
EXPECTED_NEW_15_VAL_N = 600
EXPECTED_LEGACY_50_VAL_N = 1996
EXPECTED_ALL_65_VAL_N = 2596

# The known, previously-established B4 result for northeast_v1_b4_dev_v2
# (see TODO.md / docs/plans/northeast-expansion-v1.md). A one-time, explicit
# review assertion -- call assert_known_b4_aggregate_counts(report)
# separately after generation; generate_report() itself never hardcodes
# this, so the pipeline stays generic to whichever completed run it is
# pointed at.
KNOWN_B4_AGGREGATE_COUNTS = {
    "new_15": {"top1_correct": 397, "top3_correct": 483, "n": 600},
    "legacy_50": {"top1_correct": 1369, "top3_correct": 1668, "n": 1996},
    "all_65": {"top1_correct": 1766, "top3_correct": 2151, "n": 2596},
}


class DevReportError(RuntimeError):
    """A required invariant did not hold while building the dev report.
    Always fail closed: never write a partial or adjusted report."""


# ----------------------------------------------------------------- utilities
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_relative_posix(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        # Outside the repo (e.g. a test fixture in a temp dir) -- fall back to
        # just the directory name rather than embedding an absolute path.
        return resolved.name


# ------------------------------------------------------------ run_manifest
def load_completed_run_manifest(run_dir: Path) -> tuple[dict, str]:
    """Load, schema-validate, and require status == 'completed'. Returns
    (run_manifest, run_manifest_sha256_of_the_file_bytes)."""
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise DevReportError(f"{path} does not exist.")
    raw = path.read_bytes()
    run_manifest_sha256 = _sha256_bytes(raw)
    try:
        run_manifest = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DevReportError(f"{path} is not valid JSON: {e}") from e
    try:
        run_manifest_schema.validate_run_manifest(run_manifest, stage="completed")
    except run_manifest_schema.RunManifestValidationError as e:
        raise DevReportError(f"{path} failed schema validation: {e}") from e
    if run_manifest.get("status") != "completed":
        raise DevReportError(
            f"{path} status is {run_manifest.get('status')!r}, expected 'completed' -- "
            f"refusing to build a development report for a run that has not finished."
        )
    return run_manifest, run_manifest_sha256


def verify_final_artifact_hashes(run_dir: Path, run_manifest: dict) -> dict[str, str]:
    """Recompute sha256 for every file run_manifest.final_artifact_hashes
    names and require an exact match. Returns the verified hash dict."""
    expected = run_manifest.get("final_artifact_hashes") or {}
    missing_names = [n for n in FINAL_ARTIFACT_NAMES if n not in expected]
    if missing_names:
        raise DevReportError(
            f"run_manifest.final_artifact_hashes is missing entries for {missing_names}."
        )
    mismatches = []
    for name in FINAL_ARTIFACT_NAMES:
        path = run_dir / name
        if not path.exists():
            mismatches.append(f"{name}: file does not exist at {path}")
            continue
        actual = _sha256_file(path)
        if actual != expected[name]:
            mismatches.append(
                f"{name}: on-disk sha256 {actual} != run_manifest {expected[name]}"
            )
    if mismatches:
        raise DevReportError(
            "artifact hash mismatch: " + "; ".join(mismatches)
        )
    return dict(expected)


# ------------------------------------------------------------------- data
def assert_new_species_constant_shape(new_species_slugs: frozenset) -> None:
    """The new-15-species constant itself must contain exactly 15 unique
    slugs. Checked unconditionally (this is a property of the imported
    constant, not of any particular run's data), so a future edit that
    accidentally drops/duplicates a slug is caught immediately."""
    if len(new_species_slugs) != EXPECTED_NEW_SPECIES_COUNT:
        raise DevReportError(
            f"NORTHEAST_NEW_SPECIES_SLUGS must contain exactly "
            f"{EXPECTED_NEW_SPECIES_COUNT} unique slugs, got {len(new_species_slugs)}."
        )


def assert_new_species_present_in_taxonomy(taxonomy: dict, new_species_slugs: frozenset) -> None:
    """Every new-15 slug must actually resolve to a species in this run's
    taxonomy -- a missing or misspelled slug must fail loudly rather than
    silently being counted as legacy (which would corrupt the new_15/
    legacy_50 aggregate split without any visible error)."""
    taxonomy_slugs = {entry["slug"] for entry in taxonomy.values()}
    missing = sorted(new_species_slugs - taxonomy_slugs)
    if missing:
        raise DevReportError(
            f"NORTHEAST_NEW_SPECIES_SLUGS names species not present in this run's "
            f"taxonomy: {missing}."
        )


def assert_group_denominators(aggregates: dict) -> None:
    """Fail closed unless the new_15/legacy_50/all_65 val-image counts are
    EXACTLY the frozen 65-species catalog's known shape. A silently
    misclassified species (see assert_new_species_present_in_taxonomy) would
    otherwise still slip through as a plausible-looking but wrong split."""
    expected = {"new_15": EXPECTED_NEW_15_VAL_N, "legacy_50": EXPECTED_LEGACY_50_VAL_N,
               "all_65": EXPECTED_ALL_65_VAL_N}
    mismatches = [
        f"{group}: expected n={want}, got n={aggregates[group]['n']}"
        for group, want in expected.items() if aggregates[group]["n"] != want
    ]
    if mismatches:
        raise DevReportError("group denominator mismatch: " + "; ".join(mismatches))


def assert_known_b4_aggregate_counts(report: dict) -> None:
    """One-time explicit review assertion (not part of generate_report's own
    fail-closed pipeline) that this report's aggregates exactly match the
    previously-established B4 result. Prefers integer top1_correct/
    top3_correct/n comparisons over comparing rounded decimal rates."""
    aggregates = report["aggregates"]
    mismatches = []
    for group, expected in KNOWN_B4_AGGREGATE_COUNTS.items():
        actual = aggregates[group]
        for field, want in expected.items():
            got = actual[field]
            if got != want:
                mismatches.append(f"{group}.{field}: expected {want}, got {got}")
    if mismatches:
        raise DevReportError(
            "known-B4-result mismatch: " + "; ".join(mismatches)
        )


def assert_taxonomy_has_genus(taxonomy: dict) -> None:
    """Every taxonomy entry a metric will be computed over must carry a
    nonblank genus. Isolated so it can be unit-tested without a real
    manifest/image pipeline."""
    for entry in taxonomy.values():
        genus = entry.get("genus")
        if not genus or not str(genus).strip():
            raise DevReportError(
                f"taxonomy entry {entry.get('slug')!r} has a missing/blank genus -- "
                f"refusing to compute genus metrics."
            )


def resolve_manifest_and_taxonomy(run_manifest: dict, local_data_dir: Path,
                                  manifest_csv: Path, taxonomy_json: Path,
                                  database_url: str | None):
    """Hash-verify the manifest/taxonomy/image bytes against run_manifest's
    own recorded records. Returns (samples, taxonomy, manifest_sha256,
    taxonomy_sha256)."""
    manifest_record = run_manifest["manifest"]
    taxonomy_record = run_manifest["taxonomy_source"]
    if manifest_record is None or taxonomy_record is None:
        raise DevReportError(
            "run_manifest has no recorded manifest/taxonomy_source -- this run did not "
            "use an explicit, hash-bound data source."
        )

    samples, taxonomy, manifest_sha256, taxonomy_sha256, _ = load_explicit_manifest_source(
        manifest_csv, local_data_dir, taxonomy_json,
        manifest_record["sha256"], taxonomy_record["sha256"], database_url=database_url,
    )
    assert_taxonomy_has_genus(taxonomy)
    verify_image_bytes(manifest_csv, local_data_dir)
    return samples, taxonomy, manifest_sha256, taxonomy_sha256


def resolve_pinned_val_split(run_dir: Path, run_manifest: dict, samples: list):
    """Hash-verify val_split.json against run_manifest's own recorded hash,
    then resolve exactly its pinned 'val' membership from `samples`. Isolated
    from manifest/image resolution so it is unit-testable with a plain
    in-memory sample list. Returns (val_samples, split_record)."""
    val_split_path = run_dir / "val_split.json"
    if not val_split_path.exists():
        raise DevReportError(f"{val_split_path} does not exist.")
    val_split_bytes = val_split_path.read_bytes()
    val_split_sha256 = _sha256_bytes(val_split_bytes)
    expected_val_split_sha256 = run_manifest["val_split"]["sha256"]
    if val_split_sha256 != expected_val_split_sha256:
        raise DevReportError(
            f"val-split mismatch: {val_split_path} sha256 {val_split_sha256} != "
            f"run_manifest.val_split.sha256 {expected_val_split_sha256}."
        )
    split_record = json.loads(val_split_bytes)
    val_keys = set(split_record["val"])
    val_samples = [s for s in samples
                  if f"{s.slug}/{Path(s.storage_path).stem}" in val_keys]
    if len(val_samples) != split_record["n_val"]:
        raise DevReportError(
            f"resolved {len(val_samples)} val samples from the pinned split, expected "
            f"{split_record['n_val']} (val-split mismatch)."
        )
    return val_samples, split_record


# ------------------------------------------------------------------ model
def build_model(cfg: dict, num_classes: int) -> AntIDModel:
    """Separated so tests can monkeypatch report_dev_metrics.AntIDModel with
    a tiny fixture model without downloading or constructing a real timm
    backbone."""
    return AntIDModel(
        num_classes=num_classes, backbone=cfg["model"]["backbone"], pretrained=False,
        dropout=cfg["dropout"], embedding_dim=cfg["model"]["embedding_dim"],
    )


@torch.no_grad()
def collect_raw_predictions(model, prototypes, loader, device):
    """One forward pass over `loader` (must be shuffle=False), returning:
      - per-image list of (true_label:int, top3:list[int]) in loader order
      - correct1/correct3/total per class (np.int64 arrays), for cross-check
        against evaluate.topk_accuracy()'s own accounting.
    Same algorithm as evaluate.topk_accuracy (cosine sim over L2-normalized
    prototypes/embeddings, torch.topk) -- re-derived here (not imported as a
    return value) because topk_accuracy does not expose raw per-image
    predictions, which genus metrics need.
    """
    import torch.nn as nn

    model.eval()
    protos = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
    protos = nn.functional.normalize(protos, dim=1)
    n_classes = protos.shape[0]
    k = min(3, n_classes)

    correct1 = np.zeros(n_classes, dtype=np.int64)
    correct3 = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)
    predictions: list[tuple[int, list[int]]] = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        emb = nn.functional.normalize(model.embed(imgs), dim=1)
        sims = emb @ protos.T
        top3 = sims.topk(k, dim=1).indices.cpu().numpy()
        labels_np = labels.numpy()
        for lbl, preds in zip(labels_np, top3):
            lbl = int(lbl)
            preds_list = [int(p) for p in preds]
            total[lbl] += 1
            if preds_list[0] == lbl:
                correct1[lbl] += 1
            if lbl in preds_list:
                correct3[lbl] += 1
            predictions.append((lbl, preds_list))

    return predictions, correct1, correct3, total


def _accuracy_block_matches(a: dict, b: dict) -> bool:
    """Exact equality check between two topk_accuracy()-shaped blocks
    (overall + per_species), restricted to the fields both carry."""
    if a["overall"]["n"] != b["overall"]["n"]:
        return False
    if a["overall"]["top1"] != b["overall"]["top1"]:
        return False
    if a["overall"]["top3"] != b["overall"]["top3"]:
        return False
    if set(a["per_species"]) != set(b["per_species"]):
        return False
    for slug, av in a["per_species"].items():
        bv = b["per_species"][slug]
        if av["n"] != bv["n"] or av["top1"] != bv["top1"] or av["top3"] != bv["top3"]:
            return False
    return True


def _raw_counts_to_block(correct1, correct3, total, taxonomy) -> dict:
    per_species = {}
    for idx in range(len(total)):
        slug = taxonomy[idx]["slug"]
        t = int(total[idx])
        per_species[slug] = {
            "n": t,
            "top1": float(correct1[idx] / t) if t else None,
            "top3": float(correct3[idx] / t) if t else None,
        }
    tot = int(total.sum())
    return {
        "overall": {
            "n": tot,
            "top1": float(correct1.sum() / tot) if tot else 0.0,
            "top3": float(correct3.sum() / tot) if tot else 0.0,
        },
        "per_species": per_species,
    }


# --------------------------------------------------------------- genus math
def _rate(count: int, total: int) -> dict:
    return {"count": count, "total": total, "rate": (count / total) if total else None}


def compute_genus_metrics(predictions: list[tuple[int, list[int]]], taxonomy: dict,
                          new_species_slugs: frozenset) -> dict:
    """From raw (true_label, top3_preds) pairs, compute genus_top1,
    genus_top3_any, wrong_species_but_correct_genus_top1, and
    top3_unanimous_true_genus -- overall, new-15-vs-legacy-50, per genus, and
    per species. Every rate carries its own numerator/denominator.
    """
    genus_of = {idx: taxonomy[idx]["genus"] for idx in taxonomy}
    slug_of = {idx: taxonomy[idx]["slug"] for idx in taxonomy}

    def new_stats() -> dict:
        return {
            "n": 0, "genus_top1": 0, "genus_top3_any": 0,
            "species_wrong_top1": 0, "species_wrong_but_genus_correct_top1": 0,
            "top3_unanimous_true_genus": 0,
        }

    overall = new_stats()
    group_stats = {"new_15": new_stats(), "legacy_50": new_stats()}
    per_species_stats: dict[str, dict] = {}
    per_genus_stats: dict[str, dict] = {}

    for true_label, preds in predictions:
        true_genus = genus_of[true_label]
        true_slug = slug_of[true_label]
        group = "new_15" if true_slug in new_species_slugs else "legacy_50"

        top1_pred = preds[0]
        top1_species_correct = (top1_pred == true_label)
        top1_genus_correct = (genus_of[top1_pred] == true_genus)
        any_genus_in_top3 = any(genus_of[p] == true_genus for p in preds)
        all_genus_in_top3 = all(genus_of[p] == true_genus for p in preds)

        for bucket in (overall, group_stats[group],
                      per_species_stats.setdefault(true_slug, new_stats()),
                      per_genus_stats.setdefault(true_genus, new_stats())):
            bucket["n"] += 1
            if top1_genus_correct:
                bucket["genus_top1"] += 1
            if any_genus_in_top3:
                bucket["genus_top3_any"] += 1
            if all_genus_in_top3:
                bucket["top3_unanimous_true_genus"] += 1
            if not top1_species_correct:
                bucket["species_wrong_top1"] += 1
                if top1_genus_correct:
                    bucket["species_wrong_but_genus_correct_top1"] += 1

    def finalize(bucket: dict, extra: dict | None = None) -> dict:
        n = bucket["n"]
        out = {
            "n": n,
            "genus_top1": _rate(bucket["genus_top1"], n),
            "genus_top3_any": _rate(bucket["genus_top3_any"], n),
            "top3_unanimous_true_genus": _rate(bucket["top3_unanimous_true_genus"], n),
            "wrong_species_but_correct_genus_top1": _rate(
                bucket["species_wrong_but_genus_correct_top1"], bucket["species_wrong_top1"]
            ),
        }
        if extra:
            out.update(extra)
        return out

    return {
        "overall": finalize(overall),
        "new_15_vs_legacy_50": {
            "new_15": finalize(group_stats["new_15"]),
            "legacy_50": finalize(group_stats["legacy_50"]),
        },
        "per_genus": {
            genus: finalize(stats) for genus, stats in sorted(per_genus_stats.items())
        },
        "per_species": {
            slug: finalize(stats, {"genus": taxonomy[idx]["genus"]})
            for idx in sorted(taxonomy)
            for slug in [taxonomy[idx]["slug"]]
            if slug in per_species_stats
            for stats in [per_species_stats[slug]]
        },
    }


# ------------------------------------------------------------- aggregates
def compute_group_aggregate(correct1, correct3, total, taxonomy, new_species_slugs) -> dict:
    def sums(indices):
        return (int(correct1[list(indices)].sum()), int(correct3[list(indices)].sum()),
               int(total[list(indices)].sum()))

    new_idx = [i for i in range(len(total)) if taxonomy[i]["slug"] in new_species_slugs]
    legacy_idx = [i for i in range(len(total)) if taxonomy[i]["slug"] not in new_species_slugs]
    all_idx = list(range(len(total)))

    def block(indices):
        c1, c3, n = sums(indices)
        return {"n": n, "top1": (c1 / n) if n else None, "top3": (c3 / n) if n else None,
               "top1_correct": c1, "top3_correct": c3}

    return {"new_15": block(new_idx), "legacy_50": block(legacy_idx), "all_65": block(all_idx)}


# ------------------------------------------------------------- selection nuance
def compute_selection_nuance(run_dir: Path, run_manifest: dict, top1_correct: int,
                             top1_n: int) -> dict | None:
    """Generic runner-up computation under the frozen selection rule (highest
    top1, tie -> highest top3, tie -> earliest epoch): rank every VALIDATED
    epoch in history.jsonl by that key and report the top two. The winner
    must equal run_manifest.best. Returns None if history.jsonl is absent."""
    history_path = run_dir / "history.jsonl"
    if not history_path.exists():
        return None
    rows = []
    with history_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    validated = [r for r in rows if r.get("validation_ran")]
    if len(validated) < 1:
        return None
    validated.sort(key=lambda r: (r["val_top1"], r["val_top3"], -r["epoch"]), reverse=True)
    best_row = validated[0]
    if best_row["epoch"] != run_manifest["best"]["epoch"]:
        raise DevReportError(
            f"history.jsonl's top-ranked validated epoch ({best_row['epoch']}) does not "
            f"match run_manifest.best.epoch ({run_manifest['best']['epoch']})."
        )

    def entry(row: dict, *, live_correct: int | None = None, live_n: int | None = None) -> dict:
        out = {
            "internal_epoch": row["epoch"], "human_epoch": row["epoch"] + 1,
            "top1": row["val_top1"], "top3": row["val_top3"],
        }
        if live_correct is not None:
            out["top1_correct"] = live_correct
            out["top1_n"] = live_n
            out["top1_correct_source"] = "live re-inference by this report"
        else:
            derived = round(row["val_top1"] * top1_n)
            out["top1_correct_derived"] = derived
            out["top1_n"] = top1_n
            out["top1_correct_source"] = (
                "derived from history.jsonl's recorded val_top1 -- not independently "
                "reproduced (no checkpoint remains for this superseded epoch)"
            )
        return out

    selected = entry(best_row, live_correct=top1_correct, live_n=top1_n)
    if len(validated) < 2:
        return {"selected": selected, "runner_up": None,
               "note": "only one validated epoch exists; no runner-up to compare."}

    runner_up_row = validated[1]
    runner_up = entry(runner_up_row)
    margin = selected["top1_correct"] - runner_up["top1_correct_derived"]
    return {
        "selected": selected,
        "runner_up": runner_up,
        "margin_top1_predictions": margin,
        "note": (
            "The selected epoch won mechanically under the frozen selection rule "
            "(highest top1, tie -> highest top3, tie -> earliest epoch); a margin this "
            "small is not evidence of a meaningful statistical improvement."
        ),
    }


# --------------------------------------------------------------- serialize
def serialize_report(report: dict) -> bytes:
    text = json.dumps(report, indent=2, sort_keys=True)
    return (text + "\n").replace("\r\n", "\n").encode("utf-8")


def _git_dirty_excluding(ignore_paths: frozenset[str]) -> bool | None:
    """Like train.get_git_state()'s dirty flag, but a change confined
    entirely to `ignore_paths` (repo-relative posix strings) does not count
    as dirty -- lets the not-yet-committed report file itself (which this
    very generation is about to write, or has already written for --verify)
    be present without blocking the git-clean gate meant for the SOURCE
    tree. None means git state could not be determined at all."""
    import subprocess
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        return None
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if "->" in path:  # renames: "old -> new"
            path = path.split("->")[-1].strip()
        path = path.strip('"').replace("\\", "/")
        if path not in ignore_paths:
            return True
    return False


# ----------------------------------------------------------------- generate
def generate_report(*, run_dir: Path, local_data_dir: Path, manifest_csv: Path,
                    taxonomy_json: Path, database_url: str | None, device: torch.device,
                    require_clean_git: bool = True, assert_frozen_catalog_shape: bool = True,
                    ignore_git_dirty_paths: frozenset[str] = frozenset(),
                    selection_rule: str = (
                        "highest raw-cosine top-1; tie -> highest top-3; "
                        "tie -> earliest epoch"
                    )) -> dict:
    """Run the full pipeline and return the deterministic report dict (does
    NOT write anything -- callers decide where/whether to write). Raises
    DevReportError (or lets a DataIntegrityError propagate) on any problem,
    before anything is computed that would need writing.

    require_clean_git: the generator's OWN source tree (this repo) must
    resolve a git HEAD and be clean before producing a report meant to be
    committed -- tests pass False since they don't care about this repo's
    live git state. ignore_git_dirty_paths lets the not-yet-committed report
    output itself (repo-relative posix path(s)) be present without failing
    this check -- the requirement is a clean SOURCE tree, not that the
    report has already been committed before it can be verified.
    assert_frozen_catalog_shape: enforce the real 65-species Northeast
    catalog's known new_15/legacy_50/all_65 shape -- tests pass False for
    their small synthetic fixtures.
    """
    numerics.apply_numerical_policy()

    git_commit, _ = get_git_state()
    if require_clean_git:
        git_dirty = _git_dirty_excluding(ignore_git_dirty_paths)
        require_clean_git_state(git_commit, git_dirty)

    assert_new_species_constant_shape(NORTHEAST_NEW_SPECIES_SLUGS)

    run_manifest, run_manifest_sha256 = load_completed_run_manifest(run_dir)
    final_hashes = verify_final_artifact_hashes(run_dir, run_manifest)

    samples, taxonomy, manifest_sha256, taxonomy_sha256 = resolve_manifest_and_taxonomy(
        run_manifest, local_data_dir, manifest_csv, taxonomy_json, database_url,
    )
    if assert_frozen_catalog_shape:
        assert_new_species_present_in_taxonomy(taxonomy, NORTHEAST_NEW_SPECIES_SLUGS)
    val_samples, split_record = resolve_pinned_val_split(run_dir, run_manifest, samples)
    num_classes = len(taxonomy)

    state = torch.load(run_dir / "model.pth", map_location=device, weights_only=False)
    cfg = state["config"]
    provenance = state["provenance"]
    verify_run_manifest_matches_checkpoint_provenance(run_manifest, provenance)
    effective_interpolation = cfg.get("interpolation") or "bilinear"
    resolve_interpolation(cfg)  # fail closed if the config value is unsupported

    model = build_model(cfg, num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    prototypes = np.load(run_dir / "prototypes.npy")
    expected_proto_shape = (num_classes, cfg["model"]["embedding_dim"])
    if tuple(prototypes.shape) != expected_proto_shape:
        raise DevReportError(
            f"prototypes.npy shape {tuple(prototypes.shape)} != expected {expected_proto_shape}."
        )
    proto_norms = np.linalg.norm(prototypes, axis=1)
    if not np.allclose(proto_norms, 1.0, atol=1e-4):
        raise DevReportError("prototypes.npy rows are not L2-normalized.")

    loader_for_official = DataLoader(AntDataset(val_samples, cfg, train=False),
                                     batch_size=cfg["batch_size"], shuffle=False)
    official = topk_accuracy(model, prototypes, loader_for_official, taxonomy, device)

    loader_for_raw = DataLoader(AntDataset(val_samples, cfg, train=False),
                                batch_size=cfg["batch_size"], shuffle=False)
    predictions, correct1, correct3, total = collect_raw_predictions(
        model, prototypes, loader_for_raw, device)
    own = _raw_counts_to_block(correct1, correct3, total, taxonomy)

    if not _accuracy_block_matches(official, own):
        raise DevReportError(
            "Reproduction mismatch: this report's raw-prediction recomputation does not "
            "exactly match evaluate.topk_accuracy()'s result over the same pinned val "
            "split. Refusing to write a report -- see requirement: reproduction must "
            "agree at the exact prediction-count level."
        )

    recorded_best = run_manifest["best"]["metrics"]
    if (official["overall"]["top1"] != recorded_best["top1"]
            or official["overall"]["top3"] != recorded_best["top3"]):
        raise DevReportError(
            f"Reproduction mismatch: freshly computed overall top1/top3 "
            f"({official['overall']['top1']}/{official['overall']['top3']}) does not "
            f"exactly match run_manifest.best.metrics "
            f"({recorded_best['top1']}/{recorded_best['top3']})."
        )

    eval_json_path = run_dir / "eval.json"
    if eval_json_path.exists():
        shipped_eval = json.loads(eval_json_path.read_text())
        if (shipped_eval["overall"]["top1"] != official["overall"]["top1"]
                or shipped_eval["overall"]["top3"] != official["overall"]["top3"]):
            raise DevReportError(
                "Reproduction mismatch: freshly computed overall top1/top3 does not "
                "exactly match the shipped eval.json's overall top1/top3."
            )

    top1_correct = int(correct1.sum())
    top1_n = int(total.sum())
    top3_correct = int(correct3.sum())

    aggregates = compute_group_aggregate(correct1, correct3, total, taxonomy,
                                        NORTHEAST_NEW_SPECIES_SLUGS)
    if assert_frozen_catalog_shape:
        assert_group_denominators(aggregates)
    genus_metrics = compute_genus_metrics(predictions, taxonomy, NORTHEAST_NEW_SPECIES_SLUGS)
    selection_nuance = compute_selection_nuance(run_dir, run_manifest, top1_correct, top1_n)

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "sha256": _sha256_file(Path(__file__).resolve()),
            "git_commit": git_commit,
        },
        "label": LABEL,
        "report_type": "pinned_development_split_analysis",
        "source": {
            "run_dir": _repo_relative_posix(run_dir),
            "run_manifest_sha256": run_manifest_sha256,
            "training_commit": run_manifest.get("git_head"),
            "final_artifact_hashes": final_hashes,
            "manifest_sha256": manifest_sha256,
            "taxonomy_sha256": taxonomy_sha256,
            "val_split_sha256": run_manifest["val_split"]["sha256"],
            "checkpoint": {
                "resolved_config_sha256": provenance.get("resolved_config_sha256"),
                "numerical_policy": provenance.get("numerical_policy"),
                "backbone": cfg["model"]["backbone"],
                "embedding_dim": cfg["model"]["embedding_dim"],
                "image_size": cfg["image_size"],
                "normalize": cfg["normalize"],
                "interpolation_effective": effective_interpolation,
            },
        },
        "selection": {
            "rule": selection_rule,
            "selected_internal_epoch": run_manifest["best"]["epoch"],
            "selected_human_epoch": run_manifest["best"]["epoch"] + 1,
        },
        "dataset": {
            "species_count": num_classes,
            "val_n": split_record["n_val"],
            "train_n": split_record["n_train"],
        },
        "reproduction": {
            "top1": official["overall"]["top1"],
            "top1_correct": top1_correct,
            "top1_n": top1_n,
            "top3": official["overall"]["top3"],
            "top3_correct": top3_correct,
            "top3_n": top1_n,
            "matches_evaluate_topk_accuracy": True,
            "matches_run_manifest_best_metrics": True,
            "matches_shipped_eval_json": eval_json_path.exists(),
        },
        "aggregates": aggregates,
        "selection_nuance": selection_nuance,
        "genus_metrics": genus_metrics,
    }
    return report


def verify_byte_identical(regenerated: bytes, existing_path: Path) -> None:
    if not existing_path.exists():
        raise DevReportError(f"{existing_path} does not exist -- nothing to verify against.")
    existing = existing_path.read_bytes()
    if regenerated != existing:
        raise DevReportError(
            f"Regenerated report does not byte-match {existing_path}. "
            f"regenerated_sha256={_sha256_bytes(regenerated)} "
            f"existing_sha256={_sha256_bytes(existing)}"
        )


# ----------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--local-data-dir", type=Path, required=True)
    ap.add_argument("--manifest-csv", type=Path, default=None,
                    help="Defaults to run_manifest.json's own recorded manifest path.")
    ap.add_argument("--taxonomy-json", type=Path, default=None,
                    help="Defaults to run_manifest.json's own recorded taxonomy path.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="Regenerate into a temp file and compare byte-for-byte "
                         "against --existing instead of writing --out.")
    ap.add_argument("--existing", type=Path, default=None)
    args = ap.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.run_dir.resolve()
    run_manifest_path_probe = run_dir / "run_manifest.json"
    manifest_csv = args.manifest_csv
    taxonomy_json = args.taxonomy_json
    if (manifest_csv is None or taxonomy_json is None) and run_manifest_path_probe.exists():
        rm = json.loads(run_manifest_path_probe.read_text())
        if manifest_csv is None and rm.get("manifest"):
            manifest_csv = Path(rm["manifest"]["path"])
        if taxonomy_json is None and rm.get("taxonomy_source"):
            taxonomy_json = Path(rm["taxonomy_source"]["path"])
    if manifest_csv is None or taxonomy_json is None:
        ap.error("--manifest-csv/--taxonomy-json could not be inferred from "
                 "run_manifest.json; pass them explicitly.")

    if args.verify and args.existing is None:
        ap.error("--verify requires --existing.")
    if not args.verify and args.out is None:
        ap.error("--out is required unless --verify is given.")

    # The report file this invocation is about to write (or, in --verify
    # mode, already exists on disk) is not yet committed by design (see
    # generate_report's own docstring) -- exclude ONLY that path from the
    # source-tree git-clean gate, never anything else.
    ignore_path = args.existing if args.verify else args.out
    ignore_git_dirty_paths = frozenset()
    if ignore_path is not None:
        try:
            ignore_git_dirty_paths = frozenset({ignore_path.resolve().relative_to(REPO).as_posix()})
        except ValueError:
            pass

    try:
        report = generate_report(
            run_dir=run_dir, local_data_dir=args.local_data_dir,
            manifest_csv=manifest_csv, taxonomy_json=taxonomy_json,
            database_url=database_url, device=device,
            ignore_git_dirty_paths=ignore_git_dirty_paths,
        )
    except DevReportError as e:
        print(f"[report_dev_metrics] FAILED: {e}")
        return 1

    payload = serialize_report(report)

    if args.verify:
        if args.existing is None:
            ap.error("--verify requires --existing.")
        try:
            verify_byte_identical(payload, args.existing)
        except DevReportError as e:
            print(f"[report_dev_metrics] VERIFY FAILED: {e}")
            return 1
        print(f"[report_dev_metrics] VERIFY PASSED: regenerated report is byte-identical "
             f"to {args.existing}")
        return 0

    if args.out is None:
        ap.error("--out is required unless --verify is given.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / f"{args.out.stem}.tmp{os.getpid()}{args.out.suffix}"
    tmp.write_bytes(payload)
    os.replace(tmp, args.out)
    print(f"[report_dev_metrics] wrote {args.out} ({len(payload)} bytes, "
         f"sha256={_sha256_bytes(payload)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
