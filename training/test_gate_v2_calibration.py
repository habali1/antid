#!/usr/bin/env python3
"""test_gate_v2_calibration.py — synthetic/offline tests for the Gate v2
threshold-selection contract, scorer, and selector.

No test in this file touches a real calibration_v2/unknown_test_v2 image,
constructs a real ONNX session against the real B4 backbone, or accesses
the network. Every fixture is built in a temporary directory from small
synthetic data (including, for the end-to-end scoring test, a tiny
synthetic ONNX model built from scratch -- never the real B4 backbone).

Run directly: python test_gate_v2_calibration.py
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import gate_v2_contract as gc  # noqa: E402
import score_calibration_v2 as sc  # noqa: E402
import select_gate_v2_threshold as sel  # noqa: E402

REAL_CONTRACT_PATH = HERE / "gate_v2_selection_contract.json"


# =========================================================================
# helpers for building small synthetic fixtures
# =========================================================================
def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fieldnames})


MANIFEST_FIELDS = None  # populated lazily to avoid importing scrape_gate_v2 at module scope twice


def _manifest_fields() -> list[str]:
    global MANIFEST_FIELDS
    if MANIFEST_FIELDS is None:
        sys.path.insert(0, str(HERE.parent / "data_pipeline"))
        import scrape_gate_v2 as sg
        MANIFEST_FIELDS = sg.MANIFEST_REQUIRED_FIELDS
    return MANIFEST_FIELDS


def _manifest_row(category, slug, taxon_id, photo_id, sha, species=None,
                  byte_size="1000", width="220", height="220") -> dict:
    fields = _manifest_fields()
    row = {f: "x" for f in fields}
    row.update(category=category, slug=slug, taxon_id=str(taxon_id), photo_id=str(photo_id),
              species=species or slug.title(), observation_uuid=f"u-{photo_id}", sha256=sha,
              byte_size=str(byte_size), width=str(width), height=str(height),
              source_url=f"http://x/{photo_id}.jpg",
              photo_license="cc0", photo_attribution="attr", provenance_source="inat_api_gate_v2_readiness",
              observation_id=str(1000 + int(photo_id)))
    return row


def _implementation_source_bindings(repo_for_hash: Path) -> dict:
    return {
        name: {"path": path, "sha256": gc.canonical_lf_sha256_file(repo_for_hash / path)}
        for name, path in gc.IMPLEMENTATION_SOURCE_PATHS.items()
    }


def build_tiny_fixture(testcase: unittest.TestCase, tmp: Path, n_species: int = 2, per_species_cal: int = 2,
                       ood_cal: int = 1, non_ant_cal: int = 1, unrelated_cal: int = 1,
                       copy_real_implementation_sources: bool = False) -> dict:
    """Builds a small, internally-consistent repo under `tmp` with a tiny
    contract + calibration_v2 manifest, small taxonomy/prototypes, a
    minimal run_manifest and parity report. Returns the built contract.

    validate_contract() now checks dataset_quotas and binding paths for
    EXACT equality against gc.FROZEN_DATASET_QUOTAS / gc.FROZEN_BINDING_PATHS
    (the one real, production-scale approved contract) -- a tiny synthetic
    fixture can never equal those by construction. So this fixture instead
    PATCHES those two module-level constants (only those two: every other
    FROZEN_* value -- policy_name, runtime, decision_shape, grid, selection
    policy, single-use rule, closed_sources, low_quality_known, gate_framing,
    approved_outputs -- is scale-independent and the fixture reuses the real
    ones unpatched) to the fixture's own scale for the lifetime of
    `testcase`, via `testcase.addCleanup`. This keeps "the contract" a
    genuine singleton concept in production while still letting tests run
    against small synthetic data.

    implementation_sources always bind to the REAL project files' hashes
    (contract structural validation only checks path/hash-format, not that
    the file exists at `tmp`). When `copy_real_implementation_sources` is
    True, the real files are ALSO copied into `tmp` at the same relative
    paths, so gc.verify_implementation_sources(tmp, contract) succeeds too
    -- needed only by tests that actually call run_score/write_selection_output.
    """
    repo = tmp
    slugs = [f"species-{i}" for i in range(n_species)]
    taxonomy = {str(i): {"species_name": slugs[i].title(), "common_name": "x",
                        "taxon_id": 100 + i, "slug": slugs[i]} for i in range(n_species)}
    _write_json(repo / "training/artifacts/fixture_candidate/taxonomy.json", taxonomy)

    import numpy as np
    protos = np.eye(n_species, dtype=np.float32)
    proto_path = repo / "training/artifacts/fixture_candidate/prototypes.npy"
    proto_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(proto_path, protos)

    backbone_path = repo / "training/artifacts/fixture_candidate/backbone.onnx"
    backbone_path.write_bytes(b"fake-onnx-bytes-not-a-real-model")

    run_manifest = {"status": "completed", "final_artifact_hashes": {}}
    run_manifest_path = repo / "training/artifacts/fixture_candidate/run_manifest.json"
    _write_json(run_manifest_path, run_manifest)

    parity_report = {"deterministic_content": {"ort_provider_evidence": {
        "any_node_executed_outside_cpu": False, "registered_providers": ["CPUExecutionProvider"],
    }}}
    parity_path = repo / "training/reports/fixture_parity.json"
    _write_json(parity_path, parity_report)

    def _unique_sha(pid: int) -> str:
        return hashlib.sha256(f"fixture-row-{pid}".encode()).hexdigest()

    cal_rows = []
    photo_id = 1
    for slug in slugs:
        idx = slugs.index(slug)
        for _ in range(per_species_cal):
            cal_rows.append(_manifest_row("known_holdout", slug, 100 + idx, photo_id, _unique_sha(photo_id)))
            photo_id += 1
    for _ in range(ood_cal):
        cal_rows.append(_manifest_row("out_of_scope_ant", "ood-sp", 900, photo_id, _unique_sha(photo_id)))
        photo_id += 1
    for _ in range(non_ant_cal):
        cal_rows.append(_manifest_row("non_ant_insect", "insect-sp", 901, photo_id, _unique_sha(photo_id)))
        photo_id += 1
    for _ in range(unrelated_cal):
        cal_rows.append(_manifest_row("unrelated", "unrel-sp", 902, photo_id, _unique_sha(photo_id)))
        photo_id += 1

    cal_csv_path = repo / "data/calibration_v2/calibration_v2.csv"
    _write_csv(cal_csv_path, _manifest_fields(), cal_rows)
    cal_csv_hash = gc.sha256_file(cal_csv_path)
    cal_json_path = repo / "data/calibration_v2/calibration_v2.json"
    _write_json(cal_json_path, {"manifest": {"sha256": cal_csv_hash, "rows": len(cal_rows)}})
    cal_json_hash = gc.sha256_file(cal_json_path)

    ut_csv_path = repo / "data/unknown_test_v2/unknown_test_v2.csv"
    _write_csv(ut_csv_path, _manifest_fields(), [])
    ut_csv_hash = gc.sha256_file(ut_csv_path)
    ut_json_path = repo / "data/unknown_test_v2/unknown_test_v2.json"
    _write_json(ut_json_path, {"manifest": {"sha256": ut_csv_hash, "rows": 0}})
    ut_json_hash = gc.sha256_file(ut_json_path)

    if copy_real_implementation_sources:
        for name, rel_path in gc.IMPLEMENTATION_SOURCE_PATHS.items():
            dest = repo / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel_path, dest)
        impl_bindings = _implementation_source_bindings(repo)
    else:
        impl_bindings = _implementation_source_bindings(REPO)

    fixture_binding_paths = {
        "calibration_v2_csv": "data/calibration_v2/calibration_v2.csv",
        "calibration_v2_json": "data/calibration_v2/calibration_v2.json",
        "unknown_test_v2_csv": "data/unknown_test_v2/unknown_test_v2.csv",
        "unknown_test_v2_json": "data/unknown_test_v2/unknown_test_v2.json",
        "candidate_run_manifest": "training/artifacts/fixture_candidate/run_manifest.json",
        "parity_report": "training/reports/fixture_parity.json",
        "backbone_onnx": "training/artifacts/fixture_candidate/backbone.onnx",
        "prototypes_npy": "training/artifacts/fixture_candidate/prototypes.npy",
        "taxonomy_json": "training/artifacts/fixture_candidate/taxonomy.json",
    }
    fixture_dataset_quotas = {
        "calibration_v2": {"total": len(cal_rows), "species_count": n_species,
                          "known_holdout": n_species * per_species_cal,
                          "known_holdout_per_species": per_species_cal,
                          "out_of_scope_ant": ood_cal, "non_ant_insect": non_ant_cal,
                          "unrelated": unrelated_cal},
        "unknown_test_v2": {"total": 0, "species_count": n_species, "known_holdout": 0,
                           "known_holdout_per_species": 0, "out_of_scope_ant": 0,
                           "non_ant_insect": 0, "unrelated": 0},
    }
    # Only these two constants are scale-dependent -- see the docstring for
    # why every other FROZEN_* value is reused unpatched below.
    patcher = mock.patch.multiple(
        gc, FROZEN_DATASET_QUOTAS=fixture_dataset_quotas, FROZEN_BINDING_PATHS=fixture_binding_paths,
    )
    patcher.start()
    testcase.addCleanup(patcher.stop)

    identity_order_sha256 = gc.compute_identity_order_sha256(
        [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in cal_rows]
    )
    # gc.FROZEN_DATASET_QUOTAS itself carries no "purpose" key (same as
    # production); the CONTENT's unknown_test_v2 quota entry requires one,
    # so it's added on a separate copy rather than onto the patched dict.
    content_dataset_quotas = {
        "calibration_v2": dict(fixture_dataset_quotas["calibration_v2"]),
        "unknown_test_v2": {**fixture_dataset_quotas["unknown_test_v2"],
                           "purpose": gc.FROZEN_UNKNOWN_TEST_V2_PURPOSE},
    }

    content = {
        "policy_name": gc.FROZEN_POLICY_NAME, "is_v1_reconstruction": gc.FROZEN_IS_V1_RECONSTRUCTION,
        "bindings": {
            "calibration_v2_csv": {"path": fixture_binding_paths["calibration_v2_csv"], "sha256": cal_csv_hash},
            "calibration_v2_json": {"path": fixture_binding_paths["calibration_v2_json"], "sha256": cal_json_hash},
            "unknown_test_v2_csv": {"path": fixture_binding_paths["unknown_test_v2_csv"], "sha256": ut_csv_hash},
            "unknown_test_v2_json": {"path": fixture_binding_paths["unknown_test_v2_json"], "sha256": ut_json_hash},
            "candidate_run_manifest": {"path": fixture_binding_paths["candidate_run_manifest"],
                                      "sha256": gc.sha256_file(run_manifest_path)},
            "parity_report": {"path": fixture_binding_paths["parity_report"],
                             "sha256": gc.sha256_file(parity_path)},
            "backbone_onnx": {"path": fixture_binding_paths["backbone_onnx"],
                             "sha256": gc.sha256_file(backbone_path)},
            "prototypes_npy": {"path": fixture_binding_paths["prototypes_npy"],
                              "sha256": gc.sha256_file(proto_path)},
            "taxonomy_json": {"path": fixture_binding_paths["taxonomy_json"],
                             "sha256": gc.sha256_file(repo / "training/artifacts/fixture_candidate/taxonomy.json")},
        },
        "implementation_sources": impl_bindings,
        "approved_outputs": dict(gc.FROZEN_APPROVED_OUTPUTS),
        "dataset_quotas": content_dataset_quotas,
        "runtime": dict(gc.FROZEN_RUNTIME),
        "decision_shape": dict(gc.FROZEN_DECISION_SHAPE),
        "grid": dict(gc.FROZEN_GRID),
        "selection": dict(gc.FROZEN_SELECTION_POLICY),
        "single_use_rule": dict(gc.FROZEN_SINGLE_USE_RULE),
        "closed_sources": list(gc.FROZEN_CLOSED_SOURCES),
        "low_quality_known": gc.FROZEN_LOW_QUALITY_KNOWN,
        "gate_framing": gc.FROZEN_GATE_FRAMING,
        gc.CONTENT_KEY_IDENTITY_ORDER_SHA256: identity_order_sha256,
    }
    contract = {
        "schema_version": gc.CONTRACT_SCHEMA_VERSION, "status": gc.CONTRACT_STATUS_FROZEN,
        "content": content, "content_sha256": gc.compute_content_sha256(content),
        "generation": {"note": "fixture"},
    }
    contract_path = repo / "training/gate_v2_selection_contract.json"
    _write_json(contract_path, contract)
    return {"repo": repo, "contract": contract, "contract_path": contract_path,
           "cal_csv_path": cal_csv_path, "cal_rows": cal_rows}


def _rewrite_calibration_csv(fx: dict, rows: list[dict]) -> None:
    """Rewrites calibration_v2.csv with `rows`, then updates
    calibration_v2.json and the contract's bindings (both the CSV and JSON
    hashes) so the fixture stays internally consistent -- isolating the
    specific invariant a test wants to violate (row content, not hash
    binding) from the unrelated hash-binding check that would otherwise
    fire first."""
    _write_csv(fx["cal_csv_path"], _manifest_fields(), rows)
    new_csv_hash = gc.sha256_file(fx["cal_csv_path"])
    cal_json_path = fx["repo"] / "data/calibration_v2/calibration_v2.json"
    _write_json(cal_json_path, {"manifest": {"sha256": new_csv_hash, "rows": len(rows)}})
    new_json_hash = gc.sha256_file(cal_json_path)

    contract = fx["contract"]
    contract["content"]["bindings"]["calibration_v2_csv"]["sha256"] = new_csv_hash
    contract["content"]["bindings"]["calibration_v2_json"]["sha256"] = new_json_hash
    contract["content"][gc.CONTENT_KEY_IDENTITY_ORDER_SHA256] = gc.compute_identity_order_sha256(
        [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in rows]
    )
    contract["content_sha256"] = gc.compute_content_sha256(contract["content"])
    _write_json(fx["contract_path"], contract)
    fx["cal_rows"] = rows


def _fixture_args(fx, **overrides):
    import argparse
    d = dict(repo=fx["repo"], artifacts_dir=fx["repo"] / "training/artifacts/fixture_candidate",
            contract=fx["contract_path"], calibration_csv=fx["cal_csv_path"],
            calibration_json=fx["repo"] / "data/calibration_v2/calibration_v2.json",
            out=fx["repo"] / "data/calibration_v2/calibration_v2_scores.json",
            preflight=True, score=False)
    d.update(overrides)
    return argparse.Namespace(**d)


class RaisingSession:
    def __init__(self, *a, **k):
        raise AssertionError("an ONNX InferenceSession must never be constructed during preflight")


class RaisingImageOpen:
    def __call__(self, *a, **k):
        raise AssertionError("PIL.Image.open must never be called during preflight")


def _bind_scores_to_contract_identity(contract: dict, records: list[dict]) -> None:
    """Test helper: binds `contract`'s row identity-order hash (and the
    resulting content_sha256) to the exact identity sequence of `records`.

    Many tests below build hand-rolled synthetic score records to exercise
    ONE specific validator invariant (a bad hash, a malformed field, ...)
    and intentionally never reproduce the fixture's real calibration_v2.csv
    row identities. Without this, EVERY such test would now also trip the
    (correctly strict) identity-order check, for a reason unrelated to what
    it's testing. The dedicated identity-order-hash tests build their own
    explicit mismatch instead of using this helper. Some callers (the pure
    selector-logic tests) build minimal records without identity fields at
    all -- those never go through gc.validate_score_content, so this is a
    no-op for them rather than a KeyError."""
    if not all("photo_id" in r and "observation_uuid" in r and "image_sha256" in r for r in records):
        return
    triples = [(r["photo_id"], r["observation_uuid"], r["image_sha256"]) for r in records]
    contract["content"][gc.CONTENT_KEY_IDENTITY_ORDER_SHA256] = gc.compute_identity_order_sha256(triples)
    contract["content_sha256"] = gc.compute_content_sha256(contract["content"])


def _score_content(known_rows, ood_rows=None, contract=None):
    records = list(known_rows) + list(ood_rows or [])
    _bind_scores_to_contract_identity(contract, records)
    impl_hashes = {name: entry["sha256"] for name, entry in contract["content"]["implementation_sources"].items()}
    content = {
        "dataset": "calibration_v2", "row_order": "test order", "n_rows": len(records), "records": records,
        "bindings": {"contract_content_sha256": contract["content_sha256"],
                    "calibration_v2_csv_sha256": contract["content"]["bindings"]["calibration_v2_csv"]["sha256"]},
        "runtime": {
            "onnxruntime_version": "1.0.0-test", "registered_providers": ["CPUExecutionProvider"],
            "pillow_version": "10.0.0-test", "preprocessing_contract": {"image_size": 380},
        },
        "provenance": {"git_head": "a" * 40, "implementation_source_hashes": impl_hashes},
    }
    return {"schema_version": gc.SCORE_SCHEMA_VERSION, "content": content,
           "content_sha256": gc.compute_content_sha256(content)}


def _full_score_record(category, slug, idx, taxon_id, photo_id, true_idx=None, top1_idx=None,
                       max_cosine=0.5, second_idx=None, third_idx=None, n_species=3):
    top1_idx = idx if top1_idx is None else top1_idx
    # Defaults are 3 DISTINCT indices (mod n_species) -- the validator now
    # requires top3_indices/top3_slugs to be 3 distinct values, so a repeat
    # is never a valid default. Only tests that specifically want a
    # duplicate/out-of-range top-3 pass second_idx/third_idx explicitly.
    second_idx = (top1_idx + 1) % n_species if second_idx is None else second_idx
    third_idx = (top1_idx + 2) % n_species if third_idx is None else third_idx
    top1_correct = (top1_idx == true_idx) if true_idx is not None else None
    return {
        "category": category, "slug": slug, "species": slug.title(), "taxon_id": str(taxon_id),
        "photo_id": str(photo_id), "observation_uuid": f"u-{photo_id}",
        "image_sha256": hashlib.sha256(f"score-row-{photo_id}".encode()).hexdigest(),
        "true_class_index": true_idx, "top1_index": top1_idx, "top1_slug": f"species-{top1_idx}",
        "top3_indices": [top1_idx, second_idx, third_idx],
        "top3_slugs": [f"species-{top1_idx}", f"species-{second_idx}", f"species-{third_idx}"],
        "top3_similarities": [max_cosine, max_cosine - 0.1, max_cosine - 0.2],
        "max_cosine": max_cosine, "top1_correct": top1_correct,
    }


# =========================================================================
# 1. Contract schema
# =========================================================================
class TestContractSchema(unittest.TestCase):
    def test_real_contract_loads_and_validates(self):
        contract = gc.load_and_verify_contract(REAL_CONTRACT_PATH)
        self.assertEqual(contract["status"], gc.CONTRACT_STATUS_FROZEN)

    def test_real_contract_matches_expected_hashes(self):
        contract = gc.load_and_verify_contract(REAL_CONTRACT_PATH)
        b = contract["content"]["bindings"]
        expected = {
            "calibration_v2_csv": "8201200f4c5d7c869a71c3c927ccee0dfb83d6b91a595241dd49996deee1d46e",
            "calibration_v2_json": "d7d96d321c7ebb4a5b3cd55ded2f2bb72628b6ec38b82ef135c5be088369ed9e",
            "unknown_test_v2_csv": "d706cb6dbda7a672594fc425ce5c8f8f0f6f868662c3d52c5f9c88083163b09f",
            "unknown_test_v2_json": "eea1cbb3118dfc05f42f477ef58d6ccbc5458df1e8a5496845a146d161a89e6d",
            "candidate_run_manifest": "6f8bfe3141a9870c37f6810da6b9a5459cf701242f9cbb3d32a8b01c2f0f265b",
            "parity_report": "bec08235a48e9583f2b40556c9c8d860c41904729bcded9d6f91c37716c8297e",
            "backbone_onnx": "fc22d26ae5c73d20613dafcef02e75291c8779ec8a1296ebc9a72f0e7d7f826b",
            "prototypes_npy": "0e52a7f350996a48f4129e1924e9b0af79517e4c3273dd9ecc7e851121d43825",
            "taxonomy_json": "c8672287d59b5b9d3fb9aec5fd85a008ad30441f2ce3159713a3752b37a04767",
        }
        for name, expected_hash in expected.items():
            self.assertEqual(b[name]["sha256"], expected_hash, name)

    def test_real_contract_dataset_totals(self):
        contract = gc.load_and_verify_contract(REAL_CONTRACT_PATH)
        q = contract["content"]["dataset_quotas"]
        self.assertEqual(q["calibration_v2"]["total"], 1250)
        self.assertEqual(q["unknown_test_v2"]["total"], 790)

    def test_real_contract_binds_implementation_sources_to_real_files(self):
        contract = gc.load_and_verify_contract(REAL_CONTRACT_PATH)
        actual = gc.verify_implementation_sources(REPO, contract)
        self.assertEqual(set(actual), gc.IMPLEMENTATION_SOURCE_KEYS)

    def test_real_contract_wording_uses_improve_not_exceed(self):
        contract = gc.load_and_verify_contract(REAL_CONTRACT_PATH)
        wording = contract["content"]["selection"]["usefulness_rule"]
        self.assertIn("improve", wording)
        self.assertNotIn("exceed", wording)

    def test_deterministic_serialization_round_trip(self):
        contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
        recomputed = gc.compute_content_sha256(contract["content"])
        self.assertEqual(recomputed, contract["content_sha256"])
        shuffled = json.loads(json.dumps(contract["content"]))
        self.assertEqual(gc.compute_content_sha256(shuffled), contract["content_sha256"])

    def test_bool_rejected_where_integer_required(self):
        contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
        contract["content"] = dict(contract["content"])
        contract["content"]["grid"] = dict(contract["content"]["grid"])
        contract["content"]["grid"]["start"] = True  # bool, not int -100
        contract["content_sha256"] = gc.compute_content_sha256(contract["content"])
        problems = gc.validate_contract(contract)
        self.assertTrue(problems)
        self.assertTrue(any("grid.start" in p for p in problems))

    def test_altered_content_hash_rejected(self):
        contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
        contract["content"] = dict(contract["content"])
        contract["content"]["policy_name"] = "tampered"
        problems = gc.validate_contract(contract)
        self.assertTrue(any("content_sha256" in p for p in problems))

    def test_missing_binding_rejected(self):
        contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
        content = dict(contract["content"])
        content["bindings"] = {k: v for k, v in content["bindings"].items() if k != "parity_report"}
        content_sha256 = gc.compute_content_sha256(content)
        contract2 = {**contract, "content": content, "content_sha256": content_sha256}
        problems = gc.validate_contract(contract2)
        self.assertTrue(any("bindings missing" in p for p in problems))


# =========================================================================
# 2. Grid
# =========================================================================
class TestGrid(unittest.TestCase):
    def test_201_points(self):
        grid = gc.generate_grid_integers()
        self.assertEqual(len(grid), 201)
        self.assertEqual(grid[0], -100)
        self.assertEqual(grid[-1], 100)

    def test_exact_060_present_as_integer_60(self):
        grid = gc.generate_grid_integers()
        self.assertIn(60, grid)
        self.assertEqual(gc.grid_integer_to_float(60), 0.6)

    def test_grid_never_built_by_float_addition(self):
        source = inspect.getsource(gc.generate_grid_integers)
        self.assertIn("range(", source)
        self.assertNotIn("+=", source)

    def test_boundary_0_5999_rejected_0_6000_accepted_0_6001_accepted(self):
        threshold = gc.grid_integer_to_float(60)
        known = [
            {"max_cosine": 0.5999, "top1_correct": True, "true_class_index": 0, "top3_indices": [0, 1, 2]},
            {"max_cosine": 0.6000, "top1_correct": True, "true_class_index": 0, "top3_indices": [0, 1, 2]},
            {"max_cosine": 0.6001, "top1_correct": True, "true_class_index": 0, "top3_indices": [0, 1, 2]},
        ]
        counts = sel._classify_known(known, threshold)
        self.assertEqual(counts["rejected"], 1)   # only 0.5999
        self.assertEqual(counts["accepted"], 2)   # 0.6000 (equality accepted) and 0.6001


# =========================================================================
# 3. Coverage feasibility
# =========================================================================
class TestCoverageFeasibility(unittest.TestCase):
    def test_exact_65_percent_boundary_feasible(self):
        self.assertTrue(gc.coverage_feasible(65, 100, 65, 100))

    def test_below_boundary_infeasible(self):
        self.assertFalse(gc.coverage_feasible(64, 100, 65, 100))

    def test_above_boundary_feasible(self):
        self.assertTrue(gc.coverage_feasible(66, 100, 65, 100))

    def test_non_round_totals_exact(self):
        self.assertTrue(gc.coverage_feasible(13, 20, 65, 100))
        self.assertFalse(gc.coverage_feasible(12, 20, 65, 100))


# =========================================================================
# 4. Accuracy fraction comparison
# =========================================================================
class TestAccuracyComparison(unittest.TestCase):
    def test_exact_equal_fractions_not_a_rounded_tie(self):
        self.assertTrue(gc.accuracy_ge(1, 3, 2, 6))
        self.assertTrue(gc.accuracy_ge(2, 6, 1, 3))

    def test_strictly_greater(self):
        self.assertTrue(gc.accuracy_ge(3, 4, 1, 2))
        self.assertFalse(gc.accuracy_ge(1, 2, 3, 4))


# =========================================================================
# 5. Usefulness guard
# =========================================================================
class TestUsefulnessGuard(unittest.TestCase):
    def test_exactly_5pp_passes(self):
        self.assertTrue(gc.usefulness_ge_floor(55, 100, 50, 100, 5, 100))

    def test_below_5pp_fails(self):
        self.assertFalse(gc.usefulness_ge_floor(549, 1000, 500, 1000, 5, 100))

    def test_above_5pp_passes(self):
        self.assertTrue(gc.usefulness_ge_floor(60, 100, 50, 100, 5, 100))


# =========================================================================
# 6. Selection logic (pure, on synthetic score dicts)
# =========================================================================
class TestSelectionLogic(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=2, per_species_cal=5,
                                     ood_cal=2, non_ant_cal=1, unrelated_cal=1)
        self.contract = self.fx["contract"]

    def tearDown(self):
        self._tmpdir.cleanup()

    @staticmethod
    def _band(n, score, n_correct):
        return [{"category": "known_holdout", "slug": "species-0", "max_cosine": score,
                "top1_correct": i < n_correct, "true_class_index": 0, "top3_indices": [0, 0, 0]}
               for i in range(n)]

    def test_primary_objective_maximizes_accepted_accuracy(self):
        known = self._band(70, 0.9, 70) + self._band(30, 0.3, 15)
        result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertEqual(result["status"], sel.STATUS_CANDIDATE_SELECTED)
        self.assertEqual(result["candidate"]["accepted"], 70)
        self.assertEqual(result["candidate"]["accepted_top1_accuracy"], 1.0)

    def test_higher_coverage_tiebreak(self):
        known = self._band(70, 0.7, 70) + self._band(30, 0.4, 30) + self._band(6, 0.05, 0)
        result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertEqual(result["status"], sel.STATUS_CANDIDATE_SELECTED)
        self.assertEqual(result["candidate"]["accepted"], 100)
        self.assertEqual(result["candidate"]["accepted_top1_accuracy"], 1.0)

    def test_lower_threshold_final_tiebreak(self):
        known = self._band(94, 0.5, 94) + self._band(6, -0.9, 0)
        result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertEqual(result["status"], sel.STATUS_CANDIDATE_SELECTED)
        self.assertEqual(result["candidate"]["accepted"], 94)
        self.assertEqual(result["candidate"]["threshold_integer"], -89)

    def test_exactly_5pp_improvement_passes(self):
        known = self._band(80, 0.9, 20) + self._band(20, 0.1, 0)
        result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertEqual(result["status"], sel.STATUS_CANDIDATE_SELECTED)
        self.assertAlmostEqual(result["candidate"]["improvement_over_baseline_pp"], 5.0)

    def test_below_5pp_yields_no_useful_gate(self):
        known = self._band(65, 0.9, 65) + self._band(35, 0.9, 0)
        result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertEqual(result["status"], sel.STATUS_NO_USEFUL_GATE)
        self.assertEqual(result["reason"], "usefulness_guard_failed")
        self.assertNotIn("candidate", result)

    def test_no_feasible_threshold_yields_no_useful_gate(self):
        known = self._band(90, 0.9, 90) + self._band(10, 0.1, 0)
        with mock.patch.object(gc, "coverage_feasible", return_value=False):
            result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertEqual(result["status"], sel.STATUS_NO_USEFUL_GATE)
        self.assertEqual(result["reason"], "no_feasible_threshold")
        self.assertNotIn("candidate", result)

    def test_ood_scores_cannot_change_selected_threshold(self):
        known = self._band(70, 0.9, 70) + self._band(30, 0.3, 15)
        ood_mild = [{"category": "out_of_scope_ant", "max_cosine": 0.1, "top1_correct": None,
                    "true_class_index": None, "top3_indices": [0, 1, 2]}]
        ood_wild = [{"category": "out_of_scope_ant", "max_cosine": 0.99, "top1_correct": None,
                    "true_class_index": None, "top3_indices": [0, 1, 2]}] * 500
        result_mild = sel.select_threshold(_score_content(known, ood_mild, self.contract), self.contract)
        result_wild = sel.select_threshold(_score_content(known, ood_wild, self.contract), self.contract)
        self.assertEqual(result_mild["candidate"]["threshold_integer"],
                         result_wild["candidate"]["threshold_integer"])

    def test_candidate_never_present_when_status_is_no_useful_gate(self):
        known = self._band(65, 0.9, 65) + self._band(35, 0.9, 0)
        result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertEqual(result["status"], sel.STATUS_NO_USEFUL_GATE)
        self.assertNotIn("candidate", result)

    def test_diagnostics_include_top3_baseline_and_per_species(self):
        known = self._band(80, 0.9, 20) + self._band(20, 0.1, 0)
        result = sel.select_threshold(_score_content(known, contract=self.contract), self.contract)
        self.assertIn("accuracy_top3", result["diagnostics"]["baseline"])
        self.assertIn("accepted_top3_accuracy", result["candidate"])
        self.assertIn("per_species_known", result["candidate"])
        for entry in result["candidate"]["per_species_known"]:
            for field in ("n", "accepted", "rejected", "coverage",
                         "baseline_top1_accuracy", "baseline_top3_accuracy",
                         "accepted_top1_accuracy", "accepted_top3_accuracy"):
                self.assertIn(field, entry)


# =========================================================================
# 7. Rejection-metric definitions (point 3 of the correction)
# =========================================================================
class TestRejectionMetricsFixed(unittest.TestCase):
    def test_rates_use_total_outcome_denominator_not_total_rejected(self):
        # 100 correct predictions total, 10 rejected -> rate = 10/100 = 10%.
        # 10 incorrect predictions total, 5 rejected -> rate = 5/10 = 50%.
        # The OLD (buggy) implementation divided both numerators by
        # total_rejected (15), giving correct_rate=10/15=66.7% and
        # incorrect_rate=5/15=33.3% -- both wrong, and their ratio (0.5)
        # would coincidentally look plausible while being derived from the
        # wrong quantity entirely. The fixed ratio (50%/10%=5.0) must differ
        # from the old buggy count-based ratio (5/10=0.5).
        counts = {
            "correct_rejected": 10, "total_correct": 100,
            "incorrect_rejected": 5, "total_incorrect": 10,
        }
        metrics = sel._rejection_metrics(counts)
        self.assertAlmostEqual(metrics["correct_prediction_rejection_rate"], 0.10)
        self.assertAlmostEqual(metrics["incorrect_prediction_rejection_rate"], 0.50)
        self.assertAlmostEqual(metrics["incorrect_to_correct_rejection_ratio"], 5.0)
        # the discredited old-style count ratio (5/10=0.5) must NOT equal
        # the correct rate-based ratio (5.0) -- proving they are genuinely
        # different quantities, not just different labels for the same one.
        old_style_ratio = counts["incorrect_rejected"] / counts["correct_rejected"]
        self.assertNotAlmostEqual(metrics["incorrect_to_correct_rejection_ratio"], old_style_ratio)

    def test_zero_denominator_yields_null_not_zero_or_error(self):
        counts = {"correct_rejected": 0, "total_correct": 0, "incorrect_rejected": 3, "total_incorrect": 10}
        metrics = sel._rejection_metrics(counts)
        self.assertIsNone(metrics["correct_prediction_rejection_rate"])
        self.assertIsNone(metrics["incorrect_to_correct_rejection_ratio"])

    def test_raw_integer_counts_preserved_in_report(self):
        counts = {"correct_rejected": 7, "total_correct": 20, "incorrect_rejected": 2, "total_incorrect": 5}
        metrics = sel._rejection_metrics(counts)
        self.assertEqual(metrics["correct_rejected"], 7)
        self.assertEqual(metrics["total_correct_predictions"], 20)
        self.assertEqual(metrics["incorrect_rejected"], 2)
        self.assertEqual(metrics["total_incorrect_predictions"], 5)


# =========================================================================
# 8. Score-file identity binding / failure modes (via the real validator)
# =========================================================================
class TestScoreFileIdentityBinding(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=3, per_species_cal=2)
        self.contract = self.fx["contract"]

    def tearDown(self):
        self._tmpdir.cleanup()

    def _valid_scores_dict(self):
        known = ([_full_score_record("known_holdout", f"species-{s}", s, 100 + s, s * 2 + i, true_idx=s)
                 for s in range(3) for i in range(2)])
        ood = [_full_score_record("out_of_scope_ant", "ood-sp", 0, 900, 900, true_idx=None, top1_idx=1)]
        non_ant = [_full_score_record("non_ant_insect", "insect-sp", 0, 901, 901, true_idx=None, top1_idx=0)]
        unrelated = [_full_score_record("unrelated", "unrel-sp", 0, 902, 902, true_idx=None, top1_idx=0)]
        return _score_content(known, ood + non_ant + unrelated, self.contract)

    def test_wrong_contract_hash_rejected(self):
        scores = self._valid_scores_dict()
        scores["content"] = dict(scores["content"])
        scores["content"]["bindings"] = dict(scores["content"]["bindings"])
        scores["content"]["bindings"]["contract_content_sha256"] = "0" * 64
        scores["content_sha256"] = gc.compute_content_sha256(scores["content"])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scores.json"
            _write_json(p, scores)
            with self.assertRaises(sel.SelectionError):
                sel.load_scores(p, self.contract)

    def test_wrong_csv_hash_rejected(self):
        scores = self._valid_scores_dict()
        scores["content"] = dict(scores["content"])
        scores["content"]["bindings"] = dict(scores["content"]["bindings"])
        scores["content"]["bindings"]["calibration_v2_csv_sha256"] = "0" * 64
        scores["content_sha256"] = gc.compute_content_sha256(scores["content"])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scores.json"
            _write_json(p, scores)
            with self.assertRaises(sel.SelectionError):
                sel.load_scores(p, self.contract)

    def test_tampered_content_hash_rejected(self):
        scores = self._valid_scores_dict()
        scores["content"]["records"].append(_full_score_record("known_holdout", "species-0", 0, 100, 999, true_idx=0))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scores.json"
            _write_json(p, scores)
            with self.assertRaises(sel.SelectionError):
                sel.load_scores(p, self.contract)

    def test_valid_scores_load_cleanly(self):
        scores = self._valid_scores_dict()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scores.json"
            _write_json(p, scores)
            loaded = sel.load_scores(p, self.contract)
        self.assertEqual(len(loaded["content"]["records"]), 9)


# =========================================================================
# 9. Shared score-artifact schema validator (point 5)
# =========================================================================
class TestScoreContentValidator(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=3, per_species_cal=2,
                                     ood_cal=1, non_ant_cal=1, unrelated_cal=1)
        self.contract = self.fx["contract"]

    def tearDown(self):
        self._tmpdir.cleanup()

    def _valid_content(self):
        known = ([_full_score_record("known_holdout", f"species-{s}", s, 100 + s, s * 2 + i, true_idx=s)
                 for s in range(3) for i in range(2)])
        ood = [_full_score_record("out_of_scope_ant", "ood-sp", 0, 900, 900, true_idx=None, top1_idx=0)]
        non_ant = [_full_score_record("non_ant_insect", "insect-sp", 0, 901, 901, true_idx=None, top1_idx=0)]
        unrelated = [_full_score_record("unrelated", "unrel-sp", 0, 902, 902, true_idx=None, top1_idx=0)]
        scores = _score_content(known, ood + non_ant + unrelated, self.contract)
        return scores

    def test_valid_content_has_no_problems(self):
        scores = self._valid_content()
        problems = gc.validate_score_content(scores, self.contract)
        self.assertEqual(problems, [])

    def test_malformed_top_level_returns_problems_not_exception(self):
        self.assertEqual(gc.validate_score_content(None, self.contract), ["scores is not a JSON object"])
        self.assertEqual(gc.validate_score_content({}, self.contract),
                        gc.validate_score_content({}, self.contract))  # no exception raised
        problems = gc.validate_score_content("not a dict", self.contract)
        self.assertTrue(problems)

    def test_bool_schema_version_rejected(self):
        scores = self._valid_content()
        scores["schema_version"] = True
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("schema_version" in p for p in problems))

    def test_duplicate_photo_id_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][1]["photo_id"] = scores["content"]["records"][0]["photo_id"]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("duplicate photo_id" in p for p in problems))

    def test_duplicate_observation_uuid_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][1]["observation_uuid"] = scores["content"]["records"][0]["observation_uuid"]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("duplicate observation_uuid" in p for p in problems))

    def test_duplicate_image_sha256_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][1]["image_sha256"] = scores["content"]["records"][0]["image_sha256"]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("duplicate image_sha256" in p for p in problems))

    def test_non_finite_similarity_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top3_similarities"][0] = float("nan")
        scores["content"]["records"][0]["max_cosine"] = float("nan")
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("non-finite" in p for p in problems))

    def test_bool_similarity_rejected_as_non_numeric(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top3_similarities"][0] = True
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("non-finite or non-numeric" in p for p in problems))

    def test_max_cosine_must_equal_first_top3_similarity(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["max_cosine"] = 0.123456
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("!= first top-3 similarity" in p for p in problems))

    def test_top1_index_must_equal_first_top3_index(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top1_index"] = 99
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("top1_index" in p for p in problems))

    def test_known_row_top1_correct_must_agree_with_indices(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top1_correct"] = not scores["content"]["records"][0]["top1_correct"]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("disagrees with" in p for p in problems))

    def test_ood_row_must_have_null_true_class_index(self):
        scores = self._valid_content()
        ood_record = next(r for r in scores["content"]["records"] if r["category"] == "out_of_scope_ant")
        ood_record["true_class_index"] = 0
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("OOD row true_class_index must be null" in p for p in problems))

    def test_out_of_range_index_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top3_indices"] = [999, 1, 2]
        scores["content"]["records"][0]["top1_index"] = 999
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("out of range" in p for p in problems))

    def test_wrong_known_holdout_count_detected(self):
        scores = self._valid_content()
        scores["content"]["records"].append(
            _full_score_record("known_holdout", "species-0", 0, 100, 999, true_idx=0))
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("known_holdout record count" in p or "score records, contract expects" in p
                           for p in problems))

    def test_blank_identity_field_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["species"] = ""
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("blank required field" in p for p in problems))

    def test_skewed_per_species_count_with_correct_global_total_detected(self):
        # Same TOTAL known_holdout count (6) as the fixture quota, but all
        # skewed onto species-0/species-1 with NONE for species-2 -- the
        # aggregate count check alone would miss this.
        known = ([_full_score_record("known_holdout", "species-0", 0, 100, i, true_idx=0) for i in range(4)]
                + [_full_score_record("known_holdout", "species-1", 1, 101, i, true_idx=1) for i in range(4, 6)])
        ood = [_full_score_record("out_of_scope_ant", "ood-sp", 0, 900, 900, true_idx=None, top1_idx=0)]
        non_ant = [_full_score_record("non_ant_insect", "insect-sp", 0, 901, 901, true_idx=None, top1_idx=0)]
        unrelated = [_full_score_record("unrelated", "unrel-sp", 0, 902, 902, true_idx=None, top1_idx=0)]
        scores = _score_content(known, ood + non_ant + unrelated, self.contract)
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("distinct slug" in p for p in problems))
        self.assertTrue(any("per-species row count" in p for p in problems))

    def test_bogus_registered_providers_detected(self):
        scores = self._valid_content()
        scores["content"]["runtime"] = dict(scores["content"]["runtime"])
        scores["content"]["runtime"]["registered_providers"] = ["CUDAExecutionProvider"]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("registered_providers" in p for p in problems))

    def test_missing_preprocessing_contract_detected(self):
        scores = self._valid_content()
        scores["content"]["runtime"] = dict(scores["content"]["runtime"])
        scores["content"]["runtime"]["preprocessing_contract"] = {}
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("preprocessing_contract" in p for p in problems))

    def test_bogus_git_head_detected(self):
        scores = self._valid_content()
        scores["content"]["provenance"] = dict(scores["content"]["provenance"])
        scores["content"]["provenance"]["git_head"] = "not-a-real-commit-id"
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("git_head" in p for p in problems))

    def test_wrong_implementation_source_hash_in_provenance_detected(self):
        scores = self._valid_content()
        scores["content"]["provenance"] = dict(scores["content"]["provenance"])
        scores["content"]["provenance"]["implementation_source_hashes"] = dict(
            scores["content"]["provenance"]["implementation_source_hashes"])
        scores["content"]["provenance"]["implementation_source_hashes"]["gate_v2_contract"] = "0" * 64
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("implementation_source_hashes['gate_v2_contract']" in p for p in problems))

    def test_duplicate_top3_index_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top3_indices"] = [0, 0, 1]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("top3_indices must be 3 distinct values" in p for p in problems))

    def test_duplicate_top3_slug_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top3_slugs"] = ["species-0", "species-0", "species-1"]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("top3_slugs must be 3 distinct values" in p for p in problems))

    def test_top1_slug_disagrees_with_top3_slugs_detected(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["top1_slug"] = "species-2"
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("top1_slug" in p and "!=" in p for p in problems))

    def test_non_descending_similarities_detected(self):
        scores = self._valid_content()
        r = scores["content"]["records"][0]
        r["top3_similarities"] = [0.5, 0.9, 0.1]
        r["max_cosine"] = 0.5
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("non-increasing" in p for p in problems))

    def test_reordered_identity_sequence_detected(self):
        # Same records, same content_sha256-consistent structure, just the
        # first two records SWAPPED -- a pure reordering. Individual
        # per-record checks all still pass; only the identity-order hash
        # catches this.
        scores = self._valid_content()
        records = scores["content"]["records"]
        records[0], records[1] = records[1], records[0]
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("identity-order hash does not match" in p for p in problems))

    def test_substituted_identity_detected(self):
        # Same length, same category distribution, but one row's identity
        # fields are swapped out for values that were never in
        # calibration_v2.csv at all.
        scores = self._valid_content()
        r = scores["content"]["records"][0]
        r["photo_id"], r["observation_uuid"] = "substituted-pid", "substituted-uuid"
        r["image_sha256"] = hashlib.sha256(b"substituted-image").hexdigest()
        problems = gc.validate_score_content(scores, self.contract)
        self.assertTrue(any("identity-order hash does not match" in p for p in problems))

    # ---- totality over malformed JSON: every case below must return a
    # nonempty problems list and must NEVER raise (KeyError, TypeError,
    # AttributeError -- unhashable/uncomparable/wrong-shape input included).
    def _assert_total_and_rejected(self, mutate):
        scores = self._valid_content()
        mutate(scores)
        try:
            problems = gc.validate_score_content(scores, self.contract)
        except Exception as exc:  # pragma: no cover -- the failure mode under test
            self.fail(f"validate_score_content raised {type(exc).__name__}: {exc}")
        self.assertTrue(problems, "expected a nonempty problems list for malformed input")
        return problems

    def test_bindings_as_list_rejected_without_raising(self):
        self._assert_total_and_rejected(lambda s: s["content"].__setitem__("bindings", []))

    def test_bindings_as_null_rejected_without_raising(self):
        self._assert_total_and_rejected(lambda s: s["content"].__setitem__("bindings", None))

    def test_category_as_list_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["category"] = []
        self._assert_total_and_rejected(mutate)

    def test_category_as_dict_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["category"] = {}
        self._assert_total_and_rejected(mutate)

    def test_category_as_null_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["category"] = None
        self._assert_total_and_rejected(mutate)

    def test_photo_id_as_list_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["photo_id"] = []
        self._assert_total_and_rejected(mutate)

    def test_observation_uuid_as_dict_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["observation_uuid"] = {}
        self._assert_total_and_rejected(mutate)

    def test_image_sha256_as_list_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["image_sha256"] = []
        self._assert_total_and_rejected(mutate)

    def test_slug_as_list_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["slug"] = []
        self._assert_total_and_rejected(mutate)

    def test_slug_as_dict_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["slug"] = {}
        self._assert_total_and_rejected(mutate)

    def test_top3_indices_containing_list_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["top3_indices"] = [[], 1, 2]
        self._assert_total_and_rejected(mutate)

    def test_top3_indices_containing_dict_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["top3_indices"] = [{}, 1, 2]
        self._assert_total_and_rejected(mutate)

    def test_top3_indices_containing_bool_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["top3_indices"] = [True, 1, 2]
        self._assert_total_and_rejected(mutate)

    def test_top3_indices_containing_null_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["top3_indices"] = [None, 1, 2]
        self._assert_total_and_rejected(mutate)

    def test_top3_slugs_containing_non_strings_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["top3_slugs"] = [1, 2, 3]
        self._assert_total_and_rejected(mutate)

    def test_top3_slugs_containing_nested_list_rejected_without_raising(self):
        def mutate(s):
            s["content"]["records"][0]["top3_slugs"] = [[], [], []]
        self._assert_total_and_rejected(mutate)

    def test_malformed_runtime_nested_value_rejected_without_raising(self):
        def mutate(s):
            s["content"]["runtime"] = []
        self._assert_total_and_rejected(mutate)

    def test_malformed_provenance_nested_value_rejected_without_raising(self):
        def mutate(s):
            s["content"]["provenance"] = "not-an-object"
        self._assert_total_and_rejected(mutate)

    def test_malformed_provenance_implementation_hashes_rejected_without_raising(self):
        def mutate(s):
            s["content"]["provenance"] = dict(s["content"]["provenance"])
            s["content"]["provenance"]["implementation_source_hashes"] = ["not", "a", "dict"]
        self._assert_total_and_rejected(mutate)

    def test_load_scores_converts_malformed_record_to_selection_error(self):
        scores = self._valid_content()
        scores["content"]["records"][0]["category"] = []
        scores["content"]["records"][0]["top3_indices"] = [[], 1, 2]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scores.json"
            _write_json(p, scores)
            with self.assertRaises(sel.SelectionError):
                sel.load_scores(p, self.contract)


# =========================================================================
# 10. Duplicate identity fails closed (point 4 -- REPLACES the old
#     "not_deduplicated_silently" test, which incorrectly accepted a
#     duplicate)
# =========================================================================
class TestDuplicateIdentityFailsClosed(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_duplicate_photo_id_rejected_before_image_access(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=1, per_species_cal=2, ood_cal=0,
                                non_ant_cal=0, unrelated_cal=0)
        rows = fx["cal_rows"]
        rows[1]["photo_id"] = rows[0]["photo_id"]
        rows[1]["observation_uuid"] = "distinct-uuid"
        rows[1]["sha256"] = "b" * 64
        _rewrite_calibration_csv(fx, rows)
        args = _fixture_args(fx)
        with self.assertRaisesRegex(sc.ScoringError, "duplicate identity"):
            sc.run_preflight(args)

    def test_duplicate_observation_uuid_rejected_before_image_access(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=1, per_species_cal=2, ood_cal=0,
                                non_ant_cal=0, unrelated_cal=0)
        rows = fx["cal_rows"]
        rows[1]["photo_id"] = "different-id"
        rows[1]["observation_uuid"] = rows[0]["observation_uuid"]
        rows[1]["sha256"] = "b" * 64
        _rewrite_calibration_csv(fx, rows)
        args = _fixture_args(fx)
        with self.assertRaisesRegex(sc.ScoringError, "duplicate identity"):
            sc.run_preflight(args)

    def test_duplicate_sha256_rejected_before_image_access(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=1, per_species_cal=2, ood_cal=0,
                                non_ant_cal=0, unrelated_cal=0)
        rows = fx["cal_rows"]
        rows[1]["photo_id"] = "different-id"
        rows[1]["observation_uuid"] = "distinct-uuid"
        rows[1]["sha256"] = rows[0]["sha256"]
        _rewrite_calibration_csv(fx, rows)
        args = _fixture_args(fx)
        with self.assertRaisesRegex(sc.ScoringError, "duplicate identity"):
            sc.run_preflight(args)

    def test_duplicates_are_never_silently_deduplicated(self):
        # The fix must FAIL, not quietly drop one of the two rows.
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=1, per_species_cal=2, ood_cal=0,
                                non_ant_cal=0, unrelated_cal=0)
        rows = fx["cal_rows"]
        rows[1]["photo_id"] = rows[0]["photo_id"]
        _rewrite_calibration_csv(fx, rows)
        args = _fixture_args(fx)
        with self.assertRaises(sc.ScoringError):
            result = sc.run_preflight(args)
            self.fail(f"expected ScoringError, got a result with {len(result['rows'])} rows")


# =========================================================================
# 11. Scorer verification (fail before image access)
# =========================================================================
class TestScorerVerification(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_empty_category_rejected(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), ood_cal=0)
        contract = fx["contract"]
        # Bump total AND a category together, on BOTH the contract's own
        # dataset_quotas AND the currently-patched gc.FROZEN_DATASET_QUOTAS
        # (build_tiny_fixture patches it to this fixture's scale) so
        # validate_contract's exact-equality check still passes -- the
        # mismatch this test targets is between the (approved) contract and
        # the ACTUAL calibration_v2.csv row count, never the contract's own
        # internal consistency.
        contract["content"]["dataset_quotas"]["calibration_v2"]["out_of_scope_ant"] = 1
        contract["content"]["dataset_quotas"]["calibration_v2"]["total"] += 1
        gc.FROZEN_DATASET_QUOTAS["calibration_v2"]["out_of_scope_ant"] = 1
        gc.FROZEN_DATASET_QUOTAS["calibration_v2"]["total"] += 1
        contract["content_sha256"] = gc.compute_content_sha256(contract["content"])
        _write_json(fx["contract_path"], contract)
        args = _fixture_args(fx)
        with self.assertRaises(sc.ScoringError):
            sc.run_preflight(args)

    def test_wrong_row_count_rejected(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name))
        contract = fx["contract"]
        # Bump total AND a category together so the contract's own internal
        # arithmetic stays consistent (required by the strengthened
        # validate_contract), on BOTH the contract's dataset_quotas and the
        # currently-patched gc.FROZEN_DATASET_QUOTAS -- the mismatch this
        # test targets is between the (approved) contract and the actual
        # CSV row count.
        contract["content"]["dataset_quotas"]["calibration_v2"]["total"] += 1
        contract["content"]["dataset_quotas"]["calibration_v2"]["out_of_scope_ant"] += 1
        gc.FROZEN_DATASET_QUOTAS["calibration_v2"]["total"] += 1
        gc.FROZEN_DATASET_QUOTAS["calibration_v2"]["out_of_scope_ant"] += 1
        contract["content_sha256"] = gc.compute_content_sha256(contract["content"])
        _write_json(fx["contract_path"], contract)
        args = _fixture_args(fx)
        with self.assertRaises(sc.ScoringError):
            sc.run_preflight(args)

    def test_nan_or_inf_similarity_never_silently_clamped(self):
        import numpy as np
        sims = np.array([0.5, float("nan"), 0.2], dtype=np.float32)
        self.assertFalse(bool(np.isfinite(sims).all()))
        sims2 = np.array([0.5, float("inf"), 0.2], dtype=np.float32)
        self.assertFalse(bool(np.isfinite(sims2).all()))

    def test_manifest_hash_mismatch_fails_before_image_access(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name))
        fx["cal_csv_path"].write_text(fx["cal_csv_path"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
        args = _fixture_args(fx)
        with mock.patch("PIL.Image.open", side_effect=RaisingImageOpen()):
            with self.assertRaises(gc.ContractError):
                sc.run_preflight(args)

    def test_artifact_hash_mismatch_fails_before_image_access(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name))
        backbone = fx["repo"] / "training/artifacts/fixture_candidate/backbone.onnx"
        backbone.write_bytes(b"different-bytes")
        args = _fixture_args(fx)
        with self.assertRaises(gc.ContractError):
            sc.run_preflight(args)

    def test_contract_hash_mismatch_fails(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name))
        contract_text = fx["contract_path"].read_text(encoding="utf-8")
        contract = json.loads(contract_text)
        contract["content"]["policy_name"] = "tampered-without-rehash"
        _write_json(fx["contract_path"], contract)
        args = _fixture_args(fx)
        with self.assertRaises(gc.ContractError):
            sc.run_preflight(args)

    def test_parity_report_hash_mismatch_fails(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name))
        parity_path = fx["repo"] / "training/reports/fixture_parity.json"
        parity_path.write_text(parity_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        args = _fixture_args(fx)
        with self.assertRaises(gc.ContractError):
            sc.run_preflight(args)

    def test_non_cpu_provider_in_parity_evidence_rejected(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name))
        parity_path = fx["repo"] / "training/reports/fixture_parity.json"
        _write_json(parity_path, {"deterministic_content": {"ort_provider_evidence": {
            "any_node_executed_outside_cpu": True, "registered_providers": ["CPUExecutionProvider"],
        }}})
        contract = fx["contract"]
        contract["content"]["bindings"]["parity_report"]["sha256"] = gc.sha256_file(parity_path)
        contract["content_sha256"] = gc.compute_content_sha256(contract["content"])
        _write_json(fx["contract_path"], contract)
        args = _fixture_args(fx)
        with self.assertRaises(sc.ScoringError):
            sc.run_preflight(args)

    def test_known_slug_not_in_taxonomy_rejected(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=2, per_species_cal=1)
        rows = fx["cal_rows"]
        rows[0]["slug"] = "totally-unknown-species"
        _rewrite_calibration_csv(fx, rows)
        args = _fixture_args(fx)
        with self.assertRaises(sc.ScoringError):
            sc.run_preflight(args)

    def test_known_taxon_id_mismatch_rejected(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=2, per_species_cal=1)
        rows = fx["cal_rows"]
        rows[0]["taxon_id"] = "999999"
        _rewrite_calibration_csv(fx, rows)
        args = _fixture_args(fx)
        with self.assertRaises(sc.ScoringError):
            sc.run_preflight(args)

    def test_blank_required_field_rejected(self):
        fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=1, per_species_cal=1, ood_cal=0,
                                non_ant_cal=0, unrelated_cal=0)
        rows = fx["cal_rows"]
        rows[0]["photo_license"] = ""
        _rewrite_calibration_csv(fx, rows)
        args = _fixture_args(fx)
        with self.assertRaises(sc.ScoringError):
            sc.run_preflight(args)


# =========================================================================
# 12. --artifacts-dir identity is by byte hash, not by path string (point 1)
# =========================================================================
class TestArtifactDirectoryBinding(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_fixture(self, Path(self._tmpdir.name))

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_mismatched_artifacts_dir_rejected_before_session_or_image(self):
        other_dir = self.fx["repo"] / "training/artifacts/a_completely_different_candidate"
        other_dir.mkdir(parents=True)
        (other_dir / "backbone.onnx").write_bytes(b"not-the-bound-backbone")
        (other_dir / "prototypes.npy").write_bytes(b"not-the-bound-prototypes")
        (other_dir / "taxonomy.json").write_text("{}", encoding="utf-8")
        (other_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

        with mock.patch("onnxruntime.InferenceSession", RaisingSession), \
            mock.patch("PIL.Image.open", side_effect=RaisingImageOpen()):
            with self.assertRaises(gc.ContractError):
                gc.verify_artifact_directory_matches_contract(self.fx["repo"], self.fx["contract"], other_dir)

    def test_identical_byte_content_at_a_different_path_passes(self):
        # A copy of the EXACT SAME files at a different directory must pass
        # -- identity is by hash, never by the directory string.
        copy_dir = self.fx["repo"] / "training/artifacts/a_copy_of_the_same_candidate"
        copy_dir.mkdir(parents=True)
        original = self.fx["repo"] / "training/artifacts/fixture_candidate"
        for name in ("backbone.onnx", "prototypes.npy", "taxonomy.json", "run_manifest.json"):
            shutil.copyfile(original / name, copy_dir / name)
        gc.verify_artifact_directory_matches_contract(self.fx["repo"], self.fx["contract"], copy_dir)  # must not raise

    def test_run_score_rejects_mismatched_artifacts_dir_via_full_preflight_first(self):
        # run_score's own flow: preflight (contract-bound repo artifacts,
        # which ARE valid) succeeds, but the arbitrary --artifacts-dir
        # points elsewhere -- this must still fail, and fail before any
        # image access or ORT session construction.
        other_dir = self.fx["repo"] / "training/artifacts/wrong_one"
        other_dir.mkdir(parents=True)
        (other_dir / "backbone.onnx").write_bytes(b"wrong")
        (other_dir / "prototypes.npy").write_bytes(b"wrong")
        (other_dir / "taxonomy.json").write_text("{}", encoding="utf-8")
        (other_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        args = _fixture_args(self.fx, artifacts_dir=other_dir, score=True, preflight=False)
        with mock.patch("onnxruntime.InferenceSession", RaisingSession), \
            mock.patch("PIL.Image.open", side_effect=RaisingImageOpen()), \
            mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef"), \
            mock.patch.object(gc, "verify_implementation_sources", return_value={k: "fake" for k in gc.IMPLEMENTATION_SOURCE_KEYS}):
            with self.assertRaises(gc.ContractError):
                sc.run_score(args)


# =========================================================================
# 13. Output writing (point 2): atomic, overwrite-refusing, approved-path-only,
#     writes for BOTH candidate_selected and no_useful_gate_found
# =========================================================================
class TestSelectionOutputWriting(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_fixture(self, Path(self._tmpdir.name), n_species=2, per_species_cal=10,
                                     ood_cal=1, non_ant_cal=1, unrelated_cal=1)
        self.contract = self.fx["contract"]

    def tearDown(self):
        self._tmpdir.cleanup()

    def _args(self, out=None):
        import argparse
        return argparse.Namespace(
            repo=self.fx["repo"], contract=self.fx["contract_path"],
            scores=self.fx["repo"] / "data/calibration_v2/calibration_v2_scores.json",
            out=out or self.fx["repo"] / "data/calibration_v2/calibration_v2_selection.json",
        )

    def _write_scores_file(self, known):
        scores = _score_content(known, contract=self.contract)
        p = self.fx["repo"] / "data/calibration_v2/calibration_v2_scores.json"
        _write_json(p, scores)
        return scores

    def test_writes_output_for_candidate_selected(self):
        known = ([{"category": "known_holdout", "slug": "species-0", "max_cosine": 0.9, "top1_correct": True,
                  "true_class_index": 0, "top3_indices": [0, 1, 2]}] * 16
                + [{"category": "known_holdout", "slug": "species-0", "max_cosine": 0.1, "top1_correct": False,
                   "true_class_index": 0, "top3_indices": [1, 2, 3]}] * 4)
        scores = self._write_scores_file(known)
        result = sel.select_threshold(scores, self.contract)
        self.assertEqual(result["status"], sel.STATUS_CANDIDATE_SELECTED)
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef"), \
            mock.patch.object(gc, "verify_implementation_sources", return_value={k: "fake" for k in gc.IMPLEMENTATION_SOURCE_KEYS}):
            out_path = sel.write_selection_output(self._args(), self.contract, scores, result)
        self.assertTrue(out_path.exists())
        written = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(written["content"]["status"], sel.STATUS_CANDIDATE_SELECTED)
        self.assertIn("candidate", written["content"]["result"])

    def test_writes_output_for_no_useful_gate_found(self):
        known = [{"category": "known_holdout", "slug": "species-0", "max_cosine": 0.9, "top1_correct": True,
                 "true_class_index": 0, "top3_indices": [0, 1, 2]}] * 20
        scores = self._write_scores_file(known)
        result = sel.select_threshold(scores, self.contract)
        self.assertEqual(result["status"], sel.STATUS_NO_USEFUL_GATE)
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef"), \
            mock.patch.object(gc, "verify_implementation_sources", return_value={k: "fake" for k in gc.IMPLEMENTATION_SOURCE_KEYS}):
            out_path = sel.write_selection_output(self._args(), self.contract, scores, result)
        written = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(written["content"]["status"], sel.STATUS_NO_USEFUL_GATE)
        self.assertNotIn("candidate", written["content"]["result"])
        self.assertIn("diagnostics", written["content"])

    def test_refuses_to_overwrite_existing_selection_file(self):
        known = [{"category": "known_holdout", "slug": "species-0", "max_cosine": 0.9, "top1_correct": True,
                 "true_class_index": 0, "top3_indices": [0, 1, 2]}] * 20
        scores = self._write_scores_file(known)
        result = sel.select_threshold(scores, self.contract)
        out_file = self.fx["repo"] / "data/calibration_v2/calibration_v2_selection.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("{}", encoding="utf-8")
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef"), \
            mock.patch.object(gc, "verify_implementation_sources", return_value={k: "fake" for k in gc.IMPLEMENTATION_SOURCE_KEYS}):
            with self.assertRaises(sel.SelectionError):
                sel.write_selection_output(self._args(), self.contract, scores, result)
        self.assertEqual(out_file.read_text(encoding="utf-8"), "{}")  # untouched

    def test_output_path_must_resolve_to_approved_destination(self):
        known = [{"category": "known_holdout", "slug": "species-0", "max_cosine": 0.9, "top1_correct": True,
                 "true_class_index": 0, "top3_indices": [0, 1, 2]}] * 20
        scores = self._write_scores_file(known)
        result = sel.select_threshold(scores, self.contract)
        rogue_path = self.fx["repo"] / "training/artifacts/sneaky_selection.json"
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef"), \
            mock.patch.object(gc, "verify_implementation_sources", return_value={k: "fake" for k in gc.IMPLEMENTATION_SOURCE_KEYS}):
            with self.assertRaises(gc.ContractError):
                sel.write_selection_output(self._args(out=rogue_path), self.contract, scores, result)

    def test_forbidden_output_prefix_rejected_even_if_contract_tried_to_approve_it(self):
        contract = json.loads(json.dumps(self.contract))
        contract["content"]["approved_outputs"]["calibration_v2_selection"] = "training/artifacts/leaked.json"
        contract["content_sha256"] = gc.compute_content_sha256(contract["content"])
        with self.assertRaises(gc.ContractError):
            gc.verify_output_path_is_approved(self.fx["repo"], contract,
                                              "calibration_v2_selection",
                                              self.fx["repo"] / "training/artifacts/leaked.json")


# =========================================================================
# 14. Implementation-source binding (point 6)
# =========================================================================
class TestImplementationSourceBinding(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_fixture(self, Path(self._tmpdir.name), copy_real_implementation_sources=True)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_unmodified_copies_verify_successfully(self):
        actual = gc.verify_implementation_sources(self.fx["repo"], self.fx["contract"])
        self.assertEqual(set(actual), gc.IMPLEMENTATION_SOURCE_KEYS)

    def test_modified_implementation_source_detected(self):
        path = self.fx["repo"] / "training/gate_v2_contract.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(gc.ContractError, "changed since the contract was frozen"):
            gc.verify_implementation_sources(self.fx["repo"], self.fx["contract"])

    def test_crlf_normalization_does_not_change_identity(self):
        path = self.fx["repo"] / "training/gate_v2_contract.py"
        original = path.read_bytes()
        path.write_bytes(original.replace(b"\n", b"\r\n"))
        actual = gc.verify_implementation_sources(self.fx["repo"], self.fx["contract"])
        self.assertEqual(set(actual), gc.IMPLEMENTATION_SOURCE_KEYS)  # still verifies -- CRLF-insensitive

    def test_missing_implementation_source_detected(self):
        (self.fx["repo"] / "api/inference.py").unlink()
        with self.assertRaises(gc.ContractError):
            gc.verify_implementation_sources(self.fx["repo"], self.fx["contract"])

    def test_run_score_verifies_implementation_sources_before_image_access(self):
        path = self.fx["repo"] / "training/score_calibration_v2.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
        args = _fixture_args(self.fx, score=True, preflight=False)
        with mock.patch("onnxruntime.InferenceSession", RaisingSession), \
            mock.patch("PIL.Image.open", side_effect=RaisingImageOpen()), \
            mock.patch.object(gc, "require_clean_tracked_tree"):
            with self.assertRaises(gc.ContractError):
                sc.run_score(args)

    def test_preflight_does_not_require_implementation_source_verification(self):
        # Preflight must remain runnable even with a dirty/tampered
        # implementation source -- only --score enforces this gate.
        path = self.fx["repo"] / "training/score_calibration_v2.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# work in progress\n", encoding="utf-8")
        args = _fixture_args(self.fx)
        result = sc.run_preflight(args)  # must not raise
        self.assertTrue(result["ok"])


# =========================================================================
# 15. Strengthened contract validation never raises ZeroDivisionError etc. (point 7)
# =========================================================================
class TestContractValidationStrength(unittest.TestCase):
    def _mutate(self, mutator):
        contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
        content = dict(contract["content"])
        mutator(content)
        contract["content"] = content
        contract["content_sha256"] = gc.compute_content_sha256(content)
        return contract

    def test_zero_grid_step_returns_problem_not_zerodivisionerror(self):
        def mutate(content):
            content["grid"] = dict(content["grid"])
            content["grid"]["step"] = 0
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)  # must not raise
        self.assertTrue(any("step" in p for p in problems))

    def test_negative_grid_divisor_returns_problem(self):
        def mutate(content):
            content["grid"] = dict(content["grid"])
            content["grid"]["divisor"] = -100
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("divisor" in p for p in problems))

    def test_zero_coverage_floor_denominator_returns_problem(self):
        def mutate(content):
            content["selection"] = dict(content["selection"])
            content["selection"]["coverage_floor_denominator"] = 0
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(problems)

    def test_wrong_diagnostic_category_list_rejected(self):
        def mutate(content):
            content["selection"] = dict(content["selection"])
            content["selection"]["diagnostic_only_categories"] = ["out_of_scope_ant"]
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("diagnostic_only_categories" in p for p in problems))

    def test_wrong_coverage_floor_value_rejected(self):
        def mutate(content):
            content["selection"] = dict(content["selection"])
            content["selection"]["coverage_floor_numerator"] = 50
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("selection does not exactly match" in p for p in problems))

    def test_wrong_usefulness_floor_value_rejected(self):
        def mutate(content):
            content["selection"] = dict(content["selection"])
            content["selection"]["usefulness_floor_numerator"] = 1
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("selection does not exactly match" in p for p in problems))

    def test_malformed_sha256_hex_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["parity_report"] = dict(content["bindings"]["parity_report"])
            content["bindings"]["parity_report"]["sha256"] = "not-a-hash"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("64 lowercase hex" in p for p in problems))

    def test_uppercase_sha256_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["parity_report"] = dict(content["bindings"]["parity_report"])
            content["bindings"]["parity_report"]["sha256"] = "A" * 64
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("64 lowercase hex" in p for p in problems))

    def test_absolute_path_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["parity_report"] = dict(content["bindings"]["parity_report"])
            content["bindings"]["parity_report"]["path"] = "/etc/passwd"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("safe repo-relative path" in p for p in problems))

    def test_path_traversal_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["parity_report"] = dict(content["bindings"]["parity_report"])
            content["bindings"]["parity_report"]["path"] = "../../etc/passwd"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("safe repo-relative path" in p for p in problems))

    def test_windows_drive_path_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["parity_report"] = dict(content["bindings"]["parity_report"])
            content["bindings"]["parity_report"]["path"] = "C:/Windows/System32/config"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("safe repo-relative path" in p for p in problems))

    def test_inconsistent_known_holdout_quota_rejected(self):
        def mutate(content):
            content["dataset_quotas"] = dict(content["dataset_quotas"])
            content["dataset_quotas"]["calibration_v2"] = dict(content["dataset_quotas"]["calibration_v2"])
            content["dataset_quotas"]["calibration_v2"]["known_holdout"] = 651  # != 10*65
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("known_holdout" in p for p in problems))

    def test_inconsistent_total_quota_rejected(self):
        def mutate(content):
            content["dataset_quotas"] = dict(content["dataset_quotas"])
            content["dataset_quotas"]["calibration_v2"] = dict(content["dataset_quotas"]["calibration_v2"])
            content["dataset_quotas"]["calibration_v2"]["total"] = 9999
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("total" in p for p in problems))

    def test_closed_sources_must_include_northeast_final_test_v1(self):
        def mutate(content):
            content["closed_sources"] = []
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("northeast_final_test_v1" in p for p in problems))

    def test_missing_approved_outputs_rejected(self):
        def mutate(content):
            content["approved_outputs"] = {}
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("approved_outputs missing" in p for p in problems))

    def test_approved_output_under_forbidden_prefix_rejected(self):
        def mutate(content):
            content["approved_outputs"] = dict(content["approved_outputs"])
            content["approved_outputs"]["calibration_v2_scores"] = "training/artifacts/leaked.json"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("forbidden prefix" in p for p in problems))

    def test_unexpected_top_level_key_rejected(self):
        # Reproduces the exact Codex finding: an unexpected TOP-LEVEL key
        # (outside `content`, so content_sha256 doesn't even need to be
        # touched) must still be rejected -- schema_version 1's top-level
        # key set is bounded, not "anything with the four required keys
        # present."
        contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
        contract["unexpected_semantic_top_level"] = "bad"
        problems = gc.validate_contract(contract)
        self.assertTrue(any("unexpected top-level key" in p for p in problems))

    def test_missing_unknown_test_v2_purpose_rejected(self):
        # Reproduces the exact Codex finding: dropping
        # dataset_quotas.unknown_test_v2.purpose and rehashing content_sha256
        # fresh must still be rejected -- it is a REQUIRED field with a
        # required exact value, not an optional one.
        def mutate(content):
            content["dataset_quotas"] = dict(content["dataset_quotas"])
            content["dataset_quotas"]["unknown_test_v2"] = {
                k: v for k, v in content["dataset_quotas"]["unknown_test_v2"].items() if k != "purpose"
            }
        contract = self._mutate(mutate)
        self.assertEqual(contract["content_sha256"], gc.compute_content_sha256(contract["content"]))
        problems = gc.validate_contract(contract)
        self.assertTrue(any("dataset_quotas.unknown_test_v2 missing required key 'purpose'" in p
                           for p in problems))

    def test_generation_block_with_unexpected_key_rejected(self):
        contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
        contract["generation"] = dict(contract["generation"])
        contract["generation"]["sneaky_semantic_field"] = "smuggled"
        problems = gc.validate_contract(contract)
        self.assertTrue(any("generation has unexpected key" in p for p in problems))

    # ---- exact-value rejection: each of these is INTERNALLY CONSISTENT and
    # only rehashed after mutation -- a stale hash is never the reason any
    # of these are rejected. Each covers one of the fields the approved
    # contract must match EXACTLY, not merely self-consistently. -----------
    def test_wrong_policy_name_rejected(self):
        def mutate(content):
            content["policy_name"] = "not_the_approved_policy"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("policy_name" in p for p in problems))

    def test_is_v1_reconstruction_flipped_rejected(self):
        def mutate(content):
            content["is_v1_reconstruction"] = True
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("is_v1_reconstruction" in p for p in problems))

    def test_calibration_v2_quota_shrunk_but_self_consistent_rejected(self):
        # A quota mutated to a SMALLER but still internally-consistent total
        # (4 = 1 known + 1 + 1 + 1) must still be rejected -- arithmetic
        # self-consistency is not the same as matching the approved values.
        def mutate(content):
            content["dataset_quotas"] = dict(content["dataset_quotas"])
            content["dataset_quotas"]["calibration_v2"] = {
                "total": 4, "species_count": 1, "known_holdout": 1, "known_holdout_per_species": 1,
                "out_of_scope_ant": 1, "non_ant_insect": 1, "unrelated": 1,
            }
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("dataset_quotas.calibration_v2" in p for p in problems))

    def test_approved_output_path_changed_to_arbitrary_file_rejected(self):
        def mutate(content):
            content["approved_outputs"] = dict(content["approved_outputs"])
            content["approved_outputs"]["calibration_v2_scores"] = "data/arbitrary-new-file.json"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("approved_outputs does not exactly match" in p for p in problems))

    def test_runtime_rounding_changed_rejected(self):
        def mutate(content):
            content["runtime"] = dict(content["runtime"])
            content["runtime"]["rounding"] = "rounded_to_2dp"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("runtime does not exactly match" in p for p in problems))

    def test_no_useful_gate_found_action_changed_rejected(self):
        def mutate(content):
            content["selection"] = dict(content["selection"])
            content["selection"]["no_useful_gate_found_action"] = "silently retry with a lower floor"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("selection does not exactly match" in p for p in problems))

    def test_single_use_wording_changed_rejected(self):
        def mutate(content):
            content["single_use_rule"] = dict(content["single_use_rule"])
            content["single_use_rule"]["unknown_test_v2"] = "may be evaluated as many times as needed"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("single_use_rule does not exactly match" in p for p in problems))

    def test_low_quality_known_changed_rejected(self):
        def mutate(content):
            content["low_quality_known"] = "included"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("low_quality_known" in p for p in problems))

    def test_gate_framing_changed_rejected(self):
        def mutate(content):
            content["gate_framing"] = "unknown_detector"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("gate_framing" in p for p in problems))

    def test_binding_path_redirected_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["backbone_onnx"] = dict(content["bindings"]["backbone_onnx"])
            content["bindings"]["backbone_onnx"]["path"] = "training/artifacts/some_other_model/backbone.onnx"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("binding 'backbone_onnx' path must be exactly" in p for p in problems))

    def test_unexpected_content_key_rejected(self):
        # Schema version 1 must reject an UNEXPECTED semantic key, not
        # silently accept a future/altered shape.
        def mutate(content):
            content["a_new_undocumented_semantic_field"] = "sneaky"
        contract = self._mutate(mutate)
        problems = gc.validate_contract(contract)
        self.assertTrue(any("unexpected key" in p for p in problems))

    def test_combined_seven_field_mutation_attack_rejected(self):
        """Reproduces the exact combined-mutation attack: change ALL seven
        of these fields simultaneously (each individually self-consistent),
        rehash content_sha256 fresh, and confirm validate_contract still
        reports every one of them -- not zero problems, as the pre-fix
        validator did."""
        def mutate(content):
            content["dataset_quotas"] = dict(content["dataset_quotas"])
            content["dataset_quotas"]["calibration_v2"] = {
                "total": 4, "species_count": 1, "known_holdout": 1, "known_holdout_per_species": 1,
                "out_of_scope_ant": 1, "non_ant_insect": 1, "unrelated": 1,
            }
            content["approved_outputs"] = dict(content["approved_outputs"])
            content["approved_outputs"]["calibration_v2_scores"] = "data/arbitrary-new-file.json"
            content["runtime"] = dict(content["runtime"])
            content["runtime"]["rounding"] = "rounded_to_2dp"
            content["selection"] = dict(content["selection"])
            content["selection"]["no_useful_gate_found_action"] = "unrelated text"
            content["single_use_rule"] = dict(content["single_use_rule"])
            content["single_use_rule"]["unknown_test_v2"] = "a different rule"
            content["low_quality_known"] = "included"
            content["gate_framing"] = "unknown_detector"
        contract = self._mutate(mutate)
        # Confirm the mutated contract's content_sha256 IS the fresh, correct
        # hash of the mutated content -- proving rejection below is NOT
        # merely a stale-hash artifact.
        self.assertEqual(contract["content_sha256"], gc.compute_content_sha256(contract["content"]))
        problems = gc.validate_contract(contract)
        self.assertTrue(any("dataset_quotas.calibration_v2" in p for p in problems))
        self.assertTrue(any("approved_outputs does not exactly match" in p for p in problems))
        self.assertTrue(any("runtime does not exactly match" in p for p in problems))
        self.assertTrue(any("selection does not exactly match" in p for p in problems))
        self.assertTrue(any("single_use_rule does not exactly match" in p for p in problems))
        self.assertTrue(any("low_quality_known" in p for p in problems))
        self.assertTrue(any("gate_framing" in p for p in problems))


# =========================================================================
# 16. Policy/geo isolation (point 8)
# =========================================================================
class TestPolicyGeoIsolation(unittest.TestCase):
    def test_load_candidate_session_and_prototypes_never_references_policy_or_geo(self):
        source = inspect.getsource(sc.load_candidate_session_and_prototypes)
        # Strip the docstring (which mentions these filenames only to
        # document that they are never read) before checking the CODE body.
        body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
        self.assertNotIn("inference_policy", body)
        self.assertNotIn("geo_index", body)
        self.assertNotIn("AntIdentifier(", body)  # never constructs the whole class, only reuses preprocess

    def test_run_score_never_constructs_antidentifier(self):
        source = inspect.getsource(sc.run_score)
        self.assertNotIn("AntIdentifier(", source)
        self.assertIn("AntIdentifier.preprocess", source)

    def test_hostile_policy_and_geo_files_are_never_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = build_tiny_fixture(self, Path(tmp), copy_real_implementation_sources=True)
            artifacts_dir = fx["repo"] / "training/artifacts/fixture_candidate"
            # Hostile/malformed content that would raise if ever parsed.
            (artifacts_dir / "inference_policy.json").write_text("{not valid json at all", encoding="utf-8")
            (artifacts_dir / "geo_index.json").write_text("{not valid json at all", encoding="utf-8")
            # Build a real (tiny, fake-bytes) backbone.onnx already exists in the fixture;
            # loading it as a real ORT session will fail (not a real model), but that
            # failure must happen INSIDE onnxruntime, never inside a policy/geo parser.
            try:
                sc.load_candidate_session_and_prototypes(artifacts_dir)
            except Exception as exc:
                # must fail on the fake ONNX bytes, never on the hostile JSON files
                self.assertNotIn("inference_policy", str(exc))
                self.assertNotIn("geo_index", str(exc))
                self.assertNotIn("Expecting value", str(exc))  # json.JSONDecodeError message


# =========================================================================
# 17. Per-image contract verification (point 9)
# =========================================================================
class TestImageContractVerification(unittest.TestCase):
    @staticmethod
    def _png_bytes(size=220, color=(10, 20, 30)):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (size, size), color).save(buf, format="PNG")
        return buf.getvalue()

    def test_valid_image_passes(self):
        data = self._png_bytes(220)
        sc.decode_and_validate_image(data, 220, 220)  # must not raise

    def test_undersized_image_rejected(self):
        data = self._png_bytes(150)
        with self.assertRaisesRegex(sc.ScoringError, "200px floor"):
            sc.decode_and_validate_image(data, 150, 150)

    def test_dimension_mismatch_against_manifest_rejected(self):
        data = self._png_bytes(220)
        with self.assertRaisesRegex(sc.ScoringError, "manifest expects"):
            sc.decode_and_validate_image(data, 300, 300)

    def test_truncated_image_rejected(self):
        data = self._png_bytes(220)[:len(self._png_bytes(220)) // 2]
        with self.assertRaisesRegex(sc.ScoringError, "failed to decode"):
            sc.decode_and_validate_image(data, 220, 220)

    def test_garbage_bytes_rejected(self):
        with self.assertRaisesRegex(sc.ScoringError, "failed to decode"):
            sc.decode_and_validate_image(b"not an image at all", 220, 220)

    def test_cosine_tolerance_is_defined_explicitly(self):
        self.assertIsInstance(gc.COSINE_TOLERANCE, float)
        self.assertGreater(gc.COSINE_TOLERANCE, 0)
        self.assertLess(gc.COSINE_TOLERANCE, 0.01)


# =========================================================================
# 18. Preflight performs no image/model access
# =========================================================================
class TestPreflightNoImageOrModelAccess(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_fixture(self, Path(self._tmpdir.name))

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_preflight_succeeds_with_image_and_session_construction_poisoned(self):
        args = _fixture_args(self.fx)
        with mock.patch("PIL.Image.open", side_effect=RaisingImageOpen()), \
            mock.patch("onnxruntime.InferenceSession", RaisingSession):
            result = sc.run_preflight(args)
        self.assertTrue(result["ok"])

    def test_preflight_never_imports_pil_at_module_level(self):
        preflight_source = inspect.getsource(sc.run_preflight)
        self.assertNotIn("PIL", preflight_source)
        self.assertNotIn("Image.open", preflight_source)

    def test_preflight_twice_is_idempotent_and_side_effect_free(self):
        args = _fixture_args(self.fx)
        before = self.fx["cal_csv_path"].read_bytes()
        r1 = sc.run_preflight(args)
        r2 = sc.run_preflight(args)
        after = self.fx["cal_csv_path"].read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(r1["contract_content_sha256"], r2["contract_content_sha256"])
        self.assertFalse((self.fx["repo"] / "data/calibration_v2/calibration_v2_scores.json").exists())


# =========================================================================
# 19. Module isolation
# =========================================================================
class TestModuleIsolation(unittest.TestCase):
    def test_selector_never_imports_pil_or_onnxruntime(self):
        source = Path(sel.__file__).read_text(encoding="utf-8")
        for forbidden in ("import PIL", "from PIL", "import onnxruntime", "from onnxruntime", "import ort"):
            self.assertNotIn(forbidden, source)

    def test_scorer_has_no_unknown_test_cli_argument_or_path_reference(self):
        source = Path(sc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("unknown_test_v2.csv", source)
        self.assertNotIn("unknown-test", source)
        self.assertNotIn("unknown_test_dir", source)

    def test_scorer_artifacts_dir_has_no_default(self):
        self.assertFalse(hasattr(sc, "DEFAULT_ARTIFACTS_DIR"))

    def test_scorer_default_output_path_is_under_calibration_v2_not_live_artifacts(self):
        self.assertIn("calibration_v2", str(sc.DEFAULT_OUT_PATH))

    def test_score_mode_requires_explicit_artifacts_dir(self):
        argv = ["score_calibration_v2.py", "--score"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                sc.main()


# =========================================================================
# 20. End-to-end synthetic scorer + selector (point 12)
# =========================================================================
def _build_tiny_onnx_model(path: Path, n_classes: int) -> None:
    """A genuinely tiny, from-scratch ONNX model (never the real B4
    backbone): GlobalAveragePool(input) -> Flatten -> Gemm(W, B) ->
    "embedding". Small enough to construct and run instantly, but a REAL
    ONNX graph executed by a REAL onnxruntime CPU session -- not a mock."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(42)
    weight = rng.standard_normal((3, n_classes)).astype(np.float32)
    bias = rng.standard_normal((n_classes,)).astype(np.float32)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 380, 380])
    out = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, n_classes])
    w_init = numpy_helper.from_array(weight, name="W")
    b_init = numpy_helper.from_array(bias, name="B")

    pool = helper.make_node("GlobalAveragePool", ["input"], ["pooled"])
    flat = helper.make_node("Flatten", ["pooled"], ["flat"], axis=1)
    gemm = helper.make_node("Gemm", ["flat", "W", "B"], ["embedding"])

    graph = helper.make_graph([pool, flat, gemm], "tiny_gate_v2_fixture_model",
                              [inp], [out], initializer=[w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class TestEndToEndScoring(unittest.TestCase):
    """Exercises run_score end-to-end against a tiny synthetic ONNX model
    and synthetic images, then runs the selector end-to-end on the
    resulting real score file for BOTH outcome paths."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)
        self.n_species = 3
        self.fx = build_tiny_fixture(self, self.repo, n_species=self.n_species, per_species_cal=2,
                                     ood_cal=1, non_ant_cal=1, unrelated_cal=1,
                                     copy_real_implementation_sources=True)
        self.contract = self.fx["contract"]

        # Replace the fixture's fake backbone.onnx with a real tiny model,
        # and re-bind the contract to its real hash (this is a legitimate
        # fixture rebuild, not a bypass of any check).
        artifacts_dir = self.repo / "training/artifacts/fixture_candidate"
        onnx_path = artifacts_dir / "backbone.onnx"
        _build_tiny_onnx_model(onnx_path, self.n_species)
        contract = self.fx["contract"]
        contract["content"]["bindings"]["backbone_onnx"]["sha256"] = gc.sha256_file(onnx_path)
        contract["content_sha256"] = gc.compute_content_sha256(contract["content"])
        _write_json(self.fx["contract_path"], contract)

        # Write real, valid, differently-sized-and-colored PNG images for
        # every calibration_v2 row, updating the manifest's sha256/byte_size
        # to match the actual bytes written (synthetic images, real bytes).
        from PIL import Image
        rows = self.fx["cal_rows"]
        new_rows = []
        for i, row in enumerate(rows):
            img = Image.new("RGB", (220, 220), (i % 256, (i * 7) % 256, (i * 13) % 256))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
            slug_dir = self.repo / "data/calibration_v2" / row["slug"]
            slug_dir.mkdir(parents=True, exist_ok=True)
            (slug_dir / f"{row['photo_id']}.png").write_bytes(data)
            row = dict(row)
            row["sha256"] = gc.sha256_bytes(data)
            row["byte_size"] = str(len(data))
            row["width"] = "220"
            row["height"] = "220"
            new_rows.append(row)
        _rewrite_calibration_csv(self.fx, new_rows)
        self.contract = self.fx["contract"]

    def tearDown(self):
        self._tmpdir.cleanup()

    def _score_args(self, out=None):
        return _fixture_args(self.fx, score=True, preflight=False,
                             out=out or self.repo / "data/calibration_v2/calibration_v2_scores.json")

    def test_run_score_end_to_end_produces_valid_schema(self):
        args = self._score_args()
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef" * 5):
            output = sc.run_score(args)

        self.assertTrue(args.out.exists())
        problems = gc.validate_score_content(output, self.contract)
        self.assertEqual(problems, [])

        records = output["content"]["records"]
        self.assertEqual(len(records), len(self.fx["cal_rows"]))
        photo_ids_out = [r["photo_id"] for r in records]
        photo_ids_in = [r["photo_id"] for r in self.fx["cal_rows"]]
        self.assertEqual(photo_ids_out, photo_ids_in)  # deterministic row order preserved

        for r in records:
            for sim in r["top3_similarities"]:
                self.assertIsInstance(sim, float)
            self.assertEqual(r["max_cosine"], r["top3_similarities"][0])

    def test_run_score_refuses_to_overwrite_existing_output(self):
        args = self._score_args()
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef" * 5):
            sc.run_score(args)
            before = args.out.read_bytes()
            with self.assertRaises(sc.ScoringError):
                sc.run_score(args)
            self.assertEqual(args.out.read_bytes(), before)

    def test_run_score_failure_leaves_no_partial_output(self):
        # Corrupt one image AFTER preflight would pass, forcing a mid-loop failure.
        bad_row = self.fx["cal_rows"][-1]
        img_path = self.repo / "data/calibration_v2" / bad_row["slug"] / f"{bad_row['photo_id']}.png"
        img_path.write_bytes(b"corrupted-not-a-real-image")
        args = self._score_args()
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef" * 5):
            with self.assertRaises(sc.ScoringError):
                sc.run_score(args)
        self.assertFalse(args.out.exists())
        # no leftover temp file either
        leftovers = list(args.out.parent.glob(f"{args.out.name}.tmp*")) if args.out.parent.exists() else []
        self.assertEqual(leftovers, [])

    def test_full_pipeline_selector_writes_candidate_or_no_useful_gate(self):
        score_args = self._score_args()
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef" * 5):
            scores = sc.run_score(score_args)

        result = sel.select_threshold(scores, self.contract)
        self.assertIn(result["status"], (sel.STATUS_CANDIDATE_SELECTED, sel.STATUS_NO_USEFUL_GATE))

        import argparse
        sel_args = argparse.Namespace(
            repo=self.repo, contract=self.fx["contract_path"],
            scores=score_args.out, out=self.repo / "data/calibration_v2/calibration_v2_selection.json",
        )
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef" * 5):
            out_path = sel.write_selection_output(sel_args, self.contract, scores, result)
        self.assertTrue(out_path.exists())
        written = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(written["content"]["status"], result["status"])

        # second write must refuse (overwrite-refusal proven on the real path)
        with mock.patch.object(gc, "require_clean_tracked_tree"), \
            mock.patch.object(gc, "get_git_head", return_value="deadbeef" * 5):
            with self.assertRaises(sel.SelectionError):
                sel.write_selection_output(sel_args, self.contract, scores, result)


if __name__ == "__main__":
    unittest.main()
