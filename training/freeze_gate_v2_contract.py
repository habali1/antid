#!/usr/bin/env python3
"""freeze_gate_v2_contract.py — builds/checks training/gate_v2_selection_contract.json
from this file's literal, committed content.

Two modes:
  (default) / --check   Reconstructs the expected content in memory and
                         compares it byte-for-byte against the existing file
                         on disk. Writes NOTHING. Importing this module also
                         writes nothing -- all of the above lives inside
                         main(), never at module scope.
  --write                Atomically writes the reconstructed contract.
                         Refuses to run if a real calibration_v2_scores.json
                         or calibration_v2_selection.json already exists --
                         the contract must never be rewritten once real
                         scoring has started.

Usage:
    python freeze_gate_v2_contract.py            # --check (default)
    python freeze_gate_v2_contract.py --check
    python freeze_gate_v2_contract.py --write
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import gate_v2_contract as gc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONTRACT_PATH = HERE / "gate_v2_selection_contract.json"
REAL_SCORES_PATH = REPO / "data/calibration_v2/calibration_v2_scores.json"
REAL_SELECTION_PATH = REPO / "data/calibration_v2/calibration_v2_selection.json"
CALIBRATION_V2_CSV_SHA256 = "8201200f4c5d7c869a71c3c927ccee0dfb83d6b91a595241dd49996deee1d46e"


class FreezeError(RuntimeError):
    """Refuses to write. Always fails closed."""


def compute_calibration_identity_order_sha256(repo: Path) -> str:
    """Binds record order to the frozen calibration_v2.csv WITHOUT opening
    any image: verifies the CSV's own byte hash against the frozen binding
    first, then reads only its (photo_id, observation_uuid, sha256) columns
    -- CSV metadata, never image bytes -- in file row order, and hashes that
    ordered identity sequence via the one canonical serialization shared
    with the scorer and the validator."""
    csv_path = repo / gc.FROZEN_BINDING_PATHS["calibration_v2_csv"]
    actual_csv_hash = gc.sha256_file(csv_path)
    if actual_csv_hash != CALIBRATION_V2_CSV_SHA256:
        raise FreezeError(f"{csv_path} sha256 {actual_csv_hash} != frozen binding "
                          f"{CALIBRATION_V2_CSV_SHA256} -- refusing to derive an identity-order "
                          f"hash from a calibration_v2.csv that does not match the frozen binding")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    triples = [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in rows]
    return gc.compute_identity_order_sha256(triples)


def build_content(implementation_source_hashes: dict[str, str], identity_order_sha256: str) -> dict:
    """Pure function: no filesystem access except reading the four
    implementation source hashes and the identity-order hash the caller
    already computed. Everything else is a literal value written here (or a
    constant imported from gate_v2_contract.py, the ONE place these frozen
    semantic values are defined), so re-running this function is
    deterministic by construction."""
    return {
        "policy_name": gc.FROZEN_POLICY_NAME,
        "is_v1_reconstruction": gc.FROZEN_IS_V1_RECONSTRUCTION,
        "notes": (
            "New deterministic v2 contract. v1 recorded the selected 0.60 result "
            "(data/calibration_v1/calibration_v1.json:frozen_candidate_abstention_threshold) "
            "but not a reproducible selection algorithm -- eval_unknown_test.py's own docstring "
            "states 0.60 was a hardcoded constant, added to machine_readable_rule after the fact "
            "purely for reproduction of the value, not as a record of how it was chosen. This "
            "contract is not an attempt to reconstruct that undocumented process; it is a new, "
            "fully deterministic algorithm for v2."
        ),
        "bindings": {
            "calibration_v2_csv": {
                "path": gc.FROZEN_BINDING_PATHS["calibration_v2_csv"],
                "sha256": CALIBRATION_V2_CSV_SHA256,
            },
            "calibration_v2_json": {
                "path": gc.FROZEN_BINDING_PATHS["calibration_v2_json"],
                "sha256": "d7d96d321c7ebb4a5b3cd55ded2f2bb72628b6ec38b82ef135c5be088369ed9e",
            },
            "unknown_test_v2_csv": {
                "path": gc.FROZEN_BINDING_PATHS["unknown_test_v2_csv"],
                "sha256": "d706cb6dbda7a672594fc425ce5c8f8f0f6f868662c3d52c5f9c88083163b09f",
            },
            "unknown_test_v2_json": {
                "path": gc.FROZEN_BINDING_PATHS["unknown_test_v2_json"],
                "sha256": "eea1cbb3118dfc05f42f477ef58d6ccbc5458df1e8a5496845a146d161a89e6d",
            },
            "candidate_run_manifest": {
                "path": gc.FROZEN_BINDING_PATHS["candidate_run_manifest"],
                "sha256": "6f8bfe3141a9870c37f6810da6b9a5459cf701242f9cbb3d32a8b01c2f0f265b",
            },
            "parity_report": {
                "path": gc.FROZEN_BINDING_PATHS["parity_report"],
                "sha256": "bec08235a48e9583f2b40556c9c8d860c41904729bcded9d6f91c37716c8297e",
                "applicability": (
                    "65-class candidate (northeast_v1_b4_dev_v2); pinned development-split "
                    "260-image sample only (4/species x 65, val split, seed 42) -- not "
                    "benchmark, not calibration, not a population-level equivalence bound. "
                    "Satisfies the pre-calibration ONNX-CPU parity prerequisite; a new parity "
                    "run is not required before scoring calibration_v2."
                ),
            },
            "backbone_onnx": {
                "path": gc.FROZEN_BINDING_PATHS["backbone_onnx"],
                "sha256": "fc22d26ae5c73d20613dafcef02e75291c8779ec8a1296ebc9a72f0e7d7f826b",
            },
            "prototypes_npy": {
                "path": gc.FROZEN_BINDING_PATHS["prototypes_npy"],
                "sha256": "0e52a7f350996a48f4129e1924e9b0af79517e4c3273dd9ecc7e851121d43825",
            },
            "taxonomy_json": {
                "path": gc.FROZEN_BINDING_PATHS["taxonomy_json"],
                "sha256": "c8672287d59b5b9d3fb9aec5fd85a008ad30441f2ce3159713a3752b37a04767",
            },
        },
        "implementation_sources": {
            "api_inference": {
                "path": gc.IMPLEMENTATION_SOURCE_PATHS["api_inference"],
                "sha256": implementation_source_hashes["api_inference"],
            },
            "gate_v2_contract": {
                "path": gc.IMPLEMENTATION_SOURCE_PATHS["gate_v2_contract"],
                "sha256": implementation_source_hashes["gate_v2_contract"],
            },
            "score_calibration_v2": {
                "path": gc.IMPLEMENTATION_SOURCE_PATHS["score_calibration_v2"],
                "sha256": implementation_source_hashes["score_calibration_v2"],
            },
            "select_gate_v2_threshold": {
                "path": gc.IMPLEMENTATION_SOURCE_PATHS["select_gate_v2_threshold"],
                "sha256": implementation_source_hashes["select_gate_v2_threshold"],
            },
        },
        "approved_outputs": dict(gc.FROZEN_APPROVED_OUTPUTS),
        "dataset_quotas": {
            "calibration_v2": dict(gc.FROZEN_DATASET_QUOTAS["calibration_v2"]),
            "unknown_test_v2": {
                **gc.FROZEN_DATASET_QUOTAS["unknown_test_v2"],
                "purpose": gc.FROZEN_UNKNOWN_TEST_V2_PURPOSE,
            },
        },
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


def build_contract() -> dict:
    """Computes the four implementation-source hashes and the calibration_v2
    row identity-order hash fresh (this is the ONE place these are allowed
    to be written INTO the contract -- everywhere else only verifies them)
    and assembles the full frozen contract dict, self-validated before being
    returned. Reads calibration_v2.csv's METADATA ONLY (photo_id,
    observation_uuid, sha256 columns) for the identity-order hash -- never
    opens an image."""
    impl_hashes = {
        name: gc.canonical_lf_sha256_file(REPO / path)
        for name, path in gc.IMPLEMENTATION_SOURCE_PATHS.items()
    }
    identity_order_sha256 = compute_calibration_identity_order_sha256(REPO)
    content = build_content(impl_hashes, identity_order_sha256)
    content_sha256 = gc.compute_content_sha256(content)
    contract = {
        "schema_version": gc.CONTRACT_SCHEMA_VERSION,
        "status": gc.CONTRACT_STATUS_FROZEN,
        "content": content,
        "content_sha256": content_sha256,
        "generation": {
            "note": "generation metadata is NOT part of content and NOT covered by content_sha256",
            "generator": "training/freeze_gate_v2_contract.py",
        },
    }
    problems = gc.validate_contract(contract)
    if problems:
        raise FreezeError(f"reconstructed contract failed self-validation: {problems}")
    return contract


def serialize(contract: dict) -> str:
    return json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def cmd_check() -> int:
    contract = build_contract()
    expected_text = serialize(contract)
    if not CONTRACT_PATH.exists():
        print(f"[check] FAILED: {CONTRACT_PATH} does not exist")
        return 1
    actual_text = CONTRACT_PATH.read_text(encoding="utf-8")
    if actual_text != expected_text:
        print(f"[check] FAILED: {CONTRACT_PATH} does not match the reconstructed content byte-for-byte")
        return 1
    print(f"[check] PASS: {CONTRACT_PATH} matches the reconstructed content byte-for-byte")
    print(f"content_sha256: {contract['content_sha256']}")
    return 0


def cmd_write() -> int:
    if REAL_SCORES_PATH.exists() or REAL_SELECTION_PATH.exists():
        raise FreezeError(
            f"refusing to (re)write {CONTRACT_PATH}: real scoring output already exists "
            f"({REAL_SCORES_PATH if REAL_SCORES_PATH.exists() else REAL_SELECTION_PATH}). "
            f"The contract must never change after real scoring has started."
        )
    contract = build_contract()
    text = serialize(contract)
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + f".tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, CONTRACT_PATH)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"wrote {CONTRACT_PATH}")
    print(f"content_sha256: {contract['content_sha256']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args()

    try:
        return cmd_write() if args.write else cmd_check()
    except FreezeError as e:
        print(f"[freeze_gate_v2_contract] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
