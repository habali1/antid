#!/usr/bin/env python3
"""freeze_gate_v2_evaluation_contract.py — builds/checks
training/gate_v2_evaluation_contract.json from this file's literal,
committed content (via gate_v2_evaluation_contract.py's FROZEN_* constants).

Two modes:
  (default) / --check   Reconstructs the expected content in memory and
                         compares it byte-for-byte against the existing file
                         on disk. Writes NOTHING.
  --write                Atomically writes the reconstructed contract.
                         Refuses to run if the unknown_test_v2 evaluation
                         attempt marker or output already exists -- this
                         contract must never be rewritten once the single
                         permitted evaluation has started.

Usage:
    python freeze_gate_v2_evaluation_contract.py            # --check (default)
    python freeze_gate_v2_evaluation_contract.py --check
    python freeze_gate_v2_evaluation_contract.py --write
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import gate_v2_contract as gc
import gate_v2_evaluation_contract as ec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONTRACT_PATH = HERE / "gate_v2_evaluation_contract.json"
REAL_ATTEMPT_MARKER_PATH = REPO / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
REAL_EVAL_OUTPUT_PATH = REPO / "data/unknown_test_v2/unknown_test_v2_eval.json"
UNKNOWN_TEST_V2_CSV_SHA256 = ec.FROZEN_UNKNOWN_TEST_V2_HASHES["unknown_test_v2_csv"]


class FreezeError(RuntimeError):
    """Refuses to write. Always fails closed."""


def compute_unknown_test_v2_identity_order_sha256(repo: Path) -> str:
    """Binds record order to the frozen unknown_test_v2.csv WITHOUT opening
    any image: verifies the CSV's own byte hash against the frozen binding
    first, then reads only its (photo_id, observation_uuid, sha256)
    columns -- CSV metadata, never image bytes -- in file row order."""
    csv_path = repo / ec.FROZEN_UNKNOWN_TEST_V2_BINDING_PATHS["unknown_test_v2_csv"]
    actual_csv_hash = gc.sha256_file(csv_path)
    if actual_csv_hash != UNKNOWN_TEST_V2_CSV_SHA256:
        raise FreezeError(f"{csv_path} sha256 {actual_csv_hash} != frozen binding "
                          f"{UNKNOWN_TEST_V2_CSV_SHA256} -- refusing to derive an identity-order "
                          f"hash from an unknown_test_v2.csv that does not match the frozen binding")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    triples = [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in rows]
    return gc.compute_identity_order_sha256(triples)


def build_content(implementation_source_hashes: dict[str, str], identity_order_sha256: str) -> dict:
    """Pure function: no filesystem access except reading the six
    implementation source hashes and the identity-order hash the caller
    already computed. Everything else is a constant imported from
    gate_v2_evaluation_contract.py, the ONE place these frozen values are
    defined, so re-running this function is deterministic by construction."""
    return {
        "policy_name": ec.FROZEN_POLICY_NAME,
        "notes": (
            "Single-use, independent evaluation of the ALREADY-SELECTED and reviewed Phase 5C2 "
            "calibration_v2 candidate threshold (0.61) against unknown_test_v2. This contract "
            "never re-selects, sweeps, or searches a threshold -- it applies the one frozen "
            "candidate exactly once and reports precommitted pass/fail criteria."
        ),
        "selection_contract": {"content_sha256": ec.FROZEN_SELECTION_CONTRACT_CONTENT_SHA256},
        "scores_binding": {"byte_sha256": ec.FROZEN_SCORES_BYTE_SHA256,
                           "content_sha256": ec.FROZEN_SCORES_CONTENT_SHA256},
        "selection_binding": {"byte_sha256": ec.FROZEN_SELECTION_BYTE_SHA256,
                              "content_sha256": ec.FROZEN_SELECTION_CONTENT_SHA256},
        "threshold": {
            "threshold_integer": ec.FROZEN_THRESHOLD_INTEGER,
            "threshold": ec.FROZEN_THRESHOLD,
            "source": ec.FROZEN_THRESHOLD_SOURCE,
            "comparison": ec.FROZEN_DECISION_SHAPE["comparison"],
            "equal_threshold_action": ec.FROZEN_DECISION_SHAPE["equal_threshold_action"],
            "required_status": ec.FROZEN_THRESHOLD_STATUS_REQUIRED,
        },
        "bindings": {
            name: {"path": ec.FROZEN_BINDING_PATHS[name], "sha256": ec.FROZEN_BINDING_HASHES[name]}
            for name in ec.REQUIRED_BINDING_KEYS
        },
        "implementation_sources": {
            name: {"path": ec.IMPLEMENTATION_SOURCE_PATHS[name], "sha256": implementation_source_hashes[name]}
            for name in ec.IMPLEMENTATION_SOURCE_KEYS
        },
        "approved_outputs": dict(ec.FROZEN_APPROVED_OUTPUTS),
        "dataset_quota": dict(ec.FROZEN_DATASET_QUOTA),
        "runtime": dict(ec.FROZEN_RUNTIME),
        "decision_shape": dict(ec.FROZEN_DECISION_SHAPE),
        "validation_criteria": dict(ec.FROZEN_VALIDATION_CRITERIA),
        "single_use_rule": dict(ec.FROZEN_SINGLE_USE_RULE),
        "closed_sources": list(ec.FROZEN_CLOSED_SOURCES),
        "gate_framing": ec.FROZEN_GATE_FRAMING,
        ec.CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256: identity_order_sha256,
    }


def build_contract() -> dict:
    """Computes the six implementation-source hashes and the unknown_test_v2
    row identity-order hash fresh (the ONE place these are allowed to be
    written INTO the contract) and assembles the full frozen contract dict,
    self-validated before being returned. Reads unknown_test_v2.csv's
    METADATA ONLY -- never opens an image."""
    impl_hashes = {
        name: gc.canonical_lf_sha256_file(REPO / path)
        for name, path in ec.IMPLEMENTATION_SOURCE_PATHS.items()
    }
    identity_order_sha256 = compute_unknown_test_v2_identity_order_sha256(REPO)
    content = build_content(impl_hashes, identity_order_sha256)
    content_sha256 = ec.compute_content_sha256(content)
    contract = {
        "schema_version": ec.EVAL_CONTRACT_SCHEMA_VERSION,
        "status": ec.EVAL_CONTRACT_STATUS_FROZEN,
        "content": content,
        "content_sha256": content_sha256,
        "generation": {
            "note": "generation metadata is NOT part of content and NOT covered by content_sha256",
            "generator": "training/freeze_gate_v2_evaluation_contract.py",
        },
    }
    problems = ec.validate_evaluation_contract(contract)
    if problems:
        raise FreezeError(f"reconstructed evaluation contract failed self-validation: {problems}")
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
    if REAL_ATTEMPT_MARKER_PATH.exists() or REAL_EVAL_OUTPUT_PATH.exists():
        raise FreezeError(
            f"refusing to (re)write {CONTRACT_PATH}: the unknown_test_v2 evaluation attempt marker "
            f"or output already exists "
            f"({REAL_ATTEMPT_MARKER_PATH if REAL_ATTEMPT_MARKER_PATH.exists() else REAL_EVAL_OUTPUT_PATH}). "
            f"The evaluation contract must never change once the single permitted evaluation has started."
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
        print(f"[freeze_gate_v2_evaluation_contract] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
