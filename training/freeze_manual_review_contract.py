#!/usr/bin/env python3
"""freeze_manual_review_contract.py — builds/checks
training/manual_review_contract.json, binding the ALREADY-PUBLISHED
training/manual_review_queue.csv and training/manual_review_queue_summary.json
(generated first, independently, by generate_manual_review_queue.py --write)
by their exact byte hashes, row count, and ordered-identity hash.

Two modes:
  --check (default)  Reconstructs the expected content in memory and
                      compares it byte-for-byte against the existing file
                      on disk. Writes NOTHING.
  --write             Atomically and exclusively writes the reconstructed
                      contract -- refuses to overwrite an existing file.

Usage:
    python freeze_manual_review_contract.py            # --check (default)
    python freeze_manual_review_contract.py --check
    python freeze_manual_review_contract.py --write
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402

CONTRACT_PATH = REPO / mrc.CONTRACT_REL_PATH

IMPLEMENTATION_SOURCE_PATHS = {
    "manual_review_contract": "training/manual_review_contract.py",
    "append_only_ledger": "training/append_only_ledger.py",
    "generate_manual_review_queue": "training/generate_manual_review_queue.py",
    "freeze_manual_review_contract": "training/freeze_manual_review_contract.py",
    "perceptual_hash": "training/perceptual_hash.py",
    "manual_review_tool": "training/manual_review_tool.py",
    "scan_perceptual_duplicates": "training/scan_perceptual_duplicates.py",
    "adjudicate_pairs": "training/adjudicate_pairs.py",
    "finalize_stop_status": "training/finalize_stop_status.py",
}


class FreezeError(RuntimeError):
    """Refuses to write. Always fails closed."""


def _build_queue_artifact() -> dict:
    queue_path = REPO / mrc.QUEUE_ARTIFACT_REL_PATH
    if not queue_path.exists():
        raise FreezeError(f"queue CSV does not exist yet -- run generate_manual_review_queue.py "
                          f"--write first: {queue_path}")
    queue_bytes = queue_path.read_bytes()
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    for row in rows:
        row["queue_index"] = int(row["queue_index"])
    rows.sort(key=lambda r: r["queue_index"])
    if len(rows) != mrc.N_QUEUE_ROWS:
        raise FreezeError(f"{queue_path} has {len(rows)} rows, expected exactly {mrc.N_QUEUE_ROWS}")
    identity_order_sha256 = mrc.compute_queue_identity_order_sha256(rows)
    return {
        "path": mrc.QUEUE_ARTIFACT_REL_PATH,
        "byte_sha256": mrc.sha256_bytes(queue_bytes),
        "row_count": len(rows),
        "identity_order_sha256": identity_order_sha256,
    }


def _build_summary_artifact() -> dict:
    summary_path = REPO / mrc.SUMMARY_ARTIFACT_REL_PATH
    if not summary_path.exists():
        raise FreezeError(f"summary JSON does not exist yet -- run generate_manual_review_queue.py "
                          f"--write first: {summary_path}")
    summary_bytes = summary_path.read_bytes()
    try:
        summary = json.loads(summary_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise FreezeError(f"{summary_path} is not valid JSON: {exc}") from exc
    content_sha256 = summary.get("content_sha256")
    if not mrc.is_sha256_hex(content_sha256):
        raise FreezeError(f"{summary_path} content_sha256 is missing or malformed")
    return {
        "path": mrc.SUMMARY_ARTIFACT_REL_PATH,
        "byte_sha256": mrc.sha256_bytes(summary_bytes),
        "content_sha256": content_sha256,
    }


def build_contract() -> dict:
    """Reads the already-published queue.csv/summary.json (never an image)
    and hashes the eight implementation source files (the ONE place those
    hashes are allowed to be written INTO the contract) to assemble the
    full frozen contract dict, self-validated before being returned."""
    queue_artifact = _build_queue_artifact()
    summary_artifact = _build_summary_artifact()
    implementation_sources = {
        name: {"path": path, "sha256": mrc.canonical_lf_sha256_file(REPO / path)}
        for name, path in sorted(IMPLEMENTATION_SOURCE_PATHS.items())
    }
    content = mrc.build_contract_content(
        queue_artifact=queue_artifact, summary_artifact=summary_artifact,
        implementation_sources=implementation_sources,
    )
    content_sha256 = mrc.compute_content_sha256(content)
    contract = {
        "schema_version": mrc.SCHEMA_VERSION,
        "status": mrc.CONTRACT_STATUS_FROZEN,
        "content": content,
        "content_sha256": content_sha256,
        "generation": {
            "note": "generation metadata is NOT part of content and NOT covered by content_sha256",
            "generator": "training/freeze_manual_review_contract.py",
        },
    }
    problems = mrc.validate_contract_structure(contract)
    if problems:
        raise FreezeError(f"reconstructed contract failed self-validation: {problems}")
    return contract


def serialize(contract: dict) -> str:
    return json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def cmd_check() -> int:
    contract = build_contract()
    expected_bytes = serialize(contract).encode("utf-8")
    if not CONTRACT_PATH.exists():
        print(f"[check] FAILED: {CONTRACT_PATH} does not exist")
        return 1
    actual_bytes = CONTRACT_PATH.read_bytes()
    if actual_bytes != expected_bytes:
        print(f"[check] FAILED: {CONTRACT_PATH} does not match the reconstructed content byte-for-byte")
        return 1
    print(f"[check] PASS: {CONTRACT_PATH} matches the reconstructed content byte-for-byte")
    print(f"content_sha256: {contract['content_sha256']}")
    return 0


def cmd_write() -> int:
    if CONTRACT_PATH.exists():
        raise FreezeError(f"refusing to overwrite existing contract: {CONTRACT_PATH}")
    contract = build_contract()
    data = serialize(contract).encode("utf-8")
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + f".tmp{os.getpid()}")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(str(tmp), str(CONTRACT_PATH))
    finally:
        tmp.unlink(missing_ok=True)
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
        print(f"[freeze_manual_review_contract] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
