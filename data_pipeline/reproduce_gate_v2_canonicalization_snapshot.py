#!/usr/bin/env python3
"""Deterministic, network-free reproduction/check of the known_holdout
parent-species-collapse correction applied to
data/gate_v2_readiness/candidates.csv and gate_v2_readiness.json.

Immutable input source: the ORIGINAL (pre-correction) Phase 5A snapshot
committed at e30fe4e, read via `git show e30fe4e:<path>` -- never a working
copy on disk, so this can never accidentally rerun against already-corrected
data and never needs a second full-size CSV committed to the repo.

This tool makes NO network access whatsoever. All taxonomic ancestry
knowledge it needs comes from the already-captured, hash-bound lineage
evidence file at data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json
(read from the current working tree, since that evidence file did not exist
at e30fe4e and is not itself being reproduced here -- only consumed).

It:
  1. Reads candidates.csv and gate_v2_readiness.json as they existed at
     e30fe4e and verifies the ORIGINAL candidates.csv sha256 matches the
     expected pre-correction hash.
  2. Applies the exact same deterministic transform used for the real
     correction: add source_taxon_id/source_taxon_name/source_taxon_rank to
     every row; canonicalize species/taxon_id on every known_holdout row
     whose taxon_id is a verified source taxon in the lineage evidence.
  3. Applies the same minimal JSON patch to gate_v2_readiness.json
     (candidates_csv record + known_holdout_taxonomic_canonicalization
     block), deriving every value from the transform's own output -- never
     hardcoding the affected row count, slug, or taxon id.
  4. Writes both reproduced files to a temporary directory and compares them
     byte-for-byte against the current working-tree files.
  5. Verifies the reproduced candidates.csv sha256 matches the expected
     post-correction hash.

Exits nonzero and explains clearly if the source commit/blob is missing, if
either expected hash does not match, or if either reproduced file differs
from the working tree.

Usage:
    python reproduce_gate_v2_canonicalization_snapshot.py
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from audit_gate_v2_readiness import (  # noqa: E402
    CANDIDATE_CSV_FIELDS,
    sha256_bytes,
    sha256_file,
    write_json,
)

SOURCE_COMMIT = "e30fe4e"
CANDIDATES_REL_PATH = "data/gate_v2_readiness/candidates.csv"
READINESS_REL_PATH = "data/gate_v2_readiness/gate_v2_readiness.json"
LINEAGE_EVIDENCE_REL_PATH = "data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json"

EXPECTED_ORIGINAL_CANDIDATES_SHA256 = "01c2c37c785d1e1b1a33efefe2709e3927cabb372ff5b0e4755121658a4b0a77"
EXPECTED_CORRECTED_CANDIDATES_SHA256 = "eed811a31c2febb98fb2c8f4f4b958e105c1b7fe7aa80b3f3de1e2ef8152c4f9"
EXPECTED_ROW_COUNT = 4012


class ReproductionError(RuntimeError):
    """The deterministic reproduction could not be verified. Always fails closed."""


def git_show_bytes(repo: Path, commit: str, rel_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{rel_path}"], cwd=repo, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ReproductionError(
            f"could not read {rel_path!r} at commit {commit} -- is the commit/blob available? "
            f"git stderr: {exc.stderr.decode('utf-8', errors='replace')}"
        ) from exc


def apply_canonicalization(original_csv_bytes: bytes, lineage_evidence: dict[str, Any]
                           ) -> tuple[bytes, int, dict[str, Any]]:
    """Pure function: original candidates.csv bytes + lineage evidence ->
    (corrected candidates.csv bytes, affected row count, canonicalization
    summary). No filesystem or network access."""
    reader = csv.DictReader(io.StringIO(original_csv_bytes.decode("utf-8")))
    rows = list(reader)
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ReproductionError(f"original candidates.csv has {len(rows)} rows, expected {EXPECTED_ROW_COUNT}")

    verified_source_taxa = lineage_evidence.get("verified_source_taxa", {})
    canonical_targets = lineage_evidence.get("canonical_targets", {})

    affected_slugs: set[str] = set()
    affected_taxon_ids: set[int] = set()
    affected_count = 0
    for r in rows:
        r["source_taxon_id"] = r["taxon_id"]
        r["source_taxon_name"] = r["species"]
        if r["category"] == "known_holdout" and r["taxon_id"] in verified_source_taxa:
            source_info = verified_source_taxa[r["taxon_id"]]
            target_id = source_info["canonical_target_taxon_id"]
            target_info = canonical_targets[str(target_id)]
            r["source_taxon_rank"] = source_info["rank"]
            r["taxon_id"] = str(target_id)
            r["species"] = target_info["name"]
            affected_count += 1
            affected_slugs.add(r["slug"])
            affected_taxon_ids.add(target_id)
        elif r["category"] == "known_holdout":
            r["source_taxon_rank"] = "species"
        else:
            r["source_taxon_rank"] = ""

    if len(affected_slugs) > 1 or len(affected_taxon_ids) > 1:
        raise ReproductionError(
            f"lineage evidence implies multiple distinct affected slugs/targets "
            f"({sorted(affected_slugs)} / {sorted(affected_taxon_ids)}) -- this reproduction tool "
            f"assumes a single-target canonicalization policy"
        )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CANDIDATE_CSV_FIELDS)  # default \r\n, matching the committed file
    writer.writeheader()
    writer.writerows(rows)
    corrected_bytes = buf.getvalue().encode("utf-8")

    summary = {
        "affected_known_holdout_rows": affected_count,
        "affected_slug": next(iter(affected_slugs)) if affected_slugs else None,
        "canonical_taxon_id": next(iter(affected_taxon_ids)) if affected_taxon_ids else None,
        "source_taxon_ids_verified": sorted(int(x) for x in verified_source_taxa),
    }
    return corrected_bytes, affected_count, summary


def apply_readiness_patch(original_readiness: dict[str, Any], corrected_csv_hash: str,
                          lineage_evidence_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Pure function: mutates only the two things the real correction
    touched in gate_v2_readiness.json -- never regenerates any other field
    (known_species_readiness, exclusion_sources, etc. are left untouched,
    exactly matching the real correction's scope)."""
    patched = json.loads(json.dumps(original_readiness))  # deep copy
    patched["candidates_csv"]["sha256"] = corrected_csv_hash
    patched["candidates_csv"]["rows"] = EXPECTED_ROW_COUNT
    patched["known_holdout_taxonomic_canonicalization"] = {
        "policy": "parent_species_collapse_v1",
        "lineage_evidence": {
            "path": LINEAGE_EVIDENCE_REL_PATH,
            "sha256": sha256_file(lineage_evidence_path),
        },
        "affected_known_holdout_rows": summary["affected_known_holdout_rows"],
        "affected_slug": summary["affected_slug"],
        "canonical_taxon_id": summary["canonical_taxon_id"],
        "source_taxon_ids_verified": summary["source_taxon_ids_verified"],
        "candidates_csv_added_columns": ["source_taxon_id", "source_taxon_name", "source_taxon_rank"],
    }
    return patched


def reproduce(repo: Path, out_dir: Path) -> dict[str, Any]:
    original_csv_bytes = git_show_bytes(repo, SOURCE_COMMIT, CANDIDATES_REL_PATH)
    actual_original_hash = sha256_bytes(original_csv_bytes)
    if actual_original_hash != EXPECTED_ORIGINAL_CANDIDATES_SHA256:
        raise ReproductionError(
            f"original candidates.csv at {SOURCE_COMMIT} has sha256 {actual_original_hash}, "
            f"expected {EXPECTED_ORIGINAL_CANDIDATES_SHA256} -- refusing to reproduce from an "
            f"unexpected source snapshot")

    original_readiness_bytes = git_show_bytes(repo, SOURCE_COMMIT, READINESS_REL_PATH)
    original_readiness = json.loads(original_readiness_bytes.decode("utf-8"))

    lineage_path = repo / LINEAGE_EVIDENCE_REL_PATH
    if not lineage_path.exists():
        raise ReproductionError(f"lineage evidence file missing (no network access permitted here): {lineage_path}")
    lineage_evidence = json.loads(lineage_path.read_text(encoding="utf-8"))

    corrected_csv_bytes, affected_count, summary = apply_canonicalization(original_csv_bytes, lineage_evidence)
    corrected_hash = sha256_bytes(corrected_csv_bytes)
    if corrected_hash != EXPECTED_CORRECTED_CANDIDATES_SHA256:
        raise ReproductionError(
            f"reproduced candidates.csv has sha256 {corrected_hash}, expected "
            f"{EXPECTED_CORRECTED_CANDIDATES_SHA256} -- reproduction does not match the committed correction")

    patched_readiness = apply_readiness_patch(original_readiness, corrected_hash, lineage_path, summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "candidates.csv"
    out_csv.write_bytes(corrected_csv_bytes)
    out_readiness = out_dir / "gate_v2_readiness.json"
    write_json(out_readiness, patched_readiness)

    return {
        "out_csv": out_csv, "out_readiness": out_readiness,
        "original_hash": actual_original_hash, "corrected_hash": corrected_hash,
        "affected_count": affected_count, "summary": summary,
    }


def compare_with_working_tree(repo: Path, reproduced: dict[str, Any]) -> list[str]:
    mismatches = []
    working_csv = repo / CANDIDATES_REL_PATH
    working_readiness = repo / READINESS_REL_PATH
    if reproduced["out_csv"].read_bytes() != working_csv.read_bytes():
        mismatches.append(f"{CANDIDATES_REL_PATH} differs from reproduction")
    if reproduced["out_readiness"].read_bytes() != working_readiness.read_bytes():
        mismatches.append(f"{READINESS_REL_PATH} differs from reproduction")
    return mismatches


def main() -> int:
    repo = REPO
    with tempfile.TemporaryDirectory() as tmp:
        try:
            reproduced = reproduce(repo, Path(tmp))
        except ReproductionError as e:
            print(f"[reproduce] FAILURE: {e}")
            return 1

        print(f"[reproduce] original candidates.csv sha256 verified: {reproduced['original_hash']}")
        print(f"[reproduce] reproduced candidates.csv sha256 verified: {reproduced['corrected_hash']}")
        print(f"[reproduce] {reproduced['affected_count']} known_holdout rows canonicalized: "
             f"{reproduced['summary']}")
        print(f"[reproduce] row order/selection_status/intended_dataset/allocations preserved "
             f"(only species/taxon_id on affected rows, plus 3 new columns on every row)")

        mismatches = compare_with_working_tree(repo, reproduced)
        if mismatches:
            print(f"[reproduce] FAILURE: reproduced output differs from working tree: {mismatches}")
            return 1
        print("[reproduce] PASS: reproduced candidates.csv and gate_v2_readiness.json are "
             "byte-for-byte identical to the working tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
