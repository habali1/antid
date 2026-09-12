#!/usr/bin/env python3
"""Capture/check tool for the known_holdout parent-species-collapse lineage
evidence file (data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json).

This performs the MINIMUM metadata-only iNaturalist taxa lookups needed to
verify that a source (observed) taxon is a taxonomic descendant of a
canonical target species before that source's known_holdout rows are
relabeled to the canonical species. No image is ever requested -- only
GET https://api.inaturalist.org/v1/taxa/{id}[,{id2},...] calls.

For one canonicalization (one canonical target with N source taxa), exactly
TWO metadata GET requests are made: one canonical-target lookup plus one
batched source-taxa lookup covering all N source ids -- never "one GET" per
canonicalization, and never one GET per source id.

Two mutually exclusive modes:
  (default) capture   Fetches metadata and writes the evidence file --
                       ATOMICALLY (temp file + os.replace), and ONLY if the
                       file does not already exist. Refuses to overwrite an
                       existing, hash-bound evidence file; use --check to
                       verify one instead. generated_at_utc is stamped once,
                       at first capture, and never rewritten afterward.
  --check              Read-only: fetches the same metadata, semantically
                       compares it against the frozen evidence file already
                       on disk (ignoring the timestamp and hash fields,
                       which are expected to be stable/irrelevant to
                       semantic content), and writes NOTHING. Exits nonzero
                       on any semantic disagreement.

Usage:
    python build_known_holdout_lineage_evidence.py            # first capture only
    python build_known_holdout_lineage_evidence.py --check    # verify, no writes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
API = "https://api.inaturalist.org/v1"
USER_AGENT = "AntID-gate-v2-lineage-check/1.0 (educational project, metadata-only)"
POLICY = "parent_species_collapse_v1"

# canonical_target_taxon_id -> [source (observed) taxon ids that must be
# verified as descendants before their known_holdout rows are canonicalized]
CANONICALIZATIONS = {
    126838: [735984, 313487],  # Eciton burchellii <- E. b. parvispinum, E. b. foreli
}

OUTPUT_PATH = REPO / "data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json"

# Fields that carry the actual taxonomic claim -- compared semantically by
# --check. generated_at_utc and evidence_sha256 are deliberately excluded:
# the timestamp is expected to differ from any fresh fetch, and the hash is
# a function of the content, not content itself.
SEMANTIC_FIELDS = ("policy", "canonical_targets", "verified_source_taxa")


class LineageEvidenceError(RuntimeError):
    """A capture or check-mode failure -- always fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compute_evidence_hash(evidence_without_hash: dict) -> str:
    canonical = json.dumps(evidence_without_hash, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fetch_lineage_metadata(client: httpx.Client) -> tuple[dict[str, dict], dict[str, dict], int]:
    """Performs exactly 2 metadata GET requests per canonicalization group
    (one canonical-target lookup, one batched source-taxa lookup) and
    returns (canonical_targets, verified_source_taxa, request_count).
    Raises LineageEvidenceError if any source taxon's ancestry does not
    actually contain its claimed canonical target -- this function never
    fabricates or assumes ancestry."""
    verified_source_taxa: dict[str, dict] = {}
    canonical_targets: dict[str, dict] = {}
    request_count = 0

    for target_id, source_ids in CANONICALIZATIONS.items():
        time.sleep(1.05)
        resp = client.get(f"{API}/taxa/{target_id}")
        resp.raise_for_status()
        request_count += 1
        target_results = resp.json().get("results") or []
        if not target_results:
            raise LineageEvidenceError(f"canonical target taxon {target_id} not found via iNaturalist API")
        target_result = target_results[0]
        canonical_targets[str(target_id)] = {
            "taxon_id": target_id, "name": target_result["name"], "rank": target_result["rank"],
        }

        time.sleep(1.05)
        ids_param = ",".join(str(i) for i in source_ids)
        resp = client.get(f"{API}/taxa/{ids_param}")
        resp.raise_for_status()
        request_count += 1
        results = {r["id"]: r for r in resp.json().get("results") or []}
        for source_id in source_ids:
            r = results.get(source_id)
            if r is None:
                raise LineageEvidenceError(f"source taxon {source_id} not found via iNaturalist API")
            ancestor_ids = r.get("ancestor_ids") or []
            verified_source_taxa[str(source_id)] = {
                "taxon_id": source_id, "name": r["name"], "rank": r["rank"],
                "parent_id": r.get("parent_id"), "ancestor_ids": ancestor_ids,
                "canonical_target_taxon_id": target_id,
                "target_in_ancestor_ids": target_id in ancestor_ids,
            }
            if target_id not in ancestor_ids:
                raise LineageEvidenceError(
                    f"source taxon {source_id} ({r['name']}) does NOT have canonical target "
                    f"{target_id} in its ancestor_ids -- refusing to canonicalize"
                )

    return canonical_targets, verified_source_taxa, request_count


def build_evidence(canonical_targets: dict, verified_source_taxa: dict, generated_at_utc: str) -> dict[str, Any]:
    evidence = {
        "policy": POLICY,
        "generated_at_utc": generated_at_utc,
        "canonical_targets": canonical_targets,
        "verified_source_taxa": verified_source_taxa,
    }
    evidence["evidence_sha256"] = compute_evidence_hash(
        {k: v for k, v in evidence.items() if k != "evidence_sha256"})
    return evidence


def cmd_capture(args) -> int:
    if OUTPUT_PATH.exists():
        raise LineageEvidenceError(
            f"{OUTPUT_PATH} already exists -- refusing to overwrite a hash-bound evidence file. "
            f"Use --check to verify it instead, or remove it explicitly first if a genuine "
            f"re-capture is intended."
        )
    client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
    try:
        canonical_targets, verified_source_taxa, request_count = fetch_lineage_metadata(client)
    finally:
        client.close()

    evidence = build_evidence(canonical_targets, verified_source_taxa, utc_now())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, OUTPUT_PATH)

    n_targets = len(CANONICALIZATIONS)
    n_sources = sum(len(v) for v in CANONICALIZATIONS.values())
    print(f"wrote {OUTPUT_PATH}")
    print(f"made {request_count} metadata GET request(s): {n_targets} canonical-target lookup(s) + "
         f"{n_targets} batched source-taxa lookup(s) covering {n_sources} source taxon/taxa total "
         f"(never one GET per source id)")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def cmd_check(args) -> int:
    if not OUTPUT_PATH.exists():
        raise LineageEvidenceError(f"{OUTPUT_PATH} does not exist -- nothing to check; run a capture first")
    frozen = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    before_bytes = OUTPUT_PATH.read_bytes()

    client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
    try:
        canonical_targets, verified_source_taxa, request_count = fetch_lineage_metadata(client)
    finally:
        client.close()

    fresh = {"policy": POLICY, "canonical_targets": canonical_targets,
            "verified_source_taxa": verified_source_taxa}
    frozen_semantic = {k: frozen.get(k) for k in SEMANTIC_FIELDS}
    mismatches = []
    for field in SEMANTIC_FIELDS:
        if fresh[field] != frozen_semantic[field]:
            mismatches.append(field)

    after_bytes = OUTPUT_PATH.read_bytes()
    if after_bytes != before_bytes:
        raise LineageEvidenceError("--check must never modify the evidence file, but its bytes changed")

    n_targets = len(CANONICALIZATIONS)
    n_sources = sum(len(v) for v in CANONICALIZATIONS.values())
    print(f"made {request_count} metadata GET request(s): {n_targets} canonical-target lookup(s) + "
         f"{n_targets} batched source-taxa lookup(s) covering {n_sources} source taxon/taxa total")
    print("no writes performed")

    if mismatches:
        print(f"[check] FAILED: semantic disagreement in field(s): {mismatches}")
        return 1
    print("[check] PASS: fresh iNaturalist metadata matches the frozen evidence file exactly "
         "(ignoring generated_at_utc/evidence_sha256, which are not semantic content)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                   help="read-only: verify the existing evidence file against fresh metadata, write nothing")
    args = ap.parse_args()
    try:
        return cmd_check(args) if args.check else cmd_capture(args)
    except LineageEvidenceError as e:
        print(f"[build_known_holdout_lineage_evidence] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
