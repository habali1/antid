#!/usr/bin/env python3
"""scrape_gate_v2.py — Phase 5B1: download/freeze calibration_v2 and
unknown_test_v2 from the approved Gate v2 readiness snapshot.

Source of truth: data/gate_v2_readiness/candidates.csv (hash-bound against
data/gate_v2_readiness/gate_v2_readiness.json). This script NEVER
rediscovers candidates or queries iNaturalist for substitutes outside that
CSV -- every row it ever downloads or reuses already exists as a row there.

Three mutually exclusive modes:
  --preflight   default, safe: reproduces the exact deterministic allocation
                plan (selected + ordered reserve queues) from the committed
                CSV, verifies all reused-row hashes, re-binds the 5
                exclusion sources and the 65-species taxonomy against Phase
                5A's own recorded hashes -- NO network access, NO writes.
  --download    the real cache/download/freeze run: downloads fresh images
                (skipping anything already cached), copies+re-verifies
                reused calibration_v1 images, applies every overlap gate,
                substitutes from a STATEFUL shared deterministic reserve
                queue on rejection, writes a non-frozen attempt/status
                report every time, and -- ONLY if every quota is fully
                satisfied -- stages, fully re-validates, and publishes both
                data/calibration_v2/ and data/unknown_test_v2/ as a
                best-effort atomic pair (rolling back the first if
                publishing the second fails).
  --restore     re-populate ONE already-frozen dataset's images from its own
                (byte-identical, read-only) manifest.

Any source-contract problem aborts immediately, before any network access.
Ordinary per-image failures do not abort the run -- they consume a reserve
slot (from a single shared, stateful queue -- never an independent copy)
and the run keeps going until every quota is met or every reserve is
exhausted.

No confidence threshold is ever imported, applied, or assumed here.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from common import slugify  # noqa: E402,F401

from audit_gate_v2_readiness import (  # noqa: E402
    DOMAIN_LABELS,
    MASTER_SEED,
    SOURCE_TAXON_RANK_NOT_RECORDED,
    deterministic_select,
    read_csv,
    sha256_bytes,
    sha256_file,
    write_json,
)

API = "https://api.inaturalist.org/v1"
USER_AGENT = "AntID-gate-v2-downloader/1.0 (educational project)"
MIN_REQUEST_INTERVAL_SECONDS = 1.05
MIN_DIMENSION_PX = 200

CANDIDATES_CSV_PATH = "data/gate_v2_readiness/candidates.csv"
READINESS_JSON_PATH = "data/gate_v2_readiness/gate_v2_readiness.json"
EXPECTED_CANDIDATES_SHA256 = "eed811a31c2febb98fb2c8f4f4b958e105c1b7fe7aa80b3f3de1e2ef8152c4f9"
EXPECTED_CANDIDATES_ROWS = 4012
TAXONOMY_PATH = "data/northeast_expansion_v1/northeast_taxonomy_v1.json"
REQUIRED_TAXONOMY_SPECIES_COUNT = 65

CACHE_DIR_DEFAULT = "data/gate_v2_cache"       # outside data/calibration_v2 and data/unknown_test_v2
WORK_DIR_DEFAULT = "data/gate_v2_work"          # non-frozen attempt/status reports

QUOTAS = {
    ("calibration_v2", "known_holdout"): 10,   # per species, x65 = 650
    ("unknown_test_v2", "known_holdout"): 6,   # per species, x65 = 390
    ("calibration_v2", "out_of_scope_ant"): 300,
    ("calibration_v2", "non_ant_insect"): 150,
    ("calibration_v2", "unrelated"): 150,
    ("unknown_test_v2", "out_of_scope_ant"): 200,
    ("unknown_test_v2", "non_ant_insect"): 100,
    ("unknown_test_v2", "unrelated"): 100,
}
DATASET_TOTALS = {"calibration_v2": 1250, "unknown_test_v2": 790}
COMBINED_TOTAL = 2040
EXPECTED_REUSED_COUNTS = {"out_of_scope_ant": 229, "non_ant_insect": 87, "unrelated": 119}

EXCLUSION_SOURCES = {
    "northeast_manifest": "data/northeast_expansion_v1/manifest_all_northeast_v1.csv",
    "benchmark_v1": "data/benchmark_v1/benchmark_v1.csv",
    "calibration_v1": "data/calibration_v1/calibration_v1.csv",
    "unknown_test_v1": "data/unknown_test_v1/unknown_test_v1.csv",
    "northeast_final_test_v1": "data/northeast_final_test_v1/northeast_final_test_v1.csv",
}

# The exact fields the task requires nonblank (or the explicit sentinel) on
# every frozen row, plus a small number of ADDITIONAL provenance fields this
# correction pass adds (image_source, candidate_source_url,
# enriched_source_url) -- never fewer than required, sometimes more.
MANIFEST_REQUIRED_FIELDS = [
    "category", "species", "slug", "taxon_id", "observation_id", "observation_uuid",
    "photo_id", "source_url", "photo_license", "photo_attribution", "sha256",
    "byte_size", "width", "height", "provenance_source",
    "origin_dataset", "origin_csv_sha256", "origin_row_number", "origin_photo_id",
    "origin_observation_uuid", "origin_sha256",
    "source_taxon_id", "source_taxon_name", "source_taxon_rank",
]
MANIFEST_EXTRA_FIELDS = ["image_source", "candidate_source_url", "enriched_source_url"]
MANIFEST_ALL_FIELDS = MANIFEST_REQUIRED_FIELDS + MANIFEST_EXTRA_FIELDS

# Explicit sentinel for a genuinely inapplicable field (a fresh row has no
# calibration_v1 origin) -- never blank, never an invented real value.
NOT_APPLICABLE = "n/a_fresh_row"
LOCAL_REUSE_SOURCE_URL = "local_copy_verified:data/calibration_v1"


class SourceContractError(RuntimeError):
    """A source-contract problem -- always aborts before any network access."""


# ----------------------------------------------------------------- utilities
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def to_large_url(url: str) -> str:
    for size in ("square", "small", "medium", "thumb"):
        if f"/{size}." in url:
            return url.replace(f"/{size}.", "/large.")
    return url


def ext_from_url(url: str) -> str:
    tail = url.split("/")[-1].split("?")[0]
    if "." in tail:
        ext = tail.rsplit(".", 1)[1].lower()
        if ext in ("jpg", "jpeg", "png", "webp"):
            return ext
    return "jpg"


def is_valid_sha256(value: str | None) -> bool:
    if not value or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def check_no_stray_manifest_files(repo: Path) -> None:
    """Refuses to proceed if the earlier manifest-placement bug's stray
    root-level outputs exist -- data/{dataset}.csv / data/{dataset}.json
    instead of data/{dataset}/{dataset}.csv / .json."""
    stray = []
    for dataset in ("calibration_v2", "unknown_test_v2"):
        for ext in ("csv", "json"):
            path = repo / "data" / f"{dataset}.{ext}"
            if path.exists():
                stray.append(str(path))
    if stray:
        raise SourceContractError(f"stray root-level manifest file(s) from a prior bug: {stray}")


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


# ------------------------------------------------------------- source contract
def load_and_verify_candidates(repo: Path) -> tuple[list[dict[str, Any]], dict, str]:
    csv_path = repo / CANDIDATES_CSV_PATH
    json_path = repo / READINESS_JSON_PATH
    if not csv_path.exists():
        raise SourceContractError(f"{csv_path} does not exist")
    if not json_path.exists():
        raise SourceContractError(f"{json_path} does not exist")

    actual_hash = sha256_file(csv_path)
    if actual_hash != EXPECTED_CANDIDATES_SHA256:
        raise SourceContractError(
            f"candidates.csv sha256 {actual_hash} != expected {EXPECTED_CANDIDATES_SHA256}"
        )
    rows = read_csv(csv_path)
    if len(rows) != EXPECTED_CANDIDATES_ROWS:
        raise SourceContractError(
            f"candidates.csv has {len(rows)} rows, expected {EXPECTED_CANDIDATES_ROWS}"
        )

    readiness = json.loads(json_path.read_text(encoding="utf-8"))
    recorded = readiness.get("candidates_csv", {})
    if recorded.get("sha256") != EXPECTED_CANDIDATES_SHA256 or recorded.get("rows") != EXPECTED_CANDIDATES_ROWS:
        raise SourceContractError(
            f"gate_v2_readiness.json's candidates_csv record {recorded} does not match "
            f"the expected hash/row-count"
        )
    if "candidate_rows" in readiness:
        raise SourceContractError(
            "gate_v2_readiness.json embeds candidate_rows -- refusing to trust a "
            "readiness snapshot that duplicates the CSV instead of referencing it"
        )
    return rows, readiness, actual_hash


def verify_exclusion_sources_match_readiness(repo: Path, readiness: dict[str, Any]
                                             ) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Recompute each of the 5 exclusion sources' path/rows/sha256 and
    require them to match Phase 5A's own recorded values in
    gate_v2_readiness.json -- any drift since the audit aborts before any
    network access. A blank or malformed sha256 cell is a hard failure
    (never silently dropped from the hash set)."""
    recorded = readiness.get("exclusion_sources", {})
    sets: dict[str, set[str]] = {}
    provenance: dict[str, Any] = {}
    problems = []
    for name, rel_path in EXCLUSION_SOURCES.items():
        path = repo / rel_path
        if not path.exists():
            raise SourceContractError(f"required exclusion source missing: {path}")
        rows = read_csv(path)
        hashes: set[str] = set()
        for i, r in enumerate(rows):
            raw = r.get("sha256")
            if not is_valid_sha256(raw):
                problems.append(f"{name} row {i + 2}: blank/malformed sha256 {raw!r}")
                continue
            hashes.add(raw)
        sets[name] = hashes
        actual_hash = sha256_file(path)
        provenance[name] = {"path": rel_path, "sha256": actual_hash, "rows": len(rows),
                            "unique_sha256": len(hashes)}
        rec = recorded.get(name, {})
        if rec.get("sha256") != actual_hash or rec.get("rows") != len(rows) or rec.get("path") != rel_path:
            problems.append(
                f"{name}: current (path={rel_path}, rows={len(rows)}, sha256={actual_hash}) "
                f"!= Phase 5A recorded {rec}"
            )
    if problems:
        raise SourceContractError(f"exclusion-source binding failure: {problems[:10]}")
    return sets, provenance


EXPECTED_CANONICALIZATION_POLICY = "parent_species_collapse_v1"


def load_and_verify_lineage_evidence(repo: Path, readiness: dict[str, Any]) -> dict[str, Any] | None:
    """Loads the known_holdout parent-species-collapse lineage evidence file
    referenced (path + sha256) from gate_v2_readiness.json, verifies both
    the file's own hash and its internal evidence_sha256 self-check, and
    returns (evidence, policy_block). Returns None when no canonicalization
    policy is recorded (nothing to verify -- every known_holdout row must
    then already carry the canonical label with no source/target split).

    `target_in_ancestor_ids` inside each verified_source_taxa entry is a
    stored ASSERTION, never trusted on its own: this function recomputes
    `canonical_target_taxon_id in ancestor_ids` for every entry and requires
    any stored boolean to agree, replacing the stored value with the
    recomputed one for all downstream use."""
    policy_block = readiness.get("known_holdout_taxonomic_canonicalization")
    if not policy_block:
        return None
    if policy_block.get("policy") != EXPECTED_CANONICALIZATION_POLICY:
        raise SourceContractError(
            f"unrecognized canonicalization policy {policy_block.get('policy')!r}, "
            f"expected {EXPECTED_CANONICALIZATION_POLICY!r}")
    ref = policy_block.get("lineage_evidence", {})
    path = repo / ref.get("path", "")
    if not path.exists():
        raise SourceContractError(f"lineage evidence file missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != ref.get("sha256"):
        raise SourceContractError(
            f"lineage evidence file hash {actual_hash} != recorded {ref.get('sha256')}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("policy") != EXPECTED_CANONICALIZATION_POLICY:
        raise SourceContractError(
            f"lineage evidence declares policy {evidence.get('policy')!r}, "
            f"expected {EXPECTED_CANONICALIZATION_POLICY!r}")
    recorded_evidence_hash = evidence.get("evidence_sha256")
    without_hash = {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    recomputed = sha256_bytes(json.dumps(without_hash, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    if recomputed != recorded_evidence_hash:
        raise SourceContractError(
            "lineage evidence file's own evidence_sha256 does not match its content -- "
            "the evidence has been altered since it was generated"
        )

    verified_source_taxa = evidence.get("verified_source_taxa", {})
    for source_id_str, info in verified_source_taxa.items():
        target_id = info.get("canonical_target_taxon_id")
        ancestor_ids = set(info.get("ancestor_ids") or [])
        computed = target_id in ancestor_ids
        stored = info.get("target_in_ancestor_ids")
        if stored is not None and bool(stored) != computed:
            raise SourceContractError(
                f"lineage evidence for source taxon {source_id_str}: stored target_in_ancestor_ids="
                f"{stored} does not match recomputed value {computed} from ancestor_ids -- "
                f"refusing to trust an asserted boolean over the underlying data")
        info["_computed_target_in_ancestor_ids"] = computed

    declared_verified = {str(x) for x in policy_block.get("source_taxon_ids_verified", [])}
    evidence_keys = set(verified_source_taxa.keys())
    if declared_verified != evidence_keys:
        raise SourceContractError(
            f"readiness source_taxon_ids_verified {sorted(declared_verified)} != evidence's own "
            f"verified_source_taxa keys {sorted(evidence_keys)}")

    return {"evidence": evidence, "policy_block": policy_block, "verified_source_taxa": verified_source_taxa}


def verify_taxonomy_matches_candidates(repo: Path, rows: list[dict[str, Any]],
                                       readiness: dict[str, Any]) -> dict[str, Any]:
    """The current 65-species taxonomy must agree (slug, taxon_id, species
    name) with EVERY known_holdout candidate row, regardless of
    selection_status -- including ineligible rows, since the canonicalization
    pass rewrote every known_holdout row's label -- not just the first one
    encountered per slug, which silently let later rows disagree unnoticed.
    Allocation completeness (every species has at least one selected/reserve
    row) remains a separate check scoped to selected/reserve rows only.

    A row whose source_taxon_id differs from its (canonical) taxon_id is
    only accepted when the recorded, hash-bound lineage evidence explicitly
    proves that source taxon's ancestry includes the canonical target --
    AND the evidence's own claimed identity (taxon_id/name/rank) matches
    what the candidate row itself claims for source_taxon_id/name/rank, AND
    the evidence's canonical target identity matches the 65-species
    taxonomy exactly. Slug equality, name-prefix matching, and slugify()
    are never accepted as ancestry proof."""
    path = repo / TAXONOMY_PATH
    if not path.exists():
        raise SourceContractError(f"{path} does not exist")
    raw = json.loads(path.read_text(encoding="utf-8"))
    taxonomy_by_slug = {v["slug"]: v for v in raw.values()}
    if len(taxonomy_by_slug) != REQUIRED_TAXONOMY_SPECIES_COUNT:
        raise SourceContractError(
            f"expected {REQUIRED_TAXONOMY_SPECIES_COUNT} taxonomy species, found {len(taxonomy_by_slug)}")

    lineage = load_and_verify_lineage_evidence(repo, readiness)
    verified_source_taxa = (lineage or {}).get("verified_source_taxa", {})
    evidence_content = (lineage or {}).get("evidence", {})
    canonical_targets = evidence_content.get("canonical_targets", {})

    # Label/lineage validation runs over EVERY known_holdout candidate row --
    # including ineligible ones -- because the canonicalization pass touched
    # all of them (the frozen snapshot's own labels were rewritten
    # regardless of eligibility). Allocation/quota completeness, by
    # contrast, is only meaningful for rows actually in the plan, so that
    # check stays scoped to selected/reserve below.
    all_known_holdout = [r for r in rows if r["category"] == "known_holdout"]
    selected_or_reserve = [r for r in all_known_holdout if r["selection_status"] in ("selected", "reserve")]
    problems = []
    affected_rows: list[dict[str, Any]] = []  # rows whose source differs from canonical target
    for r in all_known_holdout:
        slug = r["slug"]
        tax = taxonomy_by_slug.get(slug)
        if tax is None:
            problems.append(f"{slug}/{r['photo_id']}: no such species in the 65-species taxonomy")
            continue
        if int(r["taxon_id"]) != int(tax["taxon_id"]):
            problems.append(f"{slug}/{r['photo_id']}: taxon_id {r['taxon_id']} != taxonomy {tax['taxon_id']}")
            continue
        if r["species"] != tax["species_name"]:
            problems.append(f"{slug}/{r['photo_id']}: species {r['species']!r} != taxonomy {tax['species_name']!r}")
            continue
        source_id = r.get("source_taxon_id") or r["taxon_id"]
        if str(source_id) == str(r["taxon_id"]):
            continue  # observation was already identified at the canonical species itself

        affected_rows.append(r)
        evidence = verified_source_taxa.get(str(source_id))
        if evidence is None:
            problems.append(
                f"{slug}/{r['photo_id']}: source_taxon_id {source_id} differs from canonical "
                f"{r['taxon_id']} with no recorded lineage evidence -- refusing to canonicalize")
            continue
        if int(evidence.get("canonical_target_taxon_id", -1)) != int(tax["taxon_id"]):
            problems.append(
                f"{slug}/{r['photo_id']}: lineage evidence for source taxon {source_id} targets "
                f"{evidence.get('canonical_target_taxon_id')}, not {tax['taxon_id']}")
            continue
        if not evidence.get("_computed_target_in_ancestor_ids"):
            problems.append(
                f"{slug}/{r['photo_id']}: lineage evidence for source taxon {source_id} does not "
                f"confirm {tax['taxon_id']} is among its ancestors")
            continue
        if str(evidence.get("taxon_id")) != str(source_id):
            problems.append(
                f"{slug}/{r['photo_id']}: lineage evidence taxon_id {evidence.get('taxon_id')} "
                f"!= candidate's own source_taxon_id {source_id}")
            continue
        if evidence.get("name") != r.get("source_taxon_name"):
            problems.append(
                f"{slug}/{r['photo_id']}: lineage evidence name {evidence.get('name')!r} != "
                f"candidate's source_taxon_name {r.get('source_taxon_name')!r}")
            continue
        if evidence.get("rank") != r.get("source_taxon_rank"):
            problems.append(
                f"{slug}/{r['photo_id']}: lineage evidence rank {evidence.get('rank')!r} != "
                f"candidate's source_taxon_rank {r.get('source_taxon_rank')!r}")
            continue
        canonical_target_info = canonical_targets.get(str(tax["taxon_id"]))
        if canonical_target_info is None:
            problems.append(
                f"{slug}/{r['photo_id']}: lineage evidence has no canonical_targets entry for "
                f"{tax['taxon_id']}")
            continue
        if int(canonical_target_info.get("taxon_id", -1)) != int(tax["taxon_id"]):
            problems.append(f"{slug}/{r['photo_id']}: canonical_targets taxon_id mismatch")
            continue
        if canonical_target_info.get("name") != tax["species_name"]:
            problems.append(
                f"{slug}/{r['photo_id']}: canonical_targets name {canonical_target_info.get('name')!r} "
                f"!= taxonomy {tax['species_name']!r}")
            continue
        if canonical_target_info.get("rank") != "species":
            problems.append(
                f"{slug}/{r['photo_id']}: canonical_targets rank {canonical_target_info.get('rank')!r} "
                f"!= expected 'species'")
            continue

    # Allocation-completeness (every species actually has a selected/reserve
    # row) is scoped to selected/reserve only -- ineligible rows for a
    # species that also has a real selected/reserve row don't matter here,
    # and a species with only ineligible rows is a genuine allocation gap.
    seen_slugs = {r["slug"] for r in selected_or_reserve if r["slug"] in taxonomy_by_slug}
    missing_species = set(taxonomy_by_slug) - seen_slugs
    if missing_species:
        problems.append(f"no known_holdout selected/reserve rows found for: {sorted(missing_species)}")

    if lineage is not None and not problems:
        policy_block = lineage["policy_block"]
        actual_source_ids = {str(r.get("source_taxon_id")) for r in affected_rows}
        evidence_keys = set(verified_source_taxa.keys())
        declared_verified = {str(x) for x in policy_block.get("source_taxon_ids_verified", [])}
        if actual_source_ids != evidence_keys:
            problems.append(
                f"actual differing source_taxon_id set {sorted(actual_source_ids)} found in candidates "
                f"!= lineage evidence's verified_source_taxa keys {sorted(evidence_keys)}")
        if actual_source_ids != declared_verified:
            problems.append(
                f"actual differing source_taxon_id set {sorted(actual_source_ids)} found in candidates "
                f"!= readiness source_taxon_ids_verified {sorted(declared_verified)}")
        affected_slugs = {r["slug"] for r in affected_rows}
        affected_taxon_ids = {int(r["taxon_id"]) for r in affected_rows}
        if len(affected_slugs) > 1 or len(affected_taxon_ids) > 1:
            problems.append(
                f"readiness records a single-target canonicalization policy but affected rows span "
                f"multiple slugs/targets: {sorted(affected_slugs)} / {sorted(affected_taxon_ids)}")
        else:
            if int(policy_block.get("affected_known_holdout_rows", -1)) != len(affected_rows):
                problems.append(
                    f"readiness affected_known_holdout_rows {policy_block.get('affected_known_holdout_rows')} "
                    f"!= actual mismatch count {len(affected_rows)}")
            if affected_slugs and policy_block.get("affected_slug") != next(iter(affected_slugs)):
                problems.append(
                    f"readiness affected_slug {policy_block.get('affected_slug')!r} != actual "
                    f"affected slug {next(iter(affected_slugs))!r}")
            if affected_taxon_ids and int(policy_block.get("canonical_taxon_id", -1)) != next(iter(affected_taxon_ids)):
                problems.append(
                    f"readiness canonical_taxon_id {policy_block.get('canonical_taxon_id')} != actual "
                    f"canonical taxon_id {next(iter(affected_taxon_ids))}")

    if problems:
        raise SourceContractError(f"taxonomy/candidate disagreement: {problems[:10]}")
    return {"path": TAXONOMY_PATH, "sha256": sha256_file(path), "species_count": len(taxonomy_by_slug),
           "known_holdout_rows_taxonomically_checked": len(all_known_holdout),
           "selected_or_reserve_known_holdout_rows": len(selected_or_reserve),
           "canonicalized_source_rows": len(affected_rows),
           "lineage_evidence": (readiness["known_holdout_taxonomic_canonicalization"]["lineage_evidence"]
                               if lineage else None)}


def verify_reused_rows(repo: Path, rows: list[dict[str, Any]],
                       exclusion_hashes: dict[str, set[str]]) -> dict[str, Any]:
    reused = [r for r in rows if r["selection_status"] == "reused"]
    root = repo / "data/calibration_v1"
    problems = []
    verified = 0
    by_category: Counter = Counter()
    for r in reused:
        slug, photo_id, expected = r["slug"], r["origin_photo_id"], r["origin_sha256"]
        existing = [p for ext in ("jpg", "jpeg", "png", "webp")
                   if (p := root / slug / f"{photo_id}.{ext}").exists()]
        if len(existing) != 1:
            problems.append(f"{slug}/{photo_id}: expected exactly one local file, found {len(existing)}")
            continue
        actual = sha256_file(existing[0])
        if actual != expected:
            problems.append(f"{slug}/{photo_id}: hash {actual} != origin_sha256 {expected}")
            continue
        ok, reason, width, height = validate_and_decode(existing[0].read_bytes())
        if not ok:
            problems.append(f"{slug}/{photo_id}: local origin file failed validation ({reason})")
            continue
        other_sources = [name for name, hashes in exclusion_hashes.items()
                         if name != "calibration_v1" and actual in hashes]
        if other_sources:
            problems.append(
                f"{slug}/{photo_id}: hash {actual} unexpectedly also present in "
                f"{other_sources} -- hard integrity failure"
            )
            continue
        verified += 1
        by_category[r["category"]] += 1
    if len(reused) != sum(EXPECTED_REUSED_COUNTS.values()):
        problems.append(
            f"expected {sum(EXPECTED_REUSED_COUNTS.values())} reused rows total, found {len(reused)}"
        )
    for cat, expected_n in EXPECTED_REUSED_COUNTS.items():
        if by_category.get(cat, 0) != expected_n and not problems:
            problems.append(f"reused/{cat}: expected {expected_n} verified, got {by_category.get(cat, 0)}")
    if problems:
        raise SourceContractError(
            f"reused-row integrity failure ({len(problems)} problem(s)): {problems[:10]}"
        )
    return {"rows_checked": len(reused), "rows_verified": verified,
           "by_category": dict(sorted(by_category.items()))}


# ------------------------------------------------------ allocation reconstruction
def _key(r: dict[str, Any]) -> tuple:
    """Must sort identically to the original audit's in-memory key
    (int photo_id, observation_uuid) -- candidates.csv stores photo_id as
    text, and a naive string sort would silently reorder e.g. "9" after
    "10", producing a different deterministic shuffle than the one that
    actually ran."""
    return (int(r["photo_id"]), r["observation_uuid"])


def reconstruct_known_holdout_plan(rows: list[dict[str, Any]], supported_slugs: list[str]
                                   ) -> dict[str, dict[str, Any]]:
    plan: dict[str, dict[str, Any]] = {}
    by_slug: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] == "known_holdout" and r["selection_status"] in ("selected", "reserve"):
            by_slug[r["slug"]].append(r)

    if set(by_slug) != set(supported_slugs):
        raise SourceContractError(
            f"known_holdout species mismatch: CSV has {sorted(set(by_slug) - set(supported_slugs))} "
            f"extra and is missing {sorted(set(supported_slugs) - set(by_slug))}"
        )

    for slug in supported_slugs:
        eligible = by_slug[slug]
        cal_selected, remainder = deterministic_select(
            eligible, "calibration_v2_known", QUOTAS[("calibration_v2", "known_holdout")], key=_key)
        ut_selected, reserve_queue = deterministic_select(
            remainder, "unknown_test_v2_known", QUOTAS[("unknown_test_v2", "known_holdout")], key=_key)

        csv_cal = {_key(r) for r in eligible
                  if r["selection_status"] == "selected" and r["intended_dataset"] == "calibration_v2"}
        csv_ut = {_key(r) for r in eligible
                 if r["selection_status"] == "selected" and r["intended_dataset"] == "unknown_test_v2"}
        if {_key(r) for r in cal_selected} != csv_cal or {_key(r) for r in ut_selected} != csv_ut:
            raise SourceContractError(
                f"quota plan disagreement for known_holdout/{slug}: reconstructed selection "
                f"does not match the committed CSV"
            )
        plan[slug] = {"calibration_v2": cal_selected, "unknown_test_v2": ut_selected,
                     "reserve_queue": reserve_queue}
    return plan


def reconstruct_category_plan(rows: list[dict[str, Any]], category: str,
                              cal_domain: str, ut_domain: str,
                              cal_target: int, ut_target: int,
                              species_partition: bool) -> dict[str, Any]:
    category_rows = [r for r in rows if r["category"] == category
                     and r["selection_status"] in ("selected", "reserve")]

    if species_partition:
        cal_species = {r["slug"] for r in category_rows
                       if r["selection_status"] == "selected" and r["intended_dataset"] == "calibration_v2"}
        cal_pool = [r for r in category_rows if r["slug"] in cal_species]
        ut_pool = [r for r in category_rows if r["slug"] not in cal_species]

        cal_selected, cal_reserve = deterministic_select(cal_pool, cal_domain, cal_target, key=_key)
        ut_selected, ut_reserve = deterministic_select(ut_pool, ut_domain, ut_target, key=_key)

        csv_cal = {_key(r) for r in category_rows
                  if r["selection_status"] == "selected" and r["intended_dataset"] == "calibration_v2"}
        csv_ut = {_key(r) for r in category_rows
                 if r["selection_status"] == "selected" and r["intended_dataset"] == "unknown_test_v2"}
        if {_key(r) for r in cal_selected} != csv_cal or {_key(r) for r in ut_selected} != csv_ut:
            raise SourceContractError(f"quota plan disagreement for {category} (species-partitioned)")
        return {
            "calibration_v2": {"selected": cal_selected, "reserve_queue": cal_reserve},
            "unknown_test_v2": {"selected": ut_selected, "reserve_queue": ut_reserve},
        }

    cal_selected, remainder = deterministic_select(category_rows, cal_domain, cal_target, key=_key)
    ut_selected, reserve_queue = deterministic_select(remainder, ut_domain, ut_target, key=_key)
    csv_cal = {_key(r) for r in category_rows
              if r["selection_status"] == "selected" and r["intended_dataset"] == "calibration_v2"}
    csv_ut = {_key(r) for r in category_rows
             if r["selection_status"] == "selected" and r["intended_dataset"] == "unknown_test_v2"}
    if {_key(r) for r in cal_selected} != csv_cal or {_key(r) for r in ut_selected} != csv_ut:
        raise SourceContractError(f"quota plan disagreement for {category} (shared pool)")
    return {
        "calibration_v2": {"selected": cal_selected, "reserve_queue": []},
        "unknown_test_v2": {"selected": ut_selected, "reserve_queue": []},
        "shared_reserve_queue": reserve_queue,  # ONE shared list -- calibration draws first
    }


def reconstruct_allocation_plan(repo: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from audit_gate_v2_readiness import load_supported_taxonomy
    supported = load_supported_taxonomy(repo)
    supported_slugs = sorted(supported)

    known_plan = reconstruct_known_holdout_plan(rows, supported_slugs)
    ood_plan = reconstruct_category_plan(
        rows, "out_of_scope_ant", "calibration_v2_ood", "unknown_test_v2_ood",
        QUOTAS[("calibration_v2", "out_of_scope_ant")] - EXPECTED_REUSED_COUNTS["out_of_scope_ant"],
        QUOTAS[("unknown_test_v2", "out_of_scope_ant")], species_partition=True)
    non_ant_plan = reconstruct_category_plan(
        rows, "non_ant_insect", "calibration_v2_non_ant_insect", "unknown_test_v2_non_ant_insect",
        QUOTAS[("calibration_v2", "non_ant_insect")] - EXPECTED_REUSED_COUNTS["non_ant_insect"],
        QUOTAS[("unknown_test_v2", "non_ant_insect")], species_partition=False)
    unrelated_plan = reconstruct_category_plan(
        rows, "unrelated", "calibration_v2_unrelated", "unknown_test_v2_unrelated",
        QUOTAS[("calibration_v2", "unrelated")] - EXPECTED_REUSED_COUNTS["unrelated"],
        QUOTAS[("unknown_test_v2", "unrelated")], species_partition=False)

    cal_ood_species = {r["slug"] for r in rows if r["category"] == "out_of_scope_ant"
                      and ((r["selection_status"] == "reused")
                           or (r["selection_status"] == "selected" and r["intended_dataset"] == "calibration_v2"))}
    cal_ood_taxon_ids = {int(r["taxon_id"]) for r in rows if r["slug"] in cal_ood_species
                        and r["category"] == "out_of_scope_ant"}

    return {
        "known_holdout": known_plan, "out_of_scope_ant": ood_plan,
        "non_ant_insect": non_ant_plan, "unrelated": unrelated_plan,
        "supported_slugs": supported_slugs,
        "calibration_v2_ood_species": cal_ood_species,
        "calibration_v2_ood_taxon_ids": cal_ood_taxon_ids,
    }


# ----------------------------------------------------------------- HTTP client
class PacedImageClient:
    def __init__(self, interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS, retries: int = 5):
        if interval_seconds < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError(f"interval_seconds must be >= {MIN_REQUEST_INTERVAL_SECONDS}")
        self.interval_seconds = interval_seconds
        self.retries = retries
        self._last_request = 0.0
        self.request_count = 0
        self._client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60)

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.interval_seconds:
            time.sleep(self.interval_seconds - elapsed)
        self._last_request = time.monotonic()

    def get_json(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(self.retries):
            self._pace()
            self.request_count += 1
            try:
                r = self._client.get(url, params=params)
            except httpx.TransportError:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2.0 * (attempt + 1))
                continue
            r.raise_for_status()
        raise RuntimeError(f"request failed after {self.retries} attempts: {url}")

    def get_bytes(self, url: str) -> bytes | None:
        for attempt in range(self.retries):
            self._pace()
            self.request_count += 1
            try:
                r = self._client.get(url)
            except httpx.TransportError:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2.0 * (attempt + 1))
                continue
            return None
        return None

    def close(self) -> None:
        self._client.close()


# ------------------------------------------------------------------- validation
def validate_and_decode(data: bytes) -> tuple[bool, str, int, int]:
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img2 = Image.open(io.BytesIO(data))
        img2.load()
        width, height = img2.size
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "decode_failure", 0, 0
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        return False, "under_200px", width, height
    return True, "", width, height


# --------------------------------------------------------------------- cache
def find_cached(cache_dir: Path, photo_id: str) -> Path:
    """Returns the cached file for `photo_id`, or None if absent.

    Multiple extensions for one photo_id is a hard failure (never silently
    picks the first). A cached file that fails to decode is treated as
    corrupt: it is removed (so exactly one normal re-fetch can happen next)
    and this call reports it as absent, never poisoning the run permanently.
    """
    matches = [p for p in cache_dir.glob(f"{photo_id}.*") if not p.name.endswith(".part")]
    if len(matches) > 1:
        raise SourceContractError(
            f"cache has {len(matches)} files for photo_id {photo_id}: {[p.name for p in matches]}"
        )
    if not matches:
        return None
    path = matches[0]
    ok, _, _, _ = validate_and_decode(path.read_bytes())
    if not ok:
        path.unlink(missing_ok=True)
        return None
    return path


def write_to_cache(cache_dir: Path, photo_id: str, ext: str, data: bytes) -> Path:
    """`.part` file, flush/close (implicit on write_bytes), THEN atomic
    replace onto the real cache target -- a crash mid-write can never leave
    a half-written file at the real cache path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / f"{photo_id}.{ext}"
    part = cache_dir / f"{photo_id}.{ext}.part"
    part.write_bytes(data)
    os.replace(part, final)
    return final


# ----------------------------------------------------------------------- CLI
def cmd_preflight(args) -> int:
    repo = args.repo.resolve()
    check_no_stray_manifest_files(repo)
    rows, readiness, csv_hash = load_and_verify_candidates(repo)
    exclusion_hashes, exclusion_provenance = verify_exclusion_sources_match_readiness(repo, readiness)
    taxonomy_provenance = verify_taxonomy_matches_candidates(repo, rows, readiness)
    reuse_report = verify_reused_rows(repo, rows, exclusion_hashes)
    plan = reconstruct_allocation_plan(repo, rows)

    for path_name in ("data/calibration_v2", "data/unknown_test_v2"):
        if (repo / path_name).exists():
            raise SourceContractError(f"unexpected existing frozen output: {repo / path_name}")

    fresh_needed = {
        "known_holdout": sum(len(v["calibration_v2"]) + len(v["unknown_test_v2"])
                            for v in plan["known_holdout"].values()),
        "out_of_scope_ant": (len(plan["out_of_scope_ant"]["calibration_v2"]["selected"])
                            + len(plan["out_of_scope_ant"]["unknown_test_v2"]["selected"])),
        "non_ant_insect": (len(plan["non_ant_insect"]["calibration_v2"]["selected"])
                          + len(plan["non_ant_insect"]["unknown_test_v2"]["selected"])),
        "unrelated": (len(plan["unrelated"]["calibration_v2"]["selected"])
                     + len(plan["unrelated"]["unknown_test_v2"]["selected"])),
    }
    total_fresh = sum(fresh_needed.values())
    total_reused = sum(EXPECTED_REUSED_COUNTS.values())

    report = {
        "mode": "preflight", "generated_at_utc": utc_now(),
        "candidates_csv_sha256": csv_hash, "candidates_rows": len(rows),
        "master_seed": MASTER_SEED, "domain_labels": list(DOMAIN_LABELS),
        "taxonomy": taxonomy_provenance,
        "reused_row_verification": reuse_report,
        "fresh_needed_by_category": fresh_needed,
        "total_fresh_expected": total_fresh, "total_reused_expected": total_reused,
        "combined_total_expected": total_fresh + total_reused,
        "exclusion_sources": exclusion_provenance,
        "no_network_access": True, "no_writes": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[preflight] PASS: allocation plan reconstructed and self-consistent with "
         f"the committed CSV; exclusion sources and taxonomy bound to Phase 5A. "
         f"{total_reused} reused + {total_fresh} fresh = {total_reused + total_fresh} "
         f"planned rows. No network access, nothing written.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default; read-only, no network")
    mode.add_argument("--download", action="store_true", help="the real download/freeze run")
    mode.add_argument("--restore", choices=["calibration_v2", "unknown_test_v2"], default=None,
                      help="restore one already-frozen dataset from its own manifest")
    ap.add_argument("--cache-dir", type=Path, default=Path(CACHE_DIR_DEFAULT))
    ap.add_argument("--work-dir", type=Path, default=Path(WORK_DIR_DEFAULT))
    ap.add_argument("--request-interval", type=float, default=MIN_REQUEST_INTERVAL_SECONDS)
    args = ap.parse_args()

    if args.download and args.restore:
        ap.error("--download and --restore are mutually exclusive")
    if args.request_interval < MIN_REQUEST_INTERVAL_SECONDS:
        ap.error(f"--request-interval must be >= {MIN_REQUEST_INTERVAL_SECONDS}")

    try:
        if args.restore:
            return cmd_restore(args)
        if args.download:
            return cmd_download(args)
        return cmd_preflight(args)
    except SourceContractError as e:
        print(f"[scrape_gate_v2] SOURCE CONTRACT FAILURE: {e}")
        return 1


# ------------------------------------------------------------------ download
def cmd_download(args) -> int:
    repo = args.repo.resolve()
    check_no_stray_manifest_files(repo)
    rows, readiness, csv_hash = load_and_verify_candidates(repo)
    exclusion_hashes, exclusion_provenance = verify_exclusion_sources_match_readiness(repo, readiness)
    taxonomy_provenance = verify_taxonomy_matches_candidates(repo, rows, readiness)
    reuse_report = verify_reused_rows(repo, rows, exclusion_hashes)
    plan = reconstruct_allocation_plan(repo, rows)
    for path_name in ("data/calibration_v2", "data/unknown_test_v2"):
        if (repo / path_name).exists():
            raise SourceContractError(f"unexpected existing frozen output: {repo / path_name}")

    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else repo / args.cache_dir
    work_dir = args.work_dir if args.work_dir.is_absolute() else repo / args.work_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = PacedImageClient(interval_seconds=args.request_interval)
    try:
        outcome = run_download_plan(repo, plan, rows, cache_dir, client, exclusion_hashes)
    finally:
        client.close()

    write_work_reports(work_dir, outcome)

    if not outcome["success"]:
        print(f"[download] FAILED: {len(outcome['shortfalls'])} bucket(s) short. "
             f"Cache preserved. No frozen manifest written.")
        return 1

    publish(repo, outcome, rows, readiness, csv_hash, exclusion_provenance,
           taxonomy_provenance, cache_dir, plan, exclusion_hashes)
    finalize_work_report_after_publish(work_dir, repo, cache_dir, outcome)
    return 0


def _empty_dataset_counters() -> dict[str, int]:
    return {"network_transferred_bytes": 0, "network_response_count": 0,
           "cache_accept_count": 0, "calibration_v1_reuse_count": 0,
           "accepted_image_bytes": 0}


def run_download_plan(repo, plan, rows, cache_dir, client, exclusion_hashes) -> dict[str, Any]:
    """Attempts every bucket to completion; never stops early. Returns a
    result dict (always -- success or not) with accepted rows, the full
    attempt log, shortfalls, per-bucket achieved/target results, and
    per-dataset byte-accounting counters, so a non-frozen report can always
    be written."""
    accepted: dict[str, list[dict[str, Any]]] = {"calibration_v2": [], "unknown_test_v2": []}
    accepted_hashes: dict[str, set[str]] = {"calibration_v2": set(), "unknown_test_v2": set()}
    attempted_photo_ids: set[str] = set()
    shortfalls: list[dict[str, Any]] = []
    bucket_results: list[dict[str, Any]] = []
    attempt_log: list[dict[str, Any]] = []
    counters = {"calibration_v2": _empty_dataset_counters(), "unknown_test_v2": _empty_dataset_counters()}

    def log(dataset, category, species, row, role, outcome, reason, detail,
           image_source="", transferred_bytes=0, sha256_val=""):
        attempt_log.append({
            "dataset": dataset, "category": category, "species": species or "",
            "photo_id": str(row.get("photo_id") or row.get("origin_photo_id") or ""),
            "observation_uuid": row.get("observation_uuid") or row.get("origin_observation_uuid") or "",
            "role": role, "outcome": outcome, "reason": reason, "detail": detail,
            "image_source": image_source, "transferred_bytes": transferred_bytes,
            "sha256": sha256_val,
        })

    def try_row(dataset: str, category: str, species: str | None, row: dict[str, Any],
               was_reserve: bool) -> bool:
        role = "reserve" if was_reserve else "selected"
        pid = str(row["photo_id"])
        if pid in attempted_photo_ids:
            # Already attempted (accepted, rejected, or is mid-flight in
            # another bucket) -- never a second final allocation for the
            # same candidate.
            log(dataset, category, species, row, role, "rejected", "already_attempted", "")
            return False
        attempted_photo_ids.add(pid)

        cached = find_cached(cache_dir, pid)
        candidate_url = row["source_url"]
        actual_url = to_large_url(candidate_url)
        transferred = 0
        if cached is not None:
            data = cached.read_bytes()
            image_source = "cache"
            counters[dataset]["cache_accept_count"] += 1
        else:
            raw = client.get_bytes(actual_url)
            counters[dataset]["network_response_count"] += 1
            if raw is None:
                log(dataset, category, species, row, role, "rejected", "download_failure", actual_url)
                return False
            data = raw
            image_source = "network"
            transferred = len(data)
            counters[dataset]["network_transferred_bytes"] += transferred

        ok, reason, width, height = validate_and_decode(data)
        if not ok:
            log(dataset, category, species, row, role, "rejected", reason, "",
               image_source=image_source, transferred_bytes=transferred)
            return False
        digest = sha256_bytes(data)
        colliding = [name for name, hashes in exclusion_hashes.items() if digest in hashes]
        if colliding:
            log(dataset, category, species, row, role, "rejected", "frozen_set_hash_collision",
               ",".join(colliding), image_source=image_source, transferred_bytes=transferred,
               sha256_val=digest)
            return False
        if dataset == "unknown_test_v2" and digest in accepted_hashes["calibration_v2"]:
            log(dataset, category, species, row, role, "rejected", "cross_v2_hash_collision", "",
               image_source=image_source, transferred_bytes=transferred, sha256_val=digest)
            return False
        if digest in accepted_hashes[dataset]:
            log(dataset, category, species, row, role, "rejected", "internal_hash_duplicate", "",
               image_source=image_source, transferred_bytes=transferred, sha256_val=digest)
            return False

        if cached is None:
            write_to_cache(cache_dir, pid, ext_from_url(actual_url), data)
        row = dict(row)
        row.update(sha256=digest, byte_size=len(data), width=width, height=height,
                  image_source=image_source, candidate_source_url=candidate_url,
                  enriched_source_url="", source_url=actual_url)
        accepted[dataset].append(row)
        accepted_hashes[dataset].add(digest)
        counters[dataset]["accepted_image_bytes"] += len(data)
        log(dataset, category, species, row, role, "accepted", "", image_source,
           image_source=image_source, transferred_bytes=transferred, sha256_val=digest)
        return True

    def fill(dataset: str, category: str, target: int, selected: list[dict],
            reserve_queue: list[dict], species: str | None = None) -> None:
        """`reserve_queue` is mutated in place (never copied) -- calibration
        (called first for every bucket) permanently removes what it
        consumes or rejects, so unknown-test only ever sees what is
        genuinely still available."""
        n_ok = 0
        for row in selected:
            if try_row(dataset, category, species, row, was_reserve=False):
                n_ok += 1
                continue
            while reserve_queue:
                substitute = reserve_queue.pop(0)
                if try_row(dataset, category, species, substitute, was_reserve=True):
                    n_ok += 1
                    break
        bucket_results.append({"dataset": dataset, "category": category, "species": species or "",
                              "target": target, "achieved": n_ok, "shortfall": max(0, target - n_ok)})
        if n_ok < target:
            shortfalls.append({"dataset": dataset, "category": category, "species": species,
                              "target": target, "achieved": n_ok, "short_by": target - n_ok})

    # reused rows: copy from calibration_v1, already hash-verified upfront.
    for r in [x for x in rows if x["selection_status"] == "reused"]:
        root = repo / "data/calibration_v1" / r["slug"]
        src = next(root / f"{r['origin_photo_id']}.{e}" for e in ("jpg", "jpeg", "png", "webp")
                  if (root / f"{r['origin_photo_id']}.{e}").exists())
        data = src.read_bytes()
        from PIL import Image
        with Image.open(src) as img:
            width, height = img.size
        row = dict(r)
        row.update(sha256=r["origin_sha256"], byte_size=len(data), width=width, height=height,
                  image_source="calibration_v1_reuse", source_url=LOCAL_REUSE_SOURCE_URL,
                  candidate_source_url=LOCAL_REUSE_SOURCE_URL, enriched_source_url=r["source_url"])
        accepted["calibration_v2"].append(row)
        accepted_hashes["calibration_v2"].add(r["origin_sha256"])
        counters["calibration_v2"]["calibration_v1_reuse_count"] += 1
        counters["calibration_v2"]["accepted_image_bytes"] += len(data)
        attempted_photo_ids.add(str(r["origin_photo_id"]))
        log("calibration_v2", r["category"], r["slug"], row, "selected", "accepted", "",
           "calibration_v1_reuse", image_source="calibration_v1_reuse", transferred_bytes=0,
           sha256_val=r["origin_sha256"])

    for slug in plan["supported_slugs"]:
        species_plan = plan["known_holdout"][slug]
        fill("calibration_v2", "known_holdout", QUOTAS[("calibration_v2", "known_holdout")],
            species_plan["calibration_v2"], species_plan["reserve_queue"], species=slug)
        fill("unknown_test_v2", "known_holdout", QUOTAS[("unknown_test_v2", "known_holdout")],
            species_plan["unknown_test_v2"], species_plan["reserve_queue"], species=slug)

    ood = plan["out_of_scope_ant"]
    fill("calibration_v2", "out_of_scope_ant",
        QUOTAS[("calibration_v2", "out_of_scope_ant")] - EXPECTED_REUSED_COUNTS["out_of_scope_ant"],
        ood["calibration_v2"]["selected"], ood["calibration_v2"]["reserve_queue"])
    fill("unknown_test_v2", "out_of_scope_ant", QUOTAS[("unknown_test_v2", "out_of_scope_ant")],
        ood["unknown_test_v2"]["selected"], ood["unknown_test_v2"]["reserve_queue"])

    for cat in ("non_ant_insect", "unrelated"):
        cat_plan = plan[cat]
        shared_queue = cat_plan.get("shared_reserve_queue", [])
        fill("calibration_v2", cat, QUOTAS[("calibration_v2", cat)] - EXPECTED_REUSED_COUNTS[cat],
            cat_plan["calibration_v2"]["selected"], shared_queue)
        fill("unknown_test_v2", cat, QUOTAS[("unknown_test_v2", cat)],
            cat_plan["unknown_test_v2"]["selected"], shared_queue)  # SAME list -- already drained

    return {
        "success": not shortfalls, "accepted": accepted, "attempt_log": attempt_log,
        "shortfalls": shortfalls, "bucket_results": bucket_results, "counters": counters,
    }


def _overall_counters(counters: dict[str, dict[str, int]]) -> dict[str, int]:
    overall = _empty_dataset_counters()
    for per_dataset in counters.values():
        for k, v in per_dataset.items():
            overall[k] += v
    return overall


def write_work_reports(work_dir: Path, outcome: dict[str, Any]) -> None:
    """Written on EVERY --download invocation, success or shortfall --
    atomically, outside either final dataset name. Uses csv.DictWriter so a
    comma inside a rejection/collision `detail` field can never corrupt the
    schema."""
    work_dir.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "category", "species", "photo_id", "observation_uuid", "role",
             "outcome", "reason", "detail", "image_source", "transferred_bytes", "sha256"]
    attempts_path = work_dir / "download_attempts.csv"
    tmp = attempts_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for e in outcome["attempt_log"]:
            w.writerow({f: e.get(f, "") for f in fields})
    os.replace(tmp, attempts_path)

    rejection_breakdown = Counter(e["reason"] for e in outcome["attempt_log"] if e["outcome"] == "rejected")
    collisions = [e for e in outcome["attempt_log"]
                 if e.get("reason") in ("frozen_set_hash_collision", "cross_v2_hash_collision",
                                       "internal_hash_duplicate")]
    status = {
        "generated_at_utc": utc_now(), "success": outcome["success"],
        "bucket_results": outcome["bucket_results"],
        "shortfalls": outcome["shortfalls"],
        "rejection_breakdown": dict(sorted(rejection_breakdown.items())),
        "collisions": collisions,
        "counters_by_dataset": outcome["counters"],
        "counters_overall": _overall_counters(outcome["counters"]),
        "accepted_counts": {d: len(rows) for d, rows in outcome["accepted"].items()},
    }
    atomic_write_text(work_dir / "download_status.json",
                      json.dumps(status, indent=2, sort_keys=True, default=str) + "\n")


def finalize_work_report_after_publish(work_dir: Path, repo: Path, cache_dir: Path,
                                       outcome: dict[str, Any]) -> dict[str, Any]:
    """Called only after BOTH final directories exist. Adds actual overall
    network bytes transferred this invocation, cache size, and each final
    directory's real on-disk size to the non-frozen work status -- never
    embedded inside a frozen manifest JSON (which would need to measure its
    own not-yet-written size)."""
    status_path = work_dir / "download_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    sizes = {
        "network_transferred_bytes_this_invocation": _overall_counters(outcome["counters"])["network_transferred_bytes"],
        "cache_size_bytes": dir_size_bytes(cache_dir),
        "calibration_v2_dir_bytes": dir_size_bytes(repo / "data/calibration_v2"),
        "unknown_test_v2_dir_bytes": dir_size_bytes(repo / "data/unknown_test_v2"),
    }
    status["post_publish_sizes"] = sizes
    atomic_write_text(status_path, json.dumps(status, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[download] published. post-publish sizes: {sizes}")
    return sizes


# ------------------------------------------------------- final-pair validator
def validate_final_pair(cal_rows: list[dict[str, Any]], ut_rows: list[dict[str, Any]],
                        supported_slugs: list[str], exclusion_hashes: dict[str, set[str]],
                        cal_ood_species: set[str], cal_ood_taxon_ids: set[int],
                        staged_dirs: dict[str, Path] | None = None) -> None:
    """Independently re-verifies the ACCEPTED manifest rows (not just the
    plan/quota constants) immediately before publication. Raises
    SourceContractError naming the first class of problem found."""
    if len(cal_rows) != DATASET_TOTALS["calibration_v2"]:
        raise SourceContractError(f"calibration_v2 total {len(cal_rows)} != {DATASET_TOTALS['calibration_v2']}")
    if len(ut_rows) != DATASET_TOTALS["unknown_test_v2"]:
        raise SourceContractError(f"unknown_test_v2 total {len(ut_rows)} != {DATASET_TOTALS['unknown_test_v2']}")

    for dataset, rows in (("calibration_v2", cal_rows), ("unknown_test_v2", ut_rows)):
        counts = Counter(r["category"] for r in rows)
        for (d, cat), target in QUOTAS.items():
            if d != dataset or cat == "known_holdout":
                continue
            if counts.get(cat, 0) != target:
                raise SourceContractError(f"{dataset}/{cat}: {counts.get(cat, 0)} != {target}")

    for dataset, rows, per_species_target in (
        ("calibration_v2", cal_rows, QUOTAS[("calibration_v2", "known_holdout")]),
        ("unknown_test_v2", ut_rows, QUOTAS[("unknown_test_v2", "known_holdout")]),
    ):
        known = [r for r in rows if r["category"] == "known_holdout"]
        per_species = Counter(r["slug"] for r in known)
        if set(per_species) != set(supported_slugs):
            raise SourceContractError(f"{dataset} known_holdout species set mismatch")
        for slug in supported_slugs:
            if per_species[slug] != per_species_target:
                raise SourceContractError(
                    f"{dataset} known_holdout/{slug}: {per_species[slug]} != {per_species_target}")

    for dataset, rows in (("calibration_v2", cal_rows), ("unknown_test_v2", ut_rows)):
        for r in rows:
            for field in MANIFEST_REQUIRED_FIELDS:
                if r.get(field) in (None, ""):
                    raise SourceContractError(f"{dataset}: manifest row missing {field!r}: {r}")
        for id_field in ("photo_id", "observation_uuid", "sha256"):
            values = [r[id_field] for r in rows if r[id_field] != NOT_APPLICABLE]
            if len(values) != len(set(values)):
                raise SourceContractError(f"{dataset}: duplicate {id_field}")

    cal_photo_ids = {r["photo_id"] for r in cal_rows}
    ut_photo_ids = {r["photo_id"] for r in ut_rows}
    cal_uuids = {r["observation_uuid"] for r in cal_rows if r["observation_uuid"] != NOT_APPLICABLE}
    ut_uuids = {r["observation_uuid"] for r in ut_rows if r["observation_uuid"] != NOT_APPLICABLE}
    cal_hashes = {r["sha256"] for r in cal_rows}
    ut_hashes = {r["sha256"] for r in ut_rows}
    if cal_photo_ids & ut_photo_ids:
        raise SourceContractError(f"cross-dataset photo_id overlap: {cal_photo_ids & ut_photo_ids}")
    if cal_uuids & ut_uuids:
        raise SourceContractError(f"cross-dataset observation_uuid overlap: {cal_uuids & ut_uuids}")
    if cal_hashes & ut_hashes:
        raise SourceContractError(f"cross-dataset sha256 overlap: {cal_hashes & ut_hashes}")

    cal_ood_rows = [r for r in cal_rows if r["category"] == "out_of_scope_ant"]
    ut_ood_rows = [r for r in ut_rows if r["category"] == "out_of_scope_ant"]
    cal_ood_slugs = {r["slug"] for r in cal_ood_rows}
    ut_ood_slugs = {r["slug"] for r in ut_ood_rows}
    ut_ood_taxa = {int(r["taxon_id"]) for r in ut_ood_rows}
    if cal_ood_slugs & set(supported_slugs):
        raise SourceContractError(f"calibration OOD overlaps supported species: {cal_ood_slugs & set(supported_slugs)}")
    if ut_ood_slugs & set(supported_slugs):
        raise SourceContractError(f"unknown-test OOD overlaps supported species: {ut_ood_slugs & set(supported_slugs)}")
    if ut_ood_slugs & cal_ood_species or ut_ood_taxa & cal_ood_taxon_ids:
        raise SourceContractError("unknown-test OOD overlaps calibration OOD species/taxon set")

    for dataset, rows in (("calibration_v2", cal_rows), ("unknown_test_v2", ut_rows)):
        for r in rows:
            is_reused = r["provenance_source"] == "calibration_v1_reuse"
            if not is_reused:
                colliding = [name for name, h in exclusion_hashes.items() if r["sha256"] in h]
                if colliding:
                    raise SourceContractError(
                        f"{dataset}/{r['photo_id']}: fresh row hash present in {colliding}")
            else:
                if r["sha256"] != r["origin_sha256"]:
                    raise SourceContractError(f"{dataset}/{r['photo_id']}: reused hash != origin_sha256")
                other = [name for name, h in exclusion_hashes.items()
                        if name != "calibration_v1" and r["sha256"] in h]
                if other:
                    raise SourceContractError(f"{dataset}/{r['photo_id']}: reused hash also in {other}")

    if staged_dirs:
        for dataset, rows in (("calibration_v2", cal_rows), ("unknown_test_v2", ut_rows)):
            stage_dir = staged_dirs[dataset]
            for r in rows:
                path = stage_dir / r["slug"] / f"{r['photo_id']}.{Path(r['source_url']).suffix.lstrip('.') or 'jpg'}"
                candidates = list((stage_dir / r["slug"]).glob(f"{r['photo_id']}.*"))
                if len(candidates) != 1:
                    raise SourceContractError(f"{dataset}/{r['photo_id']}: expected exactly one staged file")
                staged = candidates[0]
                data = staged.read_bytes()
                if len(data) != r["byte_size"]:
                    raise SourceContractError(f"{dataset}/{r['photo_id']}: staged byte_size mismatch")
                actual_hash = sha256_bytes(data)
                if actual_hash != r["sha256"]:
                    raise SourceContractError(f"{dataset}/{r['photo_id']}: staged sha256 mismatch")
                ok, reason, w, h = validate_and_decode(data)
                if not ok:
                    raise SourceContractError(f"{dataset}/{r['photo_id']}: staged file {reason}")
                if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
                    raise SourceContractError(f"{dataset}/{r['photo_id']}: staged file under {MIN_DIMENSION_PX}px")


def build_manifest_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": r["category"], "species": r["species"], "slug": r["slug"],
        "taxon_id": r["taxon_id"], "observation_id": r.get("observation_id") or NOT_APPLICABLE,
        "observation_uuid": r["observation_uuid"] or r["origin_observation_uuid"],
        "photo_id": r["photo_id"] or r["origin_photo_id"],
        "source_url": r["source_url"], "photo_license": r["photo_license"],
        "photo_attribution": r["photo_attribution"], "sha256": r["sha256"],
        "byte_size": r["byte_size"], "width": r["width"], "height": r["height"],
        "provenance_source": r["provenance_source"],
        "origin_dataset": r["origin_dataset"] or NOT_APPLICABLE,
        "origin_csv_sha256": r["origin_csv_sha256"] or NOT_APPLICABLE,
        "origin_row_number": r["origin_row_number"] or NOT_APPLICABLE,
        "origin_photo_id": r["origin_photo_id"] or NOT_APPLICABLE,
        "origin_observation_uuid": r["origin_observation_uuid"] or NOT_APPLICABLE,
        "origin_sha256": r["origin_sha256"] or NOT_APPLICABLE,
        "image_source": r["image_source"],
        "candidate_source_url": r.get("candidate_source_url") or NOT_APPLICABLE,
        "enriched_source_url": r.get("enriched_source_url") or NOT_APPLICABLE,
        "source_taxon_id": r.get("source_taxon_id") or r["taxon_id"],
        "source_taxon_name": r.get("source_taxon_name") or r["species"],
        "source_taxon_rank": r.get("source_taxon_rank") or SOURCE_TAXON_RANK_NOT_RECORDED,
    }


def publish(repo, outcome, rows, readiness, csv_hash, exclusion_provenance,
           taxonomy_provenance, cache_dir, plan, exclusion_hashes) -> None:
    staging_root = repo / "data" / ".staging_gate_v2"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staged_dirs = {}
    manifests = {}
    for dataset, dataset_rows in outcome["accepted"].items():
        stage_dir = staging_root / dataset
        staged_dirs[dataset] = stage_dir
        manifest_rows = []
        for r in dataset_rows:
            slug_dir = stage_dir / r["slug"]
            slug_dir.mkdir(parents=True, exist_ok=True)
            if r["image_source"] == "calibration_v1_reuse":
                root = repo / "data/calibration_v1" / r["slug"]
                src = next(root / f"{r['origin_photo_id']}.{e}" for e in ("jpg", "jpeg", "png", "webp")
                          if (root / f"{r['origin_photo_id']}.{e}").exists())
            else:
                src = find_cached(cache_dir, str(r["photo_id"]))
                if src is None:
                    raise SourceContractError(f"accepted row {r['photo_id']} missing from cache at publish time")
            dest = slug_dir / f"{r['photo_id']}.{src.suffix.lstrip('.')}"
            shutil.copyfile(src, dest)
            actual = sha256_file(dest)
            if actual != r["sha256"]:
                raise SourceContractError(f"staged file {dest} sha256 {actual} != expected {r['sha256']}")
            manifest_rows.append(build_manifest_row(r))
        manifests[dataset] = manifest_rows

    validate_final_pair(manifests["calibration_v2"], manifests["unknown_test_v2"],
                       plan["supported_slugs"], exclusion_hashes, plan["calibration_v2_ood_species"],
                       plan["calibration_v2_ood_taxon_ids"], staged_dirs=staged_dirs)

    for dataset, manifest_rows in manifests.items():
        write_dataset_manifest(repo, dataset, manifest_rows, staged_dirs[dataset], readiness,
                               csv_hash, exclusion_provenance, taxonomy_provenance, cache_dir, outcome)

    publish_pair_with_rollback(repo, staged_dirs)
    shutil.rmtree(staging_root, ignore_errors=True)


def publish_pair_with_rollback(repo: Path, staged_dirs: dict[str, Path]) -> None:
    """Best-effort atomic pair publication: if the SECOND move fails, the
    first is moved back out of its final name -- never leaves one dataset
    presented as frozen while the other failed."""
    final_paths = {dataset: repo / "data" / dataset for dataset in staged_dirs}
    moved: list[str] = []
    try:
        for dataset, stage_dir in staged_dirs.items():
            shutil.move(str(stage_dir), str(final_paths[dataset]))
            moved.append(dataset)
    except Exception:
        for dataset in moved:
            if final_paths[dataset].exists():
                shutil.move(str(final_paths[dataset]), str(staged_dirs[dataset]))
        raise


def write_dataset_manifest(repo, dataset, manifest_rows, stage_dir: Path, readiness, csv_hash,
                           exclusion_provenance, taxonomy_provenance, cache_dir, outcome) -> None:
    """Writes {dataset}.csv and {dataset}.json INSIDE `stage_dir` (which is
    itself later moved to data/{dataset}/) -- never as siblings of it."""
    csv_path = stage_dir / f"{dataset}.csv"
    with csv_path.open("wb") as fh:
        text_fh = io.TextIOWrapper(fh, encoding="utf-8", newline="")
        w = csv.DictWriter(text_fh, fieldnames=MANIFEST_ALL_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows)
        text_fh.flush()

    license_counts = Counter(r["photo_license"] for r in manifest_rows)
    source_counts = Counter(r["image_source"] for r in manifest_rows)
    total_image_bytes = sum(r["byte_size"] for r in manifest_rows)
    dataset_shortfalls = [s for s in outcome["shortfalls"] if s["dataset"] == dataset]
    dataset_collisions = [e for e in outcome["attempt_log"]
                         if e["dataset"] == dataset and e.get("reason") in
                         ("frozen_set_hash_collision", "cross_v2_hash_collision", "internal_hash_duplicate")]
    dataset_rejections = Counter(e["reason"] for e in outcome["attempt_log"]
                                if e["dataset"] == dataset and e["outcome"] == "rejected")
    metadata = {
        "dataset": dataset, "generated_at_utc": utc_now(),
        "candidates_csv": {"path": CANDIDATES_CSV_PATH, "rows": EXPECTED_CANDIDATES_ROWS, "sha256": csv_hash},
        "readiness_json": {"path": READINESS_JSON_PATH, "sha256": sha256_file(repo / READINESS_JSON_PATH)},
        "taxonomy": taxonomy_provenance,
        "downloader_git_revision": get_git_commit(),
        "master_seed": MASTER_SEED, "domain_labels": list(DOMAIN_LABELS),
        "quotas": {f"{d}/{c}": q for (d, c), q in QUOTAS.items() if d == dataset},
        "achieved_counts": dict(Counter(r["category"] for r in manifest_rows)),
        "source_counts": dict(sorted(source_counts.items())),
        "reused_count": source_counts.get("calibration_v1_reuse", 0),
        "fresh_count": source_counts.get("network", 0) + source_counts.get("cache", 0),
        "license_distribution": dict(sorted(license_counts.items())),
        "exclusion_sources": exclusion_provenance,
        "manifest": {"path": f"data/{dataset}/{dataset}.csv", "rows": len(manifest_rows), "sha256": None},
        "byte_accounting": {
            "network_transferred_bytes": outcome["counters"][dataset]["network_transferred_bytes"],
            "network_response_count": outcome["counters"][dataset]["network_response_count"],
            "cache_accept_count": outcome["counters"][dataset]["cache_accept_count"],
            "calibration_v1_reuse_count": outcome["counters"][dataset]["calibration_v1_reuse_count"],
            "accepted_image_bytes": outcome["counters"][dataset]["accepted_image_bytes"],
            "accepted_image_bytes_this_dataset": total_image_bytes,
            "cache_size_bytes": dir_size_bytes(cache_dir),
        },
        "shortfalls_during_this_run": dataset_shortfalls,
        "rejection_breakdown": dict(sorted(dataset_rejections.items())),
        "collisions": dataset_collisions,
        "cache_dir": str(cache_dir),
        "restore_command": f"python scrape_gate_v2.py --restore {dataset}",
        "license_scope": "personal, non-commercial phase only",
    }
    metadata["manifest"]["sha256"] = sha256_file(csv_path)
    write_json(stage_dir / f"{dataset}.json", metadata)


# ------------------------------------------------------------------- restore
def cmd_restore(args) -> int:
    repo = args.repo.resolve()
    dataset = args.restore
    manifest_path = repo / "data" / dataset / f"{dataset}.csv"
    if not manifest_path.exists():
        raise SourceContractError(f"{manifest_path} does not exist -- nothing to restore")
    original_manifest_bytes = manifest_path.read_bytes()
    rows = read_csv(manifest_path)

    client = PacedImageClient(interval_seconds=args.request_interval)
    errors: list[str] = []
    n_ok = 0
    try:
        needing_fetch: list[dict[str, Any]] = []
        already_ok: set[str] = set()
        blocked: set[str] = set()
        resolved_locally: dict[str, tuple[bytes, str]] = {}
        for r in rows:
            slug, pid, want_hash = r["slug"], r["photo_id"], r["sha256"]
            sdir = repo / "data" / dataset / slug
            existing = [p for p in sdir.glob(f"{pid}.*")] if sdir.exists() else []
            if len(existing) > 1:
                errors.append(f"{slug}/{pid}: {len(existing)} ambiguous local files -- remove manually")
                blocked.add(pid)
                continue
            if len(existing) == 1 and sha256_file(existing[0]) == want_hash:
                already_ok.add(pid)
                continue
            # Prefer the verified local calibration_v1 origin for reused
            # rows -- resolved with zero network access. Only rows whose
            # origin is unavailable or mismatched fall through to the
            # generic observation/photo API lookup below.
            if r.get("provenance_source") == "calibration_v1_reuse":
                origin_root = repo / "data/calibration_v1" / slug
                origin_pid = r.get("origin_photo_id")
                origin_matches = [origin_root / f"{origin_pid}.{e}" for e in ("jpg", "jpeg", "png", "webp")
                                  if (origin_root / f"{origin_pid}.{e}").exists()]
                if origin_matches and sha256_file(origin_matches[0]) == r.get("origin_sha256"):
                    resolved_locally[pid] = (origin_matches[0].read_bytes(),
                                            origin_matches[0].suffix.lstrip("."))
                    continue
            needing_fetch.append(r)

        obs_by_uuid: dict[str, dict] = {}
        api_uuids = sorted({r["observation_uuid"] for r in needing_fetch
                           if r.get("observation_uuid") and r["observation_uuid"] != NOT_APPLICABLE})
        for i in range(0, len(api_uuids), 100):
            batch = api_uuids[i:i + 100]
            payload = client.get_json(f"{API}/observations", {"uuid": ",".join(batch), "per_page": len(batch)})
            for obs in payload.get("results", []):
                if obs.get("uuid"):
                    obs_by_uuid[obs["uuid"]] = obs

        for r in rows:
            slug, pid, want_hash = r["slug"], r["photo_id"], r["sha256"]
            if pid in blocked:
                # Already recorded as ambiguous above -- never attempt it
                # through an API lookup that was never populated for it.
                continue
            sdir = repo / "data" / dataset / slug
            sdir.mkdir(parents=True, exist_ok=True)
            if pid in already_ok:
                n_ok += 1
                continue

            data = None
            ext = None
            if pid in resolved_locally:
                data, ext = resolved_locally[pid]
            if data is None:
                obs = obs_by_uuid.get(r["observation_uuid"])
                if obs is None:
                    errors.append(f"{slug}/{pid}: observation_uuid not found via API")
                    continue
                photo = next((p for p in (obs.get("photos") or []) if str(p.get("id")) == str(pid)), None)
                if photo is None or not photo.get("url"):
                    errors.append(f"{slug}/{pid}: photo not found in current observation")
                    continue
                url = to_large_url(photo["url"])
                data = client.get_bytes(url)
                if data is None:
                    errors.append(f"{slug}/{pid}: download failed")
                    continue
                ext = ext_from_url(url)

            ok, reason, w, h = validate_and_decode(data)
            if not ok:
                errors.append(f"{slug}/{pid}: {reason}")
                continue
            if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
                errors.append(f"{slug}/{pid}: under {MIN_DIMENSION_PX}px")
                continue
            digest = sha256_bytes(data)
            if digest != want_hash:
                errors.append(f"{slug}/{pid}: sha256 mismatch after fetch")
                continue

            for stray in sdir.glob(f"{pid}.*"):
                stray.unlink()
            final_path = sdir / f"{pid}.{ext}"
            part_path = sdir / f"{pid}.{ext}.part"
            part_path.write_bytes(data)
            os.replace(part_path, final_path)
            if sha256_file(final_path) != want_hash:
                errors.append(f"{slug}/{pid}: on-disk hash mismatch after write")
                final_path.unlink(missing_ok=True)
                continue
            n_ok += 1
    finally:
        client.close()

    if manifest_path.read_bytes() != original_manifest_bytes:
        raise SourceContractError("restore mutated the manifest -- this must never happen")

    print(f"[restore] {n_ok}/{len(rows)} verified OK, {len(errors)} error(s)")
    for e in errors:
        print(f"  ! {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
