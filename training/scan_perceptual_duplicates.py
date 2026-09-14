#!/usr/bin/env python3
"""scan_perceptual_duplicates.py — generates perceptual-duplicate candidate
pairs and metadata-leakage findings across the frozen 36 comparison domains
(manual_review_contract.COMPARISON_DOMAINS), contract-bound.

CLI modes:
  --preflight  loads and verifies the contract; metadata-only row counts
               per domain part from the manifests; reports an estimated
               comparison scale for the real scan. ZERO image reads.
  --scan       the real scan (NOT run in Phase 5F1): opens every row's
               image (verified against its manifest sha256), computes
               pHash/dHash via perceptual_hash.py, indexes with a BK-tree
               per hash type per domain side, generates candidate pairs
               and metadata-leakage findings for all 36 domains, and
               publishes all five reports as ONE atomic/exclusive
               publication (all-or-nothing, like the queue/summary pair).
  --check      validates the ALREADY-COMPLETED scan reports (self-
               consistency, content_sha256) WITHOUT rerunning or
               replacing them.

Metadata loading and leakage detection never open an image.
generate_candidate_pairs_indexed is the ONE function that touches
already-computed hash dicts (not images) and does the BK-tree search;
generate_candidate_pairs_bruteforce is kept ONLY as a reference for small
synthetic tests, proven equal to the indexed output.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import perceptual_hash as ph  # noqa: E402

# Manifests for the "other evidence" datasets (photo_id-under-slug layout).
OTHER_EVIDENCE_MANIFEST_PATHS = {
    "benchmark_v1": "data/benchmark_v1/benchmark_v1.csv",
    "calibration_v1": "data/calibration_v1/calibration_v1.csv",
    "unknown_test_v1": "data/unknown_test_v1/unknown_test_v1.csv",
    "calibration_v2": "data/calibration_v2/calibration_v2.csv",
    "unknown_test_v2": "data/unknown_test_v2/unknown_test_v2.csv",
}


class ScanError(RuntimeError):
    """A manifest, contract, or image problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise ScanError(msg)


def row_identity(dataset: str, slug: str, split: str, photo_id: str) -> str:
    return f"{dataset}:{slug}:{split}:{photo_id}"


def load_identity_rows(csv_path: Path, dataset: str, *, split_column: str = "split",
                       default_split: str | None = None) -> list[dict]:
    """Metadata-only CSV read. Never opens an image."""
    path = Path(csv_path)
    if not path.exists():
        _fail(f"manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    out = []
    for row in rows:
        split = row.get(split_column) if split_column in fieldnames else default_split
        if split is None:
            _fail(f"{path} row has no split and no default_split was given")
        for field in ("slug", "species", "taxon_id", "observation_uuid", "photo_id", "sha256"):
            if row.get(field) in (None, ""):
                _fail(f"{path} row is missing/blank required field {field!r}")
        out.append({
            "dataset": dataset, "slug": row["slug"], "split": split,
            "species": row["species"], "taxon_id": row["taxon_id"],
            "observation_uuid": row["observation_uuid"], "photo_id": str(int(row["photo_id"])),
            "sha256": row["sha256"],
            "identity": row_identity(dataset, row["slug"], split, str(int(row["photo_id"]))),
        })
    return out


def _population_count(rows_by_part: dict[str, list[dict]], part: str) -> int:
    return len(rows_by_part[part])


def load_domain_part_rows(repo: Path, verified: dict, part: str) -> list[dict]:
    """Metadata-only row loading for one of the 8 DOMAIN_PARTS.

    expansion_train/expansion_development are the PERCEPTUAL-SCAN
    population, not the 600-row manual-review sample: they must load ALL
    rows from the frozen northeast_train_dev_v1.csv manifest (3,000 train +
    600 development), filtered by split. The manual-review queue only ever
    sampled 75/75 of those for human review -- using the queue here would
    silently scan a tiny fraction of the real expansion data.

    final_test correctly stays sourced from the queue: the queue already
    contains ALL 450 final_test rows unsampled, so it matches the frozen
    final_test manifest exactly.

    The five other-evidence parts are read from their own raw manifests."""
    if part in (mrc.DOMAIN_PART_EXPANSION_TRAIN, mrc.DOMAIN_PART_EXPANSION_DEVELOPMENT):
        split = (mrc.EXPANSION_TRAIN_SPLIT if part == mrc.DOMAIN_PART_EXPANSION_TRAIN
                else mrc.EXPANSION_DEV_SPLIT)
        all_rows = load_identity_rows(repo / mrc.EXPANSION_MANIFEST_REL_PATH, mrc.EXPANSION_DATASET)
        rows = [r for r in all_rows if r["split"] == split]
        expected = mrc.EXPANSION_SCAN_SPLIT_ROW_COUNTS[split]
        if len(rows) != expected:
            _fail(f"{part}: expected {expected} rows with split={split!r} in "
                  f"{mrc.EXPANSION_MANIFEST_REL_PATH}, found {len(rows)}")
        return rows
    elif part == mrc.DOMAIN_PART_FINAL_TEST:
        rows = [r for r in verified["queue_rows"] if r["dataset"] == mrc.FINAL_TEST_DATASET]
        return [{**r, "identity": row_identity(r["dataset"], r["slug"], r["split"], r["photo_id"])} for r in rows]
    elif part in OTHER_EVIDENCE_MANIFEST_PATHS:
        return load_identity_rows(repo / OTHER_EVIDENCE_MANIFEST_PATHS[part], part, default_split="known")
    else:
        _fail(f"unknown domain part: {part!r}")
        return []


def find_metadata_leakage(rows_by_domain: dict[str, list[dict]]) -> list[dict]:
    """Pure metadata function: groups every row by observation_uuid across
    ALL supplied domains and reports any observation_uuid that appears in
    more than one DISTINCT domain -- independent of whether the underlying
    photographs are perceptually similar. Never opens an image."""
    from collections import defaultdict
    by_uuid: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for domain_name, rows in rows_by_domain.items():
        for row in rows:
            by_uuid[row["observation_uuid"]].append((domain_name, row))

    findings = []
    for uuid, entries in by_uuid.items():
        domains_present = {d for d, _ in entries}
        if len(domains_present) > 1:
            findings.append({
                "observation_uuid": uuid,
                "domains": sorted(domains_present),
                "entries": [
                    {"domain": d, "dataset": r["dataset"], "slug": r["slug"], "split": r["split"],
                    "photo_id": r["photo_id"], "identity": r["identity"]}
                    for d, r in entries
                ],
            })
    findings.sort(key=lambda f: f["observation_uuid"])
    return findings


def compute_hashes_for_rows(rows: list[dict], image_root: Path, image_layouts: dict,
                           domain_part: str) -> tuple[dict[str, dict], dict]:
    """Opens and hashes each row's image (resolved via the shared extension
    resolver against the verified contract's `image_layouts` for this
    domain part, sha256 verified against its manifest entry BEFORE
    decoding). THIS is the function real --scan calls; it is the ONLY
    function in this module that touches image bytes. Returns (per-identity
    hash entries shaped exactly to mrc.HASH_ENTRY_KEYS, {"pillow_version",
    "numpy_version"} common to every entry this run)."""
    hashes: dict[str, dict] = {}
    versions: dict = {}
    for row in rows:
        path = mrc.resolve_image_path(image_root, image_layouts, domain_part, row["slug"], row["photo_id"])
        data = Path(path).read_bytes()
        actual_sha256 = ph.sha256_bytes(data)
        if actual_sha256 != row["sha256"]:
            _fail(f"{path} sha256 {actual_sha256} != manifest-bound {row['sha256']!r} for "
                  f"{row['identity']} -- refusing to hash unverified image bytes")
        computed = ph.compute_all_hashes(data, row["sha256"])
        versions = {"pillow_version": computed["pillow_version"], "numpy_version": computed["numpy_version"]}
        hashes[row["identity"]] = {"sha256": computed["sha256"], "phashes": computed["phashes"],
                                   "dhashes": computed["dhashes"]}
    return hashes, versions


def generate_candidate_pairs_bruteforce(rows_a: list[dict], hashes_a: dict[str, dict],
                                        rows_b: list[dict], hashes_b: dict[str, dict], *,
                                        phash_max: int = mrc.PHASH_MAX_HAMMING_DISTANCE,
                                        dhash_max: int = mrc.DHASH_MAX_HAMMING_DISTANCE,
                                        exclude_self: bool = False) -> list[dict]:
    """O(|A| x |B| x 64) reference implementation. Retained ONLY for small
    synthetic tests proving the indexed implementation below produces the
    identical result -- never used for the real dataset scale."""
    pairs = []
    for row_a in rows_a:
        ha = hashes_a[row_a["identity"]]
        for row_b in rows_b:
            if exclude_self and row_a["identity"] == row_b["identity"]:
                continue
            hb = hashes_b[row_b["identity"]]
            is_candidate, p_dist, d_dist = ph.is_candidate_pair(
                ha["phashes"], ha["dhashes"], hb["phashes"], hb["dhashes"],
                phash_max=phash_max, dhash_max=dhash_max)
            if is_candidate:
                pairs.append({
                    "pair_id": mrc.compute_pair_id(row_a["identity"], row_b["identity"]),
                    "identity_a": row_a["identity"], "identity_b": row_b["identity"],
                    "phash_distance": p_dist, "dhash_distance": d_dist,
                })
    pairs.sort(key=lambda p: p["pair_id"])
    return pairs


def generate_candidate_pairs_indexed(rows_a: list[dict], hashes_a: dict[str, dict],
                                     rows_b: list[dict], hashes_b: dict[str, dict], *,
                                     phash_max: int = mrc.PHASH_MAX_HAMMING_DISTANCE,
                                     dhash_max: int = mrc.DHASH_MAX_HAMMING_DISTANCE,
                                     exclude_self: bool = False) -> list[dict]:
    """The production implementation: indexes every orientation hash of the
    B side into one BK-tree per hash type, then queries with every
    orientation hash of the A side at the appropriate radius. Preserves
    EXACT inclusive threshold semantics (a BK-tree query is an exact radius
    search, not an approximation) -- proven equal to the brute-force
    reference on randomized synthetic fixtures. Deterministic output
    ordering (sorted by pair_id, exactly like the brute-force version)."""
    phash_tree = ph.BKTree()
    dhash_tree = ph.BKTree()
    for row_b in rows_b:
        hb = hashes_b[row_b["identity"]]
        for h in hb["phashes"]:
            phash_tree.add(h, row_b["identity"])
        for h in hb["dhashes"]:
            dhash_tree.add(h, row_b["identity"])

    best_distance: dict[tuple[str, str], list[int]] = {}
    for row_a in rows_a:
        ha = hashes_a[row_a["identity"]]
        matched_identities_b: set[str] = set()
        for h in ha["phashes"]:
            matched_identities_b.update(phash_tree.query(h, phash_max))
        for h in ha["dhashes"]:
            matched_identities_b.update(dhash_tree.query(h, dhash_max))
        for identity_b in matched_identities_b:
            if exclude_self and identity_b == row_a["identity"]:
                continue
            hb = hashes_b[identity_b]
            p = ph.min_hamming_across_orientations(ha["phashes"], hb["phashes"])
            d = ph.min_hamming_across_orientations(ha["dhashes"], hb["dhashes"])
            if p <= phash_max or d <= dhash_max:
                key = (row_a["identity"], identity_b)
                best_distance[key] = [p, d]

    pairs = []
    for (identity_a, identity_b), (p, d) in best_distance.items():
        pairs.append({
            "pair_id": mrc.compute_pair_id(identity_a, identity_b),
            "identity_a": identity_a, "identity_b": identity_b,
            "phash_distance": p, "dhash_distance": d,
        })
    pairs.sort(key=lambda pr: pr["pair_id"])
    return pairs


def _canonicalize_and_dedupe_within_pairs(pairs: list[dict]) -> list[dict]:
    """A within-set search over (rows, rows) with exclude_self=True still
    finds each true pair from BOTH directions (A->B and B->A), which share
    one pair_id (compute_pair_id is order-independent) but would otherwise
    be emitted twice with swapped identity_a/identity_b. This canonicalizes
    identity_a/identity_b into one sorted order and keeps exactly one entry
    per pair_id -- every unordered pair is represented exactly once.
    Hamming distance is symmetric, so the kept phash/dhash distances are
    identical regardless of which direction is kept."""
    canonical: dict[str, dict] = {}
    for pair in pairs:
        a, b = sorted([pair["identity_a"], pair["identity_b"]])
        pid = mrc.compute_pair_id(a, b)
        if pid in canonical:
            continue
        canonical[pid] = {"pair_id": pid, "identity_a": a, "identity_b": b,
                          "phash_distance": pair["phash_distance"], "dhash_distance": pair["dhash_distance"]}
    return sorted(canonical.values(), key=lambda p: p["pair_id"])


def generate_candidate_pairs_within_bruteforce(rows: list[dict], hashes: dict[str, dict], *,
                                               phash_max: int = mrc.PHASH_MAX_HAMMING_DISTANCE,
                                               dhash_max: int = mrc.DHASH_MAX_HAMMING_DISTANCE) -> list[dict]:
    """Brute-force within-set search: examines every unordered pair exactly
    once. Test-only reference, proven equal to the indexed version below."""
    raw = generate_candidate_pairs_bruteforce(rows, hashes, rows, hashes,
                                              phash_max=phash_max, dhash_max=dhash_max, exclude_self=True)
    return _canonicalize_and_dedupe_within_pairs(raw)


def generate_candidate_pairs_within_indexed(rows: list[dict], hashes: dict[str, dict], *,
                                            phash_max: int = mrc.PHASH_MAX_HAMMING_DISTANCE,
                                            dhash_max: int = mrc.DHASH_MAX_HAMMING_DISTANCE) -> list[dict]:
    """The production within-domain search: examines every unordered pair
    exactly once, identity_a/identity_b stored in one canonical (sorted)
    order, each pair_id emitted exactly once. This is what cmd_scan calls
    for every `within_*` domain."""
    raw = generate_candidate_pairs_indexed(rows, hashes, rows, hashes,
                                           phash_max=phash_max, dhash_max=dhash_max, exclude_self=True)
    return _canonicalize_and_dedupe_within_pairs(raw)


def evaluate_stop_conditions(domain_name: str, confirmed_same_source_pair_ids: set[str],
                             candidate_pairs: list[dict], leakage_findings: list[dict]) -> dict:
    """Pure decision function: a CONFIRMED same_source_image adjudication,
    or a metadata-leakage finding, in a stop-mandating domain requires a
    stop before any northeast_final_test_v1 model inference. A mere
    perceptual CANDIDATE (not yet adjudicated) never triggers a stop on its
    own."""
    reasons: list[str] = []
    if domain_name in mrc.STOP_BEFORE_INFERENCE_DOMAINS:
        confirmed_here = [p for p in candidate_pairs if p["pair_id"] in confirmed_same_source_pair_ids]
        if confirmed_here:
            reasons.append(mrc.STOP_REASON_SAME_SOURCE_IMAGE)
        if leakage_findings:
            reasons.append(mrc.STOP_REASON_MATCHING_OBSERVATION_UUID)
    return {"domain": domain_name, "stop_before_inference": bool(reasons), "reasons": reasons}


# --------------------------------------------------------------- publish --
def _stage_bytes(dest: Path, data: bytes) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return tmp


def publish_reports_atomically(report_bytes_by_dest: dict[Path, bytes]) -> None:
    """Publishes ALL report files as one all-or-nothing transaction: every
    file is staged+fsynced first, then all are published exclusively; if
    ANY publish fails, every already-published file in this batch is
    rolled back."""
    for dest in report_bytes_by_dest:
        if dest.exists():
            _fail(f"refusing to overwrite existing completed report: {dest}")
    staged = {dest: _stage_bytes(dest, data) for dest, data in report_bytes_by_dest.items()}
    published: list[Path] = []
    try:
        for dest, tmp in staged.items():
            os.link(str(tmp), str(dest))
            published.append(dest)
    except Exception:
        for dest in published:
            dest.unlink(missing_ok=True)
        raise
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)


def _report_envelope(content: dict) -> dict:
    content_sha256 = mrc.compute_content_sha256(content)
    return {"schema_version": mrc.SCHEMA_VERSION, "content": content, "content_sha256": content_sha256}


def _report_path(repo: Path, name: str) -> Path:
    return repo / mrc.APPROVED_OUTPUT_PATHS[name]


# --------------------------------------------------------------- commands --
def cmd_scan(args) -> dict:
    """The real scan: loads/verifies the contract, loads the full metadata
    population for all 8 domain parts (item 1: expansion_train/
    expansion_development load ALL 3,000/600 frozen rows, never the 600-row
    manual-review sample), resolves and hashes every row's image (verified
    sha256 before decoding, shared extension resolver -- item 3), runs the
    indexed candidate search across all 36 domains, computes metadata
    leakage, assembles and strictly cross-validates all five reports, then
    publishes them as ONE atomic/exclusive transaction. Never calls
    generate_candidate_pairs_bruteforce (test-only)."""
    verified = mrc.load_and_verify_contract(args.repo)
    contract = verified["contract"]
    contract_content_sha256 = contract["content_sha256"]
    image_root = Path(args.repo) / "data"
    image_layouts = contract["content"]["domain_part_image_layouts"]

    rows_by_part: dict[str, list[dict]] = {}
    hashes: dict[str, dict] = {}
    versions: dict = {}
    for part in mrc.DOMAIN_PARTS:
        rows = load_domain_part_rows(args.repo, verified, part)
        rows_by_part[part] = rows
        part_hashes, part_versions = compute_hashes_for_rows(rows, image_root, image_layouts, part)
        hashes.update(part_hashes)
        if part_versions:
            versions = part_versions

    hashes_report = _report_envelope({
        "contract_content_sha256": contract_content_sha256,
        "pillow_version": versions.get("pillow_version"),
        "numpy_version": versions.get("numpy_version"),
        "hashes": hashes,
    })

    leakage_findings = find_metadata_leakage(rows_by_part)
    leakage_report = _report_envelope({
        "contract_content_sha256": contract_content_sha256,
        "findings": leakage_findings,
    })

    candidate_domains: dict[str, list[dict]] = {}
    domain_summary_entries: dict[str, dict] = {}
    for domain in mrc.COMPARISON_DOMAINS:
        name = domain["name"]
        if domain["kind"] == "within":
            part = domain["parts"][0]
            rows_side = rows_by_part[part]
            pairs = generate_candidate_pairs_within_indexed(rows_side, hashes)
            a_count = b_count = _population_count(rows_by_part, part)
        else:
            a, b = domain["parts"]
            rows_a, rows_b = rows_by_part[a], rows_by_part[b]
            pairs = generate_candidate_pairs_indexed(rows_a, hashes, rows_b, hashes)
            a_count, b_count = _population_count(rows_by_part, a), _population_count(rows_by_part, b)
        candidate_domains[name] = pairs

        if domain["kind"] == "within":
            leakage_count = 0
        else:
            a, b = domain["parts"]
            leakage_count = sum(1 for f in leakage_findings if {a, b} <= set(f["domains"]))
        domain_summary_entries[name] = {
            "a_count": a_count, "b_count": b_count,
            "candidate_count": len(pairs), "leakage_count": leakage_count,
        }

    candidate_pairs_report = _report_envelope({
        "contract_content_sha256": contract_content_sha256,
        "hashes_report_content_sha256": hashes_report["content_sha256"],
        "domains": candidate_domains,
    })

    domain_summary_report = _report_envelope({
        "contract_content_sha256": contract_content_sha256,
        "candidate_pairs_report_content_sha256": candidate_pairs_report["content_sha256"],
        "metadata_leakage_report_content_sha256": leakage_report["content_sha256"],
        "domains": domain_summary_entries,
    })

    stop_domains: dict[str, dict] = {}
    any_stop = False
    for domain in mrc.COMPARISON_DOMAINS:
        name = domain["name"]
        if domain["kind"] == "within":
            domain_leakage_findings = []
        else:
            a, b = domain["parts"]
            domain_leakage_findings = [f for f in leakage_findings if {a, b} <= set(f["domains"])]
        # --scan runs before any adjudication -- no pair can yet be a
        # CONFIRMED same_source_image, so this is always empty here.
        result = evaluate_stop_conditions(name, confirmed_same_source_pair_ids=set(),
                                          candidate_pairs=candidate_domains[name],
                                          leakage_findings=domain_leakage_findings)
        stop_domains[name] = {"stop_before_inference": result["stop_before_inference"], "reasons": result["reasons"]}
        any_stop = any_stop or result["stop_before_inference"]

    stop_status_report = _report_envelope({
        "contract_content_sha256": contract_content_sha256,
        "domain_summary_report_content_sha256": domain_summary_report["content_sha256"],
        "domains": stop_domains,
        "overall_stop_before_inference": any_stop,
    })

    reports = {
        "perceptual_hashes_report": hashes_report,
        "metadata_leakage_report": leakage_report,
        "candidate_pairs_report": candidate_pairs_report,
        "domain_summary_report": domain_summary_report,
        "stop_status_report": stop_status_report,
    }
    problems = mrc.validate_scan_report_bundle(reports, contract)
    if problems:
        _fail(f"freshly generated scan report bundle failed its own strict schema/binding "
              f"validation (this is a bug, not a data problem): {problems} -- publishing "
              f"NOTHING; a corrected later --scan remains possible")

    problems = validate_reports_against_population(reports, verified, rows_by_part)
    if problems:
        _fail(f"freshly generated scan report bundle failed re-derivation against the frozen "
              f"population (this is a bug, not a data problem): {problems} -- publishing "
              f"NOTHING; a corrected later --scan remains possible")

    publish_reports_atomically({
        _report_path(args.repo, name): (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        for name, report in reports.items()
    })

    return {
        "ok": True, "contract_content_sha256": contract_content_sha256,
        "part_row_counts": {part: len(rows) for part, rows in rows_by_part.items()},
        "reports": {name: {"path": str(_report_path(args.repo, name)), "content_sha256": report["content_sha256"]}
                   for name, report in reports.items()},
        "overall_stop_before_inference": any_stop,
    }


def cmd_preflight(args) -> dict:
    """Metadata-only: loads/verifies the contract, counts rows per domain
    part from the manifests (never an image), and reports an ESTIMATED
    comparison scale for the real --scan."""
    verified = mrc.load_and_verify_contract(args.repo)
    part_counts: dict[str, int] = {}
    for part in mrc.DOMAIN_PARTS:
        rows = load_domain_part_rows(args.repo, verified, part)
        part_counts[part] = len(rows)

    domain_estimates = []
    total_comparisons = 0
    for domain in mrc.COMPARISON_DOMAINS:
        if domain["kind"] == "within":
            n = part_counts[domain["parts"][0]]
            estimate = n * (n - 1) // 2
        else:
            a, b = domain["parts"]
            estimate = part_counts[a] * part_counts[b]
        total_comparisons += estimate
        domain_estimates.append({"domain": domain["name"], "kind": domain["kind"], "estimated_pairs": estimate})

    return {
        "ok": True,
        "contract_content_sha256": verified["contract"]["content_sha256"],
        "part_row_counts": part_counts,
        "domain_estimates": domain_estimates,
        "total_brute_force_comparisons_estimate": total_comparisons,
        "note": "Brute-force comparison count shown for scale reference only; the real --scan "
                "uses generate_candidate_pairs_indexed (a BK-tree exact radius search per hash "
                "type), whose cost scales with index depth rather than |A|x|B|.",
    }


def cmd_resolve_check(args) -> dict:
    """Metadata/filesystem-only: loads/verifies the contract, loads every
    row for all 8 domain parts, and resolves EVERY row's image path via the
    shared resolver (Path.exists() only -- zero image bytes ever opened).
    Proves the dataset-specific layout binding (item 1) is correct against
    the real repository: fails closed on the first zero-match or
    ambiguous-match row, and otherwise reports per-domain-part row and
    extension counts."""
    verified = mrc.load_and_verify_contract(args.repo)
    image_root = Path(args.repo) / "data"
    image_layouts = verified["contract"]["content"]["domain_part_image_layouts"]

    part_results = {}
    total_rows = 0
    for part in mrc.DOMAIN_PARTS:
        rows = load_domain_part_rows(args.repo, verified, part)
        ext_counts: dict[str, int] = {}
        for row in rows:
            path = mrc.resolve_image_path(image_root, image_layouts, part, row["slug"], row["photo_id"])
            ext_counts[path.suffix] = ext_counts.get(path.suffix, 0) + 1
        part_results[part] = {"row_count": len(rows), "extension_counts": ext_counts}
        total_rows += len(rows)

    return {
        "ok": True, "contract_content_sha256": verified["contract"]["content_sha256"],
        "total_rows": total_rows, "part_results": part_results,
    }


def validate_reports_against_population(reports: dict, verified: dict, rows_by_part: dict[str, list[dict]]) -> list[str]:
    """The genuinely-derived checks that mrc.validate_scan_report_bundle
    cannot perform on its own (it only sees the 5 reports, never the real
    manifests): reconstructs the exact expected row population from the
    frozen manifests (metadata/filesystem-only -- never opens an image) and
    recomputes everything a report CAN be recomputed from without image
    bytes, rejecting any report whose claims disagree -- so a fully empty
    but internally rehashed bundle (or one with missing/invented/altered
    entries) is rejected, not merely one with self-inconsistent hashes."""
    problems: list[str] = []
    contract = verified["contract"]

    expected_rows_by_identity: dict[str, dict] = {}
    identity_domain_part: dict[str, str] = {}
    for part, rows in rows_by_part.items():
        for row in rows:
            expected_rows_by_identity[row["identity"]] = row
            identity_domain_part[row["identity"]] = part
    expected_identities = set(expected_rows_by_identity)

    hashes_content = reports["perceptual_hashes_report"]["content"]
    reported_hashes = hashes_content.get("hashes")
    if not isinstance(reported_hashes, dict):
        return problems + ["perceptual_hashes_report.content.hashes is not a JSON object"]
    reported_identities = set(reported_hashes)

    missing = expected_identities - reported_identities
    invented = reported_identities - expected_identities
    if missing:
        problems.append(f"perceptual_hashes_report is missing {len(missing)} identity(ies) present "
                        f"in the frozen population, e.g. {sorted(missing)[:5]}")
    if invented:
        problems.append(f"perceptual_hashes_report has {len(invented)} invented identity(ies) not in "
                        f"the frozen population, e.g. {sorted(invented)[:5]}")
    for identity in reported_identities & expected_identities:
        expected_sha256 = expected_rows_by_identity[identity]["sha256"]
        actual_sha256 = reported_hashes[identity].get("sha256") if isinstance(reported_hashes[identity], dict) else None
        if actual_sha256 != expected_sha256:
            problems.append(f"perceptual_hashes_report.content.hashes[{identity!r}].sha256 "
                            f"{actual_sha256!r} != manifest-bound {expected_sha256!r}")
    for field in ("pillow_version", "numpy_version"):
        value = hashes_content.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"perceptual_hashes_report.content.{field} must be a nonblank string, got {value!r}")

    if problems:
        # Missing/invented/mistyped hashes make every downstream recomputation
        # meaningless (KeyErrors waiting to happen) -- stop here.
        return problems

    recomputed_leakage = find_metadata_leakage(rows_by_part)
    reported_leakage = reports["metadata_leakage_report"]["content"].get("findings")
    if reported_leakage != recomputed_leakage:
        problems.append("metadata_leakage_report.content.findings does not exactly match the "
                        "deterministic recomputation from the frozen metadata (entries and/or order differ)")

    reported_domains = reports["candidate_pairs_report"]["content"].get("domains", {})
    global_pair_ids: dict[str, str] = {}
    for domain in mrc.COMPARISON_DOMAINS:
        name = domain["name"]
        reported_pairs = reported_domains.get(name)
        if not isinstance(reported_pairs, list):
            problems.append(f"candidate_pairs_report.content.domains[{name!r}] is not a list")
            continue

        if domain["kind"] == "within":
            part = domain["parts"][0]
            recomputed_pairs = generate_candidate_pairs_within_indexed(rows_by_part[part], reported_hashes)
            member_parts = {part}
        else:
            a, b = domain["parts"]
            recomputed_pairs = generate_candidate_pairs_indexed(rows_by_part[a], reported_hashes,
                                                                 rows_by_part[b], reported_hashes)
            member_parts = {a, b}

        if reported_pairs != recomputed_pairs:
            reported_ids = {p.get("pair_id") for p in reported_pairs if isinstance(p, dict)}
            recomputed_ids = {p["pair_id"] for p in recomputed_pairs}
            missing_ids = recomputed_ids - reported_ids
            invented_ids = reported_ids - recomputed_ids
            problems.append(
                f"candidate_pairs_report.content.domains[{name!r}] does not exactly match the "
                f"deterministic indexed recomputation from the reported hashes -- "
                f"missing {len(missing_ids)}, invented {len(invented_ids)}, or a changed distance/order")

        seen_in_domain: set = set()
        for pair in reported_pairs:
            if not isinstance(pair, dict):
                continue
            pid = pair.get("pair_id")
            if pid in seen_in_domain:
                problems.append(f"candidate_pairs_report.content.domains[{name!r}] has duplicate "
                                f"pair_id {pid!r} within the same domain")
            seen_in_domain.add(pid)
            if pid in global_pair_ids and global_pair_ids[pid] != name:
                problems.append(f"candidate_pairs_report pair_id {pid!r} appears in both "
                                f"{global_pair_ids[pid]!r} and {name!r} -- duplicate across the "
                                f"complete report")
            global_pair_ids.setdefault(pid, name)

            ia, ib = pair.get("identity_a"), pair.get("identity_b")
            if domain["kind"] == "within" and isinstance(ia, str) and isinstance(ib, str) and ia > ib:
                problems.append(f"candidate_pairs_report.content.domains[{name!r}] pair {pid!r} "
                                f"identity_a/identity_b are not in canonical sorted order")
            for identity, label in ((ia, "identity_a"), (ib, "identity_b")):
                actual_part = identity_domain_part.get(identity)
                if actual_part not in member_parts:
                    problems.append(f"candidate_pairs_report.content.domains[{name!r}] pair {pid!r} "
                                    f"{label} {identity!r} does not belong to domain part(s) "
                                    f"{sorted(member_parts)} (actual: {actual_part!r})")

    ds_domains = reports["domain_summary_report"]["content"].get("domains", {})
    for domain in mrc.COMPARISON_DOMAINS:
        name = domain["name"]
        entry = ds_domains.get(name)
        if not isinstance(entry, dict):
            continue
        if domain["kind"] == "within":
            expected_a = expected_b = len(rows_by_part[domain["parts"][0]])
        else:
            a, b = domain["parts"]
            expected_a, expected_b = len(rows_by_part[a]), len(rows_by_part[b])
        if entry.get("a_count") != expected_a:
            problems.append(f"domain_summary_report.content.domains[{name!r}].a_count "
                            f"{entry.get('a_count')!r} != actual frozen population count {expected_a}")
        if entry.get("b_count") != expected_b:
            problems.append(f"domain_summary_report.content.domains[{name!r}].b_count "
                            f"{entry.get('b_count')!r} != actual frozen population count {expected_b}")

    return problems


def load_and_verify_full_scan_state(repo: Path) -> dict:
    """THE one shared full-verification entry point -- used by --check
    here, and by every mode of adjudicate_pairs.py and
    finalize_stop_status.py, so none of them can trust a scan-report
    bundle that is merely internally self-consistent (schema-valid,
    cross-report hashes bound to each other) without ALSO being genuinely
    derived from the real frozen manifests.

    Does, in order, and only this:
      1. loads and verifies the frozen contract;
      2. loads all five scan reports (fails closed if any is missing/not
         valid JSON);
      3. runs mrc.validate_scan_report_bundle (schema + cross-report hash
         binding);
      4. reconstructs all eight frozen populations (metadata/filesystem
         only, via load_domain_part_rows -- never opens an image);
      5. runs validate_reports_against_population (genuine re-derivation:
         identity sets, source SHAs, leakage, candidate lists/distances,
         population counts).

    Returns {"verified": <load_and_verify_contract() result>,
    "reports": {...}, "rows_by_part": {...}} -- ONLY once every one of the
    above has passed. Raises ScanError (never partially trusts a bundle)
    on any failure. Callers needing ledger verification (adjudicate_pairs,
    finalize_stop_status) layer that on top of this, since neither the
    scan reports nor the population depend on the ledger."""
    verified = mrc.load_and_verify_contract(repo)
    reports = {}
    for name in mrc.SCAN_REPORT_NAMES:
        path = _report_path(repo, name)
        if not path.exists():
            _fail(f"{name} does not exist yet: {path} -- run --scan first (a separately authorized phase)")
        try:
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fail(f"{path} is not valid JSON: {exc}")

    problems = mrc.validate_scan_report_bundle(reports, verified["contract"])
    if problems:
        _fail(f"scan report bundle failed schema/binding validation: {problems}")

    rows_by_part = {part: load_domain_part_rows(repo, verified, part) for part in mrc.DOMAIN_PARTS}
    problems = validate_reports_against_population(reports, verified, rows_by_part)
    if problems:
        _fail(f"scan report bundle failed re-derivation against the frozen population: {problems}")

    return {"verified": verified, "reports": reports, "rows_by_part": rows_by_part}


def cmd_check(args) -> dict:
    """Metadata/filesystem-only validation of the completed scan reports
    WITHOUT rerunning --scan or opening any image: delegates entirely to
    load_and_verify_full_scan_state (schema/binding, THEN full
    re-derivation against the real frozen manifests), so a fully empty (or
    otherwise fabricated) but internally rehashed bundle is rejected, not
    merely one with self-inconsistent hashes."""
    state = load_and_verify_full_scan_state(args.repo)
    results = {name: {"path": str(_report_path(args.repo, name)), "content_sha256": report["content_sha256"]}
              for name, report in state["reports"].items()}
    return {"ok": True, "contract_content_sha256": state["verified"]["contract"]["content_sha256"],
            "reports": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--resolve-check", action="store_true")
    mode.add_argument("--scan", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        if args.preflight:
            result = cmd_preflight(args)
        elif args.resolve_check:
            result = cmd_resolve_check(args)
        elif args.scan:
            result = cmd_scan(args)
        else:
            result = cmd_check(args)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ScanError, mrc.ContractError) as e:
        print(f"[scan_perceptual_duplicates] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
