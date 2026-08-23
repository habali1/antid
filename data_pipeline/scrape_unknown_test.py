#!/usr/bin/env python3
"""scrape_unknown_test.py — build unknown_test_v1, the FINAL independent test
set for the frozen abstention rule (max_sim < 0.60, see
data/calibration_v1/calibration_v1.json:frozen_candidate_abstention_threshold).

This is not another calibration round. calibration_v1 CHOSE the 0.60
threshold; unknown_test_v1 exists to grade it exactly once, on data that
never influenced that choice in any way:

  - known_holdout, out_of_scope_ant, non_ant_insect, unrelated -- the same
    four category definitions as calibration_v1 (low_quality_known is not
    rebuilt here: its whole purpose was exploring an ambiguous case during
    calibration, not final grading).
  - out_of_scope_ant is SPECIES-disjoint from calibration_v1's out_of_scope_ant,
    not just image-disjoint: none of the 165 species calibration_v1 sampled
    are eligible here (via without_taxon_id on 50 known + 165 calibrated-against
    taxon_ids). This tests whether the rejection rule generalizes to a
    genuinely novel ant species, not just a novel photo of a species the
    threshold was implicitly calibrated against.
  - Every category is also excluded by photo_id against training,
    benchmark_v1, AND calibration_v1, and by observation_uuid against
    benchmark_v1 and calibration_v1 (both recorded it; training did not, so
    the 2026-06-16 date-cutoff structural guarantee carries that half, same
    as calibration_v1 used).

Same rigor as calibration_v1: photo_id, observation_uuid, sha256 recorded per
row; verified zero overlap after the fact against training, benchmark_v1, AND
calibration_v1 on all three axes, plus the species-level check for
out_of_scope_ant. --restore re-fetches exactly what the frozen CSV lists and
never rewrites it, same as scrape_benchmark.py / scrape_calibration.py.

Usage:
    python scrape_unknown_test.py --taxonomy ../training/artifacts/taxonomy.json \
        --training-manifest ../data/manifest_all.csv \
        --benchmark-csv ../data/benchmark_v1/benchmark_v1.csv \
        --calibration-csv ../data/calibration_v1/calibration_v1.csv \
        --out ../data/unknown_test_v1

    python scrape_unknown_test.py --restore --out ../data/unknown_test_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from scrape_calibration import (  # noqa: E402
    API, CSV_FIELDS as _CALIB_CSV_FIELDS, DEFAULT_CUTOFF, HEADERS,
    TAXON_FORMICIDAE, TAXON_INSECTA, UNRELATED_TAXA,
    blur_score, download_candidates, fetch_diverse_candidates,
    fetch_known_species_candidates, fetch_observations_by_uuid, log,
    restore_from_csv,
)

# unknown_test_v1 doesn't build a low_quality_known category, so no blur_score
# column is needed -- but download_candidates always computes one (cheap,
# harmless) so keep the same CSV shape as calibration_v1 for tooling reuse.
CSV_FIELDS = _CALIB_CSV_FIELDS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--restore-csv", type=Path, default=None)
    ap.add_argument("--taxonomy", type=Path)
    ap.add_argument("--training-manifest", type=Path)
    ap.add_argument("--benchmark-csv", type=Path)
    ap.add_argument("--calibration-csv", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cutoff-date", default=DEFAULT_CUTOFF)
    ap.add_argument("--n-known-holdout", type=int, default=200)
    ap.add_argument("--n-out-of-scope-ant", type=int, default=200)
    ap.add_argument("--n-non-ant-insect", type=int, default=100)
    ap.add_argument("--n-unrelated", type=int, default=100)
    args = ap.parse_args()

    if args.restore:
        csv_path = args.restore_csv or (args.out / "unknown_test_v1.csv")
        if not csv_path.exists():
            raise SystemExit(f"--restore: {csv_path} does not exist")
        args.out.mkdir(parents=True, exist_ok=True)
        with httpx.Client(headers=HEADERS, timeout=90, follow_redirects=True) as client:
            n_ok, errors = restore_from_csv(client, csv_path, args.out)
        log(f"[restore] {n_ok} verified OK, {len(errors)} error(s)")
        for e in errors[:50]:
            log(f"  ! {e}")
        if errors:
            raise SystemExit(f"[restore] FAILED: {len(errors)} row(s) could not be verified.")
        return

    if not all([args.taxonomy, args.training_manifest, args.benchmark_csv, args.calibration_csv]):
        raise SystemExit("--taxonomy, --training-manifest, --benchmark-csv, --calibration-csv "
                         "are required unless --restore is set")

    taxonomy = json.loads(args.taxonomy.read_text())
    known_species = sorted(({"slug": v["slug"], "species": v["species_name"],
                             "taxon_id": v["taxon_id"]} for v in taxonomy.values()),
                           key=lambda s: s["slug"])
    known_taxon_ids = {str(s["taxon_id"]) for s in known_species}

    train_photo_ids = {r["photo_id"] for r in csv.DictReader(
        args.training_manifest.open(newline="", encoding="utf-8"))}
    bench_rows = list(csv.DictReader(args.benchmark_csv.open(newline="", encoding="utf-8")))
    bench_photo_ids = {r["photo_id"] for r in bench_rows}
    bench_obs_uuids = {r["observation_uuid"] for r in bench_rows if r.get("observation_uuid")}
    calib_rows = list(csv.DictReader(args.calibration_csv.open(newline="", encoding="utf-8")))
    calib_photo_ids = {r["photo_id"] for r in calib_rows}
    calib_obs_uuids = {r["observation_uuid"] for r in calib_rows if r.get("observation_uuid")}

    exclude_photo_ids = train_photo_ids | bench_photo_ids | calib_photo_ids
    exclude_obs_uuids = bench_obs_uuids | calib_obs_uuids

    calib_ood_ant_taxon_ids = {r["taxon_id"] for r in calib_rows
                               if r["category"] == "out_of_scope_ant"}
    ood_ant_without_ids = ",".join(sorted(known_taxon_ids | calib_ood_ant_taxon_ids))

    log(f"[unknown-test] excluding {len(train_photo_ids)} training + {len(bench_photo_ids)} "
       f"benchmark_v1 + {len(calib_photo_ids)} calibration_v1 photo_ids "
       f"({len(exclude_photo_ids)} unique); {len(exclude_obs_uuids)} unique observation_uuids")
    log(f"[unknown-test] out_of_scope_ant additionally excludes calibration_v1's "
       f"{len(calib_ood_ant_taxon_ids)} out-of-scope-ant species by taxon_id "
       f"({len(known_taxon_ids | calib_ood_ant_taxon_ids)} total taxon_ids excluded)")

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    shortfalls: list[tuple[str, int, int]] = []

    with httpx.Client(headers=HEADERS, timeout=90, follow_redirects=True) as client:
        # ---- known_holdout ----
        per_sp = max(1, -(-args.n_known_holdout // len(known_species)))
        log(f"\n[unknown-test] known_holdout: target {args.n_known_holdout} (~{per_sp}/species)")
        cands = []
        for sp in known_species:
            c = fetch_known_species_candidates(
                client, sp["taxon_id"], sp["species"], sp["slug"], per_sp,
                args.cutoff_date, exclude_photo_ids, exclude_obs_uuids)
            cands.extend(c)
        cands = cands[:args.n_known_holdout]
        if len(cands) < args.n_known_holdout:
            shortfalls.append(("known_holdout", len(cands), args.n_known_holdout))
        rows = download_candidates(client, cands, args.out, "known_holdout", "inat_api_unknown_test")
        log(f"[unknown-test] known_holdout: saved {len(rows)}/{len(cands)}")
        all_rows.extend(rows)

        # ---- out_of_scope_ant (species-disjoint from calibration_v1's 165) ----
        log(f"\n[unknown-test] out_of_scope_ant: target {args.n_out_of_scope_ant} "
           f"(species-disjoint from calibration_v1)")
        cands, n_sp = fetch_diverse_candidates(
            client, TAXON_FORMICIDAE, args.n_out_of_scope_ant, args.cutoff_date,
            exclude_photo_ids, exclude_obs_uuids, without_taxon_ids=ood_ant_without_ids)
        if len(cands) < args.n_out_of_scope_ant:
            shortfalls.append(("out_of_scope_ant", len(cands), args.n_out_of_scope_ant))
        rows = download_candidates(client, cands, args.out, "out_of_scope_ant", "inat_api_unknown_test")
        log(f"[unknown-test] out_of_scope_ant: saved {len(rows)}/{len(cands)} across {n_sp} species")
        all_rows.extend(rows)

        # ---- non_ant_insect ----
        log(f"\n[unknown-test] non_ant_insect: target {args.n_non_ant_insect}")
        cands, n_sp = fetch_diverse_candidates(
            client, TAXON_INSECTA, args.n_non_ant_insect, args.cutoff_date,
            exclude_photo_ids, exclude_obs_uuids, without_taxon_ids=str(TAXON_FORMICIDAE))
        if len(cands) < args.n_non_ant_insect:
            shortfalls.append(("non_ant_insect", len(cands), args.n_non_ant_insect))
        rows = download_candidates(client, cands, args.out, "non_ant_insect", "inat_api_unknown_test")
        log(f"[unknown-test] non_ant_insect: saved {len(rows)}/{len(cands)} across {n_sp} species")
        all_rows.extend(rows)

        # ---- unrelated (split across Aves/Plantae/Mammalia) ----
        log(f"\n[unknown-test] unrelated: target {args.n_unrelated} across {list(UNRELATED_TAXA)}")
        per_taxon = -(-args.n_unrelated // len(UNRELATED_TAXA))
        for name, tid in UNRELATED_TAXA.items():
            cands, n_sp = fetch_diverse_candidates(
                client, tid, per_taxon, args.cutoff_date, exclude_photo_ids, exclude_obs_uuids)
            rows = download_candidates(client, cands, args.out, "unrelated", "inat_api_unknown_test")
            log(f"[unknown-test] unrelated/{name}: saved {len(rows)}/{len(cands)} across {n_sp} species")
            all_rows.extend(rows)

    csv_path = args.out / "unknown_test_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    log(f"\n[unknown-test] wrote {csv_path} ({len(all_rows)} rows)")

    import collections
    cat_counts = collections.Counter(r["category"] for r in all_rows)
    for cat, n in cat_counts.items():
        log(f"  {cat}: {n}")
    if shortfalls:
        log(f"[unknown-test] WARNING: {len(shortfalls)} categories below target:")
        for cat, got, want in shortfalls:
            log(f"    {cat}: {got}/{want}")


if __name__ == "__main__":
    main()
