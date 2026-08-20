#!/usr/bin/env python3
"""rebuild_taxonomy.py — refresh artifacts/taxonomy.json from the current data
source, without retraining.

Class indices are assigned by sorting species slugs, so as long as the slug set
is unchanged, taxonomy entry i still describes prototypes.npy row i. That makes
it safe to backfill metadata a previous run could not know — notably taxon_id,
which the bare-directory loader has no way to recover and therefore writes as
null for every species.

REFUSES to write if the class count or slug ordering would change, since that
would silently misalign taxonomy.json against the existing prototypes.

Usage (from training/):
    MANIFEST_CSV=../data/manifest_all.csv LOCAL_DATA_DIR=../data/clean \
        python rebuild_taxonomy.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from data import load_manifest

HERE = Path(__file__).resolve().parent
FIELDS = ("species_name", "common_name", "taxon_id")

INAT_API = "https://api.inaturalist.org/v1"
INAT_HEADERS = {"User-Agent": "AntID-pipeline/1.0 (educational project)"}


def fetch_common_names(taxon_ids: list[int]) -> dict[int, str]:
    """taxon_id -> preferred_common_name, via the iNaturalist taxa endpoint.

    Neither the manifest CSV nor the directory walk carries a common name, so
    this is the only way to fill the field without a database. Species with no
    common name on iNat are simply omitted — plenty of ants genuinely have
    none, and a made-up name is worse than null.
    """
    import httpx

    out: dict[int, str] = {}
    with httpx.Client(headers=INAT_HEADERS, timeout=60,
                      follow_redirects=True) as client:
        for start in range(0, len(taxon_ids), 30):      # endpoint takes 30 ids
            batch = taxon_ids[start:start + 30]
            ids = ",".join(str(t) for t in batch)
            for attempt in range(4):
                try:
                    r = client.get(f"{INAT_API}/taxa/{ids}")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2.0 * (attempt + 1))
            else:
                print(f"  WARNING: lookup failed for {len(batch)} taxon ids; "
                      f"leaving those common names unchanged")
                continue
            for res in r.json().get("results", []):
                name = res.get("preferred_common_name")
                if name:
                    out[int(res["id"])] = name
            time.sleep(1.0)                              # be polite to the API
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", type=Path, default=HERE / "artifacts")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    ap.add_argument("--fetch-common-names", action="store_true",
                    help="Look up missing common names on the iNaturalist API "
                         "(needs network). No data source here carries them.")
    args = ap.parse_args()

    art = args.artifacts_dir if args.artifacts_dir.is_absolute() else HERE / args.artifacts_dir
    tax_path, proto_path = art / "taxonomy.json", art / "prototypes.npy"

    _, taxonomy = load_manifest({})
    new = {int(k): v for k, v in taxonomy.items()}
    new_slugs = [new[i]["slug"] for i in sorted(new)]

    if args.fetch_common_names:
        wanted = sorted({v["taxon_id"] for v in new.values()
                         if v.get("taxon_id") and not v.get("common_name")})
        if not wanted:
            print("No species need a common name lookup.")
        else:
            print(f"Looking up common names for {len(wanted)} taxon ids…")
            found = fetch_common_names(wanted)
            for v in new.values():
                if not v.get("common_name") and v.get("taxon_id") in found:
                    v["common_name"] = found[v["taxon_id"]]
            print(f"iNaturalist returned {len(found)} common names.")

    # ---- alignment guards -------------------------------------------------
    if proto_path.exists():
        n_rows = int(np.load(proto_path, mmap_mode="r").shape[0])
        if n_rows != len(new):
            sys.exit(f"REFUSING: the data source has {len(new)} classes but "
                     f"{proto_path.name} has {n_rows} rows. Retrain instead.")

    old: dict[int, dict] = {}
    if tax_path.exists():
        old = {int(k): v for k, v in json.loads(tax_path.read_text()).items()}
        old_slugs = [old[i]["slug"] for i in sorted(old)]
        if old_slugs != new_slugs:
            added = sorted(set(new_slugs) - set(old_slugs))
            removed = sorted(set(old_slugs) - set(new_slugs))
            sys.exit(
                "REFUSING: slug ordering would change, so prototype rows would no "
                "longer line up with taxonomy entries.\n"
                f"  added:   {added or '-'}\n"
                f"  removed: {removed or '-'}\n"
                "  Retrain so prototypes and taxonomy are regenerated together."
            )

    # ---- report -----------------------------------------------------------
    changed = []
    for i in sorted(new):
        o, n = old.get(i, {}), new[i]
        diffs = {k: (o.get(k), n.get(k)) for k in FIELDS if o.get(k) != n.get(k)}
        if diffs:
            changed.append((i, n["slug"], diffs))

    if not changed:
        print(f"taxonomy.json already matches the data source "
              f"({len(new)} classes). Nothing to do.")
        return

    print(f"{len(changed)}/{len(new)} classes would change:")
    for i, slug, diffs in changed[:8]:
        bits = ", ".join(f"{k} {o!r} -> {n!r}" for k, (o, n) in diffs.items())
        print(f"  [{i:>2}] {slug}: {bits}")
    if len(changed) > 8:
        print(f"  ... and {len(changed) - 8} more")

    for field in ("taxon_id", "common_name"):
        still_null = [n["slug"] for i, n in sorted(new.items()) if not n.get(field)]
        if still_null:
            hint = ""
            if field == "common_name":
                hint = (" iNaturalist has no common name for these."
                        if args.fetch_common_names
                        else " Re-run with --fetch-common-names to fill these.")
            print(f"\nNOTE: {len(still_null)} species still have a null {field} "
                  f"(e.g. {', '.join(still_null[:3])}).{hint}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    tax_path.write_text(
        json.dumps({str(k): new[k] for k in sorted(new)}, indent=2))
    print(f"\nwrote {tax_path} ({len(new)} classes, slug order preserved)")


if __name__ == "__main__":
    main()
