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
from pathlib import Path

import numpy as np

from data import load_manifest

HERE = Path(__file__).resolve().parent
FIELDS = ("species_name", "common_name", "taxon_id")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", type=Path, default=HERE / "artifacts")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    art = args.artifacts_dir if args.artifacts_dir.is_absolute() else HERE / args.artifacts_dir
    tax_path, proto_path = art / "taxonomy.json", art / "prototypes.npy"

    _, taxonomy = load_manifest({})
    new = {int(k): v for k, v in taxonomy.items()}
    new_slugs = [new[i]["slug"] for i in sorted(new)]

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

    still_null = [n["slug"] for i, n in sorted(new.items())
                  if n.get("taxon_id") is None]
    if still_null:
        print(f"\nNOTE: {len(still_null)} species still have a null taxon_id "
              f"(e.g. {', '.join(still_null[:3])}).")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    tax_path.write_text(
        json.dumps({str(k): new[k] for k in sorted(new)}, indent=2))
    print(f"\nwrote {tax_path} ({len(new)} classes, slug order preserved)")


if __name__ == "__main__":
    main()
