#!/usr/bin/env python3
"""build_geo_from_manifest.py — build geo_index.json from a scraper manifest.

Companion to scrape_inat_api.py, which records lat/lon per image. Bins each
species' observation coordinates into a grid (default 1 degree) and writes the
same geo_index.json format api/inference.py consumes. (build_geo_index.py does
the same thing from the S3 open-data observations manifest; this does it from
the API scraper's CSV, so the geo index doesn't need the multi-GB S3 download.)

Usage:
    python build_geo_from_manifest.py --manifest ../data/raw/manifest_inat.csv \
        --out ../training/artifacts/geo_index.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-deg", type=float, default=1.0)
    ap.add_argument("--min-obs-per-cell", type=int, default=2,
                    help="Drop cells with fewer observations (noise filter).")
    args = ap.parse_args()

    cs = args.cell_deg
    counts: dict[str, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    n_rows = n_coords = 0
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            lat, lon = row.get("lat"), row.get("lon")
            if not lat or not lon:
                continue
            try:
                la, lo = float(lat), float(lon)
            except ValueError:
                continue
            n_coords += 1
            cell = (math.floor(la / cs), math.floor(lo / cs))
            counts[row["slug"]][cell] += 1

    cells = {
        slug: sorted([list(c) for c, n in cc.items() if n >= args.min_obs_per_cell])
        for slug, cc in counts.items()
    }
    cells = {k: v for k, v in cells.items() if v}  # drop species with no cells left

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cell_size_deg": cs, "cells": cells}, indent=1))
    total = sum(len(v) for v in cells.values())
    print(f"rows={n_rows}  with_coords={n_coords}")
    print(f"wrote {out}: {total} cells across {len(cells)} species "
          f"(min_obs_per_cell={args.min_obs_per_cell})")


if __name__ == "__main__":
    main()
