#!/usr/bin/env python3
"""build_geo_index.py — build geo_index.json from iNaturalist observation data.

Reads the same gzipped manifests scrape_inat.py downloads (observations.csv.gz
has latitude/longitude per research-grade observation) and bins each species'
observations into 1-degree lat/lon grid cells. The API uses this index to
boost species that have actually been observed near the user's location.

Output format (training/artifacts/geo_index.json):
{
  "cell_size_deg": 1.0,
  "cells": {
    "<species-slug>": [[cell_lat, cell_lon], ...]   # ints: floor(deg / size)
  }
}

Usage:
    python build_geo_index.py --metadata-dir ../data/inat_metadata \
        --species-file species_list.txt \
        --out ../training/artifacts/geo_index.json

Requires the taxa + observations manifests already downloaded (scrape_inat.py
does this; pass the same --metadata-dir).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from common import load_species_list

HERE = Path(__file__).resolve().parent
CHUNK = 500_000


def resolve_taxon_ids(taxa_gz: Path, wanted: dict[str, str]) -> dict[int, str]:
    """Map taxon_id -> slug for the wanted {scientific_name: slug} species."""
    import pandas as pd

    found: dict[int, str] = {}
    for chunk in pd.read_csv(taxa_gz, sep="\t", usecols=["taxon_id", "name", "rank"],
                             chunksize=CHUNK):
        hit = chunk[(chunk["rank"] == "species") & (chunk["name"].isin(wanted))]
        for _, row in hit.iterrows():
            found[int(row["taxon_id"])] = wanted[row["name"]]
        if len(found) == len(wanted):
            break
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-dir", type=Path, required=True,
                    help="Dir holding taxa.csv.gz and observations.csv.gz")
    ap.add_argument("--species-file", type=Path, default=HERE / "species_list.txt")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / "training" / "artifacts" / "geo_index.json")
    ap.add_argument("--cell-deg", type=float, default=1.0)
    ap.add_argument("--min-obs-per-cell", type=int, default=2,
                    help="Ignore cells with fewer observations (noise filter).")
    args = ap.parse_args()

    import pandas as pd  # heavy import kept off the --help path

    species = load_species_list(args.species_file)
    wanted = {sp.scientific_name: sp.slug for sp in species}

    taxa_gz = args.metadata_dir / "taxa.csv.gz"
    obs_gz = args.metadata_dir / "observations.csv.gz"
    for p in (taxa_gz, obs_gz):
        if not p.exists():
            raise SystemExit(f"Missing manifest: {p} (run scrape_inat.py first)")

    taxon_to_slug = resolve_taxon_ids(taxa_gz, wanted)
    print(f"[geo] resolved {len(taxon_to_slug)}/{len(wanted)} species to taxon ids")

    cs = args.cell_deg
    counts: dict[str, dict[tuple[int, int], int]] = {s: {} for s in wanted.values()}

    cols = ["taxon_id", "latitude", "longitude", "quality_grade"]
    n_rows = 0
    for chunk in pd.read_csv(obs_gz, sep="\t", usecols=cols, chunksize=CHUNK):
        chunk = chunk[(chunk["quality_grade"] == "research")
                      & chunk["taxon_id"].isin(taxon_to_slug)
                      & chunk["latitude"].notna() & chunk["longitude"].notna()]
        for _, row in chunk.iterrows():
            slug = taxon_to_slug[int(row["taxon_id"])]
            cell = (math.floor(row["latitude"] / cs), math.floor(row["longitude"] / cs))
            counts[slug][cell] = counts[slug].get(cell, 0) + 1
        n_rows += len(chunk)
    print(f"[geo] binned {n_rows} research-grade observations")

    cells = {
        slug: sorted([list(c) for c, n in cell_counts.items()
                      if n >= args.min_obs_per_cell])
        for slug, cell_counts in counts.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cell_size_deg": cs, "cells": cells}, indent=1))
    total = sum(len(v) for v in cells.values())
    print(f"[geo] wrote {args.out} ({total} cells across {len(cells)} species)")


if __name__ == "__main__":
    main()
