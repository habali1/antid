#!/usr/bin/env python3
"""scrape_inat.py — primary data source: iNaturalist Open Data on AWS S3.

iNaturalist publishes its full observation dataset under the AWS Open Data
program at s3://inaturalist-open-data (public, no credentials, --no-sign-request).
This is far faster than the API for bulk pulls and has no rate limits.

Pipeline:
  1. Ensure the three gzipped metadata manifests are present locally
     (taxa.csv.gz, observations.csv.gz, photos.csv.gz). These are large
     (observations is multi-GB) so they're streamed/chunked, not loaded whole.
  2. Resolve target species -> iNat taxon_id via taxa manifest
     (or --auto-discover the top-N most-observed Formicidae).
  3. Join research-grade observations -> photos to build the download list.
  4. Download up to --images-per-species photos per species to --out.
  5. Record metadata in PostgreSQL if DATABASE_URL is set, else write a
     manifest CSV alongside the images.

The metadata files use TAB separators despite the .csv name.

Examples
--------
  python scrape_inat.py --dry-run
  python scrape_inat.py --species-limit 30 --images-per-species 200
  python scrape_inat.py --auto-discover --species-limit 50
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# Lightweight, dependency-free imports only at module top.
sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    StorageClient, TargetSpecies, get_db_connection, insert_image,
    load_species_list, md5_of_bytes, slugify, upsert_species,
)

INAT_BUCKET = "inaturalist-open-data"
FORMICIDAE_TAXON_ID = 47336  # family Formicidae on iNaturalist
METADATA_FILES = ("taxa.csv.gz", "observations.csv.gz", "photos.csv.gz")
PHOTO_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos"


# --------------------------------------------------------------------------- #
# Metadata acquisition
# --------------------------------------------------------------------------- #
def ensure_metadata(metadata_dir: Path) -> None:
    """Download the gzipped manifests from the public bucket if missing."""
    import boto3                      # lazy
    from botocore import UNSIGNED     # lazy
    from botocore.config import Config

    metadata_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    for name in METADATA_FILES:
        dest = metadata_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [cache] {name} already present")
            continue
        print(f"  [download] s3://{INAT_BUCKET}/{name} -> {dest}")
        s3.download_file(INAT_BUCKET, name, str(dest))


# --------------------------------------------------------------------------- #
# Species resolution
# --------------------------------------------------------------------------- #
def resolve_taxon_ids(metadata_dir: Path,
                      targets: list[TargetSpecies]) -> dict[str, int]:
    """Map scientific_name (lowercased) -> taxon_id using taxa.csv.gz."""
    import gzip
    import pandas as pd  # lazy

    want = {t.scientific_name.lower() for t in targets}
    found: dict[str, int] = {}
    taxa_path = metadata_dir / "taxa.csv.gz"
    with gzip.open(taxa_path, "rt", encoding="utf-8") as fh:
        for chunk in pd.read_csv(fh, sep="\t", usecols=["taxon_id", "name", "rank"],
                                 chunksize=500_000, dtype=str):
            hit = chunk[chunk["name"].str.lower().isin(want)]
            for _, row in hit.iterrows():
                key = row["name"].lower()
                if key not in found:
                    found[key] = int(row["taxon_id"])
            if len(found) == len(want):
                break
    return found


def discover_top_species(metadata_dir: Path, limit: int) -> list[tuple[int, str, int]]:
    """Return [(taxon_id, name, observation_count)] for the most-observed
    Formicidae species, descending. Requires scanning observations + taxa."""
    import gzip
    from collections import Counter
    import pandas as pd  # lazy

    # 1. Find all species-rank taxa whose ancestry includes Formicidae.
    formicid_species: dict[int, str] = {}
    with gzip.open(metadata_dir / "taxa.csv.gz", "rt", encoding="utf-8") as fh:
        for chunk in pd.read_csv(fh, sep="\t",
                                 usecols=["taxon_id", "ancestry", "rank", "name"],
                                 chunksize=500_000, dtype=str):
            mask = (chunk["rank"] == "species") & chunk["ancestry"].fillna("").str.contains(
                fr"(?:^|/){FORMICIDAE_TAXON_ID}(?:/|$)")
            for _, row in chunk[mask].iterrows():
                formicid_species[int(row["taxon_id"])] = row["name"]

    # 2. Count research-grade observations per species taxon.
    counts: Counter[int] = Counter()
    valid = set(formicid_species)
    with gzip.open(metadata_dir / "observations.csv.gz", "rt", encoding="utf-8") as fh:
        for chunk in pd.read_csv(fh, sep="\t",
                                 usecols=["taxon_id", "quality_grade"],
                                 chunksize=1_000_000, dtype=str):
            rg = chunk[chunk["quality_grade"] == "research"]
            ids = pd.to_numeric(rg["taxon_id"], errors="coerce").dropna().astype(int)
            counts.update(ids[ids.isin(valid)].tolist())

    top = counts.most_common(limit)
    return [(tid, formicid_species[tid], n) for tid, n in top]


# --------------------------------------------------------------------------- #
# Photo list + download
# --------------------------------------------------------------------------- #
def build_photo_list(metadata_dir: Path, taxon_ids: dict[int, str],
                     images_per_species: int) -> dict[int, list[tuple[str, str]]]:
    """taxon_id -> [(photo_id, extension, lat, lon)], capped at images_per_species."""
    import gzip
    import pandas as pd  # lazy

    wanted = set(taxon_ids)

    # observation_uuid -> (taxon_id, lat, lon), for research-grade obs of targets.
    obs_to_taxon: dict[str, tuple[int, Optional[float], Optional[float]]] = {}
    with gzip.open(metadata_dir / "observations.csv.gz", "rt", encoding="utf-8") as fh:
        for chunk in pd.read_csv(fh, sep="\t",
                                 usecols=["observation_uuid", "taxon_id",
                                          "quality_grade", "latitude", "longitude"],
                                 chunksize=1_000_000, dtype=str):
            rg = chunk[chunk["quality_grade"] == "research"].copy()
            rg["tid"] = pd.to_numeric(rg["taxon_id"], errors="coerce")
            rg = rg[rg["tid"].isin(wanted)]
            rg["lat"] = pd.to_numeric(rg["latitude"], errors="coerce")
            rg["lon"] = pd.to_numeric(rg["longitude"], errors="coerce")
            for _, row in rg.iterrows():
                la = None if pd.isna(row["lat"]) else float(row["lat"])
                lo = None if pd.isna(row["lon"]) else float(row["lon"])
                obs_to_taxon[row["observation_uuid"]] = (int(row["tid"]), la, lo)

    # Walk photos manifest, collecting up to N per taxon.
    out: dict[int, list[tuple[str, str, Optional[float], Optional[float]]]] = {
        tid: [] for tid in wanted
    }
    target_uuids = set(obs_to_taxon)
    with gzip.open(metadata_dir / "photos.csv.gz", "rt", encoding="utf-8") as fh:
        for chunk in pd.read_csv(fh, sep="\t",
                                 usecols=["photo_id", "observation_uuid", "extension"],
                                 chunksize=1_000_000, dtype=str):
            # Vectorized filter to our target observations BEFORE iterating rows;
            # iterrows over the full ~250M-row photo manifest would take hours.
            hit = chunk[chunk["observation_uuid"].isin(target_uuids)]
            for _, row in hit.iterrows():
                rec = obs_to_taxon.get(row["observation_uuid"])
                if rec is None:
                    continue
                tid, la, lo = rec
                if len(out[tid]) >= images_per_species:
                    continue
                out[tid].append((row["photo_id"], row["extension"] or "jpg", la, lo))
            if all(len(v) >= images_per_species for v in out.values()):
                break
    return out


def download_photo(photo_id: str, ext: str) -> Optional[bytes]:
    """Download a single 'medium' photo from the public bucket."""
    import boto3                      # lazy
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    key = f"photos/{photo_id}/medium.{ext}"
    try:
        obj = s3.get_object(Bucket=INAT_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! failed {key}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    metadata_dir = Path(args.metadata_dir)
    out_dir = Path(args.out)

    # ---- Resolve the target species set -------------------------------------
    if args.auto_discover:
        if args.dry_run and not (metadata_dir / "observations.csv.gz").exists():
            print("DRY RUN — auto-discover mode")
            print(f"  Would scan s3://{INAT_BUCKET} metadata for the top "
                  f"{args.species_limit} most-observed Formicidae species.")
            print("  (metadata not cached locally; nothing downloaded)")
            return 0
        ensure_metadata(metadata_dir)
        discovered = discover_top_species(metadata_dir, args.species_limit)
        targets = [TargetSpecies(name, None) for _, name, _ in discovered]
        taxon_by_name = {name.lower(): tid for tid, name, _ in discovered}
        print(f"Auto-discovered {len(targets)} species:")
        for tid, name, n in discovered:
            print(f"  {name:<32} taxon_id={tid:<8} observations={n}")
    else:
        targets = load_species_list(Path(args.species_file))[: args.species_limit]
        if args.dry_run:
            print(f"DRY RUN — {len(targets)} target species "
                  f"(from {args.species_file}):")
            for i, t in enumerate(targets):
                common = f"  ({t.common_name})" if t.common_name else ""
                print(f"  {i:>2}. {t.scientific_name}{common}")
            print(f"\nWould download up to {args.images_per_species} images/species "
                  f"from s3://{INAT_BUCKET} into {out_dir}/")
            return 0
        ensure_metadata(metadata_dir)
        taxon_by_name = resolve_taxon_ids(metadata_dir, targets)
        missing = [t.scientific_name for t in targets
                   if t.scientific_name.lower() not in taxon_by_name]
        if missing:
            print(f"  ! no iNat taxon match for: {', '.join(missing)}",
                  file=sys.stderr)

    # ---- Build the per-taxon photo list -------------------------------------
    name_to_target = {t.scientific_name.lower(): t for t in targets}
    taxon_ids = {tid: name for name, tid in taxon_by_name.items()}
    photo_list = build_photo_list(metadata_dir, taxon_ids, args.images_per_species)

    # ---- Optional sinks ------------------------------------------------------
    conn = get_db_connection()
    storage = StorageClient(args.bucket) if args.bucket else None
    manifest_rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_idx, (lname, target) in enumerate(name_to_target.items()):
        tid = taxon_by_name.get(lname)
        if tid is None:
            continue
        species_id = None
        if conn is not None:
            species_id = upsert_species(conn, target, tid, class_idx)
        photos = photo_list.get(tid, [])
        slug = target.slug
        local_species_dir = out_dir / slug
        local_species_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{target.scientific_name}] downloading {len(photos)} photos...")
        seen_hashes: set[str] = set()
        for photo_id, ext, lat, lon in photos:
            data = download_photo(photo_id, ext)
            if not data:
                continue
            digest = md5_of_bytes(data)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            fname = f"{photo_id}.{ext}"
            (local_species_dir / fname).write_bytes(data)
            storage_path = f"{slug}/{fname}"
            if storage is not None:
                storage_path = storage.upload_bytes(f"{slug}/{fname}", data)
            split = "val" if (total % 5 == 0) else "train"  # ~80/20
            if conn is not None and species_id is not None:
                insert_image(conn, species_id, "inat", storage_path, split,
                             None, None, lat, lon)
            manifest_rows.append({
                "species": target.scientific_name, "slug": slug,
                "taxon_id": tid, "source": "inat",
                "storage_path": storage_path, "split": split,
                "lat": lat, "lon": lon,
            })
            total += 1

    if conn is not None:
        conn.close()
    if manifest_rows:
        manifest_path = out_dir / "manifest_inat.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nWrote manifest: {manifest_path}")
    print(f"Done. {total} images across {len(name_to_target)} species.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species-file",
                   default=str(Path(__file__).parent / "species_list.txt"),
                   help="Target species list (default: species_list.txt)")
    p.add_argument("--species-limit", type=int, default=30,
                   help="Max number of species to process")
    p.add_argument("--images-per-species", type=int, default=200,
                   help="Max images to download per species")
    p.add_argument("--auto-discover", action="store_true",
                   help="Ignore species file; pull the top-N most-observed "
                        "Formicidae from iNat metadata instead")
    p.add_argument("--metadata-dir", default="./data/inat_metadata",
                   help="Where the gzipped manifests are cached")
    p.add_argument("--out", default="./data/raw/inat",
                   help="Local output directory for downloaded images")
    p.add_argument("--bucket", default=None,
                   help="Optional gs:// or s3:// bucket to also upload to "
                        "(else only local + DB)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the target species list and exit; download nothing")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
