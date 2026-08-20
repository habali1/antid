#!/usr/bin/env python3
"""scrape_gbif.py — tertiary source: GBIF occurrences (gap-fill rare species).

Used for species where iNat has too few images. Flow per species:
  1. Resolve scientific name -> GBIF usageKey via the species/match endpoint.
  2. Page through occurrence/search for records that carry StillImage media.
  3. Download the media URLs.

GBIF media licensing varies per record; the API returns a `license` field
which we pass through to the manifest so downstream filtering is possible.

Examples
--------
  python scrape_gbif.py --dry-run
  python scrape_gbif.py --min-images 50 --images-per-species 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    StorageClient, get_db_connection, insert_image, load_species_list,
    md5_of_bytes, upsert_species,
)

MATCH_URL = "https://api.gbif.org/v1/species/match"
OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"


def resolve_usage_key(name: str) -> Optional[int]:
    import requests  # lazy
    try:
        r = requests.get(MATCH_URL, params={"name": name, "rank": "SPECIES"},
                         timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("matchType") not in (None, "NONE"):
            return data.get("usageKey")
    except Exception as exc:  # noqa: BLE001
        print(f"    ! GBIF match failed for {name}: {exc}", file=sys.stderr)
    return None


def fetch_media_urls(usage_key: int, limit: int) -> list[tuple[str, str]]:
    """Return [(image_url, license)] for occurrences carrying StillImage media."""
    import requests  # lazy

    out: list[tuple[str, str]] = []
    offset, page = 0, 100
    while len(out) < limit:
        params = {"taxonKey": usage_key, "mediaType": "StillImage",
                  "limit": page, "offset": offset}
        try:
            r = requests.get(OCCURRENCE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            print(f"    ! GBIF occurrence query failed: {exc}", file=sys.stderr)
            break
        for rec in data.get("results", []):
            for media in rec.get("media", []):
                url = media.get("identifier")
                if url and (media.get("type") == "StillImage" or media.get("format", "").startswith("image")):
                    out.append((url, media.get("license", "")))
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
        if data.get("endOfRecords") or not data.get("results"):
            break
        offset += page
    return out[:limit]


def download(url: str) -> Optional[bytes]:
    import requests  # lazy
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as exc:  # noqa: BLE001
        print(f"    ! download failed {url}: {exc}", file=sys.stderr)
        return None


def run(args: argparse.Namespace) -> int:
    targets = load_species_list(Path(args.species_file))[: args.species_limit]
    out_dir = Path(args.out)

    if args.dry_run:
        print(f"DRY RUN — GBIF gap-fill, {len(targets)} species "
              f"(target >= {args.min_images} images each):")
        for t in targets:
            print(f"  match '{t.scientific_name}' -> usageKey "
                  f"-> {OCCURRENCE_URL}?taxonKey=..&mediaType=StillImage")
        return 0

    conn = get_db_connection()
    storage = StorageClient(args.bucket) if args.bucket else None
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_idx, t in enumerate(targets):
        key = resolve_usage_key(t.scientific_name)
        if key is None:
            print(f"[{t.scientific_name}] no GBIF key; skipping")
            continue
        media = fetch_media_urls(key, max(args.images_per_species, args.min_images))
        print(f"[{t.scientific_name}] usageKey={key}, {len(media)} media records")
        species_id = upsert_species(conn, t, None, class_idx) if conn else None
        species_dir = out_dir / t.slug
        species_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        for i, (url, _license) in enumerate(media):
            data = download(url)
            if not data:
                continue
            digest = md5_of_bytes(data)
            if digest in seen:
                continue
            seen.add(digest)
            fname = f"gbif_{i:03d}.jpg"
            (species_dir / fname).write_bytes(data)
            storage_path = f"{t.slug}/{fname}"
            if storage is not None:
                storage_path = storage.upload_bytes(f"{t.slug}/{fname}", data)
            if conn is not None and species_id is not None:
                insert_image(conn, species_id, "gbif", storage_path,
                             "train", None, None)
            total += 1
            time.sleep(args.delay)

    if conn is not None:
        conn.close()
    print(f"Done. {total} GBIF images across {len(targets)} species.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species-file",
                   default=str(Path(__file__).parent / "species_list.txt"))
    p.add_argument("--species-limit", type=int, default=30)
    p.add_argument("--images-per-species", type=int, default=60)
    p.add_argument("--min-images", type=int, default=50,
                   help="Target floor of images for gap-filled species")
    p.add_argument("--out", default="./data/raw/gbif")
    p.add_argument("--bucket", default=None)
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
