#!/usr/bin/env python3
"""scrape_antweb.py — secondary source: AntWeb (museum-quality specimen images).

AntWeb (California Academy of Sciences) exposes a v2 JSON API at
https://www.antweb.org/api/v2/. Specimen images come in controlled angles:
  h = head (full-face), p = profile (lateral), d = dorsal.
Those controlled shots are exactly what we want for a clean training set.

NOTE: AntWeb's public API response shape has shifted over the years; this
client is defensive about missing keys and skips records it can't parse.

Examples
--------
  python scrape_antweb.py --dry-run
  python scrape_antweb.py --species-file species_list.txt --images-per-species 40
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

API_BASE = "https://www.antweb.org/api/v2"
SHOT_LABELS = {"h": "head", "p": "profile", "d": "dorsal"}


def fetch_specimen_images(genus: str, species: str, shot_types: list[str],
                          limit: int) -> list[str]:
    """Return image URLs for a genus/species, preferring the requested angles."""
    import requests  # lazy

    url = f"{API_BASE}/images"
    params = {"genus": genus, "species": species, "limit": max(limit * 3, 30)}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! AntWeb request failed for {genus} {species}: {exc}",
              file=sys.stderr)
        return []

    # The v2 payload nests specimen image records; structures vary, so we
    # walk defensively and collect any (shot_type, url) pairs we can find.
    candidates: list[tuple[str, str]] = []
    records = payload.get("specimenImages") or payload.get("images") or []
    if isinstance(records, dict):
        records = list(records.values())
    for rec in records:
        if not isinstance(rec, dict):
            continue
        shot = (rec.get("shotType") or rec.get("shot_type") or "").lower()
        img = (rec.get("image") or rec.get("url") or rec.get("imageURL")
               or rec.get("originalThumbnailUrl") or "")
        if isinstance(img, dict):
            img = img.get("original") or img.get("high") or next(iter(img.values()), "")
        if img:
            candidates.append((shot, img))

    # Prefer requested shot types, then fall back to anything else.
    preferred = [u for s, u in candidates if s in shot_types]
    other = [u for s, u in candidates if s not in shot_types]
    ordered = preferred + other
    # De-dup while preserving order.
    seen, out = set(), []
    for u in ordered:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= limit:
            break
    return out


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
    shot_types = [s.strip().lower() for s in args.shot_types.split(",") if s.strip()]
    out_dir = Path(args.out)

    if args.dry_run:
        print(f"DRY RUN — AntWeb, {len(targets)} species, "
              f"shot types {shot_types} ({', '.join(SHOT_LABELS[s] for s in shot_types if s in SHOT_LABELS)}):")
        for t in targets:
            print(f"  GET {API_BASE}/images?genus={t.genus}&species={t.species_epithet}")
        return 0

    conn = get_db_connection()
    storage = StorageClient(args.bucket) if args.bucket else None
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_idx, t in enumerate(targets):
        urls = fetch_specimen_images(t.genus, t.species_epithet, shot_types,
                                     args.images_per_species)
        print(f"[{t.scientific_name}] {len(urls)} AntWeb images")
        species_id = upsert_species(conn, t, None, class_idx) if conn else None
        species_dir = out_dir / t.slug
        species_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        for i, url in enumerate(urls):
            data = download(url)
            if not data:
                continue
            digest = md5_of_bytes(data)
            if digest in seen:
                continue
            seen.add(digest)
            fname = f"antweb_{i:03d}.jpg"
            (species_dir / fname).write_bytes(data)
            storage_path = f"{t.slug}/{fname}"
            if storage is not None:
                storage_path = storage.upload_bytes(f"{t.slug}/{fname}", data)
            if conn is not None and species_id is not None:
                insert_image(conn, species_id, "antweb", storage_path,
                             "train", None, None)
            total += 1
            time.sleep(args.delay)  # be polite to the API

    if conn is not None:
        conn.close()
    print(f"Done. {total} AntWeb images across {len(targets)} species.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species-file",
                   default=str(Path(__file__).parent / "species_list.txt"))
    p.add_argument("--species-limit", type=int, default=30)
    p.add_argument("--images-per-species", type=int, default=40)
    p.add_argument("--shot-types", default="h,p,d",
                   help="Preferred angles, comma-separated (h=head,p=profile,d=dorsal)")
    p.add_argument("--out", default="./data/raw/antweb")
    p.add_argument("--bucket", default=None)
    p.add_argument("--delay", type=float, default=0.25, help="Seconds between downloads")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
