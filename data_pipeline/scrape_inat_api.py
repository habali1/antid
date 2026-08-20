#!/usr/bin/env python3
"""scrape_inat_api.py — pull training images via the iNaturalist API.

A faster alternative to scrape_inat.py for small species sets on slow links:
instead of downloading the multi-GB S3 open-data manifests, it queries the iNat
API for research-grade observations that have photos and downloads the *medium*
image of each. Captures lat/lon for the geo index. Retries every network call,
since slow links drop connections intermittently.

Output mirrors scrape_inat.py so clean.py / training run unchanged:
    {out}/{species_slug}/{photo_id}.{ext}
    {out}/manifest_inat.csv   (species, slug, taxon_id, photo_id, source, lat, lon, split)

Usage:
    python scrape_inat_api.py --species-file species_list.txt \
        --images-per-species 200 --out ../data/raw
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from common import load_species_list, md5_of_bytes  # noqa: E402

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "AntID-pipeline/1.0 (educational project)"}
IMG_EXTS = ("jpg", "jpeg", "png", "webp")
RETRY_EXC = (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout,
             httpx.ConnectError, httpx.ReadError, httpx.PoolTimeout, httpx.WriteError)
RETRY_STATUS = {429, 500, 502, 503, 504}


def log(msg: str) -> None:
    print(msg, flush=True)  # flush so background progress is visible


def get_with_retry(client: httpx.Client, url: str, params=None, tries=5, backoff=2.0):
    last = None
    for i in range(tries):
        try:
            r = client.get(url, params=params)
            if r.status_code in RETRY_STATUS:
                last = httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r
        except RETRY_EXC as e:
            last = e
            time.sleep(backoff * (i + 1))
    raise last if last else RuntimeError("get_with_retry exhausted")


def resolve_taxon_id(client: httpx.Client, name: str):
    r = get_with_retry(client, f"{API}/taxa", params={
        "q": name, "rank": "species", "is_active": "true", "per_page": 10})
    results = r.json().get("results", [])
    for res in results:
        if res.get("name", "").lower() == name.lower():
            return res["id"]
    return results[0]["id"] if results else None


def _ext_from_url(url: str) -> str:
    tail = url.split("/")[-1].split("?")[0]
    if "." in tail:
        e = tail.rsplit(".", 1)[1].lower()
        if e in IMG_EXTS:
            return e
    return "jpg"


def fetch_candidates(client: httpx.Client, taxon_id: int, target: int):
    """Return up to `target` [(photo_id, medium_url, ext, lat, lon)]."""
    cands, seen, page = [], set(), 1
    while len(cands) < target and page <= 4:
        r = get_with_retry(client, f"{API}/observations", params={
            "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
            "per_page": 200, "page": page, "order_by": "created_at", "order": "desc"})
        results = r.json().get("results", [])
        if not results:
            break
        for obs in results:
            lat = lon = None
            loc = obs.get("location")
            if loc and "," in loc:
                try:
                    a, b = loc.split(",")
                    lat, lon = float(a), float(b)
                except ValueError:
                    pass
            photos = obs.get("photos") or []
            if not photos:
                continue
            p = photos[0]
            url, pid = p.get("url"), p.get("id")
            if not url or pid in seen:
                continue
            seen.add(pid)
            cands.append((pid, url.replace("square", "medium"), _ext_from_url(url), lat, lon))
            if len(cands) >= target:
                break
        page += 1
        time.sleep(1.0)
    return cands[:target]


def download_one(client: httpx.Client, photo_id, url, ext, dest_dir: Path, tries=3):
    for i in range(tries):
        try:
            r = client.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                fname = f"{photo_id}.{ext}"
                (dest_dir / fname).write_bytes(r.content)
                return (fname, md5_of_bytes(r.content))
            if r.status_code in RETRY_STATUS:
                time.sleep(1.5 * (i + 1))
                continue
            return None
        except RETRY_EXC:
            time.sleep(1.5 * (i + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species-file", default=str(Path(__file__).parent / "species_list.txt"))
    ap.add_argument("--images-per-species", type=int, default=200)
    ap.add_argument("--out", default="./data/raw")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    species = load_species_list(Path(args.species_file))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = args.images_per_species

    manifest: list[dict] = []
    total = 0
    limits = httpx.Limits(max_connections=args.workers + 2, max_keepalive_connections=2)
    with httpx.Client(headers=HEADERS, timeout=90, follow_redirects=True, limits=limits) as client:
        log(f"[api] resolving {len(species)} taxon ids...")
        resolved = []
        for sp in species:
            try:
                tid = resolve_taxon_id(client, sp.scientific_name)
            except Exception as e:  # noqa: BLE001
                tid = None
                log(f"  ! resolve failed for {sp.scientific_name}: {e!r}")
            resolved.append((sp, tid))
            log(f"  {sp.scientific_name:34} -> taxon_id={tid}")
            time.sleep(0.4)

        for sp, tid in resolved:
            sdir = out_dir / sp.slug
            sdir.mkdir(parents=True, exist_ok=True)
            if tid is None:
                log(f"[{sp.scientific_name}] no taxon id, skipping")
                continue
            try:
                cands = fetch_candidates(client, tid, target)
            except Exception as e:  # noqa: BLE001
                log(f"[{sp.scientific_name}] fetch failed after retries: {e!r}; skipping")
                continue
            log(f"[{sp.scientific_name}] {len(cands)} candidates; downloading...")
            results = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(download_one, client, c[0], c[1], c[2], sdir): c
                        for c in cands}
                for fut in as_completed(futs):
                    res = fut.result()
                    if res:
                        results.append((futs[fut], res))

            seen_hash, saved = set(), 0
            for c, (fname, digest) in results:
                if digest in seen_hash:
                    (sdir / fname).unlink(missing_ok=True)
                    continue
                seen_hash.add(digest)
                manifest.append({
                    "species": sp.scientific_name, "slug": sp.slug, "taxon_id": tid,
                    "photo_id": c[0], "source": "inat_api", "lat": c[3], "lon": c[4],
                    "split": "val" if (total % 5 == 0) else "train",
                })
                saved += 1
                total += 1
            log(f"[{sp.scientific_name}] saved {saved} images")

    if manifest:
        mpath = out_dir / "manifest_inat.csv"
        with open(mpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)
        log(f"[api] wrote manifest: {mpath}")
    log(f"[api] DONE. {total} images across {len(species)} species.")


if __name__ == "__main__":
    main()
