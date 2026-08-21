#!/usr/bin/env python3
"""scrape_benchmark.py — build a frozen, provably-unseen evaluation set.

The training set (manifest_all.csv) never recorded observation_uuid, and its
own val split was invalidated when the manifest was regenerated after training
(see training/artifacts/README.md). This script sidesteps both problems by
drawing an entirely fresh sample rather than trying to reconstruct the old
split: for each of the 50 trained species, it pulls iNaturalist observations
CREATED AFTER the training scrape finished (--cutoff-date), which is a
structural guarantee, not a best-effort filter -- an observation that did not
exist yet when training scraped the site cannot possibly be one of the photos
training saw. Training's own scrape completed 2026-06-14 16:49 -04:00 (wave 2);
the default cutoff of 2026-06-16 leaves a full day of margin.

On top of that structural guarantee, every candidate is also checked against
the explicit set of training photo_ids (data/manifest_all.csv) before being
kept, and every candidate's parent observation_uuid is recorded so a
downstream verification step can confirm zero overlap directly rather than
taking the date argument on faith.

One photo per observation (photos[0], matching scrape_inat_api.py's
convention) -- this keeps the benchmark itself free of the same
same-observation-in-both-splits risk that motivated this script.

Output: {out}/{slug}/{photo_id}.{ext}  and  {out}/benchmark_v1.csv with columns
    photo_id, observation_uuid, species, slug, taxon_id, lat, lon, source, sha256

Usage:
    python scrape_benchmark.py --taxonomy ../training/artifacts/taxonomy.json \
        --manifest ../data/manifest_all.csv --out ../data/benchmark_v1 \
        --per-species 35
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import httpx

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "AntID-pipeline/1.0 (educational project)"}
IMG_EXTS = ("jpg", "jpeg", "png", "webp")
RETRY_EXC = (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout,
             httpx.ConnectError, httpx.ReadError, httpx.PoolTimeout, httpx.WriteError)
RETRY_STATUS = {429, 500, 502, 503, 504}
DEFAULT_CUTOFF = "2026-06-16"  # day after wave 2 (training scrape) completed


def log(msg: str) -> None:
    print(msg, flush=True)


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


def _ext_from_url(url: str) -> str:
    tail = url.split("/")[-1].split("?")[0]
    if "." in tail:
        e = tail.rsplit(".", 1)[1].lower()
        if e in IMG_EXTS:
            return e
    return "jpg"


def fetch_fresh_candidates(client: httpx.Client, taxon_id: int, target: int,
                           cutoff_date: str, exclude_photo_ids: set[str],
                           max_pages: int = 10):
    """[(photo_id, url, ext, lat, lon, observation_uuid, created_at)], one per obs."""
    cands, page = [], 1
    while len(cands) < target and page <= max_pages:
        r = get_with_retry(client, f"{API}/observations", params={
            "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
            "per_page": 200, "page": page, "order_by": "created_at", "order": "asc",
            "created_d1": cutoff_date})
        results = r.json().get("results", [])
        if not results:
            break
        for obs in results:
            photos = obs.get("photos") or []
            if not photos:
                continue
            p = photos[0]
            url, pid = p.get("url"), p.get("id")
            if not url or pid is None:
                continue
            pid = str(pid)
            if pid in exclude_photo_ids:
                log(f"    ! skipping photo_id {pid}: present in training manifest "
                    f"despite date filter (unexpected -- investigate)")
                continue
            lat = lon = None
            loc = obs.get("location")
            if loc and "," in loc:
                try:
                    a, b = loc.split(",")
                    lat, lon = float(a), float(b)
                except ValueError:
                    pass
            cands.append((pid, url.replace("square", "medium"), _ext_from_url(url),
                          lat, lon, obs.get("uuid"), obs.get("created_at")))
            if len(cands) >= target:
                break
        page += 1
        time.sleep(1.0)
    return cands[:target], page - 1


def download_one(client, photo_id, url, ext, dest_dir: Path, tries=3):
    for i in range(tries):
        try:
            r = client.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                fname = f"{photo_id}.{ext}"
                (dest_dir / fname).write_bytes(r.content)
                return fname, hashlib.sha256(r.content).hexdigest()
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
    ap.add_argument("--taxonomy", required=True, type=Path,
                    help="taxonomy.json to source species/taxon_id from")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="Training manifest CSV, for the photo_id exclusion set")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-species", type=int, default=35)
    ap.add_argument("--cutoff-date", default=DEFAULT_CUTOFF,
                    help="Only observations CREATED on/after this date are eligible "
                         "(YYYY-MM-DD). Must be after the training scrape completed.")
    args = ap.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text())
    species = sorted(({"slug": v["slug"], "species": v["species_name"],
                       "taxon_id": v["taxon_id"]} for v in taxonomy.values()),
                     key=lambda s: s["slug"])

    exclude_photo_ids = {r["photo_id"] for r in csv.DictReader(
        args.manifest.open(newline="", encoding="utf-8"))}
    log(f"[bench] {len(species)} species, excluding {len(exclude_photo_ids)} known "
        f"training photo_ids, cutoff={args.cutoff_date}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    shortfalls: list[tuple[str, int, int]] = []

    with httpx.Client(headers=HEADERS, timeout=90, follow_redirects=True) as client:
        for sp in species:
            sdir = args.out / sp["slug"]
            sdir.mkdir(parents=True, exist_ok=True)
            try:
                cands, pages_used = fetch_fresh_candidates(
                    client, sp["taxon_id"], args.per_species, args.cutoff_date,
                    exclude_photo_ids)
            except Exception as e:  # noqa: BLE001
                log(f"[{sp['species']}] fetch failed: {e!r}; skipping")
                continue
            log(f"[{sp['species']}] {len(cands)} fresh candidates "
                f"({pages_used} page(s) scanned)")
            if len(cands) < args.per_species:
                shortfalls.append((sp["slug"], len(cands), args.per_species))

            saved = 0
            for pid, url, ext, lat, lon, obs_uuid, created_at in cands:
                res = download_one(client, pid, url, ext, sdir)
                if res is None:
                    continue
                fname, digest = res
                rows.append({
                    "photo_id": pid, "observation_uuid": obs_uuid,
                    "species": sp["species"], "slug": sp["slug"],
                    "taxon_id": sp["taxon_id"], "lat": lat, "lon": lon,
                    "source": "inat_api_benchmark", "sha256": digest,
                    "created_at": created_at,
                })
                saved += 1
            log(f"[{sp['species']}] saved {saved}/{len(cands)}")

    csv_path = args.out / "benchmark_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "photo_id", "observation_uuid", "species", "slug", "taxon_id",
            "lat", "lon", "source", "sha256", "created_at"])
        w.writeheader()
        w.writerows(rows)
    log(f"[bench] wrote {csv_path} ({len(rows)} images)")

    if shortfalls:
        log(f"[bench] WARNING: {len(shortfalls)} species below target:")
        for slug, got, want in shortfalls:
            log(f"    {slug}: {got}/{want}")


if __name__ == "__main__":
    main()
