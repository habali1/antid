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

--restore mode (the one you almost always want) does not select anything --
it reads the already-committed benchmark_v1.csv, re-fetches those EXACT
observation_uuid/photo_id records from the iNat API, and verifies every
downloaded file's sha256 against the value already recorded in the CSV. It
never writes or modifies benchmark_v1.csv; the CSV is the frozen ground truth
and this mode only ever reproduces the image files it describes. Use this to
repopulate data/benchmark_v1/{slug}/*.jpg on a fresh clone or after deleting
the local images -- NOT the default (candidate-selecting) mode above, which
would pick a different, newer set of observations if run today.

Usage:
    # Build a NEW benchmark version (selects fresh candidates -- do not use
    # this to reproduce the existing frozen benchmark_v1.csv):
    python scrape_benchmark.py --taxonomy ../training/artifacts/taxonomy.json \
        --manifest ../data/manifest_all.csv --out ../data/benchmark_v1 \
        --per-species 35

    # Restore the exact images the committed benchmark_v1.csv describes:
    python scrape_benchmark.py --restore --out ../data/benchmark_v1
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


# --------------------------------------------------------------------- restore
def fetch_observations_by_uuid(client: httpx.Client, uuids: list[str],
                                batch_size: int = 100) -> dict[str, dict]:
    """observation_uuid -> observation object, for exactly the given UUIDs.

    Batched (comma-separated `uuid=` accepts many at once) rather than one
    request per row -- ~16 requests for a 1591-row benchmark instead of 1591.
    """
    found: dict[str, dict] = {}
    for start in range(0, len(uuids), batch_size):
        batch = uuids[start:start + batch_size]
        r = get_with_retry(client, f"{API}/observations", params={
            "uuid": ",".join(batch), "per_page": batch_size})
        for obs in r.json().get("results", []):
            u = obs.get("uuid")
            if u:
                found[u] = obs
        time.sleep(1.0)
    return found


def restore_from_csv(client: httpx.Client, csv_path: Path, out_dir: Path):
    """Re-download exactly the images benchmark_v1.csv already describes.

    Never selects new candidates and never touches the CSV -- it is read-only
    ground truth here. Returns (n_ok, errors) where errors is a list of
    human-readable strings; the caller decides whether that's fatal.
    """
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    log(f"[restore] {len(rows)} rows in {csv_path}")

    uuids = sorted({r["observation_uuid"] for r in rows if r.get("observation_uuid")})
    obs_by_uuid = fetch_observations_by_uuid(client, uuids)
    log(f"[restore] resolved {len(obs_by_uuid)}/{len(uuids)} observations from the API")

    errors: list[str] = []
    n_ok = 0
    for r in rows:
        slug, pid, want_hash = r["slug"], r["photo_id"], r["sha256"]
        sdir = out_dir / slug
        sdir.mkdir(parents=True, exist_ok=True)

        existing = list(sdir.glob(f"{pid}.*"))
        if len(existing) == 1:
            digest = hashlib.sha256(existing[0].read_bytes()).hexdigest()
            if digest == want_hash:
                n_ok += 1
                continue
            log(f"    ! {slug}/{pid}: local file hash mismatch, re-downloading")
            existing[0].unlink()
        elif len(existing) > 1:
            errors.append(f"{slug}/{pid}: {len(existing)} ambiguous local files, "
                          f"expected exactly one -- resolve manually")
            continue

        obs = obs_by_uuid.get(r["observation_uuid"])
        if obs is None:
            errors.append(f"{slug}/{pid}: observation_uuid {r['observation_uuid']} "
                          f"not found via the API (deleted, hidden, or moved)")
            continue
        photo = next((p for p in (obs.get("photos") or [])
                     if str(p.get("id")) == pid), None)
        if photo is None or not photo.get("url"):
            errors.append(f"{slug}/{pid}: observation found but no matching "
                          f"photo_id in its current photos[] -- photo may have "
                          f"been removed from the observation")
            continue

        url = photo["url"].replace("square", "medium")
        ext = _ext_from_url(url)
        res = download_one(client, pid, url, ext, sdir)
        if res is None:
            errors.append(f"{slug}/{pid}: download failed after retries")
            continue
        fname, digest = res
        if digest != want_hash:
            errors.append(f"{slug}/{pid}: downloaded but sha256 mismatch "
                          f"(got {digest[:12]}..., expected {want_hash[:12]}...) "
                          f"-- the photo behind this URL has changed since the "
                          f"benchmark was frozen")
            (sdir / fname).unlink(missing_ok=True)
            continue
        n_ok += 1

    return n_ok, errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore", action="store_true",
                    help="Re-download the exact images the committed benchmark_v1.csv "
                         "already describes, verifying sha256 against it. Read-only "
                         "with respect to the CSV -- never selects new candidates and "
                         "never writes it. Only --out and --restore-csv apply.")
    ap.add_argument("--restore-csv", type=Path, default=None,
                    help="CSV to restore from (--restore only; default: {out}/benchmark_v1.csv)")
    ap.add_argument("--taxonomy", type=Path,
                    help="taxonomy.json to source species/taxon_id from (fresh-scrape mode only)")
    ap.add_argument("--manifest", type=Path,
                    help="Training manifest CSV, for the photo_id exclusion set (fresh-scrape mode only)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-species", type=int, default=35)
    ap.add_argument("--cutoff-date", default=DEFAULT_CUTOFF,
                    help="Only observations CREATED on/after this date are eligible "
                         "(YYYY-MM-DD). Must be after the training scrape completed.")
    args = ap.parse_args()

    if args.restore:
        csv_path = args.restore_csv or (args.out / "benchmark_v1.csv")
        if not csv_path.exists():
            raise SystemExit(f"--restore: {csv_path} does not exist")
        args.out.mkdir(parents=True, exist_ok=True)
        with httpx.Client(headers=HEADERS, timeout=90, follow_redirects=True) as client:
            n_ok, errors = restore_from_csv(client, csv_path, args.out)
        log(f"[restore] {n_ok} verified OK, {len(errors)} error(s)")
        for e in errors[:50]:
            log(f"  ! {e}")
        if len(errors) > 50:
            log(f"  ... and {len(errors) - 50} more")
        if errors:
            raise SystemExit(
                f"[restore] FAILED: {len(errors)} row(s) could not be verified. "
                f"benchmark_v1.csv was NOT modified. eval_benchmark.py will refuse "
                f"to run against an incomplete local copy -- fix these before evaluating."
            )
        return

    if not args.taxonomy or not args.manifest:
        raise SystemExit("--taxonomy and --manifest are required unless --restore is set")

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
