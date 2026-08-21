#!/usr/bin/env python3
"""scrape_calibration.py — build calibration_v1, a frozen set for studying
unknown/out-of-scope rejection. Separate from benchmark_v1 and never used to
tune anything else -- this exists to look at similarity-score distributions,
not to report a headline accuracy number.

Five categories, one CSV, distinguished by the `category` column:

  known_holdout       the 50 trained species, good quality, held out from
                       BOTH training and benchmark_v1 -- baseline for false
                       rejection rate (a good photo of a known species that
                       still gets rejected).
  out_of_scope_ant     Formicidae species NOT in the 50 (excluded by
                       taxon_id via without_taxon_id) -- the main case: an
                       ant the model was never trained to recognize.
  non_ant_insect       non-Formicidae Insecta -- "pointed the app at the
                       wrong bug" case.
  unrelated            non-insect photos (Aves / Plantae / Mammalia) --
                       sanity-check case, should be the easiest to reject.
  low_quality_known    photos of the 50 species sourced from quality_grade
                       "needs_id" (not "research"), then filtered down to the
                       blurriest tier by a measured Laplacian-variance
                       sharpness score (see blur_score()) -- tests whether
                       "bad photo of a known species" is distinguishable from
                       "genuinely unknown species" using similarity alone.

Same rigor as benchmark_v1: every kept row records photo_id, observation_uuid,
sha256, and created_at; every candidate is checked against the training
manifest's photo_id set AND benchmark_v1's photo_id/observation_uuid sets
before being kept; the 2026-06-16 date cutoff (training's own scrape completed
2026-06-14) gives the same structural training-disjointness guarantee
benchmark_v1 uses. One photo per observation throughout.

Like scrape_benchmark.py, --restore re-fetches exactly what the frozen CSV
already lists (verifying sha256) and never selects new candidates or rewrites
the CSV; the default mode selects and would build a different version.

Usage:
    python scrape_calibration.py --taxonomy ../training/artifacts/taxonomy.json \
        --training-manifest ../data/manifest_all.csv \
        --benchmark-csv ../data/benchmark_v1/benchmark_v1.csv \
        --out ../data/calibration_v1

    python scrape_calibration.py --restore --out ../data/calibration_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import slugify  # noqa: E402

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "AntID-pipeline/1.0 (educational project)"}
IMG_EXTS = ("jpg", "jpeg", "png", "webp")
RETRY_EXC = (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout,
             httpx.ConnectError, httpx.ReadError, httpx.PoolTimeout, httpx.WriteError)
RETRY_STATUS = {429, 500, 502, 503, 504}
DEFAULT_CUTOFF = "2026-06-16"  # same cutoff benchmark_v1 uses -- one day past
                                # training's wave 2 completion (2026-06-14 16:49 -04:00)

TAXON_INSECTA = 47158
TAXON_FORMICIDAE = 47336
UNRELATED_TAXA = {"aves": 3, "plantae": 47126, "mammalia": 40151}

CSV_FIELDS = ["photo_id", "observation_uuid", "species", "slug", "taxon_id",
             "category", "lat", "lon", "source", "sha256", "created_at",
             "blur_score"]


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


def blur_score(path: Path) -> float | None:
    """Variance of the discrete Laplacian -- lower means blurrier.

    Standard, well-known no-reference sharpness metric. Images are capped to
    512px on the long side before scoring so resolution differences between
    sources don't confound the score; this is a normalization choice, not a
    claim that it perfectly measures subjective "usability".
    """
    try:
        img = Image.open(path).convert("L")
    except Exception:  # noqa: BLE001
        return None
    img.thumbnail((512, 512))
    arr = np.asarray(img, dtype=np.float64)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return None
    lap = (-4 * arr[1:-1, 1:-1]
          + arr[:-2, 1:-1] + arr[2:, 1:-1]
          + arr[1:-1, :-2] + arr[1:-1, 2:])
    return float(lap.var())


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


# ------------------------------------------------------------- candidate fetch
def _parse_loc(obs: dict) -> tuple[float | None, float | None]:
    loc = obs.get("location")
    if loc and "," in loc:
        try:
            a, b = loc.split(",")
            return float(a), float(b)
        except ValueError:
            pass
    return None, None


def fetch_known_species_candidates(client: httpx.Client, taxon_id: int, species_name: str,
                                   slug: str, target: int, cutoff_date: str,
                                   exclude_photo_ids: set[str], exclude_obs_uuids: set[str],
                                   quality_grade: str = "research", max_pages: int = 10):
    """One of the 50 known species -- used for known_holdout and (with
    quality_grade='needs_id') the low_quality_known candidate pool."""
    cands, page = [], 1
    while len(cands) < target and page <= max_pages:
        r = get_with_retry(client, f"{API}/observations", params={
            "taxon_id": taxon_id, "quality_grade": quality_grade, "photos": "true",
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
            ouuid = obs.get("uuid")
            if pid in exclude_photo_ids or ouuid in exclude_obs_uuids:
                continue
            lat, lon = _parse_loc(obs)
            cands.append({
                "photo_id": pid, "url": url.replace("square", "medium"),
                "ext": _ext_from_url(url), "lat": lat, "lon": lon,
                "observation_uuid": ouuid, "created_at": obs.get("created_at"),
                "species": species_name, "slug": slug, "taxon_id": taxon_id,
            })
            if len(cands) >= target:
                break
        page += 1
        time.sleep(1.0)
    return cands


def fetch_diverse_candidates(client: httpx.Client, taxon_id: int, target: int,
                             cutoff_date: str, exclude_photo_ids: set[str],
                             exclude_obs_uuids: set[str], without_taxon_ids: str | None = None,
                             max_per_species: int = 15, max_pages: int = 40):
    """Candidates under taxon_id, capped per distinct observed species so one
    common species can't dominate the sample. Species-rank identifications only."""
    cands, per_species = [], {}
    page = 1
    while len(cands) < target and page <= max_pages:
        params = {
            "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
            "per_page": 200, "page": page, "order_by": "created_at", "order": "asc",
            "created_d1": cutoff_date,
        }
        if without_taxon_ids:
            params["without_taxon_id"] = without_taxon_ids
        r = get_with_retry(client, f"{API}/observations", params=params)
        results = r.json().get("results", [])
        if not results:
            break
        for obs in results:
            taxon = obs.get("taxon") or {}
            if taxon.get("rank") != "species":
                continue
            sp_name = taxon.get("name")
            if not sp_name:
                continue
            slug = slugify(sp_name)
            if per_species.get(slug, 0) >= max_per_species:
                continue
            photos = obs.get("photos") or []
            if not photos:
                continue
            p = photos[0]
            url, pid = p.get("url"), p.get("id")
            if not url or pid is None:
                continue
            pid = str(pid)
            ouuid = obs.get("uuid")
            if pid in exclude_photo_ids or ouuid in exclude_obs_uuids:
                continue
            lat, lon = _parse_loc(obs)
            cands.append({
                "photo_id": pid, "url": url.replace("square", "medium"),
                "ext": _ext_from_url(url), "lat": lat, "lon": lon,
                "observation_uuid": ouuid, "created_at": obs.get("created_at"),
                "species": sp_name, "slug": slug, "taxon_id": taxon.get("id"),
            })
            per_species[slug] = per_species.get(slug, 0) + 1
            if len(cands) >= target:
                break
        page += 1
        time.sleep(1.0)
    return cands, len(per_species)


def download_candidates(client, cands, out_dir: Path, category: str, source: str):
    rows = []
    for c in cands:
        sdir = out_dir / c["slug"]
        sdir.mkdir(parents=True, exist_ok=True)
        res = download_one(client, c["photo_id"], c["url"], c["ext"], sdir)
        if res is None:
            continue
        fname, digest = res
        rows.append({
            "photo_id": c["photo_id"], "observation_uuid": c["observation_uuid"],
            "species": c["species"], "slug": c["slug"], "taxon_id": c["taxon_id"],
            "category": category, "lat": c["lat"], "lon": c["lon"], "source": source,
            "sha256": digest, "created_at": c["created_at"],
            "blur_score": blur_score(sdir / fname),
        })
    return rows


# --------------------------------------------------------------------- restore
def fetch_observations_by_uuid(client: httpx.Client, uuids: list[str], batch_size: int = 100):
    found = {}
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
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    log(f"[restore] {len(rows)} rows in {csv_path}")
    uuids = sorted({r["observation_uuid"] for r in rows if r.get("observation_uuid")})
    obs_by_uuid = fetch_observations_by_uuid(client, uuids)
    log(f"[restore] resolved {len(obs_by_uuid)}/{len(uuids)} observations from the API")

    errors, n_ok = [], 0
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
            existing[0].unlink()
        elif len(existing) > 1:
            errors.append(f"{slug}/{pid}: {len(existing)} ambiguous local files")
            continue

        obs = obs_by_uuid.get(r["observation_uuid"])
        if obs is None:
            errors.append(f"{slug}/{pid}: observation_uuid not found via the API")
            continue
        photo = next((p for p in (obs.get("photos") or [])
                     if str(p.get("id")) == pid), None)
        if photo is None or not photo.get("url"):
            errors.append(f"{slug}/{pid}: no matching photo_id in current photos[]")
            continue
        url = photo["url"].replace("square", "medium")
        ext = _ext_from_url(url)
        res = download_one(client, pid, url, ext, sdir)
        if res is None:
            errors.append(f"{slug}/{pid}: download failed after retries")
            continue
        fname, digest = res
        if digest != want_hash:
            errors.append(f"{slug}/{pid}: sha256 mismatch after download")
            (sdir / fname).unlink(missing_ok=True)
            continue
        n_ok += 1
    return n_ok, errors


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--restore-csv", type=Path, default=None)
    ap.add_argument("--taxonomy", type=Path)
    ap.add_argument("--training-manifest", type=Path)
    ap.add_argument("--benchmark-csv", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cutoff-date", default=DEFAULT_CUTOFF)
    ap.add_argument("--n-known-holdout", type=int, default=300)
    ap.add_argument("--n-out-of-scope-ant", type=int, default=300)
    ap.add_argument("--n-non-ant-insect", type=int, default=150)
    ap.add_argument("--n-unrelated", type=int, default=150)
    ap.add_argument("--n-low-quality-pool-per-species", type=int, default=10)
    ap.add_argument("--n-low-quality-final", type=int, default=150)
    args = ap.parse_args()

    if args.restore:
        csv_path = args.restore_csv or (args.out / "calibration_v1.csv")
        if not csv_path.exists():
            raise SystemExit(f"--restore: {csv_path} does not exist")
        args.out.mkdir(parents=True, exist_ok=True)
        with httpx.Client(headers=HEADERS, timeout=90, follow_redirects=True) as client:
            n_ok, errors = restore_from_csv(client, csv_path, args.out)
        log(f"[restore] {n_ok} verified OK, {len(errors)} error(s)")
        for e in errors[:50]:
            log(f"  ! {e}")
        if errors:
            raise SystemExit(f"[restore] FAILED: {len(errors)} row(s) could not be verified.")
        return

    if not args.taxonomy or not args.training_manifest or not args.benchmark_csv:
        raise SystemExit("--taxonomy, --training-manifest, --benchmark-csv are required "
                         "unless --restore is set")

    taxonomy = json.loads(args.taxonomy.read_text())
    known_species = sorted(({"slug": v["slug"], "species": v["species_name"],
                             "taxon_id": v["taxon_id"]} for v in taxonomy.values()),
                           key=lambda s: s["slug"])
    known_taxon_ids = ",".join(str(s["taxon_id"]) for s in known_species)

    train_photo_ids = {r["photo_id"] for r in csv.DictReader(
        args.training_manifest.open(newline="", encoding="utf-8"))}
    bench_rows = list(csv.DictReader(args.benchmark_csv.open(newline="", encoding="utf-8")))
    bench_photo_ids = {r["photo_id"] for r in bench_rows}
    bench_obs_uuids = {r["observation_uuid"] for r in bench_rows if r.get("observation_uuid")}
    exclude_photo_ids = train_photo_ids | bench_photo_ids
    exclude_obs_uuids = set(bench_obs_uuids)  # training has no recorded obs_uuid to add here
    log(f"[calib] excluding {len(train_photo_ids)} training + {len(bench_photo_ids)} "
       f"benchmark_v1 photo_ids ({len(exclude_photo_ids)} unique), "
       f"{len(exclude_obs_uuids)} benchmark_v1 observation_uuids")

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    with httpx.Client(headers=HEADERS, timeout=90, follow_redirects=True) as client:
        # ---- known_holdout ----
        per_sp = max(1, -(-args.n_known_holdout // len(known_species)))
        log(f"\n[calib] known_holdout: target {args.n_known_holdout} (~{per_sp}/species)")
        cands = []
        for sp in known_species:
            c = fetch_known_species_candidates(
                client, sp["taxon_id"], sp["species"], sp["slug"], per_sp,
                args.cutoff_date, exclude_photo_ids, exclude_obs_uuids)
            cands.extend(c)
        cands = cands[:args.n_known_holdout]
        rows = download_candidates(client, cands, args.out, "known_holdout", "inat_api_calibration")
        log(f"[calib] known_holdout: saved {len(rows)}/{len(cands)}")
        all_rows.extend(rows)

        # ---- out_of_scope_ant ----
        log(f"\n[calib] out_of_scope_ant: target {args.n_out_of_scope_ant}")
        cands, n_sp = fetch_diverse_candidates(
            client, TAXON_FORMICIDAE, args.n_out_of_scope_ant, args.cutoff_date,
            exclude_photo_ids, exclude_obs_uuids, without_taxon_ids=known_taxon_ids)
        rows = download_candidates(client, cands, args.out, "out_of_scope_ant", "inat_api_calibration")
        log(f"[calib] out_of_scope_ant: saved {len(rows)}/{len(cands)} across {n_sp} species")
        all_rows.extend(rows)

        # ---- non_ant_insect ----
        log(f"\n[calib] non_ant_insect: target {args.n_non_ant_insect}")
        cands, n_sp = fetch_diverse_candidates(
            client, TAXON_INSECTA, args.n_non_ant_insect, args.cutoff_date,
            exclude_photo_ids, exclude_obs_uuids, without_taxon_ids=str(TAXON_FORMICIDAE))
        rows = download_candidates(client, cands, args.out, "non_ant_insect", "inat_api_calibration")
        log(f"[calib] non_ant_insect: saved {len(rows)}/{len(cands)} across {n_sp} species")
        all_rows.extend(rows)

        # ---- unrelated (split across Aves/Plantae/Mammalia) ----
        log(f"\n[calib] unrelated: target {args.n_unrelated} across {list(UNRELATED_TAXA)}")
        per_taxon = -(-args.n_unrelated // len(UNRELATED_TAXA))
        for name, tid in UNRELATED_TAXA.items():
            cands, n_sp = fetch_diverse_candidates(
                client, tid, per_taxon, args.cutoff_date, exclude_photo_ids, exclude_obs_uuids)
            rows = download_candidates(client, cands, args.out, "unrelated", "inat_api_calibration")
            log(f"[calib] unrelated/{name}: saved {len(rows)}/{len(cands)} across {n_sp} species")
            all_rows.extend(rows)

        # ---- low_quality_known ----
        pool_target = args.n_low_quality_pool_per_species
        log(f"\n[calib] low_quality_known: pooling ~{pool_target}/species (needs_id grade), "
           f"keeping blurriest {args.n_low_quality_final}")
        pool_cands = []
        for sp in known_species:
            c = fetch_known_species_candidates(
                client, sp["taxon_id"], sp["species"], sp["slug"], pool_target,
                args.cutoff_date, exclude_photo_ids, exclude_obs_uuids,
                quality_grade="needs_id")
            pool_cands.extend(c)
        pool_rows = download_candidates(client, pool_cands, args.out, "low_quality_known",
                                        "inat_api_calibration")
        scored = [r for r in pool_rows if r["blur_score"] is not None]
        scored.sort(key=lambda r: r["blur_score"])  # ascending: blurriest first
        kept = scored[:args.n_low_quality_final]
        discarded = scored[args.n_low_quality_final:] + [r for r in pool_rows if r["blur_score"] is None]
        log(f"[calib] low_quality_known: pooled {len(pool_rows)}, kept blurriest "
           f"{len(kept)}, discarding {len(discarded)}")
        for r in discarded:
            for ext in IMG_EXTS:
                cand = args.out / r["slug"] / f"{r['photo_id']}.{ext}"
                if cand.exists():
                    cand.unlink()
        all_rows.extend(kept)

    csv_path = args.out / "calibration_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    log(f"\n[calib] wrote {csv_path} ({len(all_rows)} rows)")

    import collections
    cat_counts = collections.Counter(r["category"] for r in all_rows)
    for cat, n in cat_counts.items():
        log(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
