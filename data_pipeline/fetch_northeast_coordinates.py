#!/usr/bin/env python3
"""fetch_northeast_coordinates.py — fetch public/obscured coordinates for the
observations already frozen in northeast_train_dev_v1.csv into a raw,
UNREVIEWED capture file. NEVER modifies northeast_train_dev_v1.csv, which is
frozen evidence.

The only path this script can ever write to:
    data/northeast_expansion_v1/northeast_coordinates_capture_v1.json

This script can NEVER write or overwrite
data/northeast_expansion_v1/northeast_coordinates_v1.json. Turning a raw
capture into that reviewed, frozen sidecar is the job of
finalize_northeast_coordinates.py, which runs entirely offline (no network
access) against a previously captured file like the one this script
produces. This split exists because an earlier version of this script wrote
directly to the frozen sidecar path before its design was reviewed.

Scope, strictly bounded:
  - Only the observation_uuid values already present in
    northeast_train_dev_v1.csv are queried -- no new observations are
    discovered or substituted, and the candidate pool is not expanded.
  - Only a public location is ever used. A source row whose geoprivacy is
    "private" is refused outright before any request is made (defense in
    depth -- none of the frozen rows are geoprivacy=private to begin with;
    this script hard-fails if that ever changes rather than silently
    skipping the check).
  - This dataset is fixed at exactly 3,600 rows. Success requires exactly
    3,600 unique source observation_uuid values and exactly 3,600 usable
    coordinates -- there is no percentage acceptance threshold. Partial
    coverage is reported and the capture is refused, never written short,
    regardless of --write.
  - Missing or unrecoverable coordinates are left absent -- never inferred or
    estimated -- which is exactly why partial coverage fails closed instead
    of being silently accepted.
  - Every fetched coordinate is validated: non-boolean, numeric, finite, and
    in range (latitude in [-90, 90], longitude in [-180, 180]) before it is
    ever stored.

Coordinate policy, corrected from an earlier draft: an obscured location is
NOT guaranteed to remain within the same 1-degree geo-index grid cell as the
true location -- iNaturalist's obscuring can displace a location across a
cell boundary. AntID's geo re-ranking checks the request cell plus its eight
neighbors, which can mitigate a one-cell displacement in the Northeast, but
this is not proof of identical placement or zero precision loss.

Paced identically to the existing scrape/restore scripts
(scrape_northeast_expansion.py): batches of 100 UUIDs per request against
iNaturalist's /observations endpoint, 1.05s between batches, same
User-Agent and retry/backoff behavior.

Usage:
    python fetch_northeast_coordinates.py            # fetch + report coverage, write nothing
    python fetch_northeast_coordinates.py --write     # fetch, validate, and write the raw
                                                       # capture -- ONLY on exactly 3,600/3,600
                                                       # coverage; refuses to write otherwise,
                                                       # regardless of --write. Refuses to
                                                       # overwrite an existing capture unless the
                                                       # freshly fetched content is byte-identical.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

API = "https://api.inaturalist.org/v1"
USER_AGENT = "AntID-pipeline/1.0 (personal non-commercial research project)"
RETRY_STATUS = {429, 500, 502, 503, 504}
BATCH_SIZE = 100
REQUEST_INTERVAL = 1.05
EXPECTED_ROW_COUNT = 3600

NORTHEAST_TRAIN_DEV = REPO / "data" / "northeast_expansion_v1" / "northeast_train_dev_v1.csv"
CAPTURE_PATH = REPO / "data" / "northeast_expansion_v1" / "northeast_coordinates_capture_v1.json"

COORDINATE_POLICY = (
    "Only a public location is used: geoprivacy=private source rows are always refused "
    "before any request is made (none exist in this dataset; this hard-fails if that ever "
    "changes). Obscured coordinates ARE included. An obscured location is NOT guaranteed to "
    "remain within the same 1-degree geo-index grid cell as the true location -- "
    "iNaturalist's obscuring can displace a location across a cell boundary. AntID's geo "
    "re-ranking checks the request cell plus its eight neighbors, which can mitigate a "
    "one-cell displacement in the Northeast, but this is not proof of identical placement or "
    "zero precision loss. This is a raw, unreviewed capture -- the geoprivacy/obscured "
    "distinction is recorded on the frozen source manifest, not re-derived here."
)


class FetchIntegrityError(RuntimeError):
    """A required invariant did not hold. Always fail closed."""


def request_bytes(url: str, tries: int = 5) -> tuple[bytes | None, str]:
    last = "unknown error"
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if data:
                return data, ""
            last = "empty response"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code not in RETRY_STATUS:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < tries:
            time.sleep(1.5 * (attempt + 1))
    return None, last


def api_json(path: str, params: dict) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    data, error = request_bytes(url)
    if data is None:
        raise RuntimeError(f"API request failed for {url}: {error}")
    return json.loads(data)


def _valid_coordinate(lat, lon) -> bool:
    if isinstance(lat, bool) or isinstance(lon, bool):
        return False
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return False
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    if not (-90.0 <= lat <= 90.0):
        return False
    if not (-180.0 <= lon <= 180.0):
        return False
    return True


def parse_public_location(obs: dict) -> tuple[float, float] | None:
    """(lat, lon) for a genuinely public/obscured location only. Never
    returns coordinates for a private observation."""
    if obs.get("geoprivacy") == "private":
        return None
    loc = obs.get("location")
    if not loc or not isinstance(loc, str) or "," not in loc:
        return None
    try:
        lat_s, lon_s = loc.split(",", 1)
        return float(lat_s), float(lon_s)
    except ValueError:
        return None


def fetch_observation_coordinates(uuids: list[str]) -> dict[str, tuple[float, float]]:
    found: dict[str, tuple[float, float]] = {}
    n_batches = -(-len(uuids) // BATCH_SIZE)
    for start in range(0, len(uuids), BATCH_SIZE):
        batch = uuids[start:start + BATCH_SIZE]
        payload = api_json("observations", {"uuid": ",".join(batch), "per_page": str(len(batch))})
        results = payload.get("results", [])
        for obs in results:
            uuid = obs.get("uuid")
            if not uuid:
                continue
            coords = parse_public_location(obs)
            if coords is not None:
                lat, lon = coords
                if not _valid_coordinate(lat, lon):
                    raise FetchIntegrityError(
                        f"observation {uuid} returned an invalid coordinate: lat={lat!r} "
                        f"lon={lon!r} -- refusing to persist it"
                    )
                found[uuid] = coords
        print(f"[fetch] batch {start // BATCH_SIZE + 1}/{n_batches}: "
             f"{len(results)} returned, {len(found)} coordinates so far", flush=True)
        if start + BATCH_SIZE < len(uuids):
            time.sleep(REQUEST_INTERVAL)
    return found


def _atomic_write_capture(capture: dict, path: Path) -> None:
    payload = (json.dumps(capture, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FetchIntegrityError(
                f"REFUSING to overwrite {path}: an existing capture is present and is NOT "
                f"byte-identical to the freshly fetched content. A fetch is not deterministic "
                f"run-to-run (retrieved_at_utc changes at least), so an existing capture must "
                f"be investigated and deliberately replaced, never silently overwritten."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Write the raw capture (only on exactly 3,600/3,600 coverage). "
                         "Default: fetch and report coverage, write nothing.")
    args = ap.parse_args()

    with NORTHEAST_TRAIN_DEV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_ROW_COUNT:
        print(f"FAIL: expected exactly {EXPECTED_ROW_COUNT} rows in {NORTHEAST_TRAIN_DEV}, "
             f"found {len(rows)}", file=sys.stderr)
        return 1

    uuids = [r["observation_uuid"] for r in rows]
    blank = [i for i, u in enumerate(uuids) if not u or not u.strip()]
    if blank:
        print(f"FAIL: {len(blank)} row(s) in {NORTHEAST_TRAIN_DEV} have a blank "
             f"observation_uuid", file=sys.stderr)
        return 1

    unique_uuids = sorted(set(uuids))
    if len(unique_uuids) != len(uuids):
        print(f"FAIL: {len(uuids)} rows but only {len(unique_uuids)} unique observation_uuid "
             f"values -- duplicate source UUIDs are not allowed", file=sys.stderr)
        return 1
    if len(unique_uuids) != EXPECTED_ROW_COUNT:
        print(f"FAIL: expected exactly {EXPECTED_ROW_COUNT} unique observation_uuid values, "
             f"found {len(unique_uuids)}", file=sys.stderr)
        return 1

    private_in_source = [r["observation_uuid"] for r in rows if r.get("geoprivacy") == "private"]
    if private_in_source:
        print(f"FAIL: {len(private_in_source)} row(s) in {NORTHEAST_TRAIN_DEV} are "
             f"geoprivacy=private; this script must never fetch coordinates for private "
             f"observations.", file=sys.stderr)
        return 1

    print(f"[fetch] querying {len(unique_uuids)} unique observations (from {len(rows)} rows) "
         f"at {REQUEST_INTERVAL}s/batch of {BATCH_SIZE}...")
    try:
        coords_by_uuid = fetch_observation_coordinates(unique_uuids)
    except FetchIntegrityError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"\n[fetch] unique observations queried: {len(unique_uuids)}")
    print(f"[fetch] observations with a valid public/obscured coordinate: {len(coords_by_uuid)}")

    if len(coords_by_uuid) != EXPECTED_ROW_COUNT:
        missing = sorted(set(unique_uuids) - set(coords_by_uuid))
        print(
            f"\n[fetch] incomplete coverage: {len(coords_by_uuid)} / {EXPECTED_ROW_COUNT} -- "
            f"refusing to write the capture regardless of --write. Missing observation_uuid(s) "
            f"(showing up to 5 of {len(missing)}): {missing[:5]}",
            file=sys.stderr,
        )
        return 1

    if not args.write:
        print("\n(report-only run -- pass --write to persist the raw capture)")
        return 0

    capture = {
        "schema_version": 1,
        "name": "northeast_coordinates_capture_v1",
        "purpose": "Raw, UNREVIEWED coordinate capture for the observations in "
                  "northeast_train_dev_v1.csv. This is NOT the frozen production sidecar -- "
                  "finalize_northeast_coordinates.py reviews and validates this offline to "
                  "produce northeast_coordinates_v1.json. Does not modify the frozen source "
                  "file.",
        "source": "https://api.inaturalist.org/v1/observations (uuid batch lookup)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "fetch_scope": "Only the observation_uuid values already present in "
                      "northeast_train_dev_v1.csv were queried -- no new observations were "
                      "discovered or substituted, and the candidate pool was not expanded.",
        "coordinate_policy": COORDINATE_POLICY,
        "coverage": {
            "rows_total": len(rows), "unique_observations_queried": len(unique_uuids),
            "observations_with_valid_coordinate": len(coords_by_uuid),
            "rows_with_coordinate": len(rows), "coverage_rate": 1.0,
        },
        "observation_coordinates": {
            uuid: {"lat": lat, "lon": lon} for uuid, (lat, lon) in sorted(coords_by_uuid.items())
        },
    }

    try:
        _atomic_write_capture(capture, CAPTURE_PATH)
    except FetchIntegrityError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"\nwrote {CAPTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
