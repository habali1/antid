#!/usr/bin/env python3
"""finalize_northeast_coordinates.py — build the reviewed, provenance-bound
Northeast coordinates sidecar OFFLINE, from a previously captured raw fetch
plus the frozen source manifest. Makes NO network requests.

This exists because the coordinate fetch that produced
northeast_coordinates_capture_v1.json ran before its design was reviewed:
that capture is raw, unreviewed input, never the frozen production sidecar.
This script is the reviewed step that turns it into one.

Inputs (read-only; never modified):
    data/northeast_expansion_v1/northeast_train_dev_v1.csv           frozen source manifest
    data/northeast_expansion_v1/northeast_coordinates_capture_v1.json preserved raw capture

Output (the only path this script ever writes to):
    data/northeast_expansion_v1/northeast_coordinates_v1.json

Requirements enforced, all fail-closed:
  - The real CLI path (`main()`, both check mode and --write) pins and
    mandatorily verifies EXPECTED_SOURCE_MANIFEST_SHA256 and
    EXPECTED_CAPTURE_SHA256 against the two raw input files' actual bytes
    before anything is parsed or built -- a substituted source manifest or a
    different/re-fetched capture is refused outright, not silently
    finalized. The pure finalize() function itself accepts these as
    optional parameters (None disables the check) so fault-injection tests
    can supply deliberately wrong values.
  - Exactly 3,600 source rows (this dataset's known, frozen size -- not a
    percentage threshold. Option A means complete coverage of the frozen
    rows, not "most of them").
  - No duplicate observation_uuid in the source manifest.
  - No source row is geoprivacy=private (refuses outright if one is; the
    captured API response itself did not preserve a per-observation
    geoprivacy/obscured field, so that side of the check cannot be repeated
    here without a new fetch -- documented as a limitation in the sidecar's
    own coordinate_policy text, not silently assumed clean).
  - Every source row is taxonomically self-consistent (species's first token
    equals its recorded genus; taxon_id and genus_id are positive integers).
  - The capture's UUID set matches the source's exactly -- no missing, no
    unexpected, no duplicate JSON keys.
  - Every coordinate is numeric, finite, and in-range (latitude in [-90, 90],
    longitude in [-180, 180]).
  - Exactly 3,600 finalized coordinate entries -- partial coverage fails
    closed, it is never silently accepted.

Usage:
    python finalize_northeast_coordinates.py            # check mode (default): rebuild
                                                          # into memory/a temp dir, diff
                                                          # byte-for-byte against the frozen
                                                          # output, write nothing, no network
    python finalize_northeast_coordinates.py --write     # finalize and write the frozen
                                                          # sidecar (refuses to overwrite an
                                                          # existing one that isn't already
                                                          # byte-identical)
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

EXPECTED_ROW_COUNT = 3600

DEFAULT_SOURCE = REPO / "data" / "northeast_expansion_v1" / "northeast_train_dev_v1.csv"
DEFAULT_CAPTURE = REPO / "data" / "northeast_expansion_v1" / "northeast_coordinates_capture_v1.json"
DEFAULT_OUTPUT = REPO / "data" / "northeast_expansion_v1" / "northeast_coordinates_v1.json"

# The frozen source manifest's and preserved capture's exact byte hashes.
# main() passes both as mandatory to finalize() -- a raw-file substitution
# (a different northeast_train_dev_v1.csv, or a different/regenerated
# capture) must never silently flow into the frozen sidecar. Settable to
# None only by tests exercising the pure finalize() function directly.
EXPECTED_SOURCE_MANIFEST_SHA256 = "9840997f7121907eabc9c5675244749f9d0eaa3908844bfd9db781d2424215f7"
EXPECTED_CAPTURE_SHA256 = "8275b487ff40d4095dfc9adc6e403299500b7185632104e2ad2f1fe2415a0677"

COORDINATE_POLICY = (
    "Coordinates come only from a public location: geoprivacy=private observations are "
    "refused outright (the frozen source manifest is checked; none are private in this "
    "dataset) and a private coordinate is never persisted. Obscured coordinates ARE "
    "included. Correction to an earlier draft of this policy: an obscured location is NOT "
    "guaranteed to remain within the same 1-degree geo-index grid cell as the true "
    "location -- iNaturalist's obscuring can displace a location across a cell boundary. "
    "AntID's geo re-ranking checks the request cell plus its eight neighbors, which can "
    "mitigate a one-cell displacement in the Northeast, but this is not proof of identical "
    "placement or zero precision loss -- it is a partial mitigation, not a guarantee. "
    "Missing or unrecoverable coordinates are left absent, never inferred or estimated "
    "(none are absent in this dataset: coverage is 3,600/3,600). Limitation: the "
    "geoprivacy/obscured values recorded here come from the frozen source manifest "
    "(northeast_train_dev_v1.csv, captured when the dataset was frozen), not from the "
    "coordinate fetch itself -- the raw capture this sidecar was finalized from recorded "
    "only latitude/longitude per observation, not the API's own per-observation "
    "geoprivacy/obscured fields, so that specific cross-check could not be repeated here "
    "without a new network request."
)


class FinalizeIntegrityError(RuntimeError):
    """A required invariant did not hold. Always fail closed."""


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _no_dup_object_pairs_hook(pairs):
    d: dict = {}
    for k, v in pairs:
        if k in d:
            raise FinalizeIntegrityError(f"duplicate JSON key {k!r} in capture file")
        d[k] = v
    return d


def finalize(source_path: Path = DEFAULT_SOURCE, capture_path: Path = DEFAULT_CAPTURE,
            expected_source_sha256: str | None = None,
            expected_capture_sha256: str | None = None,
            expected_row_count: int = EXPECTED_ROW_COUNT) -> dict:
    """Pure function: (source manifest, capture) -> finalized sidecar dict.
    No network access, no writes. Raises FinalizeIntegrityError fail-closed
    on any violated invariant."""
    if not source_path.exists():
        raise FinalizeIntegrityError(f"missing source manifest: {source_path}")
    if not capture_path.exists():
        raise FinalizeIntegrityError(f"missing capture: {capture_path}")

    source_bytes = source_path.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise FinalizeIntegrityError(
            f"{source_path} sha256 {source_sha256} does not match expected "
            f"{expected_source_sha256} -- refusing to finalize against an unexpected manifest"
        )

    source_rows = _read_csv_rows(source_path)
    if len(source_rows) != expected_row_count:
        raise FinalizeIntegrityError(
            f"expected exactly {expected_row_count} source rows, got {len(source_rows)}"
        )

    source_uuid_list = [r["observation_uuid"] for r in source_rows]
    seen: set[str] = set()
    dup_source: set[str] = set()
    for u in source_uuid_list:
        if u in seen:
            dup_source.add(u)
        else:
            seen.add(u)
    if dup_source:
        raise FinalizeIntegrityError(
            f"duplicate observation_uuid in source manifest: {sorted(dup_source)[:5]}"
        )
    source_by_uuid = {r["observation_uuid"]: r for r in source_rows}

    private_rows = [r["observation_uuid"] for r in source_rows if r.get("geoprivacy") == "private"]
    if private_rows:
        raise FinalizeIntegrityError(
            f"REFUSING: {len(private_rows)} source row(s) are geoprivacy=private -- a "
            f"private coordinate must never be persisted: {private_rows[:5]}"
        )

    for r in source_rows:
        species = (r.get("species") or "").strip()
        genus = (r.get("genus") or "").strip()
        if not species or not genus or species.split()[0] != genus:
            raise FinalizeIntegrityError(
                f"taxonomically inconsistent source row for observation "
                f"{r['observation_uuid']}: species={species!r} genus={genus!r}"
            )
        try:
            taxon_id = int(r["taxon_id"])
            genus_id = int(r["genus_id"])
        except (TypeError, ValueError, KeyError):
            raise FinalizeIntegrityError(
                f"non-integer taxon_id/genus_id for observation {r['observation_uuid']}: "
                f"taxon_id={r.get('taxon_id')!r} genus_id={r.get('genus_id')!r}"
            )
        if taxon_id <= 0 or genus_id <= 0:
            raise FinalizeIntegrityError(
                f"non-positive taxon_id/genus_id for observation {r['observation_uuid']}"
            )

    capture_bytes = capture_path.read_bytes()
    capture_sha256 = _sha256_bytes(capture_bytes)
    if expected_capture_sha256 is not None and capture_sha256 != expected_capture_sha256:
        raise FinalizeIntegrityError(
            f"{capture_path} sha256 {capture_sha256} does not match expected "
            f"{expected_capture_sha256} -- refusing to finalize against an unexpected capture"
        )
    try:
        capture = json.loads(capture_bytes, object_pairs_hook=_no_dup_object_pairs_hook)
    except json.JSONDecodeError as e:
        raise FinalizeIntegrityError(f"{capture_path} is not valid JSON: {e}")

    coords_raw = capture.get("observation_coordinates")
    if not isinstance(coords_raw, dict):
        raise FinalizeIntegrityError(f"{capture_path} has no observation_coordinates object")

    capture_uuids = set(coords_raw)
    source_uuids = set(source_by_uuid)
    missing = source_uuids - capture_uuids
    unexpected = capture_uuids - source_uuids
    if missing:
        raise FinalizeIntegrityError(
            f"{len(missing)} source UUID(s) missing from capture: {sorted(missing)[:5]}"
        )
    if unexpected:
        raise FinalizeIntegrityError(
            f"{len(unexpected)} capture UUID(s) not present in the source manifest: "
            f"{sorted(unexpected)[:5]}"
        )

    observations: dict[str, dict] = {}
    for uuid, row in source_by_uuid.items():
        entry = coords_raw[uuid]
        if not isinstance(entry, dict) or "lat" not in entry or "lon" not in entry:
            raise FinalizeIntegrityError(f"malformed capture entry for {uuid}: {entry!r}")
        lat, lon = entry["lat"], entry["lon"]
        if (isinstance(lat, bool) or isinstance(lon, bool)
                or not isinstance(lat, (int, float)) or not isinstance(lon, (int, float))):
            raise FinalizeIntegrityError(f"non-numeric coordinate for {uuid}: lat={lat!r} lon={lon!r}")
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise FinalizeIntegrityError(f"non-finite coordinate for {uuid}: lat={lat!r} lon={lon!r}")
        if not (-90.0 <= lat <= 90.0):
            raise FinalizeIntegrityError(f"latitude out of range for {uuid}: {lat!r}")
        if not (-180.0 <= lon <= 180.0):
            raise FinalizeIntegrityError(f"longitude out of range for {uuid}: {lon!r}")

        obscured_raw = (row.get("obscured") or "").strip().lower()
        observations[uuid] = {
            "observation_id": row["observation_id"],
            "slug": row["slug"],
            "taxon_id": int(row["taxon_id"]),
            "geoprivacy": row["geoprivacy"],
            "obscured": obscured_raw == "true",
            "lat": float(lat),
            "lon": float(lon),
        }

    if len(observations) != expected_row_count:
        raise FinalizeIntegrityError(
            f"expected exactly {expected_row_count} finalized coordinate entries, "
            f"got {len(observations)}"
        )

    retrieved_at_utc = capture.get("retrieved_at_utc")
    if not retrieved_at_utc:
        raise FinalizeIntegrityError(f"{capture_path} has no retrieved_at_utc")

    return {
        "schema_version": 1,
        "name": "northeast_coordinates_v1",
        "purpose": "Public/obscured coordinates for every observation in "
                  "northeast_train_dev_v1.csv, finalized offline from a previously captured "
                  "fetch. Never modifies that frozen file.",
        "source_manifest": {
            "path": "data/northeast_expansion_v1/northeast_train_dev_v1.csv",
            "sha256": source_sha256,
            "rows": len(source_rows),
        },
        "capture": {
            "path": "data/northeast_expansion_v1/northeast_coordinates_capture_v1.json",
            "sha256": capture_sha256,
            "retrieved_at_utc": retrieved_at_utc,
        },
        "coordinate_policy": COORDINATE_POLICY,
        "coverage": {
            "rows_total": len(source_rows),
            "rows_with_coordinate": len(observations),
            "coverage_rate": len(observations) / len(source_rows),
        },
        "observations": observations,
    }


def serialize(sidecar: dict) -> bytes:
    # Deterministic: sort_keys=True gives a well-defined total order at every
    # level; LF-only + trailing newline matches this directory's other
    # metadata JSON (scrape_northeast_expansion.py's write_json_atomic).
    return (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_atomic(sidecar: dict, output_path: Path) -> None:
    payload = serialize(sidecar)
    if output_path.exists():
        existing = output_path.read_bytes()
        if existing != payload:
            raise FinalizeIntegrityError(
                f"REFUSING to overwrite {output_path}: an existing file is present and is "
                f"NOT byte-identical to the freshly finalized content. Investigate before "
                f"deciding whether to replace a frozen sidecar."
            )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + f".tmp{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Write the finalized sidecar. Default: check mode, writes nothing.")
    args = ap.parse_args()

    try:
        sidecar = finalize(expected_source_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
                           expected_capture_sha256=EXPECTED_CAPTURE_SHA256)
    except FinalizeIntegrityError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"finalized: {sidecar['coverage']}")

    if args.write:
        try:
            write_atomic(sidecar, DEFAULT_OUTPUT)
        except FinalizeIntegrityError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 1
        print(f"wrote {DEFAULT_OUTPUT}")
        return 0

    if not DEFAULT_OUTPUT.exists():
        print(f"FAIL: frozen output missing: {DEFAULT_OUTPUT}", file=sys.stderr)
        return 1
    payload = serialize(sidecar)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / DEFAULT_OUTPUT.name
        tmp_path.write_bytes(payload)
        same = filecmp.cmp(tmp_path, DEFAULT_OUTPUT, shallow=False)
    print(("OK  " if same else "FAIL") +
         f" {DEFAULT_OUTPUT.name} matches freshly rebuilt output byte-for-byte")
    print("\nCHECK MODE:", "PASS" if same else "FAIL", "-- no network access, nothing written")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
