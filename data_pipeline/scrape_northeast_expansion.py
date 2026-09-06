#!/usr/bin/env python3
"""Download/freeze or restore the 15-species Northeast expansion dataset.

Fresh mode reads the audited candidates.csv and never discovers or substitutes
observations. Approved candidates are placed in a stable SHA-256 ordering using
the recorded seed. Each species is attempted in ascending reserve order. A run
that leaves any species below 270 writes only a non-frozen attempt report and
keeps its download cache; it writes no dataset manifest or metadata JSON.

When every species reaches 270 usable, positions 0:200 become train, 200:240
development, and 240:270 final-test. Final-test files and manifests live under
a separate root and never enter the train/development manifest.

Restore mode reads one already-frozen manifest, resolves its exact observation
UUID/photo ID pairs through iNaturalist, and verifies every restored file hash.
It never modifies the manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

API = "https://api.inaturalist.org/v1"
USER_AGENT = "AntID-pipeline/1.0 (personal non-commercial research project)"
SEED = 20260905
MIN_DIMENSION = 200
QUOTA = {"train": 200, "development": 40, "final_test": 30}
TOTAL_PER_SPECIES = sum(QUOTA.values())
RETRY_STATUS = {429, 500, 502, 503, 504}
APPROVED_LICENSES = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}
REQUIRED_SOURCE_FIELDS = (
    "photo_license",
    "photo_attribution",
    "observation_uuid",
    "photo_id",
    "photo_url_medium",
)
MANIFEST_FIELDS = (
    "species",
    "slug",
    "taxon_id",
    "genus",
    "genus_id",
    "split",
    "state",
    "observation_id",
    "observation_uuid",
    "observer_id",
    "observed_on",
    "created_at",
    "geoprivacy",
    "obscured",
    "photo_id",
    "photo_license",
    "photo_attribution",
    "source_url",
    "sha256",
    "byte_size",
    "width",
    "height",
    "raw_relative_path",
    "clean_relative_path",
)
FROZEN_HASH_SOURCES = {
    "benchmark_v1": Path("data/benchmark_v1/benchmark_v1.csv"),
    "calibration_v1": Path("data/calibration_v1/calibration_v1.csv"),
    "unknown_test_v1": Path("data/unknown_test_v1/unknown_test_v1.csv"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    return "-".join("".join(c.lower() if c.isalnum() else " " for c in value).split())


def extension_from_url(url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower().lstrip(".")
    return suffix if suffix in {"jpg", "jpeg", "png", "webp"} else "jpg"


def deterministic_key(row: dict[str, str], seed: int) -> tuple[str, str, str]:
    material = "\0".join(
        (str(seed), row["taxon_id"], row["observation_uuid"], row["photo_id"])
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest(), row["observation_uuid"], row["photo_id"]


def split_for_position(position: int) -> str:
    if position < QUOTA["train"]:
        return "train"
    if position < QUOTA["train"] + QUOTA["development"]:
        return "development"
    if position < TOTAL_PER_SPECIES:
        return "final_test"
    raise ValueError(f"position {position} exceeds quota")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def validate_source_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    approved: list[dict[str, str]] = []
    errors: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        if row.get("eligible_personal_nc") != "true":
            continue
        missing = [field for field in REQUIRED_SOURCE_FIELDS if not row.get(field, "").strip()]
        if missing:
            errors.append(f"row {row_number} missing {','.join(missing)}")
        if row.get("photo_license", "").lower() not in APPROVED_LICENSES:
            errors.append(f"row {row_number} has unapproved license {row.get('photo_license')!r}")
        if row.get("prior_overlap") != "false" or row.get("internal_duplicate") != "false":
            errors.append(f"row {row_number} is marked as overlapping/duplicate")
        approved.append(row)
    if errors:
        raise RuntimeError("candidate provenance/eligibility failed:\n" + "\n".join(errors[:50]))
    photo_ids = [row["photo_id"] for row in approved]
    observation_uuids = [row["observation_uuid"] for row in approved]
    if len(photo_ids) != len(set(photo_ids)) or len(observation_uuids) != len(set(observation_uuids)):
        raise RuntimeError("approved candidates are not unique by photo_id and observation_uuid")
    return approved


def group_in_run_order(rows: list[dict[str, str]], seed: int) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["species"]].append(row)
    for species_rows in grouped.values():
        species_rows.sort(key=lambda row: deterministic_key(row, seed))
    return sorted(grouped.items(), key=lambda item: (len(item[1]) - TOTAL_PER_SPECIES, item[0]))


def load_frozen_hashes(repo: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    provenance: dict[str, dict[str, Any]] = {}
    for name, relative in FROZEN_HASH_SOURCES.items():
        path = repo / relative
        rows = read_csv(path)
        missing = [index for index, row in enumerate(rows, start=2) if not row.get("sha256")]
        if missing:
            raise RuntimeError(f"{relative} has blank sha256 at rows {missing[:10]}")
        for row in rows:
            by_hash[row["sha256"].lower()].append({
                "dataset": name,
                "photo_id": row.get("photo_id", ""),
                "observation_uuid": row.get("observation_uuid", ""),
                "species": row.get("species", ""),
            })
        provenance[name] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "rows": len(rows),
        }
    return by_hash, provenance


def request_bytes(url: str, tries: int = 4) -> tuple[bytes | None, str]:
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


def cached_path(cache_dir: Path, photo_id: str) -> Path | None:
    matches = sorted(cache_dir.glob(f"{photo_id}.*"))
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous cache files for photo_id {photo_id}: {matches}")
    return matches[0] if matches else None


def download_or_reuse(row: dict[str, str], cache_dir: Path) -> tuple[Path | None, bytes | None, int, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = cached_path(cache_dir, row["photo_id"])
    if existing is not None:
        return existing, existing.read_bytes(), 0, ""
    data, error = request_bytes(row["photo_url_medium"])
    if data is None:
        return None, None, 0, error
    path = cache_dir / f"{row['photo_id']}.{extension_from_url(row['photo_url_medium'])}"
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return path, data, len(data), ""


def inspect_image(data: bytes) -> tuple[str, int | None, int | None, str]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
    except Exception as exc:  # noqa: BLE001
        return "decode_failure", None, None, f"{type(exc).__name__}: {exc}"
    if min(width, height) < MIN_DIMENSION:
        return "under_200px", width, height, f"dimensions={width}x{height}"
    return "ok", width, height, ""


def manifest_row(source: dict[str, str], accepted: dict[str, Any], split: str) -> dict[str, Any]:
    slug = slugify(source["species"])
    filename = accepted["cache_path"].name
    return {
        "species": source["species"],
        "slug": slug,
        "taxon_id": source["taxon_id"],
        "genus": source["genus"],
        "genus_id": source["genus_id"],
        "split": split,
        "state": source["state"],
        "observation_id": source["observation_id"],
        "observation_uuid": source["observation_uuid"],
        "observer_id": source["observer_id"],
        "observed_on": source["observed_on"],
        "created_at": source["created_at"],
        "geoprivacy": source["geoprivacy"],
        "obscured": source["obscured"],
        "photo_id": source["photo_id"],
        "photo_license": source["photo_license"],
        "photo_attribution": source["photo_attribution"],
        "source_url": source["photo_url_medium"],
        "sha256": accepted["sha256"],
        "byte_size": accepted["byte_size"],
        "width": accepted["width"],
        "height": accepted["height"],
        "raw_relative_path": f"raw/{slug}/{filename}",
        "clean_relative_path": f"clean/{slug}/{filename}",
    }


def copy_verified(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected_hash:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copy hash mismatch for {destination}")
    os.replace(temporary, destination)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def frozen_metadata(
    name: str,
    manifest_path: Path,
    manifest_rows: list[dict[str, Any]],
    repo: Path,
    candidates_path: Path,
    frozen_sources: dict[str, dict[str, Any]],
    seed: int,
    transferred_bytes: int,
    rejections: list[dict[str, Any]],
) -> dict[str, Any]:
    per_species = Counter(row["species"] for row in manifest_rows)
    per_split = Counter(row["split"] for row in manifest_rows)
    return {
        "schema_version": 1,
        "name": name,
        "frozen_at_utc": utc_now(),
        "purpose": "Northeast expansion train/development data" if name == "northeast_expansion_v1" else "untouched final test for Northeast expansion v1",
        "selection": {
            "seed": seed,
            "ordering": "ascending sha256(seed + NUL + taxon_id + NUL + observation_uuid + NUL + photo_id)",
            "candidate_csv": {
                "path": candidates_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(candidates_path),
            },
            "quota": QUOTA,
        },
        "manifest": {
            "path": manifest_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(manifest_path),
            "rows": len(manifest_rows),
            "required_provenance": ["photo_license", "photo_attribution", "observation_uuid", "photo_id", "source_url", "sha256"],
        },
        "counts": {
            "per_species": dict(sorted(per_species.items())),
            "per_split": dict(sorted(per_split.items())),
        },
        "frozen_evaluation_hash_sources": frozen_sources,
        "initial_run": {
            "transferred_bytes": transferred_bytes,
            "rejections": dict(sorted(Counter(row["reason"] for row in rejections).items())),
            "frozen_set_hash_collisions": sum(row["reason"] == "frozen_set_hash_collision" for row in rejections),
        },
        "restore": "Run data_pipeline/scrape_northeast_expansion.py --restore --manifest <this manifest> --out-root <manifest parent>. Restore never modifies the manifest and fails on any missing or mismatched image.",
        "license_scope": "personal, non-commercial; includes CC BY-NC and CC BY-NC-SA",
    }


def write_attempt_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["species", "photo_id", "observation_uuid", "reason", "detail", "transferred_bytes", "cache_path"]
    write_csv_atomic(path, fields, rows)


def write_status_report(
    path: Path,
    ordered_species: list[tuple[str, list[dict[str, str]]]],
    accepted_by_species: dict[str, list[dict[str, Any]]],
    attempts: list[dict[str, Any]],
    transferred_by_species: Counter[str],
) -> None:
    rejected = defaultdict(Counter)
    for row in attempts:
        rejected[row["species"]][row["reason"]] += 1
    rows = []
    for species, candidates in ordered_species:
        usable = len(accepted_by_species[species])
        counts = rejected[species]
        rows.append({
            "species": species,
            "approved_pool": len(candidates),
            "reserve": len(candidates) - TOTAL_PER_SPECIES,
            "usable": usable,
            "quota": TOTAL_PER_SPECIES,
            "shortfall": max(0, TOTAL_PER_SPECIES - usable),
            "download_failure": counts["download_failure"],
            "decode_failure": counts["decode_failure"],
            "under_200px": counts["under_200px"],
            "frozen_set_hash_collision": counts["frozen_set_hash_collision"],
            "internal_hash_duplicate": counts["internal_hash_duplicate"],
            "transferred_bytes": transferred_by_species[species],
        })
    write_csv_atomic(path, list(rows[0]), rows)


def run_fresh(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    candidates_path = args.candidates if args.candidates.is_absolute() else repo / args.candidates
    expansion_root = args.expansion_out if args.expansion_out.is_absolute() else repo / args.expansion_out
    final_root = args.final_out if args.final_out.is_absolute() else repo / args.final_out
    expected_candidate_hash = "082522f359585db13319378e14a1066c4fb7ffdda98194a9fae829e531cdc5ed"
    actual_candidate_hash = sha256_file(candidates_path)
    if actual_candidate_hash != expected_candidate_hash:
        raise RuntimeError(f"candidate CSV hash mismatch: {actual_candidate_hash}")

    approved = validate_source_rows(read_csv(candidates_path))
    ordered_species = group_in_run_order(approved, args.seed)
    if len(ordered_species) != 15:
        raise RuntimeError(f"expected 15 species, found {len(ordered_species)}")
    short_at_source = [(species, len(rows)) for species, rows in ordered_species if len(rows) < TOTAL_PER_SPECIES]
    if short_at_source:
        raise RuntimeError(f"approved source pools below quota: {short_at_source}")

    frozen_hashes, frozen_sources = load_frozen_hashes(repo)
    cache_root = expansion_root / "_download_cache"
    attempt_rows: list[dict[str, Any]] = []
    accepted_by_species: dict[str, list[dict[str, Any]]] = {}
    seen_accepted_hashes: dict[str, dict[str, str]] = {}
    transferred_bytes = 0
    transferred_by_species: Counter[str] = Counter()

    print(f"[northeast] seed={args.seed}, species={len(ordered_species)}, quota={TOTAL_PER_SPECIES}", flush=True)
    print("[northeast] run order: " + ", ".join(f"{name} (reserve {len(rows)-TOTAL_PER_SPECIES})" for name, rows in ordered_species), flush=True)
    for species, species_rows in ordered_species:
        slug = slugify(species)
        accepted: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        print(f"[{species}] approved={len(species_rows)} reserve={len(species_rows)-TOTAL_PER_SPECIES}", flush=True)
        for source in species_rows:
            if len(accepted) >= TOTAL_PER_SPECIES:
                break
            cache_path, data, transferred, error = download_or_reuse(source, cache_root / slug)
            transferred_bytes += transferred
            transferred_by_species[species] += transferred
            if data is None or cache_path is None:
                counters["download_failure"] += 1
                attempt_rows.append({"species": species, "photo_id": source["photo_id"], "observation_uuid": source["observation_uuid"], "reason": "download_failure", "detail": error, "transferred_bytes": transferred, "cache_path": ""})
                continue
            digest = sha256_bytes(data)
            status, width, height, detail = inspect_image(data)
            if status != "ok":
                counters[status] += 1
                attempt_rows.append({"species": species, "photo_id": source["photo_id"], "observation_uuid": source["observation_uuid"], "reason": status, "detail": detail, "transferred_bytes": transferred, "cache_path": str(cache_path.relative_to(repo))})
                continue
            collisions = frozen_hashes.get(digest, [])
            if collisions:
                counters["frozen_set_hash_collision"] += 1
                detail = json.dumps(collisions, sort_keys=True, separators=(",", ":"))
                attempt_rows.append({"species": species, "photo_id": source["photo_id"], "observation_uuid": source["observation_uuid"], "reason": "frozen_set_hash_collision", "detail": detail, "transferred_bytes": transferred, "cache_path": str(cache_path.relative_to(repo))})
                continue
            prior = seen_accepted_hashes.get(digest)
            if prior is not None:
                counters["internal_hash_duplicate"] += 1
                attempt_rows.append({"species": species, "photo_id": source["photo_id"], "observation_uuid": source["observation_uuid"], "reason": "internal_hash_duplicate", "detail": json.dumps(prior, sort_keys=True), "transferred_bytes": transferred, "cache_path": str(cache_path.relative_to(repo))})
                continue
            seen_accepted_hashes[digest] = {"species": species, "photo_id": source["photo_id"], "observation_uuid": source["observation_uuid"]}
            accepted.append({"source": source, "cache_path": cache_path, "sha256": digest, "byte_size": len(data), "width": width, "height": height})
        accepted_by_species[species] = accepted
        print(f"[{species}] usable={len(accepted)}/{TOTAL_PER_SPECIES}; rejected={dict(sorted(counters.items()))}", flush=True)

    report_path = expansion_root / "download_attempts.csv"
    write_attempt_report(report_path, attempt_rows)
    write_status_report(
        expansion_root / "download_status.csv",
        ordered_species,
        accepted_by_species,
        attempt_rows,
        transferred_by_species,
    )
    shortfalls = {species: TOTAL_PER_SPECIES - len(rows) for species, rows in accepted_by_species.items() if len(rows) < TOTAL_PER_SPECIES}
    print(f"[northeast] transferred_bytes={transferred_bytes}", flush=True)
    if shortfalls:
        print(f"[northeast] INCOMPLETE: {shortfalls}; no frozen manifests or JSON written", flush=True)
        print(f"[northeast] cache_size_bytes={directory_size(cache_root)}", flush=True)
        return 2

    train_dev_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for species, _ in ordered_species:
        for position, accepted in enumerate(accepted_by_species[species]):
            split = split_for_position(position)
            row = manifest_row(accepted["source"], accepted, split)
            target_root = final_root if split == "final_test" else expansion_root
            copy_verified(accepted["cache_path"], target_root / row["raw_relative_path"], row["sha256"])
            copy_verified(accepted["cache_path"], target_root / row["clean_relative_path"], row["sha256"])
            (final_rows if split == "final_test" else train_dev_rows).append(row)

    train_dev_manifest = expansion_root / "northeast_train_dev_v1.csv"
    final_manifest = final_root / "northeast_final_test_v1.csv"
    write_csv_atomic(train_dev_manifest, list(MANIFEST_FIELDS), train_dev_rows)
    write_csv_atomic(final_manifest, list(MANIFEST_FIELDS), final_rows)
    write_json_atomic(
        expansion_root / "northeast_expansion_v1.json",
        frozen_metadata("northeast_expansion_v1", train_dev_manifest, train_dev_rows, repo, candidates_path, frozen_sources, args.seed, transferred_bytes, attempt_rows),
    )
    write_json_atomic(
        final_root / "northeast_final_test_v1.json",
        frozen_metadata("northeast_final_test_v1", final_manifest, final_rows, repo, candidates_path, frozen_sources, args.seed, transferred_bytes, attempt_rows),
    )
    shutil.rmtree(cache_root)
    print(f"[northeast] FROZEN train/dev={len(train_dev_rows)}, final={len(final_rows)}", flush=True)
    print(f"[northeast] expansion_on_disk_bytes={directory_size(expansion_root)}", flush=True)
    print(f"[northeast] final_test_on_disk_bytes={directory_size(final_root)}", flush=True)
    return 0


def api_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    data, error = request_bytes(url, tries=5)
    if data is None:
        raise RuntimeError(f"API request failed: {error}")
    return json.loads(data)


def restore_observations(uuids: list[str], request_interval: float) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(uuids), 100):
        batch = uuids[start:start + 100]
        payload = api_json("observations", {"uuid": ",".join(batch), "per_page": str(len(batch))})
        for observation in payload.get("results", []):
            if observation.get("uuid"):
                found[observation["uuid"]] = observation
        if start + 100 < len(uuids):
            time.sleep(request_interval)
    return found


def run_restore(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    out_root = args.out_root if args.out_root.is_absolute() else repo / args.out_root
    rows = read_csv(manifest)
    if not rows or any(field not in rows[0] for field in MANIFEST_FIELDS):
        raise RuntimeError("restore manifest is empty or has the wrong schema")
    observations = restore_observations(sorted({row["observation_uuid"] for row in rows}), args.request_interval)
    errors: list[str] = []
    transferred = 0
    for row in rows:
        raw_path = out_root / row["raw_relative_path"]
        clean_path = out_root / row["clean_relative_path"]
        if raw_path.exists() and sha256_file(raw_path) == row["sha256"]:
            data = raw_path.read_bytes()
        else:
            observation = observations.get(row["observation_uuid"])
            photo = next((photo for photo in (observation or {}).get("photos", []) if str(photo.get("id")) == row["photo_id"]), None)
            if not photo or not photo.get("url"):
                errors.append(f"{row['species']}/{row['photo_id']}: observation/photo unavailable")
                continue
            url = photo["url"].replace("square", "medium")
            data, error = request_bytes(url)
            if data is None:
                errors.append(f"{row['species']}/{row['photo_id']}: {error}")
                continue
            transferred += len(data)
            if sha256_bytes(data) != row["sha256"]:
                errors.append(f"{row['species']}/{row['photo_id']}: restored sha256 mismatch")
                continue
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = raw_path.with_suffix(raw_path.suffix + ".part")
            temporary.write_bytes(data)
            os.replace(temporary, raw_path)
        status, _, _, detail = inspect_image(data)
        if status != "ok":
            errors.append(f"{row['species']}/{row['photo_id']}: {status} {detail}")
            continue
        copy_verified(raw_path, clean_path, row["sha256"])
    print(f"[restore] rows={len(rows)} transferred_bytes={transferred} errors={len(errors)}", flush=True)
    for error in errors:
        print(f"  ! {error}", flush=True)
    return 2 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidates", type=Path, default=Path("data/northeast_readiness_v1/candidates.csv"))
    parser.add_argument("--expansion-out", type=Path, default=Path("data/northeast_expansion_v1"))
    parser.add_argument("--final-out", type=Path, default=Path("data/northeast_final_test_v1"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--request-interval", type=float, default=1.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.restore:
        if args.manifest is None or args.out_root is None:
            raise SystemExit("--restore requires --manifest and --out-root")
        return run_restore(args)
    if args.manifest is not None or args.out_root is not None:
        raise SystemExit("--manifest/--out-root apply only with --restore")
    return run_fresh(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; downloaded cache files were preserved.", file=sys.stderr)
        raise SystemExit(130)
