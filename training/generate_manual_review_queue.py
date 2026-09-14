#!/usr/bin/env python3
"""generate_manual_review_queue.py — builds (and, once, writes) the frozen
600-row manual-review queue for Phase 5F1: every row of
northeast_final_test_v1, plus a deterministic 5-train/5-development sample
per exact approved species slug from northeast_expansion_v1.

Reads CSV manifests ONLY -- metadata (species/slug/split/observation_uuid/
photo_id/sha256), never image bytes. No mode in this module ever opens an
image.

This generator does NOT itself load the frozen contract (it produces two of
the three artifacts the contract later binds -- see
freeze_manual_review_contract.py, which must run AFTER this). Every OTHER
Phase 5F1 tool loads and verifies the contract first.

Three modes:
  --preflight  validates everything and prints the summary; writes ZERO
               bytes.
  --write      atomically publishes the queue CSV and its summary JSON as
               ONE paired transaction: both are staged and fsynced first,
               then both are published via exclusive hard-link. If the
               second publish fails for any reason, the first is rolled
               back (unlinked) so neither half of the pair is ever left on
               disk alone.
  --check      re-derives the queue from the CURRENT source manifests and
               requires the existing queue CSV + summary JSON to match
               EXACTLY byte-for-byte (raw bytes, not text-mode/universal-
               newline reads, not semantic-only JSON equality). Writes
               nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402

QUEUE_CSV_REL_PATH = mrc.QUEUE_ARTIFACT_REL_PATH
SUMMARY_JSON_REL_PATH = mrc.SUMMARY_ARTIFACT_REL_PATH

IMPLEMENTATION_SOURCE_PATHS = {
    "manual_review_contract": "training/manual_review_contract.py",
    "generate_manual_review_queue": "training/generate_manual_review_queue.py",
}


class QueueGenerationError(RuntimeError):
    """A source-manifest or generation problem. Always fails closed -- on
    any problem, nothing is written."""


def _fail(msg: str) -> None:
    raise QueueGenerationError(msg)


def _read_manifest_rows(path: Path) -> tuple[list[dict], str, int]:
    """Metadata-only CSV read. Returns (rows, byte_sha256, row_count).
    Fails closed if a required column is missing from the header."""
    if not path.exists():
        _fail(f"source manifest does not exist: {path}")
    raw = path.read_bytes()
    byte_sha256 = mrc.sha256_bytes(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"{path} is not valid UTF-8: {exc}")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        _fail(f"{path} has no header row")
    missing_cols = set(mrc.SOURCE_MANIFEST_REQUIRED_COLUMNS) - set(reader.fieldnames)
    if missing_cols:
        _fail(f"{path} is missing required column(s): {sorted(missing_cols)}")
    rows = list(reader)
    return rows, byte_sha256, len(rows)


def _require_frozen_manifest(dataset: str, path: Path, byte_sha256: str, row_count: int) -> None:
    expected = mrc.FROZEN_SOURCE_MANIFESTS[dataset]
    if byte_sha256 != expected["byte_sha256"]:
        _fail(f"{path} byte sha256 {byte_sha256} != frozen {expected['byte_sha256']!r} -- this is "
              f"not the approved source manifest")
    if row_count != expected["row_count"]:
        _fail(f"{path} has {row_count} rows, frozen contract expects exactly {expected['row_count']}")


def _require_complete_row(dataset: str, row: dict) -> None:
    for field in mrc.SOURCE_MANIFEST_REQUIRED_COLUMNS:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            _fail(f"{dataset} row is missing/blank required field {field!r}: {row!r}")
    try:
        int(row["photo_id"])
    except (TypeError, ValueError):
        _fail(f"{dataset} row has a non-numeric photo_id: {row.get('photo_id')!r}")


def _select_final_test_rows(repo: Path) -> tuple[list[dict], dict]:
    path = repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
    rows, byte_sha256, row_count = _read_manifest_rows(path)
    _require_frozen_manifest(mrc.FINAL_TEST_DATASET, path, byte_sha256, row_count)
    selected = []
    for row in rows:
        _require_complete_row(mrc.FINAL_TEST_DATASET, row)
        if row["split"] != mrc.FINAL_TEST_SPLIT:
            _fail(f"{path} row has split {row['split']!r}, expected exactly {mrc.FINAL_TEST_SPLIT!r}")
        selected.append(row)
    manifest_info = {"path": mrc.FINAL_TEST_MANIFEST_REL_PATH, "byte_sha256": byte_sha256, "row_count": row_count}
    return selected, manifest_info


def _select_expansion_sample_rows(repo: Path) -> tuple[list[dict], dict, dict]:
    """Buckets by (slug, split), requires the slug set to be EXACTLY
    mrc.APPROVED_SPECIES_SLUGS (not merely 15 distinct slugs), requires
    every bucket to have at least PER_SLUG_PER_SPLIT eligible rows, then
    selects the first five by (selection_digest, observation_uuid, numeric
    photo_id)."""
    path = repo / mrc.EXPANSION_MANIFEST_REL_PATH
    rows, byte_sha256, row_count = _read_manifest_rows(path)
    _require_frozen_manifest(mrc.EXPANSION_DATASET, path, byte_sha256, row_count)

    buckets: dict[tuple[str, str], list[dict]] = {}
    slugs: set[str] = set()
    for row in rows:
        _require_complete_row(mrc.EXPANSION_DATASET, row)
        split = row["split"]
        if split not in mrc.EXPANSION_SPLITS:
            _fail(f"{path} row has unexpected split {split!r}, expected one of {mrc.EXPANSION_SPLITS}")
        slugs.add(row["slug"])
        buckets.setdefault((row["slug"], split), []).append(row)

    if slugs != set(mrc.APPROVED_SPECIES_SLUGS):
        missing = set(mrc.APPROVED_SPECIES_SLUGS) - slugs
        extra = slugs - set(mrc.APPROVED_SPECIES_SLUGS)
        _fail(f"{path} slug set does not exactly match the {len(mrc.APPROVED_SPECIES_SLUGS)} approved "
              f"species slugs -- missing: {sorted(missing)}, unexpected: {sorted(extra)}")

    expected_bucket_keys = {(slug, split) for slug in slugs for split in mrc.EXPANSION_SPLITS}
    missing_buckets = expected_bucket_keys - set(buckets)
    if missing_buckets:
        _fail(f"{path} is missing bucket(s) entirely: {sorted(missing_buckets)}")

    selected: list[dict] = []
    per_slug_counts: dict[str, dict[str, int]] = {}
    for (slug, split), bucket_rows in sorted(buckets.items()):
        if len(bucket_rows) < mrc.PER_SLUG_PER_SPLIT:
            _fail(f"{path} bucket (slug={slug!r}, split={split!r}) has only {len(bucket_rows)} "
                  f"eligible row(s), fewer than the required {mrc.PER_SLUG_PER_SPLIT}")
        ranked = sorted(
            bucket_rows,
            key=lambda r: (
                mrc.compute_selection_digest(mrc.SEED, r["slug"], r["split"], r["observation_uuid"], r["photo_id"]),
                r["observation_uuid"],
                int(r["photo_id"]),
            ),
        )
        chosen = ranked[:mrc.PER_SLUG_PER_SPLIT]
        selected.extend(chosen)
        per_slug_counts.setdefault(slug, {})[split] = len(chosen)

    if len(selected) != mrc.N_SAMPLED_ROWS:
        _fail(f"internal error: selected {len(selected)} expansion rows, expected exactly "
              f"{mrc.N_SAMPLED_ROWS}")

    manifest_info = {"path": mrc.EXPANSION_MANIFEST_REL_PATH, "byte_sha256": byte_sha256, "row_count": row_count}
    return selected, manifest_info, per_slug_counts


def build_queue_rows(repo: Path) -> tuple[list[dict], dict]:
    """Pure over the CURRENT contents of the two source manifests (metadata
    only). Returns (queue_rows, source_manifests_info). Fails closed on any
    duplicate photo_id, observation_uuid, sha256, OR full-identity
    collision across the combined 600 rows -- not merely the composite
    identity tuple."""
    final_test_rows, final_test_manifest_info = _select_final_test_rows(repo)
    expansion_rows, expansion_manifest_info, per_slug_counts = _select_expansion_sample_rows(repo)

    combined: list[tuple[str, dict]] = (
        [(mrc.FINAL_TEST_DATASET, r) for r in final_test_rows]
        + [(mrc.EXPANSION_DATASET, r) for r in expansion_rows]
    )
    if len(combined) != mrc.N_QUEUE_ROWS:
        _fail(f"internal error: combined {len(combined)} rows, expected exactly {mrc.N_QUEUE_ROWS}")

    seen_photo_ids: set[str] = set()
    seen_uuids: set[str] = set()
    seen_sha256s: set[str] = set()
    enriched: list[dict] = []
    for dataset, row in combined:
        photo_id = str(int(row["photo_id"]))
        uuid = row["observation_uuid"]
        sha256 = row["sha256"]
        if photo_id in seen_photo_ids:
            _fail(f"duplicate photo_id across the queue: {photo_id!r}")
        if uuid in seen_uuids:
            _fail(f"duplicate observation_uuid across the queue: {uuid!r}")
        if sha256 in seen_sha256s:
            _fail(f"duplicate image sha256 across the queue: {sha256!r}")
        seen_photo_ids.add(photo_id)
        seen_uuids.add(uuid)
        seen_sha256s.add(sha256)
        enriched.append({
            "dataset": dataset,
            "slug": row["slug"],
            "split": row["split"],
            "species": row["species"],
            "taxon_id": row["taxon_id"],
            "observation_uuid": uuid,
            "photo_id": photo_id,
            "sha256": sha256,
            "selection_digest": mrc.compute_selection_digest(
                mrc.SEED, row["slug"], row["split"], row["observation_uuid"], row["photo_id"]),
            "review_order_digest": mrc.compute_review_order_digest(
                mrc.SEED, dataset, row["slug"], row["split"], row["observation_uuid"], row["photo_id"]),
        })

    enriched.sort(key=lambda r: (
        r["review_order_digest"], r["dataset"], r["slug"], r["split"], r["observation_uuid"], r["photo_id"],
    ))
    for index, row in enumerate(enriched):
        row["queue_index"] = index
        row["suggested_session"] = 1 if index < mrc.ROWS_PER_SUGGESTED_SESSION else 2

    session_counts: dict[int, int] = {}
    for row in enriched:
        session_counts[row["suggested_session"]] = session_counts.get(row["suggested_session"], 0) + 1
    for session, expected in ((1, mrc.ROWS_PER_SUGGESTED_SESSION),
                              (2, mrc.N_QUEUE_ROWS - mrc.ROWS_PER_SUGGESTED_SESSION)):
        if session_counts.get(session) != expected:
            _fail(f"internal error: suggested_session {session} has {session_counts.get(session)} "
                  f"rows, expected exactly {expected}")

    source_manifests_info = {
        mrc.FINAL_TEST_DATASET: final_test_manifest_info,
        mrc.EXPANSION_DATASET: {**expansion_manifest_info, "per_slug_counts": per_slug_counts},
    }
    return enriched, source_manifests_info


def _queue_row_to_csv_dict(row: dict) -> dict:
    return {field: row[field] for field in mrc.QUEUE_ROW_REQUIRED_FIELDS}


def serialize_queue_csv(rows: list[dict]) -> bytes:
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(mrc.QUEUE_ROW_REQUIRED_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_queue_row_to_csv_dict(row))
    return buf.getvalue().encode("utf-8")


def build_summary(rows: list[dict], source_manifests_info: dict, queue_csv_bytes: bytes,
                  identity_order_sha256: str) -> dict:
    # Implementation-source hashes always come from where THIS code actually
    # lives (module-level REPO), never from a --repo argument (which points
    # at the DATA/output location a test's synthetic fixture may not
    # mirror) -- the two coincide in every real, non-test invocation.
    impl_hashes = {name: mrc.canonical_lf_sha256_file(REPO / path)
                   for name, path in IMPLEMENTATION_SOURCE_PATHS.items()}
    counts = {
        "total": len(rows),
        "final_test": sum(1 for r in rows if r["dataset"] == mrc.FINAL_TEST_DATASET),
        "expansion_train": sum(1 for r in rows if r["dataset"] == mrc.EXPANSION_DATASET
                               and r["split"] == mrc.EXPANSION_TRAIN_SPLIT),
        "expansion_development": sum(1 for r in rows if r["dataset"] == mrc.EXPANSION_DATASET
                                     and r["split"] == mrc.EXPANSION_DEV_SPLIT),
        "suggested_session_1": sum(1 for r in rows if r["suggested_session"] == 1),
        "suggested_session_2": sum(1 for r in rows if r["suggested_session"] == 2),
    }
    content = {
        "seed": mrc.SEED,
        "approved_species_slugs": list(mrc.APPROVED_SPECIES_SLUGS),
        "source_manifests": source_manifests_info,
        "queue_csv": {
            "path": QUEUE_CSV_REL_PATH,
            "byte_sha256": mrc.sha256_bytes(queue_csv_bytes),
            "row_count": len(rows),
            "identity_order_sha256": identity_order_sha256,
        },
        "counts": counts,
        "implementation_sources": impl_hashes,
    }
    content_sha256 = mrc.compute_content_sha256(content)
    return {
        "schema_version": mrc.SCHEMA_VERSION,
        "content": content,
        "content_sha256": content_sha256,
    }


# --------------------------------------------------------------- commands --
def run_preflight(args) -> dict[str, Any]:
    rows, source_manifests_info = build_queue_rows(args.repo)
    queue_csv_bytes = serialize_queue_csv(rows)
    identity_order_sha256 = mrc.compute_queue_identity_order_sha256(rows)
    summary = build_summary(rows, source_manifests_info, queue_csv_bytes, identity_order_sha256)
    return {"ok": True, "rows": rows, "queue_csv_bytes": queue_csv_bytes,
            "identity_order_sha256": identity_order_sha256, "summary": summary}


def _stage_bytes(dest: Path, data: bytes) -> Path:
    """Writes `data` to a unique temp file in the SAME directory as `dest`,
    fsyncs it, and returns the temp path -- does not publish it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return tmp


def _publish_exclusive(tmp: Path, dest: Path) -> None:
    """Hard-links `tmp` to `dest`, refusing to overwrite an existing file.
    Always removes `tmp` afterward, success or failure."""
    try:
        os.link(str(tmp), str(dest))
    finally:
        tmp.unlink(missing_ok=True)


def publish_pair(queue_dest: Path, queue_bytes: bytes, summary_dest: Path, summary_bytes: bytes) -> None:
    """Publishes BOTH files as one paired transaction: both are staged and
    fsynced FIRST, then published exclusively one at a time. If the SECOND
    publish fails for ANY reason (including an injected failure in tests),
    the FIRST publish is rolled back (unlinked) so neither file is ever
    left on disk alone -- a reader that finds the queue CSV can always
    trust that its paired summary exists too, and vice versa."""
    queue_tmp = _stage_bytes(queue_dest, queue_bytes)
    try:
        summary_tmp = _stage_bytes(summary_dest, summary_bytes)
    except Exception:
        queue_tmp.unlink(missing_ok=True)
        raise

    try:
        try:
            os.link(str(queue_tmp), str(queue_dest))
        except FileExistsError:
            _fail(f"refusing to overwrite existing queue CSV: {queue_dest}")

        try:
            os.link(str(summary_tmp), str(summary_dest))
        except FileExistsError:
            queue_dest.unlink(missing_ok=True)
            _fail(f"refusing to overwrite existing summary JSON: {summary_dest} "
                  f"(queue publish rolled back)")
        except Exception:
            # ANY other failure publishing the SECOND file rolls back the
            # FIRST -- neither half of the pair survives.
            queue_dest.unlink(missing_ok=True)
            raise
    finally:
        queue_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)


def cmd_write(args) -> dict[str, Any]:
    queue_path = args.repo / QUEUE_CSV_REL_PATH
    summary_path = args.repo / SUMMARY_JSON_REL_PATH
    if queue_path.exists():
        _fail(f"refusing to overwrite existing queue CSV: {queue_path}")
    if summary_path.exists():
        _fail(f"refusing to overwrite existing summary JSON: {summary_path}")

    pre = run_preflight(args)
    queue_bytes = pre["queue_csv_bytes"]
    summary_bytes = (json.dumps(pre["summary"], indent=2, sort_keys=True) + "\n").encode("utf-8")

    publish_pair(queue_path, queue_bytes, summary_path, summary_bytes)

    return {"queue_path": queue_path, "summary_path": summary_path,
            "queue_csv_byte_sha256": pre["summary"]["content"]["queue_csv"]["byte_sha256"],
            "summary_content_sha256": pre["summary"]["content_sha256"]}


def cmd_check(args) -> dict[str, Any]:
    queue_path = args.repo / QUEUE_CSV_REL_PATH
    summary_path = args.repo / SUMMARY_JSON_REL_PATH
    if not queue_path.exists():
        _fail(f"queue CSV does not exist: {queue_path}")
    if not summary_path.exists():
        _fail(f"summary JSON does not exist: {summary_path}")

    # Raw bytes, deliberately NOT text-mode/universal-newline reads --
    # "byte identical" must mean byte identical.
    stored_queue_bytes = queue_path.read_bytes()
    stored_summary_bytes = summary_path.read_bytes()

    pre = run_preflight(args)
    expected_summary_bytes = (json.dumps(pre["summary"], indent=2, sort_keys=True) + "\n").encode("utf-8")

    if stored_queue_bytes != pre["queue_csv_bytes"]:
        _fail(f"{queue_path} does not match the freshly rebuilt queue byte-for-byte")
    if stored_summary_bytes != expected_summary_bytes:
        _fail(f"{summary_path} does not match the freshly rebuilt summary byte-for-byte")

    return {"queue_path": queue_path, "summary_path": summary_path,
            "content_sha256": pre["summary"]["content_sha256"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        if args.preflight:
            pre = run_preflight(args)
            summary = pre["summary"]
            print(json.dumps({
                "ok": True,
                "content_sha256": summary["content_sha256"],
                "counts": summary["content"]["counts"],
                "queue_csv_byte_sha256": summary["content"]["queue_csv"]["byte_sha256"],
                "identity_order_sha256": pre["identity_order_sha256"],
            }, indent=2))
        elif args.write:
            result = cmd_write(args)
            print(f"wrote {result['queue_path']}")
            print(f"wrote {result['summary_path']}")
            print(f"summary content_sha256: {result['summary_content_sha256']}")
        else:
            result = cmd_check(args)
            print(f"{result['queue_path']} PASS")
            print(f"{result['summary_path']} PASS")
            print(f"content_sha256: {result['content_sha256']}")
        return 0
    except QueueGenerationError as e:
        print(f"[generate_manual_review_queue] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
