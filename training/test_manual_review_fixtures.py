#!/usr/bin/env python3
"""test_manual_review_fixtures.py — shared synthetic-fixture builder for
manual_review_tool.py / scan_perceptual_duplicates.py / adjudicate_pairs.py
tests: a tiny repo with a real (patched-scale) contract + queue.csv +
summary.json that passes mrc.load_and_verify_contract() end to end.

Not a test file itself (no TestCase here) -- imported by the ones that need
a full, verifiable contract/queue/summary triple.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import generate_manual_review_queue as gq  # noqa: E402


def _synthetic_image_bytes(seed: int) -> bytes:
    """A tiny real (decodable) PNG, distinct per seed. Used only when a
    test needs `--scan` to actually open and hash images -- never real
    dataset photographs."""
    import io
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(seed)
    arr = (rng.rand(16, 16, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()

FIELDS = ("species", "slug", "taxon_id", "split", "observation_uuid", "photo_id", "sha256")
DEFAULT_SLUGS = ("fx-species-a", "fx-species-b")


def _write_csv(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path.read_bytes()


def _final_test_row(i: int) -> dict:
    return {
        "species": "Fixture Species X", "slug": "fx-final-test-species", "taxon_id": "1000",
        "split": "final_test", "observation_uuid": f"ft-uuid-{i}",
        "photo_id": str(9000 + i),
        "sha256": f"{_deterministic_digest('final_test', str(i)) % (1 << 256):064x}",
    }


def _deterministic_digest(*parts: str) -> int:
    """A REAL sha256 digest (never Python's built-in hash(), which is
    randomized per process via PYTHONHASHSEED and would make fixture
    photo_ids/sha256s flaky across runs)."""
    return int(hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest(), 16)


def _expansion_row(slug: str, split: str, i: int) -> dict:
    bucket_offset = _deterministic_digest(slug, split) % 100000
    return {
        "species": slug.replace("-", " ").title(), "slug": slug, "taxon_id": "2000",
        "split": split, "observation_uuid": f"{slug}-{split}-uuid-{i}",
        "photo_id": str(2_000_000 + bucket_offset * 100 + i),
        "sha256": f"{_deterministic_digest(slug, split, str(i)) % (1 << 256):064x}",
    }


def build_tiny_review_fixture(testcase: unittest.TestCase, repo: Path, *, slugs=DEFAULT_SLUGS,
                              n_final_test: int = 4, per_slug: int = 3, per_bucket: int = 6,
                              real_images: bool = False) -> dict:
    """Builds a tiny synthetic repo at `repo` with source manifests, a
    generated queue.csv + summary.json, and a hand-assembled (but
    structurally real) contract.json -- patches every `manual_review_
    contract` constant needed to make `mrc.load_and_verify_contract(repo)`
    succeed for the lifetime of `testcase`. Returns a dict with `repo`,
    `contract`, `queue_rows`, `ledger_path`.

    `real_images=True` (only needed by real --scan tests) writes actual
    decodable PNG bytes per row, with each row's `sha256` recomputed to
    match those real bytes, instead of empty placeholder files -- every
    other test only needs image PATHS to resolve, never opens them."""
    final_test_rows = [_final_test_row(i) for i in range(n_final_test)]
    expansion_rows = []
    for slug in slugs:
        for split in mrc.EXPANSION_SPLITS:
            for i in range(per_bucket):
                expansion_rows.append(_expansion_row(slug, split, i))

    FORCED_DUPLICATE_SEED = 9_999_999  # shared with benchmark_v1's row below

    if real_images:
        for i, row in enumerate(final_test_rows):
            row["sha256"] = mrc.sha256_bytes(_synthetic_image_bytes(1_000_000 + i))
        for i, row in enumerate(expansion_rows):
            row["sha256"] = mrc.sha256_bytes(_synthetic_image_bytes(2_000_000 + i))
        # Force one guaranteed perceptual duplicate across final_test and
        # the benchmark_v1 other-evidence set (identical bytes -> identical
        # hashes -> a candidate pair in a real STOP_BEFORE_INFERENCE domain
        # any real --scan test can rely on existing). Deliberately NOT an
        # expansion row -- that would collide with the queue generator's
        # own cross-queue duplicate-sha256 rejection, since expansion rows
        # (unlike other-evidence rows) can be sampled into the queue.
        final_test_rows[0]["sha256"] = mrc.sha256_bytes(_synthetic_image_bytes(FORCED_DUPLICATE_SEED))

    ft_bytes = _write_csv(repo / mrc.FINAL_TEST_MANIFEST_REL_PATH, final_test_rows)
    exp_bytes = _write_csv(repo / mrc.EXPANSION_MANIFEST_REL_PATH, expansion_rows)

    n_species = len(slugs)
    n_sampled = n_species * per_slug * len(mrc.EXPANSION_SPLITS)
    n_total = n_final_test + n_sampled
    rows_per_session = n_total // 2

    const_patcher = mock.patch.multiple(
        mrc,
        APPROVED_SPECIES_SLUGS=tuple(slugs), N_NEW_SPECIES=n_species, PER_SLUG_PER_SPLIT=per_slug,
        N_SAMPLED_ROWS=n_sampled, N_QUEUE_ROWS=n_total, ROWS_PER_SUGGESTED_SESSION=rows_per_session,
        EXPANSION_SCAN_SPLIT_ROW_COUNTS={mrc.EXPANSION_TRAIN_SPLIT: n_species * per_bucket,
                                        mrc.EXPANSION_DEV_SPLIT: n_species * per_bucket},
    )
    const_patcher.start()
    testcase.addCleanup(const_patcher.stop)

    frozen_manifests_patcher = mock.patch.object(mrc, "FROZEN_SOURCE_MANIFESTS", {
        mrc.FINAL_TEST_DATASET: {"path": mrc.FINAL_TEST_MANIFEST_REL_PATH,
                                 "byte_sha256": mrc.sha256_bytes(ft_bytes), "row_count": n_final_test},
        mrc.EXPANSION_DATASET: {"path": mrc.EXPANSION_MANIFEST_REL_PATH,
                                "byte_sha256": mrc.sha256_bytes(exp_bytes), "row_count": len(expansion_rows)},
    })
    frozen_manifests_patcher.start()
    testcase.addCleanup(frozen_manifests_patcher.stop)

    # The 5 "other evidence" datasets: a tiny synthetic manifest + JSON
    # provenance sidecar per dataset, contract-bound exactly like the real
    # frozen ones so load_and_verify_contract() succeeds against this
    # fixture too.
    other_evidence_manifests = {}
    other_evidence_rows_by_name = {}
    for oe_index, name in enumerate(("benchmark_v1", "calibration_v1", "unknown_test_v1",
                                     "calibration_v2", "unknown_test_v2")):
        if real_images and name == "benchmark_v1":
            oe_seed = FORCED_DUPLICATE_SEED
        else:
            oe_seed = 3_000_000 + oe_index
        sha256 = (mrc.sha256_bytes(_synthetic_image_bytes(oe_seed)) if real_images
                 else f"{_deterministic_digest(name, '0') % (1 << 256):064x}")
        rows = [{
            "species": "Fixture Species Y", "slug": "fx-other-evidence-species", "taxon_id": "3000",
            "split": "known", "observation_uuid": f"{name}-uuid-0", "photo_id": "5000",
            "sha256": sha256,
        }]
        other_evidence_rows_by_name[name] = rows
        csv_rel_path = f"data/{name}/{name}.csv"
        json_rel_path = f"data/{name}/{name}.json"
        csv_bytes = _write_csv(repo / csv_rel_path, rows)
        json_path = repo / json_rel_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_bytes = json.dumps({"dataset": name}, sort_keys=True).encode("utf-8")
        json_path.write_bytes(json_bytes)
        other_evidence_manifests[name] = {
            "path": csv_rel_path, "byte_sha256": mrc.sha256_bytes(csv_bytes), "row_count": len(rows),
            "provenance_json_path": json_rel_path, "provenance_json_byte_sha256": mrc.sha256_bytes(json_bytes),
        }
    other_evidence_patcher = mock.patch.object(mrc, "OTHER_EVIDENCE_MANIFESTS", other_evidence_manifests)
    other_evidence_patcher.start()
    testcase.addCleanup(other_evidence_patcher.stop)

    args = argparse.Namespace(repo=repo)
    gq.cmd_write(args)

    # Image files: empty placeholders by default (never opened by these
    # tests -- image_path_for_row/resolve_image_path only check existence);
    # real decodable bytes matching each row's (possibly recomputed above)
    # sha256 when real_images=True, for real --scan tests.
    for dataset, rows in ((mrc.FINAL_TEST_DATASET, final_test_rows), (mrc.EXPANSION_DATASET, expansion_rows)):
        for i, row in enumerate(rows):
            image_path = repo / "data" / dataset / "clean" / row["slug"] / f"{row['photo_id']}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if real_images:
                if dataset == mrc.FINAL_TEST_DATASET and i == 0:
                    seed = FORCED_DUPLICATE_SEED  # forced duplicate of benchmark_v1's row -- see above
                else:
                    seed = (1_000_000 if dataset == mrc.FINAL_TEST_DATASET else 2_000_000) + i
                image_path.write_bytes(_synthetic_image_bytes(seed))
            else:
                image_path.touch()
    for oe_index, (name, rows) in enumerate(other_evidence_rows_by_name.items()):
        seed = FORCED_DUPLICATE_SEED if name == "benchmark_v1" else 3_000_000 + oe_index
        for row in rows:
            # Direct layout (no "clean" subdir) -- the 5 other-evidence
            # sets store images at data/{dataset}/{slug}/{photo_id}.ext.
            image_path = repo / "data" / name / row["slug"] / f"{row['photo_id']}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if real_images:
                image_path.write_bytes(_synthetic_image_bytes(seed))
            else:
                image_path.touch()

    queue_path = repo / mrc.QUEUE_ARTIFACT_REL_PATH
    summary_path = repo / mrc.SUMMARY_ARTIFACT_REL_PATH
    queue_bytes = queue_path.read_bytes()
    queue_rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    for row in queue_rows:
        row["queue_index"] = int(row["queue_index"])
        row["suggested_session"] = int(row["suggested_session"])
    queue_rows.sort(key=lambda r: r["queue_index"])
    identity_order_sha256 = mrc.compute_queue_identity_order_sha256(queue_rows)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    queue_artifact = {"path": mrc.QUEUE_ARTIFACT_REL_PATH, "byte_sha256": mrc.sha256_bytes(queue_bytes),
                      "row_count": len(queue_rows), "identity_order_sha256": identity_order_sha256}
    summary_artifact = {"path": mrc.SUMMARY_ARTIFACT_REL_PATH,
                        "byte_sha256": mrc.sha256_bytes(summary_path.read_bytes()),
                        "content_sha256": summary["content_sha256"]}
    implementation_sources = {name: {"path": f"training/{name}.py", "sha256": "a" * 64}
                              for name in ("manual_review_contract", "append_only_ledger",
                                          "generate_manual_review_queue", "freeze_manual_review_contract",
                                          "perceptual_hash", "manual_review_tool",
                                          "scan_perceptual_duplicates", "adjudicate_pairs",
                                          "finalize_stop_status")}

    content = mrc.build_contract_content(queue_artifact=queue_artifact, summary_artifact=summary_artifact,
                                         implementation_sources=implementation_sources)
    content_sha256 = mrc.compute_content_sha256(content)
    contract = {
        "schema_version": mrc.SCHEMA_VERSION, "status": mrc.CONTRACT_STATUS_FROZEN,
        "content": content, "content_sha256": content_sha256,
        "generation": {"generator": "test fixture"},
    }
    contract_path = repo / mrc.CONTRACT_REL_PATH
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"repo": repo, "contract": contract, "queue_rows": queue_rows,
            "ledger_path": repo / mrc.APPROVED_OUTPUT_PATHS["manual_review_ledger"]}
