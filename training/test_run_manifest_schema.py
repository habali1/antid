#!/usr/bin/env python3
"""test_run_manifest_schema.py — writer/reader CONTRACT tests for the one
shared run_manifest.json schema (run_manifest_schema.py), used by both
train.py (writer) and evaluate.py (reader). Every test here calls the same
shared validate_run_manifest() -- never a reimplementation -- so a
writer/reader drift would show up here first.

Run directly: python test_run_manifest_schema.py
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_manifest_schema as rms  # noqa: E402


def _sha(c: str) -> str:
    return c * 64


def _valid_manifest_section() -> dict:
    return {"path": "/data/manifest.csv", "sha256": _sha("a"), "rows": 13581}


def _valid_taxonomy_section() -> dict:
    return {"path": "/data/taxonomy.json", "sha256": _sha("b"), "num_classes": 65}


def _valid_val_split_section() -> dict:
    return {"path": "/art/val_split.json", "sha256": _sha("c"),
           "n_train": 10985, "n_val": 2596, "n_total": 13581}


def _valid_best_section(epoch: int = 3) -> dict:
    return {"epoch": epoch, "metrics": {"top1": 0.61, "top3": 0.83},
           "filename": f"checkpoint_best_epoch_{epoch:03d}.pth", "sha256": _sha("d")}


def _valid_final_artifact_hashes() -> dict:
    return {name: _sha(str(i % 10)) for i, name in enumerate(rms.FINAL_ARTIFACT_NAMES)}


def _base_manifest(status: str = "initialized") -> dict:
    return {
        "run_manifest_schema_version": rms.RUN_MANIFEST_SCHEMA_VERSION,
        "status": status, "run_kind": "full",
        "git_head": "abc123", "git_dirty": False,
        "invocations": [{"timestamp_utc": "2026-01-01T00:00:00Z", "argv": [],
                         "resume": False, "pause_after_epoch": None}],
        "started_at_utc": "2026-01-01T00:00:00Z", "updated_at_utc": "2026-01-01T00:00:00Z",
        "finished_at_utc": None, "final_artifact_hashes": None,
    }


def _initialized_manifest() -> dict:
    return _base_manifest("initialized")


def _running_manifest() -> dict:
    m = _base_manifest("running")
    m["manifest"] = _valid_manifest_section()
    m["taxonomy_source"] = _valid_taxonomy_section()
    m["val_split"] = _valid_val_split_section()
    m["last_completed_epoch"] = 2
    m["best"] = _valid_best_section(epoch=1)
    return m


def _paused_for_smoke_manifest() -> dict:
    m = _running_manifest()
    m["status"] = "paused_for_smoke"
    m["run_kind"] = "smoke"
    return m


def _completed_manifest() -> dict:
    m = _running_manifest()
    m["status"] = "completed"
    m["finished_at_utc"] = "2026-01-02T00:00:00Z"
    m["final_artifact_hashes"] = _valid_final_artifact_hashes()
    return m


class TestValidInitializedRunning(unittest.TestCase):
    def test_valid_initialized_manifest(self):
        rms.validate_run_manifest(_initialized_manifest(), stage="initialized")  # no raise

    def test_valid_running_manifest(self):
        rms.validate_run_manifest(_running_manifest(), stage="epoch_committed")  # no raise

    def test_running_manifest_also_valid_at_any_stage(self):
        rms.validate_run_manifest(_running_manifest(), stage="any")  # no raise


class TestValidPausedForSmoke(unittest.TestCase):
    def test_valid_paused_for_smoke_manifest(self):
        rms.validate_run_manifest(_paused_for_smoke_manifest(), stage="epoch_committed")

    def test_paused_for_smoke_is_resumable(self):
        self.assertIn("paused_for_smoke", rms.RESUMABLE_STATUSES)

    def test_completed_is_not_resumable(self):
        self.assertNotIn("completed", rms.RESUMABLE_STATUSES)


class TestValidCompleted(unittest.TestCase):
    def test_valid_completed_manifest(self):
        rms.validate_run_manifest(_completed_manifest(), stage="completed")

    def test_completed_manifest_also_valid_at_any_stage(self):
        rms.validate_run_manifest(_completed_manifest(), stage="any")


class TestMissingRequiredFields(unittest.TestCase):
    def _assert_missing_fails(self, manifest: dict, stage: str, key_path: str):
        m = copy.deepcopy(manifest)
        obj = m
        parts = key_path.split(".")
        for p in parts[:-1]:
            obj = obj[p]
        del obj[parts[-1]]
        with self.assertRaises(rms.RunManifestValidationError, msg=key_path):
            rms.validate_run_manifest(m, stage=stage)

    def test_missing_schema_version(self):
        self._assert_missing_fails(_initialized_manifest(), "initialized", "run_manifest_schema_version")

    def test_missing_status(self):
        self._assert_missing_fails(_initialized_manifest(), "initialized", "status")

    def test_missing_run_kind(self):
        self._assert_missing_fails(_initialized_manifest(), "initialized", "run_kind")

    def test_missing_invocations(self):
        self._assert_missing_fails(_initialized_manifest(), "initialized", "invocations")

    def test_missing_manifest_section(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "manifest")

    def test_missing_manifest_path(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "manifest.path")

    def test_missing_manifest_sha256(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "manifest.sha256")

    def test_missing_manifest_rows(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "manifest.rows")

    def test_missing_taxonomy_source_section(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "taxonomy_source")

    def test_missing_taxonomy_source_num_classes(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "taxonomy_source.num_classes")

    def test_missing_val_split_section(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "val_split")

    def test_missing_val_split_n_train(self):
        self._assert_missing_fails(_running_manifest(), "data_verified", "val_split.n_train")

    def test_missing_last_completed_epoch(self):
        self._assert_missing_fails(_running_manifest(), "epoch_committed", "last_completed_epoch")

    def test_missing_best_section(self):
        self._assert_missing_fails(_running_manifest(), "epoch_committed", "best")

    def test_missing_best_metrics(self):
        self._assert_missing_fails(_running_manifest(), "epoch_committed", "best.metrics")

    def test_missing_final_artifact_hashes_on_completed(self):
        m = _completed_manifest()
        m["final_artifact_hashes"] = None
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="completed")

    def test_completed_status_with_missing_final_artifact_hashes_via_any_stage(self):
        m = _completed_manifest()
        m["final_artifact_hashes"] = None
        with self.assertRaises(rms.RunManifestValidationError) as cm:
            rms.validate_run_manifest(m, stage="any")
        self.assertIn("final_artifact_hashes", str(cm.exception))

    def test_final_artifact_hashes_missing_one_entry(self):
        m = _completed_manifest()
        del m["final_artifact_hashes"][rms.FINAL_ARTIFACT_NAMES[0]]
        with self.assertRaises(rms.RunManifestValidationError) as cm:
            rms.validate_run_manifest(m, stage="completed")
        self.assertIn(rms.FINAL_ARTIFACT_NAMES[0], str(cm.exception))


class TestWrongFieldTypes(unittest.TestCase):
    def test_schema_version_as_bool_rejected(self):
        m = _initialized_manifest()
        m["run_manifest_schema_version"] = True  # bool is an int subclass -- must be rejected
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="initialized")

    def test_schema_version_wrong_number_rejected(self):
        m = _initialized_manifest()
        m["run_manifest_schema_version"] = 999
        with self.assertRaises(rms.RunManifestValidationError) as cm:
            rms.validate_run_manifest(m, stage="initialized")
        self.assertIn("unsupported", str(cm.exception))

    def test_status_invalid_value_rejected(self):
        m = _initialized_manifest()
        m["status"] = "not_a_real_status"
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="initialized")

    def test_run_kind_invalid_value_rejected(self):
        m = _initialized_manifest()
        m["run_kind"] = "production"  # only "full" or "smoke" are valid
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="initialized")

    def test_git_dirty_as_string_rejected(self):
        m = _initialized_manifest()
        m["git_dirty"] = "false"  # must be a real bool or null, not a string
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="initialized")

    def test_manifest_rows_as_bool_rejected(self):
        m = _running_manifest()
        m["manifest"]["rows"] = True
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="data_verified")

    def test_manifest_rows_as_float_rejected(self):
        m = _running_manifest()
        m["manifest"]["rows"] = 13581.0
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="data_verified")

    def test_val_split_n_total_mismatch_rejected(self):
        m = _running_manifest()
        m["val_split"]["n_total"] = 99999
        with self.assertRaises(rms.RunManifestValidationError) as cm:
            rms.validate_run_manifest(m, stage="data_verified")
        self.assertIn("n_total", str(cm.exception))

    def test_last_completed_epoch_as_bool_rejected(self):
        m = _running_manifest()
        m["last_completed_epoch"] = True
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="epoch_committed")

    def test_last_completed_epoch_negative_rejected(self):
        m = _running_manifest()
        m["last_completed_epoch"] = -1
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="epoch_committed")

    def test_best_metrics_top1_as_bool_rejected(self):
        m = _running_manifest()
        m["best"]["metrics"]["top1"] = True
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="epoch_committed")


class TestMalformedHashes(unittest.TestCase):
    def test_manifest_sha256_too_short_rejected(self):
        m = _running_manifest()
        m["manifest"]["sha256"] = "abc123"
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="data_verified")

    def test_manifest_sha256_uppercase_rejected(self):
        m = _running_manifest()
        m["manifest"]["sha256"] = "A" * 64
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="data_verified")

    def test_manifest_sha256_non_hex_char_rejected(self):
        m = _running_manifest()
        m["manifest"]["sha256"] = "g" * 64
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="data_verified")

    def test_val_split_sha256_malformed_rejected(self):
        m = _running_manifest()
        m["val_split"]["sha256"] = "not-a-hash"
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="data_verified")

    def test_best_sha256_malformed_rejected(self):
        m = _running_manifest()
        m["best"]["sha256"] = "z" * 64
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="epoch_committed")

    def test_final_artifact_hash_malformed_rejected(self):
        m = _completed_manifest()
        m["final_artifact_hashes"][rms.FINAL_ARTIFACT_NAMES[0]] = "short"
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="completed")


class TestNullableExplicitSourceFields(unittest.TestCase):
    """manifest/taxonomy_source may be null (a legacy/non-explicit-source
    run), but the KEY must still be present, and val_split is never null."""

    def test_null_manifest_and_taxonomy_source_accepted(self):
        m = _running_manifest()
        m["manifest"] = None
        m["taxonomy_source"] = None
        rms.validate_run_manifest(m, stage="data_verified")  # must not raise

    def test_null_val_split_rejected(self):
        m = _running_manifest()
        m["val_split"] = None
        with self.assertRaises(rms.RunManifestValidationError):
            rms.validate_run_manifest(m, stage="data_verified")


if __name__ == "__main__":
    unittest.main()
