#!/usr/bin/env python3
"""test_scrape_gate_v2.py — unit/integration tests for scrape_gate_v2.py.

All synthetic/offline: no network access, no real image download. Download-
mode tests use small monkeypatched quota scenarios and a FakeImageClient
that never touches the network.

Run directly: python test_scrape_gate_v2.py
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PIL import Image

import scrape_gate_v2 as sg  # noqa: E402
import audit_gate_v2_readiness as g  # noqa: E402


def _solid_png_bytes(rgb: tuple[int, int, int], size: int = 220) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, format="PNG")
    return buf.getvalue()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fieldnames})


CANDIDATE_FIELDS = [
    "category", "intended_dataset", "selection_status", "species", "slug", "taxon_id",
    "observation_id", "observation_uuid", "photo_id", "source_url", "sha256", "created_at",
    "photo_license", "photo_attribution", "lat", "lon", "provenance_source",
    "origin_dataset", "origin_csv_sha256", "origin_row_number", "origin_photo_id",
    "origin_observation_uuid", "origin_sha256", "reuse_eligibility_reason",
    "rejection_reason", "source_taxon_id", "source_taxon_name", "source_taxon_rank",
]


def _fresh_row(category, intended_dataset, status, slug, taxon_id, photo_id, uuid,
              license_code="cc0", url=None, attribution="attr",
              source_taxon_id=None, source_taxon_name=None, source_taxon_rank="species"):
    species = slug.replace("-", " ").capitalize()
    return {
        "category": category, "intended_dataset": intended_dataset, "selection_status": status,
        "species": species, "slug": slug, "taxon_id": str(taxon_id),
        "observation_id": str(1000 + int(photo_id)), "observation_uuid": uuid,
        "photo_id": str(photo_id), "source_url": url or f"http://x/square.jpg?{photo_id}",
        "sha256": "", "created_at": "2026-01-01", "photo_license": license_code,
        "photo_attribution": attribution, "lat": "", "lon": "",
        "provenance_source": "inat_api_gate_v2_readiness",
        "origin_dataset": "", "origin_csv_sha256": "", "origin_row_number": "",
        "origin_photo_id": "", "origin_observation_uuid": "", "origin_sha256": "",
        "reuse_eligibility_reason": "", "rejection_reason": "",
        "source_taxon_id": str(source_taxon_id) if source_taxon_id is not None else str(taxon_id),
        "source_taxon_name": source_taxon_name if source_taxon_name is not None else species,
        "source_taxon_rank": source_taxon_rank,
    }


def argparse_ns(**kwargs):
    import argparse
    return argparse.Namespace(**kwargs)


class FakeImageClient:
    """Same .get_bytes(url)/.get_json(url, params) interface as
    PacedImageClient -- serves fixed bytes keyed by URL, never touches the
    network. Raises if asked for a URL/call it wasn't told about (catches
    accidental extra network calls)."""

    def __init__(self, bytes_by_url: dict[str, bytes] | None = None, json_by_call=None):
        self.bytes_by_url = bytes_by_url or {}
        self.json_by_call = json_by_call
        self.calls: list[str] = []
        self.request_count = 0

    def get_bytes(self, url: str) -> bytes | None:
        self.calls.append(("get_bytes", url))
        self.request_count += 1
        return self.bytes_by_url.get(url)

    def get_json(self, url, params=None):
        self.calls.append(("get_json", url, params))
        self.request_count += 1
        if self.json_by_call is not None:
            return self.json_by_call(url, params)
        raise AssertionError("get_json should not be called in this test")

    def close(self):
        pass


class RaisingClient:
    """Fails the test immediately if ANY network method is invoked."""

    def get_bytes(self, url):
        raise AssertionError(f"unexpected network get_bytes: {url}")

    def get_json(self, url, params=None):
        raise AssertionError(f"unexpected network get_json: {url}")

    def close(self):
        pass


class ClientFactory:
    """Callable that mimics PacedImageClient(interval_seconds=...) but
    always returns the same pre-built fake/raising client instance."""

    def __init__(self, instance):
        self.instance = instance

    def __call__(self, *args, **kwargs):
        return self.instance


# =========================================================================
# Source-contract / hash-binding tests (unchanged surface, still exercised
# against load_and_verify_candidates)
# =========================================================================
class TestSourceCsvHashVerification(unittest.TestCase):
    def test_wrong_csv_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_csv(repo / "data/gate_v2_readiness/candidates.csv", CANDIDATE_FIELDS, [])
            g_json = {"candidates_csv": {"path": "x", "rows": 0, "sha256": "0" * 64}}
            sg.write_json(repo / "data/gate_v2_readiness/gate_v2_readiness.json", g_json)
            with self.assertRaisesRegex(sg.SourceContractError, "sha256"):
                sg.load_and_verify_candidates(repo)

    def test_readiness_json_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            csv_path = repo / "data/gate_v2_readiness/candidates.csv"
            _write_csv(csv_path, CANDIDATE_FIELDS, [])
            real_hash = sg.sha256_file(csv_path)
            with mock.patch.object(sg, "EXPECTED_CANDIDATES_SHA256", real_hash), \
                mock.patch.object(sg, "EXPECTED_CANDIDATES_ROWS", 0):
                sg.write_json(repo / "data/gate_v2_readiness/gate_v2_readiness.json",
                             {"candidates_csv": {"path": "x", "rows": 999, "sha256": real_hash}})
                with self.assertRaisesRegex(sg.SourceContractError, "does not match"):
                    sg.load_and_verify_candidates(repo)

    def test_json_embedding_candidate_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            csv_path = repo / "data/gate_v2_readiness/candidates.csv"
            _write_csv(csv_path, CANDIDATE_FIELDS, [])
            real_hash = sg.sha256_file(csv_path)
            with mock.patch.object(sg, "EXPECTED_CANDIDATES_SHA256", real_hash), \
                mock.patch.object(sg, "EXPECTED_CANDIDATES_ROWS", 0):
                sg.write_json(repo / "data/gate_v2_readiness/gate_v2_readiness.json",
                             {"candidates_csv": {"path": "x", "rows": 0, "sha256": real_hash},
                              "candidate_rows": []})
                with self.assertRaisesRegex(sg.SourceContractError, "candidate_rows"):
                    sg.load_and_verify_candidates(repo)


# =========================================================================
# Exclusion-source binding to Phase 5A (point 3)
# =========================================================================
class TestExclusionSourceBinding(unittest.TestCase):
    def _write_excl(self, path: Path, sha_values: list[str]) -> None:
        _write_csv(path, ["sha256"], [{"sha256": s} for s in sha_values])

    def test_blank_or_malformed_sha256_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            readiness = {"exclusion_sources": {}}
            for name, rel in sg.EXCLUSION_SOURCES.items():
                self._write_excl(repo / rel, ["a" * 64] if name != "benchmark_v1" else ["", "not-hex"])
                path = repo / rel
                readiness["exclusion_sources"][name] = {
                    "path": rel, "rows": len(sg.read_csv(path)), "sha256": sg.sha256_file(path)}
            with self.assertRaisesRegex(sg.SourceContractError, "blank/malformed"):
                sg.verify_exclusion_sources_match_readiness(repo, readiness)

    def test_changed_path_rows_or_hash_since_readiness_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            readiness = {"exclusion_sources": {}}
            for name, rel in sg.EXCLUSION_SOURCES.items():
                self._write_excl(repo / rel, ["a" * 64])
                path = repo / rel
                readiness["exclusion_sources"][name] = {
                    "path": rel, "rows": len(sg.read_csv(path)), "sha256": sg.sha256_file(path)}
            # Drift one source after the readiness snapshot was recorded.
            drifted = repo / sg.EXCLUSION_SOURCES["calibration_v1"]
            self._write_excl(drifted, ["a" * 64, "b" * 64])
            with self.assertRaisesRegex(sg.SourceContractError, "exclusion-source binding failure"):
                sg.verify_exclusion_sources_match_readiness(repo, readiness)

    def test_matching_sources_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            readiness = {"exclusion_sources": {}}
            for name, rel in sg.EXCLUSION_SOURCES.items():
                self._write_excl(repo / rel, ["a" * 64])
                path = repo / rel
                readiness["exclusion_sources"][name] = {
                    "path": rel, "rows": len(sg.read_csv(path)), "sha256": sg.sha256_file(path)}
            sets, provenance = sg.verify_exclusion_sources_match_readiness(repo, readiness)
            self.assertEqual(sets["calibration_v1"], {"a" * 64})
            self.assertEqual(provenance["calibration_v1"]["rows"], 1)


# =========================================================================
# Stray root-level manifest files from the earlier placement bug (point 10)
# =========================================================================
class TestStrayManifestFiles(unittest.TestCase):
    def test_stray_root_csv_blocks_preflight_and_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "data").mkdir(parents=True)
            (repo / "data/calibration_v2.csv").write_text("x")
            with self.assertRaisesRegex(sg.SourceContractError, "stray root-level"):
                sg.check_no_stray_manifest_files(repo)

    def test_no_stray_files_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            sg.check_no_stray_manifest_files(repo)  # must not raise


# =========================================================================
# Reused-row integrity, now including full decode/dimension validation
# before any network access (point 9)
# =========================================================================
class TestVerifyReusedRows(unittest.TestCase):
    def _base_row(self, slug="sp-a", photo_id="1", sha="", category="out_of_scope_ant"):
        return {"slug": slug, "category": category, "selection_status": "reused",
               "origin_photo_id": photo_id, "origin_sha256": sha}

    def test_undecodable_local_origin_is_source_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "data/calibration_v1/sp-a"
            root.mkdir(parents=True)
            garbage = b"not an image"
            (root / "1.jpg").write_bytes(garbage)
            digest = sg.sha256_bytes(garbage)
            rows = [self._base_row(sha=digest)]
            exclusion_hashes = {"benchmark_v1": set(), "calibration_v1": set(),
                               "unknown_test_v1": set(), "northeast_final_test_v1": set(),
                               "northeast_manifest": set()}
            with mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", {"out_of_scope_ant": 1}):
                with self.assertRaisesRegex(sg.SourceContractError, "failed validation|decode_failure"):
                    sg.verify_reused_rows(repo, rows, exclusion_hashes)

    def test_undersized_local_origin_is_source_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "data/calibration_v1/sp-a"
            root.mkdir(parents=True)
            small = _solid_png_bytes((1, 2, 3), size=100)
            (root / "1.png").write_bytes(small)
            digest = sg.sha256_bytes(small)
            rows = [self._base_row(sha=digest)]
            exclusion_hashes = {"benchmark_v1": set(), "calibration_v1": set(),
                               "unknown_test_v1": set(), "northeast_final_test_v1": set(),
                               "northeast_manifest": set()}
            with mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", {"out_of_scope_ant": 1}):
                with self.assertRaises(sg.SourceContractError):
                    sg.verify_reused_rows(repo, rows, exclusion_hashes)

    def test_valid_local_origin_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "data/calibration_v1/sp-a"
            root.mkdir(parents=True)
            good = _solid_png_bytes((9, 9, 9), size=220)
            (root / "1.png").write_bytes(good)
            digest = sg.sha256_bytes(good)
            rows = [self._base_row(sha=digest)]
            exclusion_hashes = {"benchmark_v1": set(), "calibration_v1": set(),
                               "unknown_test_v1": set(), "northeast_final_test_v1": set(),
                               "northeast_manifest": set()}
            with mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", {"out_of_scope_ant": 1}):
                report = sg.verify_reused_rows(repo, rows, exclusion_hashes)
            self.assertEqual(report["rows_verified"], 1)


class TestKnownHoldoutTaxonomicCanonicalization(unittest.TestCase):
    """Ancestry-verified parent-species canonicalization: a known_holdout
    row identified to a subspecies may be canonicalized to the parent
    species label ONLY when hash-bound lineage evidence proves the
    canonical taxon is in that subspecies' ancestry -- never via slug or
    name-prefix similarity alone."""

    def _write_taxonomy(self, repo: Path) -> None:
        taxonomy = {"0": {"species_name": "Alpha ant", "common_name": "x", "taxon_id": 100, "slug": "alpha-ant"}}
        (repo / "data/northeast_expansion_v1").mkdir(parents=True, exist_ok=True)
        (repo / sg.TAXONOMY_PATH).write_text(json.dumps(taxonomy), encoding="utf-8")

    def _write_lineage_evidence(self, repo: Path, *, target=100, source=999,
                               target_in_ancestor_ids=True, corrupt_hash=False,
                               source_name="Alpha ant subspecialis", source_rank="subspecies",
                               canonical_name="Alpha ant", canonical_rank="species",
                               evidence_policy="parent_species_collapse_v1") -> dict:
        evidence = {
            "policy": evidence_policy,
            "canonical_targets": {str(target): {"taxon_id": target, "name": canonical_name, "rank": canonical_rank}},
            "verified_source_taxa": {
                str(source): {"taxon_id": source, "name": source_name, "rank": source_rank,
                             "parent_id": target, "ancestor_ids": [1, 2, target] if target_in_ancestor_ids else [1, 2],
                             "canonical_target_taxon_id": target,
                             "target_in_ancestor_ids": target_in_ancestor_ids},
            },
        }
        without_hash = dict(evidence)
        evidence["evidence_sha256"] = sg.sha256_bytes(
            json.dumps(without_hash, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        if corrupt_hash:
            evidence["evidence_sha256"] = "0" * 64
        path = repo / "data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
        return {"path": "data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json",
               "sha256": sg.sha256_file(path)}

    def _readiness_with_lineage(self, lineage_ref, *, source_taxon_ids_verified=(999,),
                               affected_known_holdout_rows=1, affected_slug="alpha-ant",
                               canonical_taxon_id=100, policy="parent_species_collapse_v1"):
        return {"known_holdout_taxonomic_canonicalization": {
            "policy": policy, "lineage_evidence": lineage_ref,
            "source_taxon_ids_verified": list(source_taxon_ids_verified),
            "affected_known_holdout_rows": affected_known_holdout_rows,
            "affected_slug": affected_slug, "canonical_taxon_id": canonical_taxon_id,
        }}

    def _row(self, taxon_id, species, source_taxon_id, status="selected", photo_id="1",
            source_taxon_name="Alpha ant subspecialis", source_taxon_rank="subspecies"):
        return {"category": "known_holdout", "selection_status": status, "slug": "alpha-ant",
               "taxon_id": str(taxon_id), "species": species, "photo_id": photo_id,
               "source_taxon_id": str(source_taxon_id),
               "source_taxon_name": source_taxon_name, "source_taxon_rank": source_taxon_rank}

    def test_canonical_species_row_passes_with_no_lineage_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            rows = [self._row(100, "Alpha ant", 100)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                result = sg.verify_taxonomy_matches_candidates(repo, rows, {})
            self.assertIsNone(result["lineage_evidence"])

    def test_verified_descendant_subspecies_accepted_and_canonicalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999)]  # already canonicalized label + differing source
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                result = sg.verify_taxonomy_matches_candidates(repo, rows, readiness)
            self.assertIsNotNone(result["lineage_evidence"])

    def test_unrelated_same_slug_taxon_rejected(self):
        """A row whose label was (incorrectly) canonicalized but whose
        source taxon has NO recorded lineage evidence at all must be
        rejected -- slug agreement alone proves nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            rows = [self._row(100, "Alpha ant", 12345)]  # source 12345 never verified
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "no recorded lineage evidence"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, {})

    def test_missing_ancestry_evidence_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            readiness = self._readiness_with_lineage({"path": "data/gate_v2_readiness/does_not_exist.json",
                                                      "sha256": "0" * 64})
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaises(sg.SourceContractError):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_target_absent_from_ancestor_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo, target_in_ancestor_ids=False)
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "does not confirm"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_forged_target_in_ancestor_ids_true_rejected(self):
        """A stored target_in_ancestor_ids=True is never trusted on its own
        -- if ancestor_ids does not actually contain the canonical target,
        this must be rejected even though the boolean claims otherwise."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            path = repo / "data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json"
            evidence = {
                "policy": "parent_species_collapse_v1",
                "canonical_targets": {"100": {"taxon_id": 100, "name": "Alpha ant", "rank": "species"}},
                "verified_source_taxa": {
                    "999": {"taxon_id": 999, "name": "Alpha ant subspecialis", "rank": "subspecies",
                           "parent_id": 100, "ancestor_ids": [1, 2, 3],  # target 100 NOT present
                           "canonical_target_taxon_id": 100,
                           "target_in_ancestor_ids": True},  # forged
                },
            }
            without_hash = dict(evidence)
            evidence["evidence_sha256"] = sg.sha256_bytes(
                json.dumps(without_hash, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
            ref = {"path": "data/gate_v2_readiness/known_holdout_taxonomic_lineage_v1.json",
                  "sha256": sg.sha256_file(path)}
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "does not match recomputed"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_altered_evidence_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo, corrupt_hash=True)
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaises(sg.SourceContractError):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_evidence_taxon_id_mismatch_with_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999, source_taxon_name="A different name")]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "source_taxon_name"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_evidence_rank_mismatch_with_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999, source_taxon_rank="variety")]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "source_taxon_rank"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_canonical_target_name_mismatch_with_taxonomy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo, canonical_name="Wrong canonical name")
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "canonical_targets name"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_canonical_target_rank_not_species_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo, canonical_rank="genus")
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "canonical_targets rank"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_wrong_policy_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref, policy="something_else_v1")
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "unrecognized canonicalization policy"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_evidence_declares_wrong_policy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo, evidence_policy="something_else_v1")
            readiness = self._readiness_with_lineage(ref)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "declares policy"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_source_taxon_ids_verified_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref, source_taxon_ids_verified=(999, 42))
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "source_taxon_ids_verified"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_affected_row_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref, affected_known_holdout_rows=2)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "affected_known_holdout_rows"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_affected_slug_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref, affected_slug="wrong-slug")
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "affected_slug"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_ineligible_known_holdout_rows_are_label_checked_too(self):
        """The 65-species/candidate label check must cover ALL known_holdout
        rows, including ineligible ones -- the canonicalization pass
        rewrote every known_holdout row's label regardless of eligibility,
        so an ineligible row with a wrong label is still a real defect."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            rows = [
                self._row(100, "Alpha ant", 100, status="selected", photo_id="1"),
                self._row(999, "Wrong ant", 999, status="ineligible", photo_id="2"),
            ]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaises(sg.SourceContractError):
                    sg.verify_taxonomy_matches_candidates(repo, rows, {})

    def test_canonicalized_source_rows_counts_ineligible_rows_too(self):
        """A row canonicalized to the parent species but marked ineligible
        (e.g. for an unrelated license reason) still counts toward
        canonicalized_source_rows and the readiness cross-checks -- the
        affected-row scope is NOT limited to selected/reserve."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref, affected_known_holdout_rows=2)
            rows = [
                self._row(100, "Alpha ant", 999, status="selected", photo_id="1"),
                self._row(100, "Alpha ant", 999, status="ineligible", photo_id="2"),
            ]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                result = sg.verify_taxonomy_matches_candidates(repo, rows, readiness)
            self.assertEqual(result["canonicalized_source_rows"], 2)
            self.assertEqual(result["known_holdout_rows_taxonomically_checked"], 2)
            self.assertEqual(result["selected_or_reserve_known_holdout_rows"], 1)

    def test_actual_source_id_set_must_equal_evidence_and_readiness(self):
        """The distinct differing source_taxon_id set actually found among
        candidate rows must equal both the lineage evidence's own keys and
        readiness.source_taxon_ids_verified -- not just each row
        individually resolving to SOME evidence entry."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            # Evidence verifies taxon 999, but readiness (falsely) also
            # claims 4242 was verified -- 4242 never appears in any actual
            # candidate row, so the actual/declared sets disagree.
            ref = self._write_lineage_evidence(repo)
            evidence_path = repo / ref["path"]
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["verified_source_taxa"]["4242"] = dict(evidence["verified_source_taxa"]["999"])
            evidence["verified_source_taxa"]["4242"]["taxon_id"] = 4242
            without_hash = {k: v for k, v in evidence.items() if k != "evidence_sha256"}
            evidence["evidence_sha256"] = sg.sha256_bytes(
                json.dumps(without_hash, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
            ref["sha256"] = sg.sha256_file(evidence_path)
            readiness = self._readiness_with_lineage(ref, source_taxon_ids_verified=(999, 4242))
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "actual differing source_taxon_id set"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_canonical_taxon_id_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            ref = self._write_lineage_evidence(repo)
            readiness = self._readiness_with_lineage(ref, canonical_taxon_id=999999)
            rows = [self._row(100, "Alpha ant", 999)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaisesRegex(sg.SourceContractError, "canonical_taxon_id"):
                    sg.verify_taxonomy_matches_candidates(repo, rows, readiness)

    def test_later_row_mismatch_detected_not_only_first_row(self):
        """Regression for the exact bug found in production: the function
        used to remember only the first candidate row seen per slug."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            rows = [
                self._row(100, "Alpha ant", 100, photo_id="1"),          # correct, seen first
                self._row(999, "Wrong ant", 999, photo_id="2"),          # wrong species, seen second
            ]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                with self.assertRaises(sg.SourceContractError):
                    sg.verify_taxonomy_matches_candidates(repo, rows, {})

    def test_taxonomy_hash_bound_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_taxonomy(repo)
            rows = [self._row(100, "Alpha ant", 100)]
            with mock.patch.object(sg, "REQUIRED_TAXONOMY_SPECIES_COUNT", 1):
                result = sg.verify_taxonomy_matches_candidates(repo, rows, {})
            self.assertEqual(result["sha256"], sg.sha256_file(repo / sg.TAXONOMY_PATH))
            self.assertEqual(result["known_holdout_rows_taxonomically_checked"], 1)
            self.assertEqual(result["selected_or_reserve_known_holdout_rows"], 1)
            self.assertEqual(result["canonicalized_source_rows"], 0)


class TestDeterministicReconstruction(unittest.TestCase):
    def _rows_for_one_species(self, slug="alpha-ant", taxon_id=1, n=20):
        rows = []
        for i in range(n):
            rows.append(_fresh_row("known_holdout", "calibration_v2+unknown_test_v2",
                                   "pending", slug, taxon_id, 1000 + i, f"u{i}"))
        return rows

    def test_reserve_selection_deterministic_across_runs(self):
        rows = self._rows_for_one_species()
        cal1, rem1 = g.deterministic_select(rows, "calibration_v2_known", 5, key=sg._key)
        cal2, rem2 = g.deterministic_select(rows, "calibration_v2_known", 5, key=sg._key)
        self.assertEqual([sg._key(r) for r in cal1], [sg._key(r) for r in cal2])
        self.assertEqual([sg._key(r) for r in rem1], [sg._key(r) for r in rem2])

    def test_key_sorts_photo_id_numerically_not_lexically(self):
        rows = [_fresh_row("known_holdout", "x", "pending", "s", 1, pid, f"u{pid}")
               for pid in (2, 10, 1, 9)]
        ordered = sorted(rows, key=sg._key)
        self.assertEqual([r["photo_id"] for r in ordered], ["1", "2", "9", "10"])

    def test_known_holdout_plan_reproduces_committed_csv(self):
        rows = self._rows_for_one_species(n=20)
        cal, remainder = g.deterministic_select(rows, "calibration_v2_known", 10, key=sg._key)
        ut, reserve = g.deterministic_select(remainder, "unknown_test_v2_known", 6, key=sg._key)
        for r in cal:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        for r in ut:
            r["selection_status"], r["intended_dataset"] = "selected", "unknown_test_v2"
        for r in reserve:
            r["selection_status"] = "reserve"
        plan = sg.reconstruct_known_holdout_plan(rows, ["alpha-ant"])
        self.assertEqual(len(plan["alpha-ant"]["calibration_v2"]), 10)
        self.assertEqual(len(plan["alpha-ant"]["unknown_test_v2"]), 6)
        self.assertEqual(len(plan["alpha-ant"]["reserve_queue"]), 4)

    def test_disagreement_with_csv_raises(self):
        rows = self._rows_for_one_species(n=20)
        for r in rows:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        with self.assertRaises(sg.SourceContractError):
            sg.reconstruct_known_holdout_plan(rows, ["alpha-ant"])

    def test_shared_pool_calibration_selected_before_unknown_test(self):
        rows = [_fresh_row("non_ant_insect", "x", "pending", f"sp-{i}", i, 2000 + i, f"v{i}")
               for i in range(20)]
        cal_selected, remainder = g.deterministic_select(
            rows, "calibration_v2_non_ant_insect", 6, key=sg._key)
        ut_selected, reserve = g.deterministic_select(
            remainder, "unknown_test_v2_non_ant_insect", 8, key=sg._key)
        cal_ids = {sg._key(r) for r in cal_selected}
        ut_ids = {sg._key(r) for r in ut_selected}
        self.assertEqual(cal_ids & ut_ids, set())
        self.assertEqual(len(reserve), 6)


class TestValidateAndDecode(unittest.TestCase):
    def test_valid_image_accepted(self):
        ok, reason, w, h = sg.validate_and_decode(_solid_png_bytes((10, 20, 30), size=220))
        self.assertTrue(ok)
        self.assertEqual((w, h), (220, 220))

    def test_garbage_bytes_decode_failure(self):
        ok, reason, w, h = sg.validate_and_decode(b"not an image at all")
        self.assertFalse(ok)
        self.assertEqual(reason, "decode_failure")

    def test_under_200px_rejected(self):
        ok, reason, w, h = sg.validate_and_decode(_solid_png_bytes((1, 2, 3), size=150))
        self.assertFalse(ok)
        self.assertEqual(reason, "under_200px")


# =========================================================================
# Cache: atomic writes, corrupt/partial-file recovery (point 5)
# =========================================================================
class TestCacheReuse(unittest.TestCase):
    def test_cached_file_used_without_network_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data = _solid_png_bytes((5, 5, 5))
            sg.write_to_cache(cache_dir, "42", "png", data)
            found = sg.find_cached(cache_dir, "42")
            self.assertIsNotNone(found)
            self.assertEqual(found.read_bytes(), data)

    def test_missing_from_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(sg.find_cached(Path(tmp), "999"))

    def test_multiple_extensions_for_one_photo_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "7.jpg").write_bytes(_solid_png_bytes((1, 1, 1)))
            (cache_dir / "7.png").write_bytes(_solid_png_bytes((2, 2, 2)))
            with self.assertRaises(sg.SourceContractError):
                sg.find_cached(cache_dir, "7")

    def test_corrupt_cached_file_is_removed_and_allows_one_refetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "8.jpg").write_bytes(b"truncated garbage")
            self.assertIsNone(sg.find_cached(cache_dir, "8"))
            self.assertFalse((cache_dir / "8.jpg").exists())  # poisoned entry removed

    def test_write_to_cache_never_leaves_a_bare_part_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data = _solid_png_bytes((3, 3, 3))
            final = sg.write_to_cache(cache_dir, "9", "png", data)
            self.assertTrue(final.exists())
            self.assertFalse((cache_dir / "9.png.part").exists())


class TestOverlapGatesViaRunDownloadPlan(unittest.TestCase):
    """Exercises run_download_plan's per-image gates directly through a
    minimal single-category plan, with a FakeImageClient (no network)."""

    def _minimal_plan_and_rows(self):
        rows = [_fresh_row("out_of_scope_ant", "unknown_test_v2", "pending",
                           "sp-a", 1, pid, f"u{pid}")
               for pid in (1, 2, 3)]
        selected, reserve = rows[:1], rows[1:]
        for r in selected:
            r["selection_status"] = "selected"
        for r in reserve:
            r["selection_status"] = "reserve"
        plan = {
            "supported_slugs": [],
            "known_holdout": {},
            "out_of_scope_ant": {
                "calibration_v2": {"selected": [], "reserve_queue": []},
                "unknown_test_v2": {"selected": selected, "reserve_queue": reserve},
            },
            "non_ant_insect": {"calibration_v2": {"selected": [], "reserve_queue": []},
                              "unknown_test_v2": {"selected": [], "reserve_queue": []},
                              "shared_reserve_queue": []},
            "unrelated": {"calibration_v2": {"selected": [], "reserve_queue": []},
                        "unknown_test_v2": {"selected": [], "reserve_queue": []},
                        "shared_reserve_queue": []},
        }
        return plan, rows

    def _zeroed_quotas(self, overrides: dict):
        zeroed = {k: 0 for k in sg.QUOTAS}
        zeroed.update(overrides)
        return zeroed

    def _no_exclusions(self):
        return {k: set() for k in
               ("benchmark_v1", "calibration_v1", "unknown_test_v1",
                "northeast_final_test_v1", "northeast_manifest")}

    def test_frozen_set_collision_rejected_and_reserve_used(self):
        plan, rows = self._minimal_plan_and_rows()
        img = _solid_png_bytes((1, 1, 1))
        frozen_hash = sg.sha256_bytes(img)
        exclusion_hashes = self._no_exclusions()
        exclusion_hashes["benchmark_v1"] = {frozen_hash}
        other_img = _solid_png_bytes((2, 2, 2))
        client = FakeImageClient({
            sg.to_large_url(rows[0]["source_url"]): img,
            sg.to_large_url(rows[1]["source_url"]): other_img,
        })
        quotas = self._zeroed_quotas({("unknown_test_v2", "out_of_scope_ant"): 1})
        reused = {k: 0 for k in sg.EXPECTED_REUSED_COUNTS}
        with mock.patch.object(sg, "QUOTAS", quotas), \
            mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", reused), \
            tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            result = sg.run_download_plan(Path(tmp), plan, rows, cache_dir, client, exclusion_hashes)
        self.assertTrue(result["success"])
        accepted = result["accepted"]["unknown_test_v2"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["sha256"], sg.sha256_bytes(other_img))
        reasons = [e["reason"] for e in result["attempt_log"] if e["outcome"] == "rejected"]
        self.assertIn("frozen_set_hash_collision", reasons)

    def test_shortfall_when_reserves_exhausted(self):
        plan, rows = self._minimal_plan_and_rows()
        exclusion_hashes = self._no_exclusions()
        client = FakeImageClient({})  # every download fails
        quotas = self._zeroed_quotas({("unknown_test_v2", "out_of_scope_ant"): 1})
        reused = {k: 0 for k in sg.EXPECTED_REUSED_COUNTS}
        with mock.patch.object(sg, "QUOTAS", quotas), \
            mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", reused), \
            tempfile.TemporaryDirectory() as tmp:
            result = sg.run_download_plan(Path(tmp), plan, rows, Path(tmp), client, exclusion_hashes)
        self.assertFalse(result["success"])
        self.assertTrue(result["shortfalls"])

    def test_internal_duplicate_within_dataset_rejected(self):
        plan, rows = self._minimal_plan_and_rows()
        exclusion_hashes = self._no_exclusions()
        same_img = _solid_png_bytes((9, 9, 9))
        selected2 = rows[:2]
        for r in selected2:
            r["selection_status"] = "selected"
        plan["out_of_scope_ant"]["unknown_test_v2"]["selected"] = selected2
        plan["out_of_scope_ant"]["unknown_test_v2"]["reserve_queue"] = rows[2:]
        client = FakeImageClient({
            sg.to_large_url(rows[0]["source_url"]): same_img,
            sg.to_large_url(rows[1]["source_url"]): same_img,
            sg.to_large_url(rows[2]["source_url"]): _solid_png_bytes((7, 7, 7)),
        })
        quotas = self._zeroed_quotas({("unknown_test_v2", "out_of_scope_ant"): 2})
        reused = {k: 0 for k in sg.EXPECTED_REUSED_COUNTS}
        with mock.patch.object(sg, "QUOTAS", quotas), \
            mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", reused), \
            tempfile.TemporaryDirectory() as tmp:
            result = sg.run_download_plan(Path(tmp), plan, rows, Path(tmp), client, exclusion_hashes)
        self.assertTrue(result["success"])
        reasons = [e["reason"] for e in result["attempt_log"] if e["outcome"] == "rejected"]
        self.assertIn("internal_hash_duplicate", reasons)

    def test_cross_v2_collision_explicit(self):
        """A photo already accepted into calibration_v2 must be rejected for
        unknown_test_v2 as cross_v2_hash_collision, never treated as a
        generic frozen-set collision or silently allowed twice."""
        img = _solid_png_bytes((4, 4, 4))
        cal_row = _fresh_row("out_of_scope_ant", "calibration_v2", "selected", "sp-cal", 1, 501, "ucal")
        ut_row = _fresh_row("out_of_scope_ant", "unknown_test_v2", "selected", "sp-ut", 2, 502, "uut")
        plan = {
            "supported_slugs": [], "known_holdout": {},
            "out_of_scope_ant": {
                "calibration_v2": {"selected": [cal_row], "reserve_queue": []},
                "unknown_test_v2": {"selected": [ut_row], "reserve_queue": []},
            },
            "non_ant_insect": {"calibration_v2": {"selected": [], "reserve_queue": []},
                              "unknown_test_v2": {"selected": [], "reserve_queue": []},
                              "shared_reserve_queue": []},
            "unrelated": {"calibration_v2": {"selected": [], "reserve_queue": []},
                        "unknown_test_v2": {"selected": [], "reserve_queue": []},
                        "shared_reserve_queue": []},
        }
        client = FakeImageClient({
            sg.to_large_url(cal_row["source_url"]): img,
            sg.to_large_url(ut_row["source_url"]): img,  # byte-identical to the calibration accept
        })
        quotas = self._zeroed_quotas({("calibration_v2", "out_of_scope_ant"): 1,
                                     ("unknown_test_v2", "out_of_scope_ant"): 1})
        reused = {k: 0 for k in sg.EXPECTED_REUSED_COUNTS}
        with mock.patch.object(sg, "QUOTAS", quotas), \
            mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", reused), \
            tempfile.TemporaryDirectory() as tmp:
            result = sg.run_download_plan(Path(tmp), plan, [cal_row, ut_row], Path(tmp),
                                          client, self._no_exclusions())
        self.assertEqual(len(result["accepted"]["calibration_v2"]), 1)
        self.assertEqual(len(result["accepted"]["unknown_test_v2"]), 0)
        reasons = [e["reason"] for e in result["attempt_log"] if e["outcome"] == "rejected"]
        self.assertIn("cross_v2_hash_collision", reasons)

    def test_shared_reserve_consumed_only_once(self):
        """calibration draws first from the shared reserve; whatever it
        consumes or rejects must be permanently gone by the time
        unknown_test_v2 runs against the SAME queue object."""
        cal_row = _fresh_row("non_ant_insect", "calibration_v2", "selected", "sp-1", 1, 601, "u1")
        reserve_row = _fresh_row("non_ant_insect", "x", "reserve", "sp-2", 2, 602, "u2")
        ut_row = _fresh_row("non_ant_insect", "unknown_test_v2", "selected", "sp-3", 3, 603, "u3")
        shared_queue = [reserve_row]
        plan = {
            "supported_slugs": [], "known_holdout": {},
            "out_of_scope_ant": {"calibration_v2": {"selected": [], "reserve_queue": []},
                                "unknown_test_v2": {"selected": [], "reserve_queue": []}},
            "non_ant_insect": {
                "calibration_v2": {"selected": [cal_row], "reserve_queue": []},
                "unknown_test_v2": {"selected": [ut_row], "reserve_queue": []},
                "shared_reserve_queue": shared_queue,
            },
            "unrelated": {"calibration_v2": {"selected": [], "reserve_queue": []},
                        "unknown_test_v2": {"selected": [], "reserve_queue": []},
                        "shared_reserve_queue": []},
        }
        # calibration's primary selected row FAILS to download, forcing it
        # to consume the one shared reserve row. unknown_test_v2's primary
        # selected row succeeds, so it should never touch the reserve at all
        # -- but we assert the reserve queue ends up empty either way,
        # proving it was drained (not independently copied).
        client = FakeImageClient({
            sg.to_large_url(reserve_row["source_url"]): _solid_png_bytes((6, 6, 6)),
            sg.to_large_url(ut_row["source_url"]): _solid_png_bytes((8, 8, 8)),
        })
        quotas = self._zeroed_quotas({("calibration_v2", "non_ant_insect"): 1,
                                     ("unknown_test_v2", "non_ant_insect"): 1})
        reused = {k: 0 for k in sg.EXPECTED_REUSED_COUNTS}
        with mock.patch.object(sg, "QUOTAS", quotas), \
            mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", reused), \
            tempfile.TemporaryDirectory() as tmp:
            result = sg.run_download_plan(Path(tmp), plan, [cal_row, reserve_row, ut_row],
                                          Path(tmp), client, self._no_exclusions())
        self.assertEqual(len(result["accepted"]["calibration_v2"]), 1)
        self.assertEqual(result["accepted"]["calibration_v2"][0]["photo_id"], "602")
        self.assertEqual(shared_queue, [])  # drained, never independently copied
        reasons_for_reserve = [e for e in result["attempt_log"] if e["photo_id"] == "602"]
        self.assertEqual(len(reasons_for_reserve), 1)  # attempted exactly once


# =========================================================================
# Byte accounting, per-dataset (point 2 / point 8) and reused rows in the
# attempt log (point 5)
# =========================================================================
class TestByteAccounting(unittest.TestCase):
    def test_network_bytes_tracked_per_dataset_cache_and_reuse_are_zero(self):
        cal_row = _fresh_row("out_of_scope_ant", "calibration_v2", "selected", "sp-1", 1, 701, "u1")
        ut_row = _fresh_row("out_of_scope_ant", "unknown_test_v2", "selected", "sp-2", 2, 702, "u2")
        plan = {
            "supported_slugs": [], "known_holdout": {},
            "out_of_scope_ant": {
                "calibration_v2": {"selected": [cal_row], "reserve_queue": []},
                "unknown_test_v2": {"selected": [ut_row], "reserve_queue": []},
            },
            "non_ant_insect": {"calibration_v2": {"selected": [], "reserve_queue": []},
                              "unknown_test_v2": {"selected": [], "reserve_queue": []},
                              "shared_reserve_queue": []},
            "unrelated": {"calibration_v2": {"selected": [], "reserve_queue": []},
                        "unknown_test_v2": {"selected": [], "reserve_queue": []},
                        "shared_reserve_queue": []},
        }
        cal_bytes = _solid_png_bytes((1, 1, 1))
        ut_bytes = _solid_png_bytes((2, 2, 2))
        client = FakeImageClient({
            sg.to_large_url(cal_row["source_url"]): cal_bytes,
            sg.to_large_url(ut_row["source_url"]): ut_bytes,
        })
        quotas = {k: 0 for k in sg.QUOTAS}
        quotas.update({("calibration_v2", "out_of_scope_ant"): 1, ("unknown_test_v2", "out_of_scope_ant"): 1})
        reused = {k: 0 for k in sg.EXPECTED_REUSED_COUNTS}
        exclusions = {k: set() for k in ("benchmark_v1", "calibration_v1", "unknown_test_v1",
                                        "northeast_final_test_v1", "northeast_manifest")}
        with mock.patch.object(sg, "QUOTAS", quotas), mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", reused), \
            tempfile.TemporaryDirectory() as tmp:
            result = sg.run_download_plan(Path(tmp), plan, [cal_row, ut_row], Path(tmp), client, exclusions)
        counters = result["counters"]
        self.assertEqual(counters["calibration_v2"]["network_transferred_bytes"], len(cal_bytes))
        self.assertEqual(counters["unknown_test_v2"]["network_transferred_bytes"], len(ut_bytes))
        self.assertEqual(counters["calibration_v2"]["calibration_v1_reuse_count"], 0)
        self.assertEqual(counters["unknown_test_v2"]["calibration_v1_reuse_count"], 0)

    def test_reused_rows_appear_in_attempt_log_with_zero_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "data/calibration_v1/sp-a"
            root.mkdir(parents=True)
            img = _solid_png_bytes((5, 5, 5))
            (root / "10.png").write_bytes(img)
            digest = sg.sha256_bytes(img)
            reused_row = _fresh_row("out_of_scope_ant", "calibration_v2", "reused", "sp-a", 1, 10, "u10")
            reused_row["origin_photo_id"] = "10"
            reused_row["origin_sha256"] = digest
            plan = {
                "supported_slugs": [], "known_holdout": {},
                "out_of_scope_ant": {"calibration_v2": {"selected": [], "reserve_queue": []},
                                    "unknown_test_v2": {"selected": [], "reserve_queue": []}},
                "non_ant_insect": {"calibration_v2": {"selected": [], "reserve_queue": []},
                                  "unknown_test_v2": {"selected": [], "reserve_queue": []},
                                  "shared_reserve_queue": []},
                "unrelated": {"calibration_v2": {"selected": [], "reserve_queue": []},
                            "unknown_test_v2": {"selected": [], "reserve_queue": []},
                            "shared_reserve_queue": []},
            }
            quotas = {k: 0 for k in sg.QUOTAS}
            reused_counts = {k: 0 for k in sg.EXPECTED_REUSED_COUNTS}
            exclusions = {k: set() for k in ("benchmark_v1", "calibration_v1", "unknown_test_v1",
                                            "northeast_final_test_v1", "northeast_manifest")}
            with mock.patch.object(sg, "QUOTAS", quotas), mock.patch.object(sg, "EXPECTED_REUSED_COUNTS", reused_counts):
                result = sg.run_download_plan(repo, plan, [reused_row], Path(tmp) / "cache",
                                              FakeImageClient({}), exclusions)
            reuse_entries = [e for e in result["attempt_log"] if e["image_source"] == "calibration_v1_reuse"]
            self.assertEqual(len(reuse_entries), 1)
            self.assertEqual(reuse_entries[0]["transferred_bytes"], 0)
            self.assertEqual(reuse_entries[0]["outcome"], "accepted")
            self.assertEqual(result["counters"]["calibration_v2"]["calibration_v1_reuse_count"], 1)
            self.assertEqual(result["counters"]["calibration_v2"]["network_transferred_bytes"], 0)


# =========================================================================
# Non-frozen work reports: CSV comma-safety, bucket results on success too
# =========================================================================
class TestWorkReports(unittest.TestCase):
    def _fake_outcome(self, detail_with_comma="frozen_set_hash_collision,extra,info"):
        attempt_log = [{
            "dataset": "calibration_v2", "category": "out_of_scope_ant", "species": "",
            "photo_id": "1", "observation_uuid": "u1", "role": "selected", "outcome": "rejected",
            "reason": "frozen_set_hash_collision", "detail": detail_with_comma,
            "image_source": "network", "transferred_bytes": 123, "sha256": "a" * 64,
        }]
        counters = {"calibration_v2": sg._empty_dataset_counters(), "unknown_test_v2": sg._empty_dataset_counters()}
        bucket_results = [{"dataset": "calibration_v2", "category": "out_of_scope_ant", "species": "",
                          "target": 1, "achieved": 0, "shortfall": 1}]
        return {"success": False, "accepted": {"calibration_v2": [], "unknown_test_v2": []},
               "attempt_log": attempt_log, "shortfalls": bucket_results, "bucket_results": bucket_results,
               "counters": counters}

    def test_comma_in_detail_field_survives_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            outcome = self._fake_outcome()
            sg.write_work_reports(work_dir, outcome)
            with (work_dir / "download_attempts.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["detail"], "frozen_set_hash_collision,extra,info")

    def test_status_json_reports_bucket_results_and_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            outcome = self._fake_outcome()
            sg.write_work_reports(work_dir, outcome)
            status = json.loads((work_dir / "download_status.json").read_text(encoding="utf-8"))
            self.assertIn("bucket_results", status)
            self.assertEqual(status["bucket_results"][0]["target"], 1)
            self.assertIn("counters_by_dataset", status)
            self.assertIn("counters_overall", status)

    def test_shortfall_creates_work_reports_but_no_frozen_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            work_dir = repo / "data/gate_v2_work"
            outcome = self._fake_outcome()
            sg.write_work_reports(work_dir, outcome)
            self.assertTrue((work_dir / "download_attempts.csv").exists())
            self.assertTrue((work_dir / "download_status.json").exists())
            self.assertFalse((repo / "data/calibration_v2").exists())
            self.assertFalse((repo / "data/unknown_test_v2").exists())


# =========================================================================
# The shared final-pair validator (point 2), including the fix for point 1:
# a real exclusion_hashes map must be independently rechecked here.
# =========================================================================
def _manifest_row(category, species, slug, taxon_id, photo_id, uuid, sha,
                  provenance="inat_api_gate_v2_readiness"):
    row = {f: sg.NOT_APPLICABLE for f in sg.MANIFEST_REQUIRED_FIELDS}
    row.update(category=category, species=species, slug=slug, taxon_id=str(taxon_id),
              observation_id=str(1000 + photo_id), observation_uuid=uuid, photo_id=str(photo_id),
              source_url=f"http://x/large.jpg?{photo_id}", photo_license="cc0",
              photo_attribution="attr", sha256=sha, byte_size=1000, width=220, height=220,
              provenance_source=provenance)
    row["image_source"] = "network"
    row["candidate_source_url"] = row["source_url"]
    row["enriched_source_url"] = sg.NOT_APPLICABLE
    return row


TINY_QUOTAS = {
    ("calibration_v2", "known_holdout"): 1, ("unknown_test_v2", "known_holdout"): 1,
    ("calibration_v2", "out_of_scope_ant"): 1, ("calibration_v2", "non_ant_insect"): 1,
    ("calibration_v2", "unrelated"): 1,
    ("unknown_test_v2", "out_of_scope_ant"): 1, ("unknown_test_v2", "non_ant_insect"): 1,
    ("unknown_test_v2", "unrelated"): 1,
}
TINY_TOTALS = {"calibration_v2": 5, "unknown_test_v2": 5}
SUPPORTED_SLUGS = ["alpha-ant", "beta-ant"]


def _hash_for(n: int) -> str:
    return format(n, "x").rjust(64, "0")


def _valid_pair():
    cal, ut = [], []
    for i, slug in enumerate(SUPPORTED_SLUGS):
        cal.append(_manifest_row("known_holdout", slug.title(), slug, 100 + i, 1 + i, f"cal-k-{i}", _hash_for(i)))
        ut.append(_manifest_row("known_holdout", slug.title(), slug, 100 + i, 10 + i, f"ut-k-{i}", _hash_for(i + 5)))
    cal.append(_manifest_row("out_of_scope_ant", "Ood cal", "ood-cal-sp", 201, 20, "cal-ood", _hash_for(100)))
    ut.append(_manifest_row("out_of_scope_ant", "Ood ut", "ood-ut-sp", 202, 21, "ut-ood", _hash_for(101)))
    cal.append(_manifest_row("non_ant_insect", "Insect", "insect-sp", 300, 22, "cal-insect", _hash_for(102)))
    ut.append(_manifest_row("non_ant_insect", "Insect2", "insect-sp2", 301, 23, "ut-insect", _hash_for(103)))
    cal.append(_manifest_row("unrelated", "Unrel", "unrel-sp", 400, 24, "cal-unrel", _hash_for(104)))
    ut.append(_manifest_row("unrelated", "Unrel2", "unrel-sp2", 401, 25, "ut-unrel", _hash_for(105)))
    return cal, ut


class TestFinalPairValidator(unittest.TestCase):
    def _run(self, cal, ut, exclusion_hashes=None, ood_species=None, ood_taxa=None):
        with mock.patch.object(sg, "DATASET_TOTALS", TINY_TOTALS), mock.patch.object(sg, "QUOTAS", TINY_QUOTAS):
            sg.validate_final_pair(cal, ut, SUPPORTED_SLUGS, exclusion_hashes or {},
                                   ood_species or {"ood-cal-sp"}, ood_taxa or {201})

    def test_valid_pair_passes(self):
        cal, ut = _valid_pair()
        self._run(cal, ut)  # must not raise

    def test_wrong_total_rejected(self):
        cal, ut = _valid_pair()
        cal.pop()
        with self.assertRaisesRegex(sg.SourceContractError, "calibration_v2 total"):
            self._run(cal, ut)

    def test_known_holdout_per_species_count_enforced(self):
        cal, ut = _valid_pair()
        cal[0]["slug"], cal[0]["species"] = "beta-ant", "Beta ant"  # now alpha-ant has 0, beta-ant has 2
        with self.assertRaises(sg.SourceContractError):
            self._run(cal, ut)

    def test_missing_required_field_rejected(self):
        cal, ut = _valid_pair()
        cal[0]["photo_license"] = ""
        with self.assertRaises(sg.SourceContractError):
            self._run(cal, ut)

    def test_duplicate_photo_id_within_dataset_rejected(self):
        cal, ut = _valid_pair()
        cal[1]["photo_id"] = cal[0]["photo_id"]
        with self.assertRaisesRegex(sg.SourceContractError, "duplicate photo_id"):
            self._run(cal, ut)

    def test_cross_dataset_hash_overlap_rejected(self):
        cal, ut = _valid_pair()
        ut[0]["sha256"] = cal[0]["sha256"]
        with self.assertRaisesRegex(sg.SourceContractError, "sha256 overlap"):
            self._run(cal, ut)

    def test_ood_overlap_with_supported_species_rejected(self):
        cal, ut = _valid_pair()
        cal[2]["slug"] = "alpha-ant"  # the calibration OOD row now collides with a supported species
        with self.assertRaises(sg.SourceContractError):
            self._run(cal, ut)

    def test_unknown_test_ood_overlaps_calibration_ood_rejected(self):
        cal, ut = _valid_pair()
        ut[2]["slug"] = "ood-cal-sp"
        ut[2]["taxon_id"] = "201"
        with self.assertRaises(sg.SourceContractError):
            self._run(cal, ut)

    def test_fresh_hash_present_in_exclusion_set_rejected_even_if_download_stage_bypassed(self):
        """This is the point-1 regression test: publish() must independently
        recheck every fresh hash against the real exclusion map, even in a
        hypothetical where an earlier per-image download-stage check never
        ran (simulated here by calling validate_final_pair directly with a
        populated exclusion map and never invoking run_download_plan at
        all)."""
        cal, ut = _valid_pair()
        colliding_hash = cal[2]["sha256"]  # the fresh OOD row's accepted hash
        exclusion_hashes = {"benchmark_v1": {colliding_hash}, "calibration_v1": set(),
                           "unknown_test_v1": set(), "northeast_final_test_v1": set(),
                           "northeast_manifest": set()}
        with self.assertRaisesRegex(sg.SourceContractError, "fresh row hash present"):
            self._run(cal, ut, exclusion_hashes=exclusion_hashes)


# =========================================================================
# Source-taxon provenance must survive into both frozen manifests, byte for
# byte, through build_manifest_row / write_dataset_manifest / restore.
# =========================================================================
class TestSourceTaxonManifestProvenance(unittest.TestCase):
    def _accepted_descendant_row(self):
        """Simulates an accepted known_holdout row whose candidates.csv
        entry was already canonicalized (species/slug/taxon_id = the
        canonical parent species) while source_taxon_* still records the
        actual observed subspecies -- exactly the shape of the 26
        corrected eciton-burchellii rows."""
        row = _fresh_row("known_holdout", "calibration_v2", "selected", "eciton-burchellii", 126838,
                         1, "u1", source_taxon_id=735984,
                         source_taxon_name="Eciton burchellii parvispinum", source_taxon_rank="subspecies")
        row.update(sha256="a" * 64, byte_size=12345, width=220, height=220,
                  image_source="network", candidate_source_url=row["source_url"],
                  enriched_source_url="")
        return row

    def test_build_manifest_row_carries_source_taxon_fields(self):
        manifest_row = sg.build_manifest_row(self._accepted_descendant_row())
        self.assertEqual(manifest_row["species"], "Eciton burchellii")  # canonical label untouched
        self.assertEqual(manifest_row["slug"], "eciton-burchellii")
        self.assertEqual(manifest_row["taxon_id"], "126838")
        self.assertEqual(manifest_row["source_taxon_id"], "735984")
        self.assertEqual(manifest_row["source_taxon_name"], "Eciton burchellii parvispinum")
        self.assertEqual(manifest_row["source_taxon_rank"], "subspecies")

    def test_build_manifest_row_defaults_rank_sentinel_when_not_recorded(self):
        row = self._accepted_descendant_row()
        row["source_taxon_rank"] = ""
        manifest_row = sg.build_manifest_row(row)
        self.assertEqual(manifest_row["source_taxon_rank"], sg.SOURCE_TAXON_RANK_NOT_RECORDED)
        self.assertNotEqual(manifest_row["source_taxon_rank"], "")

    def test_written_manifest_csv_preserves_source_taxon_fields_byte_for_byte(self):
        accepted_row = self._accepted_descendant_row()
        manifest_row = sg.build_manifest_row(accepted_row)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            stage_dir = repo / "stage"
            stage_dir.mkdir()
            (repo / sg.READINESS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
            (repo / sg.READINESS_JSON_PATH).write_text("{}", encoding="utf-8")
            outcome = {"shortfalls": [], "attempt_log": [], "counters":
                      {"calibration_v2": sg._empty_dataset_counters(), "unknown_test_v2": sg._empty_dataset_counters()}}
            readiness = {}
            sg.write_dataset_manifest(repo, "calibration_v2", [manifest_row], stage_dir, readiness,
                                      "csvhash", {}, {"path": "x", "sha256": "y"}, stage_dir, outcome)
            with (stage_dir / "calibration_v2.csv").open(encoding="utf-8") as fh:
                written = list(csv.DictReader(fh))
        self.assertEqual(len(written), 1)
        out = written[0]
        self.assertEqual(out["source_taxon_id"], "735984")
        self.assertEqual(out["source_taxon_name"], "Eciton burchellii parvispinum")
        self.assertEqual(out["source_taxon_rank"], "subspecies")
        self.assertEqual(out["species"], manifest_row["species"])
        self.assertEqual(out["taxon_id"], manifest_row["taxon_id"])

    def test_restore_preserves_source_taxon_fields_unchanged(self):
        accepted_row = self._accepted_descendant_row()
        manifest_row = sg.build_manifest_row(accepted_row)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dataset_dir = repo / "data/calibration_v2"
            slug_dir = dataset_dir / manifest_row["slug"]
            slug_dir.mkdir(parents=True)
            img = _solid_png_bytes((3, 3, 3))
            digest = sg.sha256_bytes(img)
            manifest_row["sha256"] = digest
            manifest_row["byte_size"] = len(img)
            (slug_dir / f"{manifest_row['photo_id']}.png").write_bytes(img)
            _write_csv(dataset_dir / "calibration_v2.csv", sg.MANIFEST_ALL_FIELDS, [manifest_row])
            before = (dataset_dir / "calibration_v2.csv").read_bytes()

            with mock.patch.object(sg, "PacedImageClient", ClientFactory(RaisingClient())):
                rc = sg.cmd_restore(argparse_ns(repo=repo, restore="calibration_v2", request_interval=1.05))
            after = (dataset_dir / "calibration_v2.csv").read_bytes()
            self.assertEqual(rc, 0)
            self.assertEqual(before, after)
            with (dataset_dir / "calibration_v2.csv").open(encoding="utf-8") as fh:
                row = list(csv.DictReader(fh))[0]
            self.assertEqual(row["source_taxon_id"], "735984")
            self.assertEqual(row["source_taxon_name"], "Eciton burchellii parvispinum")
            self.assertEqual(row["source_taxon_rank"], "subspecies")


# =========================================================================
# Pair-publish rollback (point 1)
# =========================================================================
class TestPublishPairRollback(unittest.TestCase):
    def test_second_move_failure_rolls_back_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            staged = {}
            for ds in ("calibration_v2", "unknown_test_v2"):
                d = repo / "data/.staging_gate_v2" / ds
                d.mkdir(parents=True)
                (d / "marker.txt").write_text(ds)
                staged[ds] = d
            real_move = shutil.move
            calls = {"n": 0}

            def flaky_move(src, dst):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError("simulated failure publishing second dataset")
                return real_move(src, dst)

            with mock.patch("scrape_gate_v2.shutil.move", side_effect=flaky_move):
                with self.assertRaises(OSError):
                    sg.publish_pair_with_rollback(repo, staged)

            self.assertFalse((repo / "data/calibration_v2").exists())
            self.assertFalse((repo / "data/unknown_test_v2").exists())
            self.assertTrue((staged["calibration_v2"] / "marker.txt").exists())
            self.assertTrue((staged["unknown_test_v2"] / "marker.txt").exists())


# =========================================================================
# End-to-end synthetic publication fixture through cmd_download + cmd_restore
# (point 12) -- the real orchestration path, not isolated helpers.
# =========================================================================
class TestEndToEndPublicationFixture(unittest.TestCase):
    def _write_exclusion_sources(self, repo: Path) -> dict:
        provenance = {}
        for name, rel in sg.EXCLUSION_SOURCES.items():
            path = repo / rel
            _write_csv(path, ["sha256"], [])
            provenance[name] = {"path": rel, "rows": 0, "sha256": sg.sha256_file(path)}
        return provenance

    def _select_known_holdout(self, slug, taxon_id, base_pid):
        rows = [_fresh_row("known_holdout", "x", "pending", slug, taxon_id, base_pid + i, f"k-{slug}-{i}")
               for i in range(4)]
        cal_sel, remainder = g.deterministic_select(rows, "calibration_v2_known", 1, key=sg._key)
        ut_sel, reserve = g.deterministic_select(remainder, "unknown_test_v2_known", 1, key=sg._key)
        for r in cal_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        for r in ut_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "unknown_test_v2"
        for r in reserve:
            r["selection_status"] = "reserve"
        return rows, cal_sel[0], ut_sel[0]

    def _select_ood_partitioned(self, base_pid):
        cal_pool = [_fresh_row("out_of_scope_ant", "x", "pending", "ood-cal-sp", 201, base_pid + i, f"oc{i}")
                   for i in range(3)]
        ut_pool = [_fresh_row("out_of_scope_ant", "x", "pending", "ood-ut-sp", 202, base_pid + 100 + i, f"ou{i}")
                  for i in range(3)]
        cal_sel, cal_reserve = g.deterministic_select(cal_pool, "calibration_v2_ood", 1, key=sg._key)
        ut_sel, ut_reserve = g.deterministic_select(ut_pool, "unknown_test_v2_ood", 1, key=sg._key)
        for r in cal_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        for r in cal_reserve:
            r["selection_status"] = "reserve"
        for r in ut_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "unknown_test_v2"
        for r in ut_reserve:
            r["selection_status"] = "reserve"
        return cal_pool + ut_pool, cal_sel[0], ut_sel[0]

    def _select_shared_pool(self, category, base_pid):
        rows = [_fresh_row(category, "x", "pending", f"{category}-{i}", 500 + i, base_pid + i, f"{category}{i}")
               for i in range(4)]
        cal_sel, remainder = g.deterministic_select(rows, f"calibration_v2_{category}", 1, key=sg._key)
        ut_sel, reserve = g.deterministic_select(remainder, f"unknown_test_v2_{category}", 1, key=sg._key)
        for r in cal_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        for r in ut_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "unknown_test_v2"
        for r in reserve:
            r["selection_status"] = "reserve"
        return rows, cal_sel[0], ut_sel[0]

    def test_full_download_publish_restore_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            known_a, cal_a, ut_a = self._select_known_holdout("alpha-ant", 101, 1000)
            known_b, cal_b, ut_b = self._select_known_holdout("beta-ant", 102, 2000)
            ood_rows, cal_ood, ut_ood = self._select_ood_partitioned(3000)
            nai_rows, cal_nai, ut_nai = self._select_shared_pool("non_ant_insect", 4000)
            unr_rows, cal_unr, ut_unr = self._select_shared_pool("unrelated", 5000)

            all_rows = known_a + known_b + ood_rows + nai_rows + unr_rows
            candidates_path = repo / sg.CANDIDATES_CSV_PATH
            _write_csv(candidates_path, CANDIDATE_FIELDS, all_rows)
            real_hash = sg.sha256_file(candidates_path)
            real_rows = len(all_rows)

            exclusion_provenance = self._write_exclusion_sources(repo)
            readiness = {
                "candidates_csv": {"path": sg.CANDIDATES_CSV_PATH, "rows": real_rows, "sha256": real_hash},
                "exclusion_sources": exclusion_provenance,
            }
            sg.write_json(repo / sg.READINESS_JSON_PATH, readiness)

            bytes_by_url = {}
            colors = iter(range(10, 250, 20))
            for row in (cal_a, ut_a, cal_b, ut_b, cal_ood, ut_ood, cal_nai, ut_nai, cal_unr, ut_unr):
                c = next(colors)
                bytes_by_url[sg.to_large_url(row["source_url"])] = _solid_png_bytes((c, c, c))

            fake_taxonomy = {"alpha-ant": {"slug": "alpha-ant", "taxon_id": 101},
                            "beta-ant": {"slug": "beta-ant", "taxon_id": 102}}

            args = argparse_ns(repo=repo, cache_dir=Path(sg.CACHE_DIR_DEFAULT),
                              work_dir=Path(sg.WORK_DIR_DEFAULT), request_interval=1.05)

            with mock.patch.object(sg, "EXPECTED_CANDIDATES_SHA256", real_hash), \
                mock.patch.object(sg, "EXPECTED_CANDIDATES_ROWS", real_rows), \
                mock.patch.object(sg, "QUOTAS", TINY_QUOTAS), \
                mock.patch.object(sg, "DATASET_TOTALS", TINY_TOTALS), \
                mock.patch.object(sg, "EXPECTED_REUSED_COUNTS",
                                  {"out_of_scope_ant": 0, "non_ant_insect": 0, "unrelated": 0}), \
                mock.patch.object(g, "load_supported_taxonomy", lambda repo_: fake_taxonomy), \
                mock.patch.object(sg, "verify_taxonomy_matches_candidates",
                                  lambda repo_, rows_, readiness_: {"path": "x", "sha256": "y", "species_count": 2}), \
                mock.patch.object(sg, "PacedImageClient", ClientFactory(FakeImageClient(bytes_by_url))):
                rc = sg.cmd_download(args)

            self.assertEqual(rc, 0)

            cal_dir = repo / "data/calibration_v2"
            ut_dir = repo / "data/unknown_test_v2"
            self.assertTrue((cal_dir / "calibration_v2.csv").exists())
            self.assertTrue((cal_dir / "calibration_v2.json").exists())
            self.assertTrue((ut_dir / "unknown_test_v2.csv").exists())
            self.assertTrue((ut_dir / "unknown_test_v2.json").exists())
            self.assertFalse((repo / "data/calibration_v2.csv").exists())
            self.assertFalse((repo / "data/calibration_v2.json").exists())
            self.assertFalse((repo / "data/unknown_test_v2.csv").exists())
            self.assertFalse((repo / "data/unknown_test_v2.json").exists())
            self.assertFalse((repo / "data/.staging_gate_v2").exists())

            with (cal_dir / "calibration_v2.csv").open(encoding="utf-8") as fh:
                cal_manifest_rows = list(csv.DictReader(fh))
            with (ut_dir / "unknown_test_v2.csv").open(encoding="utf-8") as fh:
                ut_manifest_rows = list(csv.DictReader(fh))
            self.assertEqual(len(cal_manifest_rows), 5)
            self.assertEqual(len(ut_manifest_rows), 5)

            cal_meta = json.loads((cal_dir / "calibration_v2.json").read_text(encoding="utf-8"))
            self.assertIn("byte_accounting", cal_meta)
            self.assertIn("rejection_breakdown", cal_meta)
            self.assertIn("collisions", cal_meta)

            work_dir = repo / sg.WORK_DIR_DEFAULT
            status = json.loads((work_dir / "download_status.json").read_text(encoding="utf-8"))
            self.assertTrue(status["success"])
            self.assertIn("post_publish_sizes", status)
            self.assertGreater(status["post_publish_sizes"]["calibration_v2_dir_bytes"], 0)
            self.assertGreater(status["post_publish_sizes"]["unknown_test_v2_dir_bytes"], 0)

            # Restore must reproduce byte-identical files with zero network
            # calls (everything already matches on disk) and must never
            # rewrite either manifest.
            cal_manifest_before = (cal_dir / "calibration_v2.csv").read_bytes()
            ut_manifest_before = (ut_dir / "unknown_test_v2.csv").read_bytes()
            with mock.patch.object(sg, "PacedImageClient", ClientFactory(RaisingClient())):
                rc_cal = sg.cmd_restore(argparse_ns(repo=repo, restore="calibration_v2", request_interval=1.05))
                rc_ut = sg.cmd_restore(argparse_ns(repo=repo, restore="unknown_test_v2", request_interval=1.05))
            self.assertEqual(rc_cal, 0)
            self.assertEqual(rc_ut, 0)
            self.assertEqual((cal_dir / "calibration_v2.csv").read_bytes(), cal_manifest_before)
            self.assertEqual((ut_dir / "unknown_test_v2.csv").read_bytes(), ut_manifest_before)


class TestPreflightNoNetworkNoWrites(unittest.TestCase):
    def test_preflight_never_constructs_an_http_client(self):
        import inspect
        body = inspect.getsource(sg.cmd_preflight)
        self.assertNotIn("PacedImageClient", body)
        self.assertNotIn("httpx", body)

    def _minimal_repo(self, tmp: Path) -> Path:
        repo = Path(tmp)
        known_a = [_fresh_row("known_holdout", "x", "pending", "alpha-ant", 101, 1000 + i, f"ka{i}")
                  for i in range(4)]
        cal_sel, remainder = g.deterministic_select(known_a, "calibration_v2_known", 1, key=sg._key)
        ut_sel, reserve = g.deterministic_select(remainder, "unknown_test_v2_known", 1, key=sg._key)
        for r in cal_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        for r in ut_sel:
            r["selection_status"], r["intended_dataset"] = "selected", "unknown_test_v2"
        for r in reserve:
            r["selection_status"] = "reserve"
        candidates_path = repo / sg.CANDIDATES_CSV_PATH
        _write_csv(candidates_path, CANDIDATE_FIELDS, known_a)
        real_hash = sg.sha256_file(candidates_path)
        for name, rel in sg.EXCLUSION_SOURCES.items():
            _write_csv(repo / rel, ["sha256"], [])
        readiness = {
            "candidates_csv": {"path": sg.CANDIDATES_CSV_PATH, "rows": len(known_a), "sha256": real_hash},
            "exclusion_sources": {name: {"path": rel, "rows": 0, "sha256": sg.sha256_file(repo / rel)}
                                  for name, rel in sg.EXCLUSION_SOURCES.items()},
        }
        sg.write_json(repo / sg.READINESS_JSON_PATH, readiness)
        return repo, real_hash, len(known_a)

    def test_real_preflight_call_no_network_and_filesystem_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, real_hash, n_rows = self._minimal_repo(tmp)
            fake_taxonomy = {"alpha-ant": {"slug": "alpha-ant", "taxon_id": 101}}
            args = argparse_ns(repo=repo, cache_dir=Path(sg.CACHE_DIR_DEFAULT),
                              work_dir=Path(sg.WORK_DIR_DEFAULT), request_interval=1.05)
            before = {p: p.stat().st_mtime_ns for p in repo.rglob("*") if p.is_file()}
            with mock.patch.object(sg, "EXPECTED_CANDIDATES_SHA256", real_hash), \
                mock.patch.object(sg, "EXPECTED_CANDIDATES_ROWS", n_rows), \
                mock.patch.object(sg, "QUOTAS", {k: (1 if k == ("calibration_v2", "known_holdout")
                                                     or k == ("unknown_test_v2", "known_holdout") else 0)
                                                for k in sg.QUOTAS}), \
                mock.patch.object(sg, "EXPECTED_REUSED_COUNTS",
                                  {"out_of_scope_ant": 0, "non_ant_insect": 0, "unrelated": 0}), \
                mock.patch.object(g, "load_supported_taxonomy", lambda repo_: fake_taxonomy), \
                mock.patch.object(sg, "verify_taxonomy_matches_candidates",
                                  lambda repo_, rows_, readiness_: {"path": "x", "sha256": "y", "species_count": 1}), \
                mock.patch.object(sg, "PacedImageClient", ClientFactory(RaisingClient())):
                rc = sg.cmd_preflight(args)
            after_files = {p for p in repo.rglob("*") if p.is_file()}
            after = {p: p.stat().st_mtime_ns for p in after_files}
            self.assertEqual(rc, 0)
            self.assertEqual(set(before), set(after))
            self.assertEqual(before, after)  # byte-identical: no file was rewritten
            self.assertFalse((repo / "data/calibration_v2").exists())
            self.assertFalse((repo / "data/unknown_test_v2").exists())

    def test_preflight_fails_on_stray_manifest_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, real_hash, n_rows = self._minimal_repo(tmp)
            (repo / "data/calibration_v2.csv").write_text("stray")
            args = argparse_ns(repo=repo, cache_dir=Path(sg.CACHE_DIR_DEFAULT),
                              work_dir=Path(sg.WORK_DIR_DEFAULT), request_interval=1.05)
            with self.assertRaises(sg.SourceContractError):
                sg.cmd_preflight(args)


# =========================================================================
# Restore hardening: reused-row local-origin preference + network fallback,
# ambiguous-extension blocking (points 7, 8)
# =========================================================================
class TestRestoreReusedOriginAndFallback(unittest.TestCase):
    def _manifest_and_dirs(self, tmp, dataset="calibration_v2"):
        repo = Path(tmp)
        dataset_dir = repo / "data" / dataset
        dataset_dir.mkdir(parents=True)
        return repo, dataset_dir

    def test_valid_local_origin_used_with_zero_network_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, dataset_dir = self._manifest_and_dirs(tmp)
            origin_root = repo / "data/calibration_v1/sp-a"
            origin_root.mkdir(parents=True)
            img = _solid_png_bytes((3, 3, 3))
            digest = sg.sha256_bytes(img)
            (origin_root / "77.png").write_bytes(img)

            fields = sg.MANIFEST_REQUIRED_FIELDS
            row = {f: sg.NOT_APPLICABLE for f in fields}
            row.update({"slug": "sp-a", "photo_id": "77", "sha256": digest, "observation_uuid": "u77",
                       "byte_size": len(img), "width": 220, "height": 220,
                       "provenance_source": "calibration_v1_reuse", "origin_photo_id": "77",
                       "origin_sha256": digest})
            _write_csv(dataset_dir / "calibration_v2.csv", fields, [row])
            before = (dataset_dir / "calibration_v2.csv").read_bytes()

            with mock.patch.object(sg, "PacedImageClient", ClientFactory(RaisingClient())):
                rc = sg.cmd_restore(argparse_ns(repo=repo, restore="calibration_v2", request_interval=1.05))
            self.assertEqual(rc, 0)
            self.assertEqual((dataset_dir / "calibration_v2.csv").read_bytes(), before)
            self.assertTrue((dataset_dir / "sp-a" / "77.png").exists())
            self.assertEqual(sg.sha256_file(dataset_dir / "sp-a" / "77.png"), digest)

    def test_missing_local_origin_falls_back_to_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, dataset_dir = self._manifest_and_dirs(tmp)
            # No calibration_v1 origin file exists at all.
            img = _solid_png_bytes((4, 4, 4))
            digest = sg.sha256_bytes(img)

            fields = sg.MANIFEST_REQUIRED_FIELDS
            row = {f: sg.NOT_APPLICABLE for f in fields}
            row.update({"slug": "sp-a", "photo_id": "88", "sha256": digest, "observation_uuid": "u88",
                       "byte_size": len(img), "width": 220, "height": 220,
                       "provenance_source": "calibration_v1_reuse", "origin_photo_id": "88",
                       "origin_sha256": "f" * 64})  # origin hash won't matter -- file is simply absent
            _write_csv(dataset_dir / "calibration_v2.csv", fields, [row])

            json_calls = []

            def json_by_call(url, params):
                json_calls.append((url, params))
                return {"results": [{"uuid": "u88", "photos": [{"id": 88, "url": "http://x/square.png"}]}]}

            client = FakeImageClient({"http://x/large.png": img}, json_by_call=json_by_call)
            with mock.patch.object(sg, "PacedImageClient", ClientFactory(client)):
                rc = sg.cmd_restore(argparse_ns(repo=repo, restore="calibration_v2", request_interval=1.05))
            self.assertEqual(rc, 0)
            self.assertTrue(json_calls)  # network fallback WAS used
            self.assertTrue((dataset_dir / "sp-a" / "88.png").exists())
            self.assertEqual(sg.sha256_file(dataset_dir / "sp-a" / "88.png"), digest)

    def test_ambiguous_local_files_blocked_without_cascading_api_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, dataset_dir = self._manifest_and_dirs(tmp)
            slug_dir = dataset_dir / "sp-a"
            slug_dir.mkdir(parents=True)
            (slug_dir / "5.jpg").write_bytes(_solid_png_bytes((1, 1, 1)))
            (slug_dir / "5.png").write_bytes(_solid_png_bytes((2, 2, 2)))

            fields = sg.MANIFEST_REQUIRED_FIELDS
            row = {f: sg.NOT_APPLICABLE for f in fields}
            row.update({"slug": "sp-a", "photo_id": "5", "sha256": "a" * 64, "observation_uuid": "u5",
                       "byte_size": 1, "width": 220, "height": 220})
            _write_csv(dataset_dir / "calibration_v2.csv", fields, [row])

            with mock.patch.object(sg, "PacedImageClient", ClientFactory(RaisingClient())):
                rc = sg.cmd_restore(argparse_ns(repo=repo, restore="calibration_v2", request_interval=1.05))
            self.assertEqual(rc, 1)


class TestRestoreNeverRewritesManifest(unittest.TestCase):
    def test_restore_reads_manifest_bytes_unchanged_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dataset_dir = repo / "data/calibration_v2"
            slug_dir = dataset_dir / "sp-a"
            slug_dir.mkdir(parents=True)
            img = _solid_png_bytes((3, 3, 3))
            digest = sg.sha256_bytes(img)
            (slug_dir / "1.png").write_bytes(img)
            fields = sg.MANIFEST_REQUIRED_FIELDS
            row = {f: sg.NOT_APPLICABLE for f in fields}
            row.update({"slug": "sp-a", "photo_id": "1", "sha256": digest,
                       "observation_uuid": "u1", "byte_size": len(img), "width": 220, "height": 220})
            _write_csv(dataset_dir / "calibration_v2.csv", fields, [row])
            before = (dataset_dir / "calibration_v2.csv").read_bytes()

            with mock.patch.object(sg, "PacedImageClient", ClientFactory(RaisingClient())):
                rc = sg.cmd_restore(argparse_ns(repo=repo, restore="calibration_v2", request_interval=1.05))
            after = (dataset_dir / "calibration_v2.csv").read_bytes()
            self.assertEqual(before, after)
            self.assertEqual(rc, 0)

    def test_restore_hash_mismatch_after_redownload_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dataset_dir = repo / "data/calibration_v2"
            dataset_dir.mkdir(parents=True)
            fields = sg.MANIFEST_REQUIRED_FIELDS
            row = {f: sg.NOT_APPLICABLE for f in fields}
            row.update({"slug": "sp-a", "photo_id": "1", "sha256": "f" * 64,
                       "observation_uuid": "u1", "byte_size": 1, "width": 220, "height": 220})
            _write_csv(dataset_dir / "calibration_v2.csv", fields, [row])

            def json_by_call(url, params):
                return {"results": [{"uuid": "u1", "photos": [{"id": 1, "url": "http://x/square.jpg"}]}]}

            client = FakeImageClient({"http://x/large.jpg": _solid_png_bytes((1, 1, 1))}, json_by_call=json_by_call)
            with mock.patch.object(sg, "PacedImageClient", ClientFactory(client)):
                rc = sg.cmd_restore(argparse_ns(repo=repo, restore="calibration_v2", request_interval=1.05))
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
