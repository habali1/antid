#!/usr/bin/env python3
"""Dependency-light runtime tests for Task 4 inference semantics."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

# The runtime test substitutes a fake session and does not need the native ORT
# package. Supplying a module before importing inference keeps this suite
# runnable in lightweight development environments.
fake_ort = types.ModuleType("onnxruntime")
fake_ort.InferenceSession = object
sys.modules["onnxruntime"] = fake_ort
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inference  # noqa: E402


class StubPolicy:
    def __init__(self, active: bool, threshold: float = 0.6):
        self.active = active
        self.reason = "active" if active else "policy_missing"
        self.threshold = threshold if active else None

    def classify(self, raw_max: float) -> bool | None:
        if not math.isfinite(raw_max):
            raise ValueError("non-finite")
        return raw_max < self.threshold if self.active else None


class FakeRunSession:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding.astype(np.float32)

    def run(self, _outputs, _inputs):
        return [self.embedding[None, :]]


def bare_identifier(sims: list[float], *, active: bool, geo: bool = False):
    ident = inference.AntIdentifier.__new__(inference.AntIdentifier)
    ident.session = FakeRunSession(np.array([1.0, 0.0], dtype=np.float32))
    ident.input_name = "input"
    ident.preprocess = lambda _img: np.zeros((1, 3, 380, 380), dtype=np.float32)
    ident.prototypes = np.array([[v, 0.0] for v in sims], dtype=np.float32)
    ident.taxonomy = {
        i: {"species_name": f"species-{i}", "common_name": None,
            "taxon_id": i, "slug": f"species-{i}"}
        for i in range(len(sims))
    }
    ident.species_count = len(sims)
    ident.inference_policy = StubPolicy(active)
    ident.geo_index_loaded = geo
    ident.geo_index_reason = "active" if geo else "missing"
    ident._cell_size = 1.0
    ident._geo_cells = {1: {(0, 0)}} if geo and len(sims) > 1 else {}
    return ident


class TestGatePlacement(unittest.TestCase):
    def test_active_gate_uses_unrounded_raw_global_max(self):
        ident = bare_identifier([0.59996, 0.4], active=True)
        response = ident.identify(object())
        self.assertTrue(response["gate_active"])
        self.assertTrue(response["low_confidence"])
        # Display rounding is downstream of the decision.
        self.assertEqual(response["results"][0]["similarity"], 0.6)

    def test_inactive_gate_returns_null_not_false(self):
        ident = bare_identifier([0.2, 0.1], active=False)
        response = ident.identify(object())
        self.assertFalse(response["gate_active"])
        self.assertIsNone(response["low_confidence"])

    def test_geo_reordering_does_not_change_gate_signal(self):
        ident = bare_identifier([0.61, 0.59], active=True, geo=True)
        with mock.patch.object(inference, "GEO_BOOST", 0.05):
            response = ident.identify(object(), lat=0.1, lon=0.1)
        self.assertFalse(response["low_confidence"])
        self.assertTrue(response["geo_filtered"])
        self.assertEqual(response["results"][0]["species_name"], "species-1")
        self.assertEqual(response["results"][0]["similarity"], 0.59)

    def test_nonfinite_similarity_is_controlled_error(self):
        ident = bare_identifier([float("nan"), 0.2], active=True)
        with self.assertRaises(inference.InferenceError):
            ident.identify(object())

    def test_zero_embedding_is_controlled_error(self):
        ident = bare_identifier([0.4], active=True)
        ident.session = FakeRunSession(np.zeros(2, dtype=np.float32))
        with self.assertRaises(inference.InferenceError):
            ident.identify(object())


class TestGeoIndexHealth(unittest.TestCase):
    def make_identifier(self):
        ident = inference.AntIdentifier.__new__(inference.AntIdentifier)
        ident.taxonomy = {0: {"slug": "known-ant"}}
        ident.geo_index_loaded = False
        ident.geo_index_reason = "missing"
        ident._geo_cells = {}
        ident._cell_size = 1.0
        return ident

    def test_malformed_geo_is_inactive(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text("not-json", encoding="utf-8")
            ident = self.make_identifier()
            ident._load_geo_index(path)
            self.assertFalse(ident.geo_index_loaded)
            self.assertEqual(ident.geo_index_reason, "invalid_json")

    def test_invalid_utf8_geo_is_inactive(self):
        ident = self.make_identifier()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_bytes(b"\xff\xfe")
            ident._load_geo_index(path)
            self.assertFalse(ident.geo_index_loaded)
            self.assertEqual(ident.geo_index_reason, "invalid_json")

    def test_invalid_geo_boost_disables_an_otherwise_valid_index(self):
        ident = self.make_identifier()
        original = inference.GEO_BOOST
        try:
            inference.GEO_BOOST = None
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "geo_index.json"
                path.write_text(json.dumps({
                    "cell_size_deg": 1.0,
                    "cells": {"species-0": [[0, 0]]},
                }))
                ident._load_geo_index(path)
        finally:
            inference.GEO_BOOST = original
        self.assertFalse(ident.geo_index_loaded)
        self.assertEqual(ident.geo_index_reason, "invalid_geo_boost")

    def test_geo_boost_parser_rejects_nonpositive_and_nonfinite_values(self):
        for raw in ("bad", "0", "-0.1", "nan", "inf", "-inf"):
            with self.subTest(raw=raw):
                self.assertIsNone(inference._parse_geo_boost(raw))
        self.assertEqual(inference._parse_geo_boost("0.05"), 0.05)

    def test_unknown_only_geo_is_not_functionally_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text(json.dumps({"cell_size_deg": 1, "cells": {"other": [[1, 2]]}}))
            ident = self.make_identifier()
            ident._load_geo_index(path)
            self.assertFalse(ident.geo_index_loaded)
            self.assertEqual(ident.geo_index_reason, "no_usable_cells")

    def test_one_valid_cell_activates_geo(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text(json.dumps({"cell_size_deg": 1, "cells": {"known-ant": [[1, 2]]}}))
            ident = self.make_identifier()
            ident._load_geo_index(path)
            self.assertTrue(ident.geo_index_loaded)
            self.assertEqual(ident.geo_index_reason, "active")
            self.assertEqual(ident._geo_cells, {0: {(1, 2)}})


class TestProviderSelection(unittest.TestCase):
    def test_constructor_requests_cpu_only(self):
        calls = []

        class Input:
            name = "input"

        class Session:
            def __init__(self, path, providers):
                calls.append((path, providers))

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def get_inputs(self):
                return [Input()]

        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "backbone.onnx").write_bytes(b"fake")
            np.save(art / "prototypes.npy", np.array([[1.0, 0.0]], dtype=np.float32))
            (art / "taxonomy.json").write_text(json.dumps({
                "0": {"species_name": "ant", "common_name": None,
                      "taxon_id": 1, "slug": "ant"}
            }), encoding="utf-8")
            with mock.patch.object(inference.ort, "InferenceSession", Session):
                ident = inference.AntIdentifier(art)

        self.assertEqual(calls[0][1], ["CPUExecutionProvider"])
        self.assertFalse(ident.inference_policy_loaded)
        self.assertEqual(ident.inference_policy_reason, "policy_missing")


class TestPreprocessingContract(unittest.TestCase):
    def test_contract_is_derived_from_runtime_constants_and_matches_behavior(self):
        self.assertEqual(inference.PREPROCESSING_CONTRACT["scale_divisor"],
                         inference.PIXEL_SCALE_DIVISOR)
        self.assertEqual(inference.PREPROCESSING_CONTRACT["normalize_mean"],
                         list(inference.NORMALIZE_MEAN))
        self.assertEqual(inference.PREPROCESSING_CONTRACT["normalize_std"],
                         list(inference.NORMALIZE_STD))
        self.assertIn(f"{inference.IMAGE_SIZE}x{inference.IMAGE_SIZE}",
                      inference.PREPROCESSING_CONTRACT["resize"])

        img = Image.new("RGB", (7, 3), (255, 0, 128))
        actual = inference.AntIdentifier.preprocess(None, img)
        source = np.array([255, 0, 128], dtype=np.float32)
        expected = (source / inference.PIXEL_SCALE_DIVISOR - inference.MEAN) / inference.STD
        self.assertEqual(actual.shape, (1, 3, inference.IMAGE_SIZE, inference.IMAGE_SIZE))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual[0, :, 0, 0], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
