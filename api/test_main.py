#!/usr/bin/env python3
"""Small HTTP-contract tests that avoid loading the real ONNX artifacts."""
from __future__ import annotations

import asyncio
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inference  # noqa: E402


class StubIdentifier:
    species_count = 50
    geo_index_loaded = False
    geo_index_reason = "invalid_json"
    inference_policy_loaded = False
    inference_policy_reason = "artifact_hash_mismatch"

    def species_list(self):
        return []

    def identify(self, img, lat=None, lon=None):
        return {
            "results": [],
            "inference_ms": 1,
            "geo_filtered": False,
            "gate_active": False,
            "low_confidence": None,
        }


class Upload:
    content_type = "image/png"

    def __init__(self, raw: bytes):
        self.raw = raw

    async def read(self):
        return self.raw


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class TestMainContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        cls.stub = StubIdentifier()
        with mock.patch.object(inference, "AntIdentifier", return_value=cls.stub):
            import main
        cls.main = main

    def test_health_reports_functional_optional_component_state(self):
        self.assertEqual(self.main.health(), {
            "status": "ok",
            "species_count": 50,
            "geo_index_loaded": False,
            "geo_index_reason": "invalid_json",
            "inference_policy_loaded": False,
            "inference_policy_reason": "artifact_hash_mismatch",
        })

    def test_identify_preserves_inactive_null_contract(self):
        response = asyncio.run(self.main.identify(Upload(png_bytes()), None, None))
        self.assertFalse(response["gate_active"])
        self.assertIsNone(response["low_confidence"])

    def test_inference_error_becomes_controlled_http_500(self):
        with mock.patch.object(
            self.main.identifier,
            "identify",
            side_effect=inference.InferenceError("non-finite output"),
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(self.main.identify(Upload(png_bytes()), None, None))
        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail, "Inference failed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
