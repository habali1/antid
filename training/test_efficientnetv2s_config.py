#!/usr/bin/env python3
"""test_efficientnetv2s_config.py — proves the EfficientNetV2-S candidate
config (config.efficientnetv2_s.yaml) is a correctly scoped, separate config
from the live B4 recipe (config.yaml): same training recipe, different model
identity/embedding dimension/native preprocessing only. Never downloads
pretrained weights (every model built here uses pretrained=False) and never
touches training/config.yaml.

Run directly: python test_efficientnetv2s_config.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numerics  # noqa: E402
import train  # noqa: E402
from data import build_transforms, resolve_interpolation  # noqa: E402
from data_provenance import load_explicit_manifest_source  # noqa: E402
from export import export_backbone  # noqa: E402
from model import AntIDModel  # noqa: E402

B4_CONFIG_PATH = HERE / "config.yaml"
V2S_CONFIG_PATH = HERE / "config.efficientnetv2_s.yaml"

REPO = HERE.parent
REAL_MANIFEST_CSV = REPO / "data" / "northeast_expansion_v1" / "manifest_all_northeast_v1.csv"
REAL_TAXONOMY_JSON = REPO / "data" / "northeast_expansion_v1" / "northeast_taxonomy_v1.json"
REAL_LOCAL_DATA_DIR = REPO / "data" / "clean"
REAL_MANIFEST_SHA256 = "b4f39115c80f955755987c0959dc7ed019620f1f143c64b62523f12bc8894000"
REAL_TAXONOMY_SHA256 = "5e9671fddaa1fa46dcb76a4a94787533f67c5dfb54e90841ff60432a17aec27e"
EXPECTED_VAL_SPLIT_SHA256 = "1039518efb33e43f2f97c66e11c1e24947c4a379fc54d6394ac5984c79d5ac7e"

_real_catalog_available = (
    REAL_MANIFEST_CSV.exists() and REAL_TAXONOMY_JSON.exists() and REAL_LOCAL_DATA_DIR.exists()
)


def _load(path: Path) -> dict:
    return train.load_config(path, {})


class TestB4ConfigUnchanged(unittest.TestCase):
    def setUp(self):
        self.cfg = _load(B4_CONFIG_PATH)

    def test_image_size_and_embedding_dim(self):
        self.assertEqual(self.cfg["image_size"], 380)
        self.assertEqual(self.cfg["model"]["embedding_dim"], 1792)
        self.assertEqual(self.cfg["model"]["backbone"], "tf_efficientnet_b4")

    def test_imagenet_normalization(self):
        self.assertEqual(self.cfg["normalize"]["mean"], [0.485, 0.456, 0.406])
        self.assertEqual(self.cfg["normalize"]["std"], [0.229, 0.224, 0.225])

    def test_absent_interpolation_resolves_to_bilinear(self):
        self.assertNotIn("interpolation", self.cfg)
        self.assertEqual(resolve_interpolation(self.cfg), InterpolationMode.BILINEAR)

    def test_transform_resize_matches_pre_change_default(self):
        """The explicit interpolation=... parameterization must be
        behaviorally identical to torchvision's own implicit Resize default
        -- i.e. this change is a pure no-op for B4."""
        explicit = build_transforms(self.cfg, train=False).transforms[0]
        implicit = __import__("torchvision").transforms.Resize((self.cfg["image_size"],) * 2)
        self.assertEqual(explicit.interpolation, implicit.interpolation)
        self.assertEqual(explicit.interpolation, InterpolationMode.BILINEAR)


class TestV2SConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = _load(V2S_CONFIG_PATH)

    def test_image_size_and_embedding_dim(self):
        self.assertEqual(self.cfg["image_size"], 300)
        self.assertEqual(self.cfg["model"]["embedding_dim"], 1280)
        self.assertEqual(self.cfg["model"]["backbone"], "tf_efficientnetv2_s.in21k_ft_in1k")

    def test_native_normalization(self):
        self.assertEqual(self.cfg["normalize"]["mean"], [0.5, 0.5, 0.5])
        self.assertEqual(self.cfg["normalize"]["std"], [0.5, 0.5, 0.5])

    def test_bicubic_interpolation(self):
        self.assertEqual(self.cfg["interpolation"], "bicubic")
        self.assertEqual(resolve_interpolation(self.cfg), InterpolationMode.BICUBIC)


class TestOnlyApprovedRecipeDifferences(unittest.TestCase):
    """The only fields allowed to differ between the two configs are model
    identity/embedding dimension and pretrained-native preprocessing
    (image_size, normalize, interpolation). Every other training-recipe
    field must be byte-for-byte identical."""

    APPROVED_DIFFERENCES = {"image_size", "normalize", "interpolation"}
    APPROVED_MODEL_DIFFERENCES = {"backbone", "embedding_dim"}

    def test_top_level_fields_match_except_approved(self):
        b4 = _load(B4_CONFIG_PATH)
        v2s = _load(V2S_CONFIG_PATH)
        all_keys = set(b4) | set(v2s)
        for key in sorted(all_keys - self.APPROVED_DIFFERENCES - {"model"}):
            self.assertEqual(b4.get(key), v2s.get(key),
                             f"unexpected recipe difference in top-level key {key!r}")

    def test_model_block_matches_except_approved(self):
        b4_model = _load(B4_CONFIG_PATH)["model"]
        v2s_model = _load(V2S_CONFIG_PATH)["model"]
        for key in sorted((set(b4_model) | set(v2s_model)) - self.APPROVED_MODEL_DIFFERENCES):
            self.assertEqual(b4_model.get(key), v2s_model.get(key),
                             f"unexpected recipe difference in model.{key!r}")

    def test_seed_lr_weight_decay_dropout_epochs_augmentation_match(self):
        b4 = _load(B4_CONFIG_PATH)
        v2s = _load(V2S_CONFIG_PATH)
        for key in ("seed", "lr", "weight_decay", "dropout", "epochs", "batch_size",
                   "num_workers", "val_fraction", "augmentation", "geo"):
            self.assertEqual(b4[key], v2s[key], f"{key!r} must match between B4 and V2-S")
        self.assertEqual(b4["model"]["pretrained"], v2s["model"]["pretrained"])

    def test_config_yaml_was_not_modified(self):
        """The task explicitly forbids touching training/config.yaml; a
        regression here would mean B4's own values silently drifted."""
        cfg = _load(B4_CONFIG_PATH)
        self.assertEqual(cfg["model"]["backbone"], "tf_efficientnet_b4")
        self.assertEqual(cfg["model"]["embedding_dim"], 1792)
        self.assertEqual(cfg["image_size"], 380)
        self.assertNotIn("interpolation", cfg)


class TestInterpolationRejection(unittest.TestCase):
    def test_unsupported_string_rejected(self):
        with self.assertRaises(ValueError):
            resolve_interpolation({"interpolation": "lanczos-typo"})

    def test_non_string_rejected(self):
        with self.assertRaises(ValueError):
            resolve_interpolation({"interpolation": 2})  # raw PIL constant, not a named mode

    def test_supported_values_all_resolve(self):
        for name in ("nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"):
            resolve_interpolation({"interpolation": name})  # must not raise


class _RandomImageDataset(Dataset):
    """Deterministic, synthetic (batch,3,300,300)-shaped tensors -- exercises
    the real V2-S architecture end-to-end without ever touching a real image
    file or a downloaded pretrained weight."""

    def __init__(self, n: int, num_classes: int, size: int):
        g = torch.Generator().manual_seed(2026)
        self.images = torch.rand(n, 3, size, size, generator=g)
        self.labels = torch.arange(n) % num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


class TestV2SModelShapes(unittest.TestCase):
    """Never downloads pretrained weights: pretrained=False throughout."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _load(V2S_CONFIG_PATH)
        cls.num_classes = 65
        cls.model = AntIDModel(
            num_classes=cls.num_classes, backbone=cls.cfg["model"]["backbone"],
            pretrained=False, dropout=cls.cfg["dropout"],
            embedding_dim=cls.cfg["model"]["embedding_dim"],
        )
        cls.model.eval()

    def test_embed_output_shape(self):
        x = torch.randn(2, 3, self.cfg["image_size"], self.cfg["image_size"])
        with torch.no_grad():
            emb = self.model.embed(x)
        self.assertEqual(tuple(emb.shape), (2, 1280))

    def test_prototype_shape_and_l2_normalization(self):
        numerics.apply_numerical_policy()
        ds = _RandomImageDataset(n=self.num_classes * 2, num_classes=self.num_classes,
                                 size=self.cfg["image_size"])
        loader = DataLoader(ds, batch_size=8, num_workers=0)
        protos = train.compute_prototypes(
            self.model, loader, self.num_classes, torch.device("cpu"),
            self.cfg["model"]["embedding_dim"],
        )
        self.assertEqual(protos.shape, (self.num_classes, 1280))
        import numpy as np
        norms = np.linalg.norm(protos, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_onnx_export_shapes(self):
        import onnx
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "backbone.onnx"
            export_backbone(self.model, out_path, self.cfg["image_size"], torch.device("cpu"))
            graph = onnx.load(str(out_path))

            def dims(value_info):
                shape = value_info.type.tensor_type.shape
                return [d.dim_param if d.dim_param else d.dim_value for d in shape.dim]

            input_dims = dims(graph.graph.input[0])
            output_dims = dims(graph.graph.output[0])
            self.assertEqual(input_dims[1:], [3, self.cfg["image_size"], self.cfg["image_size"]])
            self.assertEqual(output_dims[1:], [1280])
            self.assertTrue(isinstance(input_dims[0], str))  # dynamic batch
            self.assertTrue(isinstance(output_dims[0], str))  # dynamic batch


@unittest.skipUnless(_real_catalog_available,
                     "real Northeast catalog (data/clean, data/northeast_expansion_v1) "
                     "not present on this machine")
class TestSharedDatasetAcrossConfigs(unittest.TestCase):
    """B4 and V2-S must train on the exact same manifest/split/taxonomy --
    only the model/preprocessing config differs. Never hashes image bytes
    here (that's verify_image_bytes's job, already covered elsewhere) --
    only proves the split derivation itself is config-seed-driven and
    identical between the two configs."""

    def test_same_manifest_split_and_taxonomy_for_both_configs(self):
        b4_cfg = _load(B4_CONFIG_PATH)
        v2s_cfg = _load(V2S_CONFIG_PATH)
        self.assertEqual(b4_cfg["seed"], v2s_cfg["seed"])
        self.assertEqual(b4_cfg["val_fraction"], v2s_cfg["val_fraction"])

        samples, taxonomy, manifest_sha256, taxonomy_sha256, _ = load_explicit_manifest_source(
            REAL_MANIFEST_CSV, REAL_LOCAL_DATA_DIR, REAL_TAXONOMY_JSON,
            REAL_MANIFEST_SHA256, REAL_TAXONOMY_SHA256, database_url=None,
        )
        self.assertEqual(len(samples), 13581)
        self.assertEqual(len(taxonomy), 65)

        for cfg in (b4_cfg, v2s_cfg):
            train_s, val_s = train.split_samples(samples, cfg["val_fraction"], cfg["seed"])
            self.assertEqual(len(train_s), 10985)
            self.assertEqual(len(val_s), 2596)
            record = train.build_val_split_record(cfg, train_s, val_s)
            split_bytes = train.serialize_val_split(record)
            import hashlib
            self.assertEqual(hashlib.sha256(split_bytes).hexdigest(), EXPECTED_VAL_SPLIT_SHA256)


if __name__ == "__main__":
    unittest.main()
