"""inference.py — load artifacts once, preprocess, run ONNX, cosine top-3.

Preprocessing here MUST match training/data.py val transforms exactly:
  resize to (380, 380), ToTensor scaling to [0,1], normalize with ImageNet stats.

Geo re-ranking (optional): if artifacts/geo_index.json exists and the request
includes lat/lon, species observed in the user's 1-degree grid cell (or its 8
neighbors) get a small additive score boost. Ranks are computed on the boosted
score; the reported `similarity` stays the raw cosine value. `geo_boosted` is
true for a result whose final rank strictly improved because of the boost.

The optional inference_policy.json sidecar activates a selective confidence
gate only when its schema, live artifact hashes, preprocessing contract, and
CPU-only provider requirement all verify. A missing or invalid policy never
stops closest-match inference; it disables the gate and is reported in health.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from inference_policy import load_inference_policy

IMAGE_SIZE = 380
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
PIXEL_SCALE_DIVISOR = 255.0
MEAN = np.array(NORMALIZE_MEAN, dtype=np.float32)
STD = np.array(NORMALIZE_STD, dtype=np.float32)


def _parse_geo_boost(raw: str) -> float | None:
    """Return a usable boost, or None so malformed optional geo stays off."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


GEO_BOOST = _parse_geo_boost(os.environ.get("GEO_BOOST", "0.05"))
PREPROCESSING_CONTRACT = {
    "rgb_conversion": "img.convert('RGB')",
    "resize": f"squish to fixed {IMAGE_SIZE}x{IMAGE_SIZE} (both dims set, no crop)",
    "interpolation": "Pillow bilinear",
    "scale_divisor": PIXEL_SCALE_DIVISOR,
    "normalize_mean": list(NORMALIZE_MEAN),
    "normalize_std": list(NORMALIZE_STD),
    "dtype": "float32",
    "channel_layout": "RGB -> CHW, batched to NCHW",
}


class InferenceError(RuntimeError):
    """A request reached the model but could not produce finite scores."""


def _default_artifacts_dir() -> Path:
    env = os.environ.get("ARTIFACTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "training" / "artifacts"


class AntIdentifier:
    """Holds the ONNX session + prototypes + taxonomy (+ geo index)."""

    def __init__(self, artifacts_dir: Path | None = None):
        art = Path(artifacts_dir) if artifacts_dir else _default_artifacts_dir()
        onnx_path = art / "backbone.onnx"
        proto_path = art / "prototypes.npy"
        tax_path = art / "taxonomy.json"

        missing = [p.name for p in (onnx_path, proto_path, tax_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifact(s) in {art}: {', '.join(missing)}. "
                "Run training (train.py) or copy artifacts here."
            )

        # The validated serving path is ONNX Runtime CPU only. Asking for all
        # registered providers can silently select a different numerical path
        # (for example Azure/CUDA) and invalidate the gate's parity evidence.
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.runtime_providers = list(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name

        protos = np.load(proto_path).astype(np.float32)
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        self.prototypes = protos / np.clip(norms, 1e-8, None)

        raw = json.loads(tax_path.read_text())
        self.taxonomy = {int(k): v for k, v in raw.items()}
        if len(self.taxonomy) != self.prototypes.shape[0]:
            raise ValueError(
                f"taxonomy has {len(self.taxonomy)} classes but prototypes has "
                f"{self.prototypes.shape[0]} rows."
            )
        self.species_count = len(self.taxonomy)

        self.inference_policy = load_inference_policy(
            art,
            actual_providers=self.runtime_providers,
            expected_preprocessing_contract=PREPROCESSING_CONTRACT,
        )

        # ---- optional geo index -------------------------------------------
        self.geo_index_loaded = False
        self.geo_index_reason = "missing"
        self._geo_cells: dict[int, set[tuple[int, int]]] = {}
        self._cell_size = 1.0
        self._load_geo_index(art / "geo_index.json")

    @property
    def inference_policy_loaded(self) -> bool:
        return self.inference_policy.active

    @property
    def inference_policy_reason(self) -> str:
        return self.inference_policy.reason

    # ------------------------------------------------------------- preprocess
    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / PIXEL_SCALE_DIVISOR
        arr = (arr - MEAN) / STD
        arr = np.transpose(arr, (2, 0, 1))
        return arr[None, ...].astype(np.float32)

    # --------------------------------------------------------------- geo load
    def _load_geo_index(self, geo_path: Path) -> None:
        if not geo_path.exists():
            return
        if GEO_BOOST is None:
            self.geo_index_reason = "invalid_geo_boost"
            return
        try:
            geo = json.loads(geo_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.geo_index_reason = "invalid_json"
            return
        except OSError:
            self.geo_index_reason = "io_error"
            return

        if not isinstance(geo, dict):
            self.geo_index_reason = "invalid_schema"
            return
        cell_size = geo.get("cell_size_deg")
        if (not isinstance(cell_size, (int, float)) or isinstance(cell_size, bool)
                or not math.isfinite(float(cell_size)) or float(cell_size) <= 0):
            self.geo_index_reason = "invalid_cell_size"
            return
        raw_cells = geo.get("cells")
        if not isinstance(raw_cells, dict):
            self.geo_index_reason = "invalid_schema"
            return

        slug_to_idx = {v["slug"]: k for k, v in self.taxonomy.items()}
        usable: dict[int, set[tuple[int, int]]] = {}
        for slug, cells in raw_cells.items():
            idx = slug_to_idx.get(slug)
            if idx is None or not isinstance(cells, list):
                continue
            parsed: set[tuple[int, int]] = set()
            for cell in cells:
                if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                    continue
                a, b = cell
                if (not isinstance(a, (int, float)) or isinstance(a, bool)
                        or not isinstance(b, (int, float)) or isinstance(b, bool)):
                    continue
                if (not math.isfinite(float(a)) or not math.isfinite(float(b))
                        or not float(a).is_integer() or not float(b).is_integer()):
                    continue
                parsed.add((int(a), int(b)))
            if parsed:
                usable[idx] = parsed

        if not usable:
            self.geo_index_reason = "no_usable_cells"
            return
        self._cell_size = float(cell_size)
        self._geo_cells = usable
        self.geo_index_loaded = True
        self.geo_index_reason = "active"

    # -------------------------------------------------------------------- geo
    def _in_range(self, idx: int, lat: float, lon: float) -> bool:
        cells = self._geo_cells.get(idx)
        if not cells:
            return False
        cs = self._cell_size
        clat, clon = math.floor(lat / cs), math.floor(lon / cs)
        for dlat in (-1, 0, 1):          # 3x3 neighborhood: tolerant near edges
            for dlon in (-1, 0, 1):
                if (clat + dlat, clon + dlon) in cells:
                    return True
        return False

    # -------------------------------------------------------------- inference
    def identify(self, img: Image.Image,
                 lat: float | None = None, lon: float | None = None,
                 top_k: int = 3) -> dict:
        t0 = time.perf_counter()
        x = self.preprocess(img)
        emb = self.session.run(None, {self.input_name: x})[0][0]
        emb_norm = float(np.linalg.norm(emb))
        if not math.isfinite(emb_norm) or emb_norm <= 1e-8:
            raise InferenceError("model produced an invalid embedding")
        emb = emb / emb_norm
        sims = self.prototypes @ emb                       # raw cosine, (N,)
        if sims.size == 0 or not bool(np.isfinite(sims).all()):
            raise InferenceError("model produced non-finite similarity scores")

        # Gate on the global raw maximum before geo re-ranking and before the
        # per-result display rounding below. np.max returns float32 here;
        # float() widens that exact value to Python float64 for the comparison.
        raw_max_similarity = float(np.max(sims))
        low_confidence = self.inference_policy.classify(raw_max_similarity)

        geo_active = (lat is not None and lon is not None
                      and self.geo_index_loaded)
        base_order = np.argsort(-sims, kind="stable")
        base_rank = {int(c): r for r, c in enumerate(base_order)}

        if geo_active:
            in_range = np.array(
                [self._in_range(i, lat, lon) for i in range(self.species_count)]
            )
            # A functional geo index can only be loaded with a finite,
            # positive boost, so this assertion documents the invariant.
            assert GEO_BOOST is not None
            adjusted = sims + GEO_BOOST * in_range
            order = np.argsort(-adjusted, kind="stable")
        else:
            in_range = np.zeros(self.species_count, dtype=bool)
            order = base_order

        k = min(top_k, self.species_count)
        results = []
        for rank, idx in enumerate(order[:k], start=1):
            idx = int(idx)
            meta = self.taxonomy[idx]
            boosted = bool(geo_active and in_range[idx]
                           and (rank - 1) < base_rank[idx])
            results.append({
                "rank": rank,
                "species_name": meta["species_name"],
                "common_name": meta.get("common_name"),
                "taxon_id": meta.get("taxon_id"),
                "similarity": round(float(sims[idx]), 4),
                "geo_boosted": boosted,
            })
        return {
            "results": results,
            "inference_ms": int((time.perf_counter() - t0) * 1000),
            "geo_filtered": bool(geo_active),
            "gate_active": self.inference_policy.active,
            "low_confidence": low_confidence,
        }

    def species_list(self) -> list[dict]:
        return [self.taxonomy[i] for i in sorted(self.taxonomy)]
