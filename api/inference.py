"""inference.py — load artifacts once, preprocess, run ONNX, cosine top-3.

Preprocessing here MUST match training/data.py val transforms exactly:
  resize to (380, 380), ToTensor scaling to [0,1], normalize with ImageNet stats.

Geo re-ranking (optional): if artifacts/geo_index.json exists and the request
includes lat/lon, species observed in the user's 1-degree grid cell (or its 8
neighbors) get a small additive score boost. Ranks are computed on the boosted
score; the reported `similarity` stays the raw cosine value. `geo_boosted` is
true for a result whose final rank strictly improved because of the boost.
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

IMAGE_SIZE = 380
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GEO_BOOST = float(os.environ.get("GEO_BOOST", "0.05"))


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

        self.session = ort.InferenceSession(
            str(onnx_path), providers=ort.get_available_providers()
        )
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

        # ---- optional geo index -------------------------------------------
        self.geo_index_loaded = False
        self._geo_cells: dict[int, set[tuple[int, int]]] = {}
        self._cell_size = 1.0
        geo_path = art / "geo_index.json"
        if geo_path.exists():
            geo = json.loads(geo_path.read_text())
            self._cell_size = float(geo.get("cell_size_deg", 1.0))
            slug_to_idx = {v["slug"]: k for k, v in self.taxonomy.items()}
            for slug, cells in geo.get("cells", {}).items():
                idx = slug_to_idx.get(slug)
                if idx is not None:
                    self._geo_cells[idx] = {(int(a), int(b)) for a, b in cells}
            self.geo_index_loaded = True

    # ------------------------------------------------------------- preprocess
    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = np.transpose(arr, (2, 0, 1))
        return arr[None, ...].astype(np.float32)

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
        emb = emb / np.clip(np.linalg.norm(emb), 1e-8, None)
        sims = self.prototypes @ emb                       # raw cosine, (N,)

        geo_active = (lat is not None and lon is not None
                      and self.geo_index_loaded)
        base_order = np.argsort(-sims, kind="stable")
        base_rank = {int(c): r for r, c in enumerate(base_order)}

        if geo_active:
            in_range = np.array(
                [self._in_range(i, lat, lon) for i in range(self.species_count)]
            )
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
        }

    def species_list(self) -> list[dict]:
        return [self.taxonomy[i] for i in sorted(self.taxonomy)]
