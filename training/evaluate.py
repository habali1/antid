#!/usr/bin/env python3
"""evaluate.py — top-1 / top-3 accuracy under cosine-similarity inference.

This mirrors serving exactly: embed each val image, L2-normalize, take cosine
similarity against the prototype matrix, rank. Reports per-species and overall
accuracy so training and serving can't silently diverge.

The geo path (--geo) mirrors api/inference.py's re-ranking: species observed in
the image's own 1-degree grid cell or its 8 neighbours get an additive boost,
and results are re-ranked on the boosted score. Both the ranking predicate and
the cell arithmetic are re-implemented here rather than imported from the API,
for the same reason the cosine path is: if the two implementations drift, these
metrics move and the drift surfaces immediately.

Importable as `topk_accuracy(...)` (used by train.py) and runnable standalone:

    python evaluate.py                                  # cosine only
    python evaluate.py --geo                            # + geo re-ranking
    python evaluate.py --geo --geo-source train         # leak-free geo index
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------- geo
@dataclass
class GeoReranker:
    """Serving-equivalent geo boost, re-implemented for evaluation.

    `cells` maps class index -> set of occupied (lat_cell, lon_cell) integer
    cells, exactly the structure api/inference.py builds from geo_index.json.
    `coords` is per-validation-image (lat, lon), aligned with the dataset order
    the loader walks. Images without coordinates get no boost, which is what
    the API does when a request omits lat/lon.
    """

    cells: dict[int, set[tuple[int, int]]]
    cell_size: float
    coords: list[tuple[float | None, float | None]]
    boost: float
    n_classes: int
    source: str = "geo_index.json"
    _cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict, repr=False)

    def in_range(self, lat: float, lon: float) -> np.ndarray:
        """Boolean (n_classes,) mask: species recorded in this cell or a neighbour."""
        cs = self.cell_size
        key = (int(math.floor(lat / cs)), int(math.floor(lon / cs)))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        clat, clon = key
        mask = np.zeros(self.n_classes, dtype=bool)
        for idx in range(self.n_classes):
            occupied = self.cells.get(idx)
            if not occupied:
                continue
            for dlat in (-1, 0, 1):          # 3x3 neighbourhood, as in serving
                for dlon in (-1, 0, 1):
                    if (clat + dlat, clon + dlon) in occupied:
                        mask[idx] = True
                        break
                if mask[idx]:
                    break
        self._cache[key] = mask
        return mask


def load_geo_index(path: Path, taxonomy: dict[int, dict]
                   ) -> tuple[dict[int, set[tuple[int, int]]], float]:
    """Read geo_index.json into {class_idx: {(lat_cell, lon_cell)}} + cell size.

    The file is keyed by species SLUG; resolving through taxonomy is what keeps
    it aligned with prototype row order.
    """
    geo = json.loads(path.read_text())
    cell_size = float(geo.get("cell_size_deg", 1.0))
    slug_to_idx = {v["slug"]: k for k, v in taxonomy.items()}
    cells: dict[int, set[tuple[int, int]]] = {}
    for slug, cc in geo.get("cells", {}).items():
        idx = slug_to_idx.get(slug)
        if idx is not None:
            cells[idx] = {(int(a), int(b)) for a, b in cc}
    return cells, cell_size


def build_geo_index_from(samples, cell_size: float, min_obs_per_cell: int
                         ) -> dict[int, set[tuple[int, int]]]:
    """Build the cell index from a sample list, keyed by class index.

    Mirrors train.py's build_geo_index, but takes whichever split it is handed.
    Pass the TRAIN split only for a leak-free geo estimate: the index shipped in
    artifacts/ is built from train+val, so evaluating against it lets each val
    image's own coordinate vote for its own answer.
    """
    counts: dict[int, dict[tuple[int, int], int]] = {}
    for smp in samples:
        lat = smp.__dict__.get("lat")
        lon = smp.__dict__.get("lon")
        if lat is None or lon is None:
            continue
        cell = (int(math.floor(float(lat) / cell_size)),
                int(math.floor(float(lon) / cell_size)))
        per = counts.setdefault(smp.label, {})
        per[cell] = per.get(cell, 0) + 1
    return {
        label: {c for c, n in per.items() if n >= min_obs_per_cell}
        for label, per in counts.items()
        if any(n >= min_obs_per_cell for n in per.values())
    }


def _accuracy_block(correct1, correct3, total, taxonomy) -> dict:
    per_species = {}
    for idx in range(len(total)):
        slug = taxonomy[idx]["slug"] if idx in taxonomy else taxonomy[str(idx)]["slug"]
        t = int(total[idx])
        per_species[slug] = {
            "n": t,
            "top1": float(correct1[idx] / t) if t else None,
            "top3": float(correct3[idx] / t) if t else None,
        }
    tot = int(total.sum())
    return {
        "overall": {
            "n": tot,
            "top1": float(correct1.sum() / tot) if tot else 0.0,
            "top3": float(correct3.sum() / tot) if tot else 0.0,
        },
        "per_species": per_species,
    }


@torch.no_grad()
def topk_accuracy(model, prototypes, loader, taxonomy, device,
                  geo: GeoReranker | None = None) -> dict:
    """Return {'overall': {...}, 'per_species': {slug: {...}}} accuracy dict.

    When `geo` is supplied, a parallel geo-re-ranked score is computed from the
    same forward pass and reported under an extra 'geo' key. The base metrics
    are unchanged, so callers that ignore geo (train.py) see identical output.
    """
    model.eval()
    protos = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
    protos = nn.functional.normalize(protos, dim=1)

    n_classes = protos.shape[0]
    correct1 = np.zeros(n_classes, dtype=np.int64)
    correct3 = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)

    g_correct1 = np.zeros(n_classes, dtype=np.int64)
    g_correct3 = np.zeros(n_classes, dtype=np.int64)
    n_boostable = 0          # val images that actually carried coordinates
    rank_improved = 0        # true class moved up because of the boost
    rank_worsened = 0        # true class moved down because of the boost
    cursor = 0               # index into geo.coords; loader must be sequential

    k = min(3, n_classes)
    for imgs, labels in loader:
        imgs = imgs.to(device)
        emb = nn.functional.normalize(model.embed(imgs), dim=1)
        sims = emb @ protos.T                      # (B, n_classes) cosine
        top3 = sims.topk(k, dim=1).indices.cpu().numpy()
        labels = labels.numpy()
        for lbl, preds in zip(labels, top3):
            total[lbl] += 1
            if preds[0] == lbl:
                correct1[lbl] += 1
            if lbl in preds:
                correct3[lbl] += 1

        if geo is not None:
            sims_np = sims.cpu().numpy()
            for j, lbl in enumerate(labels):
                lat, lon = geo.coords[cursor + j]
                base_order = np.argsort(-sims_np[j], kind="stable")
                base_rank = int(np.where(base_order == lbl)[0][0])
                if lat is None or lon is None:
                    order = base_order
                else:
                    n_boostable += 1
                    adjusted = sims_np[j] + geo.boost * geo.in_range(lat, lon)
                    order = np.argsort(-adjusted, kind="stable")
                geo_rank = int(np.where(order == lbl)[0][0])
                if geo_rank < base_rank:
                    rank_improved += 1
                elif geo_rank > base_rank:
                    rank_worsened += 1
                if geo_rank == 0:
                    g_correct1[lbl] += 1
                if geo_rank < k:
                    g_correct3[lbl] += 1
        cursor += len(labels)

    metrics = _accuracy_block(correct1, correct3, total, taxonomy)
    if geo is not None:
        geo_block = _accuracy_block(g_correct1, g_correct3, total, taxonomy)
        base = metrics["overall"]
        metrics["geo"] = {
            "index_source": geo.source,
            "boost": geo.boost,
            "cell_size_deg": geo.cell_size,
            "n_with_coords": n_boostable,
            "species_in_index": len(geo.cells),
            "rank_improved": rank_improved,
            "rank_worsened": rank_worsened,
            "overall": geo_block["overall"],
            "per_species": geo_block["per_species"],
            "delta": {
                "top1": geo_block["overall"]["top1"] - base["top1"],
                "top3": geo_block["overall"]["top3"] - base["top3"],
            },
        }
    return metrics


def main() -> None:
    import yaml
    from data import AntDataset, load_manifest
    from model import AntIDModel
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split-file", type=Path, default=None,
                    help="val_split.json pinning the held-out set (default: "
                         "<artifacts>/val_split.json if present). Strongly "
                         "preferred: the manifest's split column and row order "
                         "are rewritten whenever the manifest is regenerated, "
                         "and a reconstructed split mixes training images back "
                         "in, inflating accuracy.")
    ap.add_argument("--geo", action="store_true",
                    help="Also report accuracy with geographic re-ranking.")
    ap.add_argument("--geo-index", type=Path, default=None,
                    help="geo_index.json to use (default: <artifacts>/geo_index.json).")
    ap.add_argument("--geo-source", choices=["file", "train"], default="file",
                    help="'file' uses the shipped index (built from train+val, so "
                         "val coordinates leak in); 'train' rebuilds it from the "
                         "train split only for a leak-free estimate.")
    ap.add_argument("--geo-boost", type=float,
                    default=float(os.environ.get("GEO_BOOST", "0.05")),
                    help="Additive boost, matching api/inference.py GEO_BOOST.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write metrics JSON here (default: <artifacts>/eval.json). "
                         "Use --no-write to skip.")
    ap.add_argument("--no-write", action="store_true",
                    help="Print metrics without writing a file.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    taxonomy_raw = json.loads((args.artifacts / "taxonomy.json").read_text())
    taxonomy = {int(k): v for k, v in taxonomy_raw.items()}
    protos = np.load(args.artifacts / "prototypes.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint or (args.artifacts / "model.pth")
    state = torch.load(ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model = AntIDModel(num_classes=len(taxonomy),
                       backbone=cfg["model"]["backbone"], pretrained=False,
                       dropout=cfg["dropout"],
                       embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    model.load_state_dict(sd)

    samples, _ = load_manifest(cfg)
    split_file = args.split_file or (args.artifacts / "val_split.json")
    if split_file.exists():
        keys = set(json.loads(split_file.read_text())["val"])
        val = [s for s in samples
               if f"{s.slug}/{Path(s.storage_path).stem}" in keys]
        if not val:
            raise SystemExit(
                f"{split_file} matched none of the {len(samples)} manifest samples. "
                "Is this split file from a different dataset?"
            )
        print(f"[eval] held-out set pinned by {split_file.name}: "
              f"{len(val)}/{len(keys)} images resolved")
    else:
        val = [s for s in samples if s.__dict__.get("split") == "val"] or samples
        print(f"[eval] WARNING: no {split_file.name}; falling back to the "
              f"manifest split ({len(val)} images). If the manifest was "
              f"regenerated after training, this set overlaps the training "
              f"data and the numbers below are inflated.")
    # shuffle=False: geo coords are matched to images by position in this list.
    loader = DataLoader(AntDataset(val, cfg, train=False),
                        batch_size=cfg["batch_size"], shuffle=False)

    geo = None
    if args.geo:
        geo_cfg = cfg.get("geo") or {}
        if args.geo_source == "train":
            train_split = [s for s in samples if s.__dict__.get("split") == "train"]
            if not train_split:
                raise SystemExit(
                    "--geo-source train needs an explicit train/val split in the "
                    "manifest; this data source did not provide one."
                )
            cell_size = float(geo_cfg.get("cell_size_deg", 1.0))
            cells = build_geo_index_from(
                train_split, cell_size, int(geo_cfg.get("min_obs_per_cell", 2)))
            source = f"rebuilt from {len(train_split)} train samples (leak-free)"
        else:
            gpath = args.geo_index or (args.artifacts / "geo_index.json")
            if not gpath.exists():
                raise SystemExit(
                    f"--geo needs a geo index but {gpath} does not exist. Train with "
                    "coordinates, or build one with data_pipeline/build_geo_index.py."
                )
            cells, cell_size = load_geo_index(gpath, taxonomy)
            source = f"{gpath.name} (built from train+val; val coordinates leak in)"

        coords = [(s.__dict__.get("lat"), s.__dict__.get("lon")) for s in val]
        geo = GeoReranker(cells=cells, cell_size=cell_size, coords=coords,
                          boost=args.geo_boost, n_classes=protos.shape[0],
                          source=source)

    metrics = topk_accuracy(model, protos, loader, taxonomy, device, geo=geo)

    o = metrics["overall"]
    print(f"[eval] cosine            n={o['n']}  "
          f"top1={o['top1']:.4f}  top3={o['top3']:.4f}")
    if geo is not None:
        g = metrics["geo"]
        go = g["overall"]
        print(f"[eval] + geo re-ranking  n={go['n']}  "
              f"top1={go['top1']:.4f}  top3={go['top3']:.4f}  "
              f"(+{g['delta']['top1']*100:.1f} / +{g['delta']['top3']*100:.1f} pts)")
        print(f"[eval] geo index: {g['index_source']}")
        print(f"[eval] boost={g['boost']} cell={g['cell_size_deg']}deg  "
              f"{g['n_with_coords']}/{go['n']} val images had coordinates  "
              f"ranks improved={g['rank_improved']} worsened={g['rank_worsened']}")

    if not args.no_write:
        out = args.out or (args.artifacts / "eval.json")
        out.write_text(json.dumps(metrics, indent=2))
        print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
