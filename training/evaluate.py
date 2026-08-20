#!/usr/bin/env python3
"""evaluate.py — top-1 / top-3 accuracy under cosine-similarity inference.

This mirrors serving exactly: embed each val image, L2-normalize, take cosine
similarity against the prototype matrix, rank. Reports per-species and overall
accuracy so training and serving can't silently diverge.

Importable as `topk_accuracy(...)` (used by train.py) and runnable standalone:

    python evaluate.py --artifacts artifacts --data-dir ../data/clean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def topk_accuracy(model, prototypes, loader, taxonomy, device) -> dict:
    """Return {'overall': {...}, 'per_species': {slug: {...}}} accuracy dict."""
    model.eval()
    protos = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
    protos = nn.functional.normalize(protos, dim=1)

    n_classes = protos.shape[0]
    correct1 = np.zeros(n_classes, dtype=np.int64)
    correct3 = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)

    for imgs, labels in loader:
        imgs = imgs.to(device)
        emb = nn.functional.normalize(model.embed(imgs), dim=1)
        sims = emb @ protos.T                      # (B, n_classes) cosine
        top3 = sims.topk(min(3, n_classes), dim=1).indices.cpu().numpy()
        labels = labels.numpy()
        for lbl, preds in zip(labels, top3):
            total[lbl] += 1
            if preds[0] == lbl:
                correct1[lbl] += 1
            if lbl in preds:
                correct3[lbl] += 1

    per_species = {}
    for idx in range(n_classes):
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
    val = [s for s in samples if s.__dict__.get("split") == "val"] or samples
    loader = DataLoader(AntDataset(val, cfg, train=False), batch_size=cfg["batch_size"])
    metrics = topk_accuracy(model, protos, loader, taxonomy, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
