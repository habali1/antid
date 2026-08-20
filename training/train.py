#!/usr/bin/env python3
"""train.py — fine-tune EfficientNet-B4 for AntID and emit serving artifacts.

Outputs (into artifacts/):
  backbone.onnx     image (1,3,380,380) → 1792-dim embedding, dynamic batch
  prototypes.npy    (num_classes, 1792) mean train embedding per species
  taxonomy.json     class_idx → {species_name, common_name, taxon_id, slug}
  geo_index.json    {cell_size_deg, cells: {slug: [[lat, lon], …]}} (if coords)
  eval.json         per-species + overall top-1 / top-3 accuracy

Device: CUDA → MPS → CPU, auto-detected.
Logging: Weights & Biases if WANDB_API_KEY is set, else tqdm + stdout.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data import AntDataset, load_manifest
from evaluate import topk_accuracy
from export import export_backbone
from model import AntIDModel

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------- config
def load_config(path: Path, overrides: dict) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_samples(samples, val_fraction: float, seed: int):
    """Use explicit DB split if present; else random split by val_fraction."""
    have_split = all("split" in s.__dict__ for s in samples)
    if have_split:
        train = [s for s in samples if s.__dict__["split"] == "train"]
        val = [s for s in samples if s.__dict__["split"] == "val"]
        if train and val:
            return train, val
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(samples), generator=g).tolist()
    n_val = max(1, int(len(samples) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


# ----------------------------------------------------------------- prototypes
import math


def build_geo_index(samples, taxonomy, cell_size_deg: float = 1.0,
                    min_obs_per_cell: int = 2) -> dict[str, list[list[int]]]:
    """Map species SLUG -> sorted [lat_cell, lon_cell] observation cells.

    Returns the inner "cells" mapping of the file api/inference.py reads:
        {"cell_size_deg": float, "cells": {slug: [[lat, lon], ...]}}

    Keyed by slug, NOT class index. The API resolves slugs through taxonomy and
    reads cells from under the "cells" key, so a flat class-index-keyed file
    loads as zero cells while still reporting geo_index_loaded: true — the geo
    boost silently becomes a no-op. Keep this in step with
    data_pipeline/build_geo_from_manifest.py, which writes the same format.

    Coordinates come from Sample.lat/lon when the source provided them (DB
    columns or a manifest CSV). Samples without coordinates are skipped; cells
    seen fewer than min_obs_per_cell times are dropped as noise.
    """
    cs = float(cell_size_deg)
    counts: dict[int, dict[tuple[int, int], int]] = {}
    for smp in samples:
        lat = smp.__dict__.get("lat")
        lon = smp.__dict__.get("lon")
        if lat is None or lon is None:
            continue
        cell = (int(math.floor(float(lat) / cs)), int(math.floor(float(lon) / cs)))
        per = counts.setdefault(smp.label, {})
        per[cell] = per.get(cell, 0) + 1

    cells: dict[str, list[list[int]]] = {}
    for label, per in counts.items():
        kept = sorted([list(c) for c, n in per.items() if n >= min_obs_per_cell])
        if kept:
            cells[taxonomy[label]["slug"]] = kept
    return cells


@torch.no_grad()
def compute_prototypes(model, loader, num_classes, device, embedding_dim):
    """Mean (L2-normalized) embedding per class over the training set."""
    model.eval()
    sums = torch.zeros(num_classes, embedding_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for imgs, labels in loader:
        imgs = imgs.to(device)
        emb = model.embed(imgs)
        emb = nn.functional.normalize(emb, dim=1)
        for c in labels.unique():
            mask = labels == c
            sums[c] += emb[mask.to(device)].sum(0)
            counts[c] += mask.sum().to(device)
    counts = counts.clamp(min=1).unsqueeze(1)
    protos = nn.functional.normalize(sums / counts, dim=1)
    return protos.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--image-size", type=int, dest="image_size")
    ap.add_argument("--num-workers", type=int, dest="num_workers",
                    help="DataLoader workers (0 = single-process; safest on Windows).")
    ap.add_argument("--artifacts-dir", type=Path, dest="artifacts_dir")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="Cap batches/epoch (smoke tests).")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "image_size": args.image_size, "num_workers": args.num_workers,
    })
    if args.artifacts_dir:
        cfg["artifacts_dir"] = str(args.artifacts_dir)

    torch.manual_seed(cfg["seed"])
    device = pick_device()
    print(f"[train] device={device}")

    samples, taxonomy = load_manifest(cfg)
    num_classes = len(taxonomy)
    train_s, val_s = split_samples(samples, cfg["val_fraction"], cfg["seed"])
    print(f"[train] {len(train_s)} train / {len(val_s)} val / {num_classes} classes")

    train_ds = AntDataset(train_s, cfg, train=True)
    val_ds = AntDataset(val_s, cfg, train=False)
    proto_ds = AntDataset(train_s, cfg, train=False)  # no aug for prototypes
    nw = cfg["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=nw, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=nw)
    proto_dl = DataLoader(proto_ds, batch_size=cfg["batch_size"], num_workers=nw)

    model = AntIDModel(
        num_classes=num_classes,
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["dropout"],
        embedding_dim=cfg["model"]["embedding_dim"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    # optional W&B
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb
            wandb.init(project="antid", config=cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[train] W&B disabled ({e})")
            use_wandb = False

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):  # type: ignore
            return x

    started = datetime.now(timezone.utc)
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for bi, (imgs, labels) in enumerate(tqdm(train_dl, desc=f"epoch {epoch+1}")):
            if args.limit_batches and bi >= args.limit_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / max(1, min(len(train_dl), args.limit_batches or len(train_dl)))
        print(f"[train] epoch {epoch+1}/{cfg['epochs']} loss={avg:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": avg})

    finished = datetime.now(timezone.utc)

    # ---- artifacts ----
    art = Path(cfg["artifacts_dir"])
    if not art.is_absolute():
        art = HERE / art
    art.mkdir(parents=True, exist_ok=True)

    torch.save({"model": model.state_dict(), "config": cfg}, art / "model.pth")

    print("[train] computing prototypes…")
    protos = compute_prototypes(model, proto_dl, num_classes, device,
                                cfg["model"]["embedding_dim"])
    np.save(art / "prototypes.npy", protos)

    (art / "taxonomy.json").write_text(
        json.dumps({str(k): v for k, v in sorted(taxonomy.items())}, indent=2)
    )

    # Geo index: grid cells per species from observation coordinates, if the
    # source carries them (DB columns or a manifest CSV). Absent coords → no
    # file written, and the API simply reports geo_index_loaded: false.
    geo_cfg = cfg.get("geo") or {}
    cell_size = float(geo_cfg.get("cell_size_deg", 1.0))
    cells = build_geo_index(samples, taxonomy, cell_size,
                            int(geo_cfg.get("min_obs_per_cell", 2)))
    geo_path = art / "geo_index.json"
    if cells:
        geo_path.write_text(json.dumps(
            {"cell_size_deg": cell_size, "cells": cells}, indent=1))
        n_cells = sum(len(v) for v in cells.values())
        print(f"[train] wrote geo_index.json "
              f"({n_cells} cells across {len(cells)} species)")
    elif geo_path.exists():
        print(f"[train] WARNING: no coordinates in this run's data, so "
              f"{geo_path.name} was left untouched — it is from an earlier run "
              f"and may not match these prototypes.")

    print("[train] exporting ONNX backbone…")
    export_backbone(model, art / "backbone.onnx", cfg["image_size"], device)

    print("[train] evaluating…")
    metrics = topk_accuracy(model, protos, val_dl, taxonomy, device)
    (art / "eval.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] top1={metrics['overall']['top1']:.3f} "
          f"top3={metrics['overall']['top3']:.3f}")

    # ---- record run in DB (best-effort) ----
    db = os.environ.get("DATABASE_URL")
    if db:
        try:
            import psycopg2
            conn = psycopg2.connect(db)
            with conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO training_runs "
                    "(started_at, finished_at, config, top1_acc, top3_acc, artifact_path) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (started, finished, json.dumps(cfg),
                     metrics["overall"]["top1"], metrics["overall"]["top3"], str(art)),
                )
            conn.close()
            print("[train] logged training_runs row")
        except Exception as e:  # noqa: BLE001
            print(f"[train] could not log run: {e}")

    if use_wandb:
        wandb.finish()
    print(f"[train] artifacts written to {art}")


if __name__ == "__main__":
    main()
