#!/usr/bin/env python3
"""eval_calibration.py — embed data/calibration_v1/ and record the two raw
signals a rejection rule would use: max cosine similarity to any of the 50
class prototypes, and the top1-top2 similarity gap. Raw cosine only, no geo
re-ranking -- this is an "is this even one of the 50" question, not a ranking
question among them.

This does NOT propose a threshold or compute FRR/FAR itself; it just produces
one verified record per image (category, max_sim, top1_idx, top2_sim, gap,
blur_score) so the analysis step can be re-run / re-sliced without re-running
the model. Same fail-closed verification as eval_benchmark.py: refuses to run
unless every calibration_v1.csv row resolves to exactly one hash-matched
local image.

Usage:
    python eval_calibration.py
    python eval_calibration.py --out ../data/calibration_v1/calibration_v1_scores.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import AntDataset, Sample
from model import AntIDModel

HERE = Path(__file__).resolve().parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_calibration(csv_path: Path, image_dir: Path, slug_to_idx: dict[str, int] | None):
    """Strict, fail-closed resolution -- mirrors eval_benchmark.load_benchmark.

    calibration_v1 includes species outside the 50 known classes (that's the
    point), so `slug_to_idx` is only used to record a known-class label where
    one exists; out-of-scope rows get label=-1 and are never used as a
    training/ranking target, only as "which known-class prototype did this
    embedding land closest to".
    """
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    samples: list[Sample] = []
    meta: list[dict] = []
    problems: list[str] = []

    for r in rows:
        slug, photo_id = r["slug"], r["photo_id"]
        matches = sorted((image_dir / slug).glob(f"{photo_id}.*")) \
            if (image_dir / slug).exists() else []
        if len(matches) == 0:
            problems.append(f"{slug}/{photo_id}: no local image file")
            continue
        if len(matches) > 1:
            problems.append(f"{slug}/{photo_id}: {len(matches)} files match (ambiguous)")
            continue
        path = matches[0]
        if path.suffix.lower() not in IMG_EXTS:
            problems.append(f"{slug}/{photo_id}: unrecognized extension {path.suffix}")
            continue
        digest = sha256_file(path)
        if digest != r["sha256"]:
            problems.append(f"{slug}/{photo_id}: sha256 mismatch")
            continue

        label = slug_to_idx.get(slug, -1) if slug_to_idx else -1
        samples.append(Sample(str(path), max(label, 0), slug))
        meta.append({
            "photo_id": photo_id, "slug": slug, "species": r["species"],
            "category": r["category"], "taxon_id": r.get("taxon_id"),
            "known_label": label, "blur_score": r.get("blur_score"),
        })

    if problems or len(samples) != len(rows):
        print(f"[calib-eval] FAILED verification: {len(problems)}/{len(rows)} row(s) "
             f"did not resolve to exactly one hash-verified image.")
        for p in problems[:50]:
            print(f"  ! {p}")
        raise SystemExit(
            f"[calib-eval] Refusing to evaluate a partial or unverified calibration set "
            f"({len(samples)}/{len(rows)} rows OK). Restore with:\n"
            f"    python ../data_pipeline/scrape_calibration.py --restore --out {image_dir}"
        )
    print(f"[calib-eval] verified {len(samples)}/{len(rows)} rows: exactly one "
         f"hash-matched image each")
    return samples, meta


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-csv", type=Path,
                    default=HERE.parent / "data" / "calibration_v1" / "calibration_v1.csv")
    ap.add_argument("--calibration-dir", type=Path,
                    default=HERE.parent / "data" / "calibration_v1")
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / "data" / "calibration_v1" / "calibration_v1_scores.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    taxonomy_raw = json.loads((args.artifacts / "taxonomy.json").read_text())
    taxonomy = {int(k): v for k, v in taxonomy_raw.items()}
    slug_to_idx = {v["slug"]: k for k, v in taxonomy.items()}
    protos_np = np.load(args.artifacts / "prototypes.npy")
    n_classes = protos_np.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or (args.artifacts / "model.pth")
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model = AntIDModel(num_classes=n_classes, backbone=cfg["model"]["backbone"],
                       pretrained=False, dropout=cfg["dropout"],
                       embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    model.load_state_dict(sd)
    model.eval()

    samples, meta = load_calibration(args.calibration_csv, args.calibration_dir, slug_to_idx)
    ds = AntDataset(samples, cfg, train=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg.get("num_workers", 0))

    protos = nn.functional.normalize(
        torch.as_tensor(protos_np, dtype=torch.float32, device=device), dim=1)

    records = []
    cursor = 0
    with torch.no_grad():
        for imgs, _labels in loader:
            imgs = imgs.to(device)
            emb = nn.functional.normalize(model.embed(imgs), dim=1)
            sims = (emb @ protos.T).cpu().numpy()  # (B, n_classes)
            for j in range(sims.shape[0]):
                s = sims[j]
                order = np.argsort(-s)
                top1_idx, top2_idx = int(order[0]), int(order[1])
                m = meta[cursor + j]
                records.append({
                    **m,
                    "max_sim": float(s[top1_idx]),
                    "top1_slug": taxonomy[top1_idx]["slug"],
                    "top1_is_true_known_class": bool(top1_idx == m["known_label"])
                                                if m["known_label"] >= 0 else None,
                    "top2_sim": float(s[top2_idx]),
                    "gap": float(s[top1_idx] - s[top2_idx]),
                })
            cursor += sims.shape[0]
    assert cursor == len(samples)

    hashes = {
        "calibration_v1_csv": sha256_file(args.calibration_csv),
        "checkpoint": sha256_file(ckpt_path),
        "prototypes_npy": sha256_file(args.artifacts / "prototypes.npy"),
        "taxonomy_json": sha256_file(args.artifacts / "taxonomy.json"),
        "config_yaml": sha256_file(args.config),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n": len(records),
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
        "hashes": hashes,
        "ground_truth_policy": {
            "positive_for_false_rejection": "known_holdout only (research-grade)",
            "negative_for_false_acceptance": "out_of_scope_ant, non_ant_insect, unrelated",
            "low_quality_known": "exploratory/diagnostic only -- needs_id species labels "
                                 "are not verified ground truth; never used as positive "
                                 "or negative ground truth in any FRR/FAR calculation",
        },
        "records": records,
    }, indent=1))
    print(f"[calib-eval] wrote {args.out} ({len(records)} records)")
    for k, v in hashes.items():
        print(f"  hash {k}: {v[:16] if v else 'MISSING'}...")

    import collections
    by_cat = collections.defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["max_sim"])
    for cat, vals in sorted(by_cat.items()):
        arr = np.array(vals)
        print(f"  {cat:20s} n={len(arr):4d}  max_sim mean={arr.mean():.3f} "
             f"median={np.median(arr):.3f} p5={np.percentile(arr,5):.3f} "
             f"p95={np.percentile(arr,95):.3f}")


if __name__ == "__main__":
    main()
