#!/usr/bin/env python3
"""eval_benchmark.py — evaluate trained artifacts against the frozen benchmark.

Unlike evaluate.py (which evaluates against the training manifest's val split,
now pinned by val_split.json), this script evaluates against
data/benchmark_v1/benchmark_v1.csv -- an entirely separate, independently
scraped set verified to share zero photo_id, zero observation_uuid, and zero
image sha256 with the training set (see data/benchmark_v1/benchmark_v1.json).
It exists because the original training-time split could not be reproduced
(see training/artifacts/README.md); this is the number to cite instead.

One forward pass over the benchmark produces both the raw-cosine metrics and
the geo-re-ranked metrics together (GeoReranker is applied per-sample using
each image's own coordinates, exactly as api/inference.py does at request
time) -- there is no retraining, tuning, or repeated runs here by design.

Usage:
    python eval_benchmark.py
    python eval_benchmark.py --benchmark-csv ../data/benchmark_v1/benchmark_v1.csv \
        --artifacts artifacts --out ../data/benchmark_v1/benchmark_v1_eval.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import AntDataset, Sample
from evaluate import GeoReranker, load_geo_index
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


def load_benchmark(csv_path: Path, image_dir: Path, slug_to_idx: dict[str, int]):
    """Resolve every benchmark_v1.csv row to exactly one verified local image.

    Fails closed, not open: if any row is missing, ambiguous (more than one
    file matching {photo_id}.*), or hash-mismatched against the CSV's own
    sha256, this raises SystemExit listing every problem found -- it never
    returns a partial sample list. A benchmark number computed over "however
    many images happened to be present" is not a number that means anything;
    silently evaluating 1400 of 1591 rows would look like a normal run and
    quietly report a different, incomparable benchmark. Fix the local copy
    with:  python ../data_pipeline/scrape_benchmark.py --restore
    """
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    samples: list[Sample] = []
    coords: list[tuple[float | None, float | None]] = []
    problems: list[str] = []

    for r in rows:
        slug, photo_id = r["slug"], r["photo_id"]
        if slug not in slug_to_idx:
            raise SystemExit(f"benchmark species {slug!r} is not in taxonomy.json "
                             f"-- benchmark and artifacts are out of sync")

        matches = sorted((image_dir / slug).glob(f"{photo_id}.*")) \
            if (image_dir / slug).exists() else []
        if len(matches) == 0:
            problems.append(f"{slug}/{photo_id}: no local image file")
            continue
        if len(matches) > 1:
            problems.append(f"{slug}/{photo_id}: {len(matches)} files match "
                            f"(ambiguous) -- {[p.name for p in matches]}")
            continue
        path = matches[0]
        if path.suffix.lower() not in IMG_EXTS:
            problems.append(f"{slug}/{photo_id}: matched file {path.name} has an "
                            f"unrecognized extension")
            continue

        digest = sha256_file(path)
        if digest != r["sha256"]:
            problems.append(f"{slug}/{photo_id}: sha256 mismatch "
                            f"(local {digest[:12] if digest else 'MISSING'}..., "
                            f"csv {r['sha256'][:12]}...)")
            continue

        samples.append(Sample(str(path), slug_to_idx[slug], slug))
        lat = float(r["lat"]) if r.get("lat") else None
        lon = float(r["lon"]) if r.get("lon") else None
        coords.append((lat, lon))

    if problems or len(samples) != len(rows):
        print(f"[bench-eval] FAILED verification: {len(problems)}/{len(rows)} "
             f"row(s) did not resolve to exactly one hash-verified image.")
        for p in problems[:50]:
            print(f"  ! {p}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        raise SystemExit(
            f"[bench-eval] Refusing to evaluate a partial or unverified benchmark "
            f"({len(samples)}/{len(rows)} rows OK). This never silently evaluates a "
            f"subset. Fix the local copy with:\n"
            f"    python ../data_pipeline/scrape_benchmark.py --restore --out {image_dir}"
        )

    print(f"[bench-eval] verified {len(samples)}/{len(rows)} rows: exactly one "
         f"hash-matched image each")
    return samples, coords


def _species_block(correct1, correct3, total, idx_to_slug) -> dict:
    per_species = {}
    for idx in range(len(total)):
        t = int(total[idx])
        if t == 0:
            continue
        per_species[idx_to_slug[idx]] = {
            "n": t,
            "top1": float(correct1[idx] / t),
            "top3": float(correct3[idx] / t),
        }
    tot = int(total.sum())
    micro = {
        "n": tot,
        "top1": float(correct1.sum() / tot) if tot else 0.0,
        "top3": float(correct3.sum() / tot) if tot else 0.0,
    }
    species_with_data = [v for v in per_species.values() if v["n"] > 0]
    macro = {
        "n_species": len(species_with_data),
        "top1": float(np.mean([v["top1"] for v in species_with_data])) if species_with_data else 0.0,
        "top3": float(np.mean([v["top3"] for v in species_with_data])) if species_with_data else 0.0,
    }
    return {"micro": micro, "macro": macro, "per_species": per_species}


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-csv", type=Path,
                    default=HERE.parent / "data" / "benchmark_v1" / "benchmark_v1.csv")
    ap.add_argument("--benchmark-dir", type=Path,
                    default=HERE.parent / "data" / "benchmark_v1")
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--geo-boost", type=float,
                    default=float(os.environ.get("GEO_BOOST", "0.05")))
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / "data" / "benchmark_v1" / "benchmark_v1_eval.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    taxonomy_raw = json.loads((args.artifacts / "taxonomy.json").read_text())
    taxonomy = {int(k): v for k, v in taxonomy_raw.items()}
    slug_to_idx = {v["slug"]: k for k, v in taxonomy.items()}
    idx_to_slug = {k: v["slug"] for k, v in taxonomy.items()}
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

    samples, coords = load_benchmark(args.benchmark_csv, args.benchmark_dir, slug_to_idx)
    print(f"[bench-eval] {len(samples)} benchmark images resolved "
         f"({len(coords) - sum(1 for a, b in coords if a is None or b is None)} with usable coordinates)")

    ds = AntDataset(samples, cfg, train=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg.get("num_workers", 0))

    geo_path = args.artifacts / "geo_index.json"
    geo = None
    if geo_path.exists():
        cells, cell_size = load_geo_index(geo_path, taxonomy)
        geo = GeoReranker(cells=cells, cell_size=cell_size, coords=coords,
                          boost=args.geo_boost, n_classes=n_classes,
                          source=geo_path.name)
    else:
        print(f"[bench-eval] WARNING: no {geo_path} -- geo results will equal raw results")

    protos = nn.functional.normalize(
        torch.as_tensor(protos_np, dtype=torch.float32, device=device), dim=1)

    labels_all = np.array([s.label for s in samples])
    n = len(samples)
    base_rank = np.full(n, -1, dtype=np.int64)
    geo_rank = np.full(n, -1, dtype=np.int64)
    has_coords = np.array([a is not None and b is not None for a, b in coords])
    k = min(3, n_classes)

    cursor = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = nn.functional.normalize(model.embed(imgs), dim=1)
            sims = (emb @ protos.T).cpu().numpy()             # (B, n_classes)
            labels_np = labels.numpy()
            for j, lbl in enumerate(labels_np):
                i = cursor + j
                order = np.argsort(-sims[j], kind="stable")
                base_rank[i] = int(np.where(order == lbl)[0][0])
                if geo is not None and has_coords[i]:
                    lat, lon = coords[i]
                    adjusted = sims[j] + geo.boost * geo.in_range(lat, lon)
                    gorder = np.argsort(-adjusted, kind="stable")
                    geo_rank[i] = int(np.where(gorder == lbl)[0][0])
                else:
                    geo_rank[i] = base_rank[i]
            cursor += len(labels_np)
    assert cursor == n and (base_rank >= 0).all()

    def block_from_ranks(ranks, mask=None):
        idx = np.where(mask)[0] if mask is not None else np.arange(n)
        correct1 = np.zeros(n_classes, dtype=np.int64)
        correct3 = np.zeros(n_classes, dtype=np.int64)
        total = np.zeros(n_classes, dtype=np.int64)
        for i in idx:
            lbl = labels_all[i]
            total[lbl] += 1
            if ranks[i] == 0:
                correct1[lbl] += 1
            if ranks[i] < k:
                correct3[lbl] += 1
        return _species_block(correct1, correct3, total, idx_to_slug)

    raw_block = block_from_ranks(base_rank)
    geo_all_block = block_from_ranks(geo_rank) if geo is not None else None
    geo_coord_block = block_from_ranks(geo_rank, mask=has_coords) if geo is not None else None
    raw_coord_block = block_from_ranks(base_rank, mask=has_coords)  # for comparison alongside geo_coord_block

    n_with_coords = int(has_coords.sum())
    csv_hash = sha256_file(args.benchmark_csv)
    artifact_hashes = {
        name: sha256_file(args.artifacts / name)
        for name in ("model.pth", "backbone.onnx", "prototypes.npy", "taxonomy.json", "geo_index.json")
    }
    config_hash = sha256_file(args.config)

    results = {
        "benchmark": "benchmark_v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_images_evaluated": n,
        "n_species": len(set(labels_all.tolist())),
        "coordinates": {
            "n_with_usable_coordinates": n_with_coords,
            "n_total": n,
            "pct_with_coordinates": round(100.0 * n_with_coords / n, 2) if n else 0.0,
        },
        "geo_config": {
            "geo_index_present": geo is not None,
            "geo_boost": args.geo_boost if geo is not None else None,
            "cell_size_deg": geo.cell_size if geo is not None else None,
        },
        "raw_cosine": {
            "description": "No geo re-ranking. All 1591 benchmark images.",
            "overall_micro_top1": raw_block["micro"]["top1"],
            "overall_micro_top3": raw_block["micro"]["top3"],
            "macro_top1": raw_block["macro"]["top1"],
            "macro_top3": raw_block["macro"]["top3"],
            "macro_n_species": raw_block["macro"]["n_species"],
            "per_species": raw_block["per_species"],
        },
        "raw_cosine_coord_subset": {
            "description": "Same raw-cosine model, restricted to the coordinate-bearing "
                           "subset only -- for comparison against geo_reranking.coord_subset.",
            "overall_micro_top1": raw_coord_block["micro"]["top1"],
            "overall_micro_top3": raw_coord_block["micro"]["top3"],
            "macro_top1": raw_coord_block["macro"]["top1"],
            "macro_top3": raw_coord_block["macro"]["top3"],
            "macro_n_species": raw_coord_block["macro"]["n_species"],
            "n": raw_coord_block["micro"]["n"],
        },
        "geo_reranking": None if geo is None else {
            "all_images": {
                "description": "Geo re-ranking applied where coordinates exist; images "
                               "without coordinates fall back to raw-cosine rank unchanged. "
                               "All 1591 benchmark images.",
                "overall_micro_top1": geo_all_block["micro"]["top1"],
                "overall_micro_top3": geo_all_block["micro"]["top3"],
                "macro_top1": geo_all_block["macro"]["top1"],
                "macro_top3": geo_all_block["macro"]["top3"],
                "macro_n_species": geo_all_block["macro"]["n_species"],
                "per_species": geo_all_block["per_species"],
            },
            "coord_subset": {
                "description": "Geo re-ranking, restricted to only the images that had "
                               "usable coordinates -- isolates the geo effect from dilution "
                               "by the no-coordinate images (which are identical to raw-cosine).",
                "overall_micro_top1": geo_coord_block["micro"]["top1"],
                "overall_micro_top3": geo_coord_block["micro"]["top3"],
                "macro_top1": geo_coord_block["macro"]["top1"],
                "macro_top3": geo_coord_block["macro"]["top3"],
                "macro_n_species": geo_coord_block["macro"]["n_species"],
                "n": geo_coord_block["micro"]["n"],
                "per_species": geo_coord_block["per_species"],
            },
        },
        "low_sample_species": {
            "note": "n<5 species have too few trials for their per-species top1/top3 to "
                   "be statistically meaningful; anoplolepis-custodiens and "
                   "polyrhachis-schistacea (n=1 each) are the extreme case.",
            "slugs_n_below_5": sorted(
                slug for slug, v in raw_block["per_species"].items() if v["n"] < 5
            ),
        },
        "hashes": {
            "benchmark_v1_csv_sha256": csv_hash,
            "config_yaml_sha256": config_hash,
            "artifacts_sha256": artifact_hashes,
        },
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\n[bench-eval] n={n}  with_coords={n_with_coords} "
         f"({results['coordinates']['pct_with_coordinates']}%)")
    print(f"[bench-eval] RAW    micro top1={raw_block['micro']['top1']:.4f} "
         f"top3={raw_block['micro']['top3']:.4f}  "
         f"macro top1={raw_block['macro']['top1']:.4f} top3={raw_block['macro']['top3']:.4f}")
    if geo is not None:
        print(f"[bench-eval] GEO(all)   micro top1={geo_all_block['micro']['top1']:.4f} "
             f"top3={geo_all_block['micro']['top3']:.4f}  "
             f"macro top1={geo_all_block['macro']['top1']:.4f} top3={geo_all_block['macro']['top3']:.4f}")
        print(f"[bench-eval] GEO(coord-subset, n={geo_coord_block['micro']['n']})  "
             f"micro top1={geo_coord_block['micro']['top1']:.4f} "
             f"top3={geo_coord_block['micro']['top3']:.4f}  "
             f"(raw on same subset: top1={raw_coord_block['micro']['top1']:.4f} "
             f"top3={raw_coord_block['micro']['top3']:.4f})")
    print(f"[bench-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
