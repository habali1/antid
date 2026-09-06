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

`--geo-source train` requires a val_split.json pinning BOTH "train" and "val"
membership (train.py's build_val_split_record writes both, additively, for
new runs) and fails closed -- rather than falling back to the manifest's raw
split column -- if that pinning is missing, incomplete, non-unique, or not
disjoint from the val set actually being evaluated. See
resolve_pinned_train_split. A legacy split file that only pins "val" cannot
prove historical train membership from its complement: the manifest's split
column is exactly what gets silently rewritten (see the reproducibility
warning in training/artifacts/README.md), so "everything not in val" is an
assumption, not a verified claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import numerics
import run_manifest_schema
from data_provenance import (
    load_explicit_manifest_source,
    resolved_config_sha256,
    taxonomy_matches_committed,
    verify_image_bytes,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "config.yaml"


def resolve_eval_config(supplied_config_path: Path | None, run_config: dict,
                        expected_config_sha256: str) -> dict:
    """Decide which config to evaluate a PROVENANCE-AWARE checkpoint with:
    the run's own recorded config by default, UNLESS the caller explicitly
    supplied a --config different from the CLI default, in which case it
    must hash-match this run's resolved_config_sha256 or this fails closed
    (never silently ignored, never silently substituted)."""
    if supplied_config_path is not None and supplied_config_path != DEFAULT_CONFIG_PATH:
        supplied_cfg = yaml.safe_load(supplied_config_path.read_text())
        actual = resolved_config_sha256(supplied_cfg)
        if actual != expected_config_sha256:
            raise SystemExit(
                f"--config {supplied_config_path} (resolved sha256 {actual}) does not "
                f"match this run's resolved_config_sha256 ({expected_config_sha256}) -- "
                f"refusing to evaluate with a mismatched config. Omit --config to use "
                f"the run's own recorded config."
            )
        return supplied_cfg
    return run_config


def load_and_validate_run_manifest(run_manifest_path: Path) -> dict:
    """Load run_manifest.json, validate it against the shared schema at the
    "data_verified" stage (evaluate.py cannot resolve a data source without
    those fields), and confirm it actually recorded an explicit source.

    Raises SystemExit with a specific, field-named message -- never lets a
    raw KeyError from a missing/malformed field propagate to the caller.
    """
    if not run_manifest_path.exists():
        raise SystemExit(f"{run_manifest_path} does not exist -- cannot resolve a "
                         f"provenance-aware run's data source.")
    run_manifest = json.loads(run_manifest_path.read_text())
    try:
        run_manifest_schema.validate_run_manifest(run_manifest, stage="data_verified")
    except run_manifest_schema.RunManifestValidationError as e:
        raise SystemExit(
            f"{run_manifest_path} failed schema validation: {e} -- refusing to read any "
            f"nested field from a run_manifest that doesn't conform to the schema train.py "
            f"and evaluate.py share (see run_manifest_schema.py)."
        ) from e
    if run_manifest["manifest"] is None or run_manifest["taxonomy_source"] is None:
        raise SystemExit(
            f"{run_manifest_path} has no recorded manifest/taxonomy_source -- this run did "
            f"not use an explicit data source, so a provenance-aware evaluation cannot "
            f"resolve one. (Its checkpoint should not have carried non-null "
            f"manifest_sha256/taxonomy_sha256 provenance in that case; this is likely a "
            f"corrupted or hand-edited run_manifest.json.)"
        )
    return run_manifest


def verify_run_manifest_matches_checkpoint_provenance(run_manifest: dict, provenance: dict) -> None:
    """run_manifest.json and the checkpoint's embedded provenance are two
    independently written records of the same three facts (manifest,
    taxonomy_source, val_split hashes) -- they must agree before either is
    trusted. Raises SystemExit naming the first field that disagrees."""
    cross_checks = (
        ("manifest", run_manifest["manifest"]["sha256"], provenance["manifest_sha256"]),
        ("taxonomy_source", run_manifest["taxonomy_source"]["sha256"], provenance["taxonomy_sha256"]),
        ("val_split", run_manifest["val_split"]["sha256"], provenance["val_split_sha256"]),
    )
    for name, from_manifest, from_provenance in cross_checks:
        if from_manifest != from_provenance:
            raise SystemExit(
                f"run_manifest.json's {name}.sha256 ({from_manifest}) does not match the "
                f"checkpoint's embedded provenance {name}_sha256 ({from_provenance}) -- "
                f"refusing to evaluate with disagreeing provenance records."
            )


def verify_final_artifacts_against_run_manifest(run_manifest: dict, artifacts_dir: Path,
                                                ckpt_path: Path) -> None:
    """When a run completed successfully, run_manifest.json's
    final_artifact_hashes is an independent record of what SHOULD be on disk
    right now -- verify model.pth/prototypes.npy/taxonomy.json/val_split.json
    still match it before evaluating. No-op if the run isn't 'completed'."""
    if run_manifest.get("status") != "completed":
        return
    final_hashes = run_manifest["final_artifact_hashes"]
    for name in ("model.pth", "prototypes.npy", "taxonomy.json", "val_split.json"):
        path = ckpt_path if name == "model.pth" else (artifacts_dir / name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != final_hashes[name]:
            raise SystemExit(
                f"{path} sha256 {actual} does not match run_manifest.json's "
                f"final_artifact_hashes[{name!r}] ({final_hashes[name]}) -- refusing to "
                f"evaluate a completed run whose artifacts have changed since finalization."
            )


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

    This return contract (cells, cell_size) is shared with eval_benchmark.py
    and is intentionally left unchanged here. An index may also carry an
    optional "source_split" provenance field (see train.py's
    write_geo_index_sidecar); this function silently ignores it, exactly like
    api/inference.py's _load_geo_index does -- see describe_geo_file_source
    for the function that reads it, for labeling only.
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
    Pass the TRAIN split only for a leak-free geo estimate. New runs' shipped
    geo_index.json is itself already train-only (see train.py's
    write_geo_index_sidecar); this function exists for evaluate.py to
    independently rebuild that same train-only index from a verified pinned
    split (see resolve_pinned_train_split) rather than trusting the shipped
    file's own claimed provenance. Older shipped indexes may have been built
    from train+val (or have unknown provenance) -- see
    describe_geo_file_source.
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


def _find_duplicates(keys) -> list:
    seen, dupes = set(), []
    for k in keys:
        if k in seen:
            dupes.append(k)
        else:
            seen.add(k)
    return dupes


def resolve_pinned_train_split(samples, split_file: Path):
    """Resolve the exact TRAIN-only sample list for a leak-free geo rebuild.

    Requires split_file to pin BOTH "train" and "val" membership (the
    additive field train.py's build_val_split_record writes for new runs),
    and requires that pinning to be complete (every key resolves to exactly
    one manifest sample), unique (no key repeated, no sample claimed twice),
    and disjoint (no key pinned in both lists) before handing back a
    train-only sample list.

    Fails closed (raises SystemExit) rather than falling back to the
    manifest's raw split column: a legacy split file that only pins "val"
    cannot prove historical train membership from its complement -- the
    manifest's split column is exactly what gets silently rewritten (see the
    reproducibility warning in training/artifacts/README.md), so "everything
    not in val" would be an assumption, not a verified claim.
    """
    if not split_file.exists():
        raise SystemExit(
            f"--geo-source train needs {split_file} pinning both 'train' and "
            f"'val' membership, but it does not exist. Retrain to produce one "
            f"(new runs pin both), or use --geo-source file."
        )
    try:
        data = json.loads(split_file.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"Cannot read pinned split {split_file}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{split_file}: root must be a JSON object.")
    if "train" not in data:
        raise SystemExit(
            f"{split_file} only pins 'val' membership (a legacy split file "
            f"from before train.py pinned both). Its complement does not "
            f"prove historical train membership -- retrain to produce a split "
            f"file with pinned 'train' membership, or use --geo-source file."
        )
    if "val" not in data:
        raise SystemExit(
            f"{split_file} has a 'train' list but no 'val' list at all -- not "
            f"a valid pinned split file."
        )
    train_keys, val_keys = data["train"], data["val"]
    for label, keys in (("train", train_keys), ("val", val_keys)):
        if (not isinstance(keys, list) or not keys
                or any(not isinstance(k, str) or not k for k in keys)):
            raise SystemExit(
                f"{split_file}: '{label}' must be a non-empty JSON array of "
                "non-empty sample-key strings."
            )

    counts = {}
    for field in ("n_total", "n_train", "n_val"):
        value = data.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SystemExit(f"{split_file}: '{field}' must be a positive integer.")
        counts[field] = value
    if counts["n_train"] != len(train_keys) or counts["n_val"] != len(val_keys):
        raise SystemExit(
            f"{split_file}: pinned-list lengths do not match n_train/n_val."
        )
    if counts["n_total"] != counts["n_train"] + counts["n_val"]:
        raise SystemExit(
            f"{split_file}: n_total does not equal n_train + n_val."
        )

    train_dupes = _find_duplicates(train_keys)
    if train_dupes:
        raise SystemExit(f"{split_file}: 'train' list has duplicate key(s), "
                         f"e.g. {train_dupes[0]!r}.")
    val_dupes = _find_duplicates(val_keys)
    if val_dupes:
        raise SystemExit(f"{split_file}: 'val' list has duplicate key(s), "
                         f"e.g. {val_dupes[0]!r}.")

    overlap = set(train_keys) & set(val_keys)
    if overlap:
        raise SystemExit(
            f"{split_file}: {len(overlap)} key(s) pinned in BOTH 'train' and "
            f"'val' (e.g. {sorted(overlap)[0]!r}) -- membership is not "
            f"disjoint."
        )

    by_key: dict[str, list] = {}
    for s in samples:
        by_key.setdefault(f"{s.slug}/{Path(s.storage_path).stem}", []).append(s)

    def resolve(keys, label):
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit(
                f"{split_file}: {len(missing)} pinned '{label}' key(s) do not "
                f"resolve to any manifest sample (e.g. {missing[0]!r}); is "
                f"this split file from a different dataset or manifest state?"
            )
        dup = [k for k in keys if len(by_key[k]) > 1]
        if dup:
            raise SystemExit(
                f"{split_file}: {len(dup)} pinned '{label}' key(s) resolve to "
                f"more than one manifest sample (e.g. {dup[0]!r}); the "
                f"manifest has duplicate slug/photo entries."
            )
        return [by_key[k][0] for k in keys]

    train = resolve(train_keys, "train")
    resolve(val_keys, "val")  # same completeness/uniqueness guarantee, so the
                              # disjointness check above actually means something
    return train


def describe_geo_file_source(geo_path: Path) -> str:
    """Human-readable provenance label for a shipped geo_index.json.

    Reads the file's own optional "source_split" field instead of asserting a
    blanket claim. This is self-declared provenance, not independent proof:
    new exports written by train.py's write_geo_index_sidecar declare
    source_split="train"; an index that predates that field, or was built by
    another script (e.g.
    data_pipeline/build_geo_index.py), has provenance this function cannot
    verify -- it must not be labeled train-only or train+val as an
    established fact just because that used to be universally true.
    """
    try:
        raw = json.loads(geo_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return f"{geo_path.name} (provenance unknown -- file could not be read for labeling)"
    source_split = raw.get("source_split") if isinstance(raw, dict) else None
    if source_split == "train":
        return (f"{geo_path.name} (declares source_split=train; declaration "
                f"not independently verified)")
    if source_split is None:
        return (f"{geo_path.name} (source_split not recorded -- provenance unknown; "
                f"may include validation coordinates, not proven leak-free)")
    return (f"{geo_path.name} (unsupported source_split={source_split!r}; "
            f"provenance unknown, not proven leak-free)")


def require_usable_geo_cells(cells) -> None:
    """Keep evaluation's `--geo` behavior aligned with serving's active state."""
    if not cells:
        raise SystemExit(
            "--geo requested, but the selected geo index has no usable "
            "species cells. Serving would keep geo inactive rather than "
            "reporting an unchanged score as geo re-ranking."
        )


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
    from data import AntDataset, load_manifest
    from model import AntIDModel
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help="For a PROVENANCE-AWARE checkpoint (one with an embedded "
                         "'provenance' key), this is ignored in favor of the run's own "
                         "recorded config UNLESS explicitly overridden, in which case it "
                         "must hash-match the run's resolved config or this refuses to "
                         "run. For a legacy (pre-provenance) checkpoint, used as before.")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--run-manifest", type=Path, dest="run_manifest", default=None,
                    help="run_manifest.json for a provenance-aware run (default: "
                         "<artifacts>/run_manifest.json). Required to evaluate a "
                         "provenance-aware checkpoint.")
    ap.add_argument("--local-data-dir", type=Path, dest="local_data_dir", default=None,
                    help="Required to evaluate a provenance-aware checkpoint -- the "
                         "image root its recorded manifest path resolves against.")
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
                    help="'file' uses the shipped index as-is; its provenance is "
                         "read from the index's own optional source_split field "
                         "(see describe_geo_file_source), not assumed. 'train' "
                         "rebuilds a leak-free index from val_split.json's pinned "
                         "train membership only -- it requires a split file "
                         "pinning BOTH 'train' and 'val' and fails closed "
                         "otherwise (see resolve_pinned_train_split), e.g. on a "
                         "legacy val-only split file.")
    ap.add_argument("--geo-boost", type=float,
                    default=float(os.environ.get("GEO_BOOST", "0.05")),
                    help="Additive boost, matching api/inference.py GEO_BOOST.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write metrics JSON here (default: <artifacts>/eval.json). "
                         "Use --no-write to skip.")
    ap.add_argument("--no-write", action="store_true",
                    help="Print metrics without writing a file.")
    args = ap.parse_args()

    numerical_policy = numerics.apply_numerical_policy()
    print(f"[eval] numerical policy: {numerical_policy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint or (args.artifacts / "model.pth")
    state = torch.load(ckpt, map_location=device, weights_only=False)  # our own trusted checkpoints
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    provenance = state.get("provenance") if isinstance(state, dict) else None
    input_hashes: dict | None = None

    if provenance is not None:
        # ---- provenance-aware path: bind evaluation to the run that produced this
        # checkpoint. Never falls back to a DB, directory walk, manifest split, or
        # global config for this path. ----
        run_manifest_path = args.run_manifest or (args.artifacts / "run_manifest.json")
        run_manifest = load_and_validate_run_manifest(run_manifest_path)
        if args.local_data_dir is None:
            raise SystemExit("--local-data-dir is required to evaluate a "
                             "provenance-aware run.")
        if os.environ.get("DATABASE_URL"):
            raise SystemExit("DATABASE_URL is set -- refusing to evaluate a "
                             "provenance-aware run against anything other than its "
                             "recorded explicit manifest.")

        verify_run_manifest_matches_checkpoint_provenance(run_manifest, provenance)

        cfg = resolve_eval_config(
            args.config if args.config != DEFAULT_CONFIG_PATH else None,
            state["config"], provenance["resolved_config_sha256"],
        )
        taxonomy_raw = json.loads((args.artifacts / "taxonomy.json").read_text())
        taxonomy = {int(k): v for k, v in taxonomy_raw.items()}

        manifest_info = run_manifest["manifest"]
        taxonomy_info = run_manifest["taxonomy_source"]
        manifest_path = Path(manifest_info["path"])
        taxonomy_path = Path(taxonomy_info["path"])
        samples, verified_taxonomy, manifest_sha256, taxonomy_sha256, _ = load_explicit_manifest_source(
            manifest_path, args.local_data_dir, taxonomy_path,
            manifest_info["sha256"], taxonomy_info["sha256"],
            database_url=os.environ.get("DATABASE_URL"),
        )
        image_stats = verify_image_bytes(manifest_path, args.local_data_dir)
        print(f"[eval] verified {image_stats['files_verified']} image files "
             f"({image_stats['total_bytes']} bytes) against recorded sha256")

        # taxonomy.json (the artifact actually used for inference below) must
        # be the exact same object as the freshly re-verified taxonomy source
        # -- full-object equality, not merely the same length.
        if not taxonomy_matches_committed(verified_taxonomy, taxonomy_raw):
            raise SystemExit(
                f"{args.artifacts / 'taxonomy.json'} does not exactly match the "
                f"hash-verified taxonomy source recorded in run_manifest.json -- refusing "
                f"to evaluate with a taxonomy.json that has drifted from its source."
            )

        val_split_path = args.artifacts / "val_split.json"
        if not val_split_path.exists():
            raise SystemExit(f"{val_split_path} does not exist -- required for a "
                             f"provenance-aware evaluation.")
        val_split_bytes = val_split_path.read_bytes()
        val_split_sha256 = hashlib.sha256(val_split_bytes).hexdigest()
        if val_split_sha256 != run_manifest["val_split"]["sha256"]:
            raise SystemExit(f"{val_split_path} sha256 {val_split_sha256} does not match "
                             f"run_manifest.json's recorded val_split.sha256 "
                             f"{run_manifest['val_split']['sha256']} -- refusing to "
                             f"evaluate against an unexpected split.")
        split_record = json.loads(val_split_bytes)
        val_keys = set(split_record["val"])
        val = [s for s in samples if f"{s.slug}/{Path(s.storage_path).stem}" in val_keys]
        if len(val) != split_record["n_val"]:
            raise SystemExit(f"resolved {len(val)} val samples from the pinned split, "
                             f"expected {split_record['n_val']}")
        print(f"[eval] provenance-aware: {len(val)} pinned val images resolved and "
             f"hash-verified (manifest={manifest_sha256}, taxonomy={taxonomy_sha256})")

        verify_final_artifacts_against_run_manifest(run_manifest, args.artifacts, Path(ckpt))
        if run_manifest.get("status") == "completed":
            print("[eval] provenance-aware: model.pth/prototypes.npy/taxonomy.json/"
                 "val_split.json all match run_manifest.json's final_artifact_hashes")

        input_hashes = {
            "manifest_sha256": manifest_sha256, "taxonomy_sha256": taxonomy_sha256,
            "val_split_sha256": val_split_sha256,
        }
        split_file = val_split_path
    else:
        # ---- LEGACY fallback: only ever used for a checkpoint with no
        # embedded provenance (e.g. the live pre-Phase-4A 50-species
        # checkpoint). Never used for a provenance-aware artifact. ----
        print(f"[eval] LEGACY MODE: {ckpt} has no embedded provenance -- using --config "
             f"{args.config} and environment-driven load_manifest(). This path is never "
             f"used for a provenance-aware run.")
        cfg = yaml.safe_load(args.config.read_text())
        taxonomy_raw = json.loads((args.artifacts / "taxonomy.json").read_text())
        taxonomy = {int(k): v for k, v in taxonomy_raw.items()}
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

    protos = np.load(args.artifacts / "prototypes.npy")
    model = AntIDModel(num_classes=len(taxonomy),
                       backbone=cfg["model"]["backbone"], pretrained=False,
                       dropout=cfg["dropout"],
                       embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    model.load_state_dict(sd)

    # shuffle=False: geo coords are matched to images by position in this list.
    loader = DataLoader(AntDataset(val, cfg, train=False),
                        batch_size=cfg["batch_size"], shuffle=False)

    geo = None
    if args.geo:
        geo_cfg = cfg.get("geo") or {}
        if args.geo_source == "train":
            train_split = resolve_pinned_train_split(samples, split_file)
            cell_size = float(geo_cfg.get("cell_size_deg", 1.0))
            cells = build_geo_index_from(
                train_split, cell_size, int(geo_cfg.get("min_obs_per_cell", 2)))
            source = (f"rebuilt from {len(train_split)} train samples pinned by "
                      f"{split_file.name} (record counts, unique resolution, "
                      f"and train/val disjointness verified; leak-free)")
        else:
            gpath = args.geo_index or (args.artifacts / "geo_index.json")
            if not gpath.exists():
                raise SystemExit(
                    f"--geo needs a geo index but {gpath} does not exist. Train with "
                    "coordinates, or build one with data_pipeline/build_geo_index.py."
                )
            cells, cell_size = load_geo_index(gpath, taxonomy)
            source = describe_geo_file_source(gpath)

        require_usable_geo_cells(cells)
        coords = [(s.__dict__.get("lat"), s.__dict__.get("lon")) for s in val]
        geo = GeoReranker(cells=cells, cell_size=cell_size, coords=coords,
                          boost=args.geo_boost, n_classes=protos.shape[0],
                          source=source)

    metrics = topk_accuracy(model, protos, loader, taxonomy, device, geo=geo)
    metrics["numerical_policy"] = numerical_policy
    if input_hashes is not None:
        metrics["input_hashes"] = {
            **input_hashes,
            "model_pth_sha256": hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
            "prototypes_npy_sha256": hashlib.sha256(
                (args.artifacts / "prototypes.npy").read_bytes()).hexdigest(),
            "taxonomy_json_sha256": hashlib.sha256(
                (args.artifacts / "taxonomy.json").read_bytes()).hexdigest(),
        }

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
