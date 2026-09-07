#!/usr/bin/env python3
"""train.py — fine-tune EfficientNet-B4 for AntID and emit serving artifacts.

Phase 4A hardening (Northeast 65-species development harness), corrected
after a dedicated review pass: explicit, fail-closed, hash-and-image-byte-
verified data-source selection; a pinned full-FP32 numerical policy (see
numerics.py); deterministic RNG seeding/checkpointing with correct resume
ordering (model/optimizer built and loaded BEFORE RNG state is restored, so a
resumed run continues the exact same random stream an uninterrupted run
would have used); per-epoch serving-mirror validation with a frozen
selection rule; crash-consistent, versioned best-checkpoint commits with
checkpoint_last.pth as the single canonical resume marker (see
checkpoint.py); a completed-run resume refusal; and a --preflight-only mode
that performs every safety check -- including full image-byte hash
verification -- without initializing a model or downloading anything.

None of this changes the training recipe itself (same optimizer,
augmentations, epoch budget, architecture) -- see docs/plans/
northeast-expansion-v1.md and TODO.md for the full protocol.

Outputs (into artifacts/, only after a successful run):
  model.pth         best-epoch model weights + config + provenance
  prototypes.npy    (num_classes, 1792) mean train embedding per species,
                    recomputed from the restored best-epoch model
  taxonomy.json     class_idx → {species_name, common_name, taxon_id, slug, genus}
  val_split.json    pinned train+val membership (sorted keys), written BEFORE
                    training starts and hash-bound into every checkpoint
  geo_index.json    {cell_size_deg, cells: {slug: [[lat, lon], …]}, source_split:
                    "train"} -- built from the train split only
  eval.json         best-epoch pinned-val top-1/top-3 accuracy (raw cosine),
                    asserted to exactly match the recorded best-epoch metrics
                    before being written
  inference_policy.json is intentionally NEVER written by this script.

Resumable, crash-consistent per-epoch state (see checkpoint.py's module
docstring for the exact commit ordering):
  checkpoint_best_epoch_NNN.pth  immutable, versioned best-model snapshot
  checkpoint_last.pth            the canonical, resumable commit marker
  history.jsonl                  a pure cache, always rederivable from
                                 checkpoint_last.pth's embedded canonical history
  checkpoint_best.pth            materialized ONLY at successful finalization
  run_manifest.json              status/provenance/environment record

Device: CUDA → MPS → CPU, auto-detected.
Logging: Weights & Biases only with explicit --wandb (never from ambient
WANDB_API_KEY alone).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import DataLoader

import checkpoint as ckpt_mod
import numerics
import run_manifest_schema
from data import AntDataset, load_manifest
from data_provenance import (
    DataIntegrityError,
    EXPECTED_CLASS_COUNT,
    NORTHEAST_NEW_SPECIES_SLUGS,
    assert_dataset_shape,
    check_per_class_counts,
    load_explicit_manifest_source,
    resolved_config_sha256,
    taxonomy_matches_committed,
    verify_image_bytes,
)
from evaluate import topk_accuracy
from export import export_backbone
from model import AntIDModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
    split_markers = ["split" in s.__dict__ for s in samples]
    if any(split_markers):
        if not all(split_markers):
            raise ValueError(
                "Only some samples carry an explicit split. Refusing to mix "
                "explicit membership with a reconstructed random split."
            )
        invalid = sorted({
            repr(s.__dict__["split"])
            for s in samples if s.__dict__["split"] not in ("train", "val")
        })
        if invalid:
            raise ValueError(
                "Explicit sample splits must be 'train' or 'val'; found "
                + ", ".join(invalid)
            )
        train = [s for s in samples if s.__dict__["split"] == "train"]
        val = [s for s in samples if s.__dict__["split"] == "val"]
        if not train or not val:
            raise ValueError("Explicit split must contain both train and val samples.")
        assert len(train) + len(val) == len(samples)
        return train, val
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(samples), generator=g).tolist()
    n_val = max(1, int(len(samples) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


# ----------------------------------------------------------------- geo index
def build_geo_index(samples, taxonomy, cell_size_deg: float = 1.0,
                    min_obs_per_cell: int = 2) -> dict[str, list[list[int]]]:
    """Map species SLUG -> sorted [lat_cell, lon_cell] observation cells.
    See api/inference.py / data_pipeline/build_geo_from_manifest.py for the
    consuming format. Callers must pass the TRAIN split only."""
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


def write_geo_index_sidecar(geo_path: Path, cells: dict[str, list[list[int]]],
                            cell_size_deg: float, source_split: str = "train") -> int:
    """Write geo_index.json unconditionally, every run, ATOMICALLY (temp file
    + os.replace). An explicitly empty {"cells": {}} is written when there is
    nothing usable, so a stale index from an unrelated earlier run can never
    silently survive. Returns the number of cells written."""
    payload = json.dumps(
        {"cell_size_deg": float(cell_size_deg), "cells": cells,
         "source_split": source_split}, indent=1).encode("utf-8")
    ckpt_mod.atomic_write_bytes(geo_path, payload)
    return sum(len(v) for v in cells.values())


def build_val_split_record(cfg: dict, train_s, val_s) -> dict:
    """The pinned split record written to val_split.json. Pins BOTH train and
    val membership as sorted "{slug}/{stem}" keys."""
    def keys(samples):
        return sorted(f"{s.slug}/{Path(s.storage_path).stem}" for s in samples)

    train_keys = keys(train_s)
    val_keys = keys(val_s)
    if not train_keys or not val_keys:
        raise ValueError("Pinned split requires non-empty train and val membership.")

    def duplicates(values):
        seen, repeated = set(), set()
        for value in values:
            if value in seen:
                repeated.add(value)
            seen.add(value)
        return sorted(repeated)

    train_dupes = duplicates(train_keys)
    val_dupes = duplicates(val_keys)
    if train_dupes or val_dupes:
        example = (train_dupes or val_dupes)[0]
        raise ValueError(
            "Sample keys must be unique within each pinned split; duplicate "
            f"key {example!r}."
        )
    overlap = set(train_keys) & set(val_keys)
    if overlap:
        raise ValueError(
            "Pinned train and val membership must be disjoint; shared key "
            f"{sorted(overlap)[0]!r}."
        )

    return {
        "seed": cfg["seed"],
        "val_fraction": cfg["val_fraction"],
        "n_total": len(train_keys) + len(val_keys),
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "train": train_keys,
        "val": val_keys,
    }


def serialize_val_split(record: dict) -> bytes:
    """Deterministic bytes for val_split.json -- hashed BEFORE writing so the
    hash can be bound into every checkpoint's provenance."""
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


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


def _atomic_np_save(path: Path, array) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp{os.getpid()}{path.suffix}"
    try:
        np.save(tmp, array)  # tmp already ends with the real suffix, so numpy won't re-append it
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ----------------------------------------------------------------- output-dir safety
def check_output_dir_safety(art: Path, resume: bool) -> None:
    """Refuse a fresh run over an existing nonempty directory; refuse
    --resume against a directory with nothing to resume from."""
    exists = art.exists()
    nonempty = exists and any(art.iterdir())
    if nonempty and not resume:
        raise SystemExit(
            f"{art} already exists and is not empty. Refusing to start a fresh run "
            f"over existing contents -- pass --resume to continue an interrupted run "
            f"in this directory, or choose a different/empty --artifacts-dir."
        )
    if resume and not nonempty:
        raise SystemExit(
            f"--resume was given but {art} does not exist or is empty -- nothing to "
            f"resume from."
        )


RESUMABLE_STATUSES = run_manifest_schema.RESUMABLE_STATUSES


def check_resumable_status(run_manifest: dict, run_manifest_path: Path) -> None:
    status = run_manifest.get("status")
    if status == "completed":
        raise SystemExit(
            f"Refusing --resume: {run_manifest_path} status is 'completed' -- this run "
            f"already finished successfully. Choose a different --artifacts-dir for a "
            f"new run."
        )
    if status not in RESUMABLE_STATUSES:
        raise SystemExit(
            f"Refusing --resume: {run_manifest_path} status is {status!r}, expected one "
            f"of {RESUMABLE_STATUSES}."
        )


def bootstrap_run_manifest(run_manifest_path: Path, *, resume: bool, invocation_record: dict,
                           run_kind: str, git_commit: str | None, git_dirty: bool | None,
                           validation_cadence: int) -> dict:
    """The exact status-transition gate main() uses for BOTH a fresh run and
    a resume: create (fresh) or load-schema-validate-and-check (resume) the
    in-memory run_manifest dict. Does NOT write it -- the caller writes once
    it has also applied the invocation-specific status ("running", set here
    for resume; the fresh-run caller does the same after this returns).

    Fails closed (SystemExit) on: --resume with no existing file, an
    existing file that fails schema validation, a schema version other than
    the current one (a schema-v1 run -- implicit cadence 1, predating this
    field entirely -- is readable for evaluation but can never be resumed
    under this harness version), or a non-resumable status (see
    check_resumable_status) -- this is what makes "paused_for_smoke ->
    --resume" (and every other real status transition) an actually-tested
    code path, not just a run_epochs-level simulation.
    """
    if resume:
        if not run_manifest_path.exists():
            raise SystemExit(f"--resume given but {run_manifest_path} does not exist.")
        run_manifest = json.loads(run_manifest_path.read_text())
        try:
            run_manifest_schema.validate_run_manifest(run_manifest, stage="any")
        except run_manifest_schema.RunManifestValidationError as e:
            raise SystemExit(f"{run_manifest_path} failed schema validation: {e}") from e
        found_version = run_manifest.get("run_manifest_schema_version")
        if found_version != run_manifest_schema.RUN_MANIFEST_SCHEMA_VERSION:
            raise SystemExit(
                f"Refusing to resume: {run_manifest_path} has "
                f"run_manifest_schema_version={found_version!r}, but this harness only "
                f"resumes run_manifest_schema_version="
                f"{run_manifest_schema.RUN_MANIFEST_SCHEMA_VERSION!r}. A schema-v1 run "
                f"(implicit cadence=1, predating validation_cadence) remains readable for "
                f"evaluation but cannot be resumed under this harness version."
            )
        check_resumable_status(run_manifest, run_manifest_path)
        run_manifest.setdefault("invocations", []).append(invocation_record)
        run_manifest["status"] = "running"
        run_manifest["updated_at_utc"] = _now_iso()
    else:
        if run_manifest_path.exists():
            raise SystemExit(
                f"{run_manifest_path} already exists but --resume was not given -- this "
                f"should have been caught by the output-directory safety check."
            )
        run_manifest = {
            "run_manifest_schema_version": run_manifest_schema.RUN_MANIFEST_SCHEMA_VERSION,
            "status": "initialized", "run_kind": run_kind,
            "validation_cadence": validate_validation_cadence(validation_cadence),
            "git_head": git_commit, "git_dirty": git_dirty,
            "invocations": [invocation_record],
            "started_at_utc": _now_iso(), "updated_at_utc": _now_iso(),
            "finished_at_utc": None, "final_artifact_hashes": None,
        }
    run_manifest_schema.validate_run_manifest(run_manifest, stage="initialized")
    return run_manifest


def persist_or_verify_data_provenance(run_manifest: dict, *, resume: bool, manifest_record: dict,
                                      taxonomy_record: dict, val_split_record: dict) -> None:
    """Mutates run_manifest in place with the manifest/taxonomy_source/
    val_split records. On a fresh run, sets them. On resume, NEVER silently
    overwrites: the freshly re-derived records must exactly equal what
    run_manifest.json already has on disk, or this aborts (SystemExit)
    naming the field that disagrees. This is a distinct check from
    checkpoint_last.pth's own provenance-hash comparison -- it independently
    guards run_manifest.json's own persisted record against drift or
    tampering."""
    if resume:
        for key, fresh in (("manifest", manifest_record), ("taxonomy_source", taxonomy_record),
                          ("val_split", val_split_record)):
            existing = run_manifest.get(key)
            if existing != fresh:
                raise SystemExit(
                    f"Refusing to resume: run_manifest.json's recorded {key!r} "
                    f"({existing!r}) does not match the freshly derived value ({fresh!r})."
                )
    else:
        run_manifest["manifest"] = manifest_record
        run_manifest["taxonomy_source"] = taxonomy_record
        run_manifest["val_split"] = val_split_record
    run_manifest_schema.validate_run_manifest(run_manifest, stage="data_verified")


# ----------------------------------------------------------------- provenance / environment
def get_git_state() -> tuple[str | None, bool | None]:
    """(commit_sha, dirty) -- both None if git is unavailable or this is not
    a git checkout. Never raises."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None, None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode()
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return commit, dirty


def require_clean_git_state(git_commit: str | None, git_dirty: bool | None) -> None:
    """A real full run and a resume must refuse a dirty or unavailable git
    state -- this run's provenance (git_commit) must reflect exactly the
    code that produced it."""
    if git_commit is None:
        raise SystemExit(
            "Cannot determine git HEAD (git unavailable, or this is not a git checkout) "
            "-- refusing to train or resume."
        )
    if git_dirty is None:
        raise SystemExit(
            "Cannot determine git working-tree cleanliness -- refusing to train or resume."
        )
    if git_dirty:
        raise SystemExit(
            "git working tree is dirty (uncommitted changes) -- refusing to train or "
            "resume. Commit or stash changes first."
        )


def collect_environment_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    try:
        import timm
        info["timm"] = timm.__version__
    except ImportError:
        info["timm"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    else:
        info["gpu_name"] = None
        info["gpu_compute_capability"] = None
    return info


def build_provenance(*, manifest_sha256: str | None, taxonomy_sha256: str | None,
                     val_split_sha256: str, cfg: dict, backbone: str, num_classes: int,
                     git_commit: str | None, numerical_policy: dict, run_kind: str,
                     limit_batches: int | None, wandb_enabled: bool,
                     validation_cadence: int) -> dict:
    return {
        "manifest_sha256": manifest_sha256,
        "taxonomy_sha256": taxonomy_sha256,
        "val_split_sha256": val_split_sha256,
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "backbone": backbone,
        "num_classes": num_classes,
        "git_commit": git_commit,
        "numerical_policy": numerical_policy,
        "run_kind": run_kind,
        "limit_batches": limit_batches,
        "wandb_enabled": wandb_enabled,
        "validation_cadence": validate_validation_cadence(validation_cadence),
    }


def _write_json_atomic(path: Path, obj) -> None:
    ckpt_mod.atomic_write_bytes(path, json.dumps(obj, indent=2, default=str).encode("utf-8"))


# ----------------------------------------------------------------- selection rule
def selection_key(top1: float, top3: float, epoch: int) -> tuple[float, float, int]:
    """Frozen selection rule: (1) highest val raw-cosine top1; (2) tie ->
    highest top3; (3) tie -> earliest epoch. Only ever compared between
    VALIDATED epochs (see should_validate) -- an unvalidated epoch never
    produces a candidate to compare."""
    return (top1, top3, -epoch)


# ----------------------------------------------------------------- validation cadence
def validate_validation_cadence(value) -> int:
    """A strict positive integer, nothing else -- bool (an int subclass in
    Python), zero, negative values, floats, and strings are all rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"validation_cadence must be a strict positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"validation_cadence must be a strict positive integer, got {value!r}")
    return value


def should_validate(epoch_number: int, total_epochs: int, validation_cadence: int) -> bool:
    """One-based epoch_number. Validate at epoch 1, every validation_cadence-
    th epoch, and always the final epoch -- frozen for the Northeast B4
    control and the later EfficientNetV2-S comparison. For (30, 3) this
    produces exactly [1,3,6,9,12,15,18,21,24,27,30]."""
    validate_validation_cadence(validation_cadence)
    return (epoch_number == 1
           or epoch_number % validation_cadence == 0
           or epoch_number == total_epochs)


# ----------------------------------------------------------------- artifact size estimate
# Empirical reference sizes from the LIVE 50-species artifacts (measured
# during the Phase 4 recon, 2026-09-06). ESTIMATES only -- never measurements.
_REF_NUM_CLASSES = 50
_REF_BACKBONE_ONNX_BYTES = 70_160_055
_REF_MODEL_PTH_BYTES = 71_278_123
_REF_PROTOTYPES_NPY_HEADER_BYTES = 128
_REF_TAXONOMY_JSON_65CLASS_BYTES = 12_218
_REF_GEO_INDEX_JSON_50CLASS_BYTES = 47_495
_REF_EVAL_JSON_50CLASS_BYTES = 6_099


def estimate_artifact_sizes(num_classes: int, embedding_dim: int) -> dict:
    """Rough, arithmetic-only size estimates for --preflight-only reporting.
    NOT measurements. Updated this pass to account for versioned best-file
    checkpointing: at steady state during a run, only ONE
    checkpoint_best_epoch_NNN.pth exists (superseded ones are cleaned up);
    at successful finalization, an additional unversioned checkpoint_best.pth
    copy is also materialized, adding roughly one more model-sized file."""
    prototypes_bytes = num_classes * embedding_dim * 4 + _REF_PROTOTYPES_NPY_HEADER_BYTES
    extra_head_params = (num_classes - _REF_NUM_CLASSES) * (embedding_dim + 1)
    model_pth_bytes = _REF_MODEL_PTH_BYTES + extra_head_params * 4
    backbone_onnx_bytes = _REF_BACKBONE_ONNX_BYTES

    checkpoint_last_bytes_estimate = model_pth_bytes * 3  # + 2 AdamW moment tensors
    checkpoint_best_versioned_bytes_estimate = model_pth_bytes + 4_096

    taxonomy_json_bytes = (_REF_TAXONOMY_JSON_65CLASS_BYTES if num_classes == 65
                           else round(_REF_TAXONOMY_JSON_65CLASS_BYTES * num_classes / 65))
    geo_index_json_bytes_range = [
        round(_REF_GEO_INDEX_JSON_50CLASS_BYTES * 1.0),
        round(_REF_GEO_INDEX_JSON_50CLASS_BYTES * 1.9),
    ]
    eval_json_bytes = round(_REF_EVAL_JSON_50CLASS_BYTES * num_classes / _REF_NUM_CLASSES)
    val_split_json_bytes_range = [500_000, 650_000]

    final_artifacts_total_bytes_approx = (
        backbone_onnx_bytes + model_pth_bytes + prototypes_bytes + taxonomy_json_bytes
        + geo_index_json_bytes_range[1] + eval_json_bytes + val_split_json_bytes_range[1]
    )
    # Steady state during the run: checkpoint_last + one versioned best file.
    directory_total_during_run_bytes_approx = (
        final_artifacts_total_bytes_approx
        + checkpoint_last_bytes_estimate + checkpoint_best_versioned_bytes_estimate
    )
    # At successful completion: PLUS the materialized unversioned checkpoint_best.pth copy.
    directory_total_after_completion_bytes_approx = (
        directory_total_during_run_bytes_approx + checkpoint_best_versioned_bytes_estimate
    )

    return {
        "note": "ESTIMATED, not measured -- preflight initializes no model.",
        "backbone_onnx_bytes": backbone_onnx_bytes,
        "model_pth_bytes": model_pth_bytes,
        "prototypes_npy_bytes": prototypes_bytes,
        "taxonomy_json_bytes": taxonomy_json_bytes,
        "geo_index_json_bytes_range": geo_index_json_bytes_range,
        "eval_json_bytes": eval_json_bytes,
        "val_split_json_bytes_range": val_split_json_bytes_range,
        "checkpoint_last_pth_bytes_estimate": checkpoint_last_bytes_estimate,
        "checkpoint_best_versioned_pth_bytes_estimate": checkpoint_best_versioned_bytes_estimate,
        "final_artifacts_total_bytes_approx": final_artifacts_total_bytes_approx,
        "directory_total_during_run_bytes_approx": directory_total_during_run_bytes_approx,
        "directory_total_after_completion_bytes_approx":
            directory_total_after_completion_bytes_approx,
    }


# ----------------------------------------------------------------- epoch orchestration
def run_epochs(*, art: Path, cfg: dict, start_epoch: int, model, opt, train_dl, proto_dl,
               val_dl, num_classes: int, embedding_dim: int, taxonomy: dict, device,
               provenance: dict, canonical_history: list[dict], best_ref: dict | None,
               train_gen, run_manifest: dict, run_manifest_path: Path,
               limit_batches: int | None, pause_after_epoch: int | None,
               validation_cadence: int, use_wandb: bool, tqdm_fn,
               on_batch=None) -> tuple[dict | None, list[dict], str]:
    """Runs cfg['epochs'] epochs starting at start_epoch (inclusive), doing a
    full crash-consistent commit (checkpoint.commit_epoch) after every epoch
    regardless of whether that epoch validated.

    Validation cadence (see should_validate): epoch 1 always validates, then
    every validation_cadence-th epoch, and always the final epoch. On a
    skipped epoch: no prototypes are constructed, the val loader never runs,
    is_best is always False, the existing best epoch/metrics are preserved
    untouched, and checkpoint_best is never rewritten -- but training still
    completes, RNG state is still captured, checkpoint_last is still
    committed atomically, and history/run_manifest are still updated, so
    interrupting on a skipped epoch remains exactly as safe to resume from as
    interrupting on a validated one.

    Returns (best_ref, canonical_history, outcome), outcome in
    {"completed", "paused"}.

    This is the SAME function the real main() and
    test_train_orchestration.py's interrupted-vs-uninterrupted integration
    test both call -- the test exercises this exact orchestration code
    against a tiny synthetic model/dataset, never a reimplementation of it.

    `on_batch(epoch, batch_index, labels, loss_value)`, if given, is called
    once per training batch purely for test observability; it has no effect
    on training.
    """
    loss_fn = nn.CrossEntropyLoss()
    total_epochs = cfg["epochs"]
    for epoch in range(start_epoch, total_epochs):
        epoch_number = epoch + 1  # one-based, for cadence and display only
        t_epoch_start = time.monotonic()
        model.train()
        running = 0.0
        n_batches = 0
        for bi, (imgs, labels) in enumerate(tqdm_fn(train_dl, desc=f"epoch {epoch_number}")):
            if limit_batches and bi >= limit_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
            if on_batch is not None:
                on_batch(epoch, bi, labels.detach().cpu().tolist(), float(loss.item()))
        avg_loss = running / max(1, n_batches)
        t_train = time.monotonic() - t_epoch_start

        do_validate = should_validate(epoch_number, total_epochs, validation_cadence)

        if do_validate:
            t0 = time.monotonic()
            protos = compute_prototypes(model, proto_dl, num_classes, device, embedding_dim)
            t_proto = time.monotonic() - t0

            t0 = time.monotonic()
            metrics = topk_accuracy(model, protos, val_dl, taxonomy, device)
            t_val = time.monotonic() - t0

            top1, top3 = metrics["overall"]["top1"], metrics["overall"]["top3"]
            print(f"[train] epoch {epoch_number}/{total_epochs} loss={avg_loss:.4f} "
                 f"val_top1={top1:.4f} val_top3={top3:.4f} "
                 f"(train={t_train:.1f}s proto={t_proto:.1f}s val={t_val:.1f}s)")
            if use_wandb:
                import wandb
                wandb.log({"epoch": epoch_number, "train_loss": avg_loss,
                          "val_top1": top1, "val_top3": top3})

            is_best = (best_ref is None or
                      selection_key(top1, top3, epoch) > selection_key(
                          best_ref["metrics"]["top1"], best_ref["metrics"]["top3"], best_ref["epoch"]))
            epoch_metrics = {"top1": top1, "top3": top3}
            best_epoch_so_far = epoch if is_best else best_ref["epoch"]
        else:
            print(f"[train] epoch {epoch_number}/{total_epochs} loss={avg_loss:.4f} "
                 f"validation skipped (cadence={validation_cadence}) (train={t_train:.1f}s)")
            if use_wandb:
                import wandb
                wandb.log({"epoch": epoch_number, "train_loss": avg_loss})

            is_best = False
            top1 = top3 = t_proto = t_val = None
            epoch_metrics = {"top1": None, "top3": None}
            best_epoch_so_far = best_ref["epoch"] if best_ref is not None else None

        history_row = {
            "epoch": epoch, "train_loss": avg_loss, "val_top1": top1, "val_top3": top3,
            "duration_seconds": {"train": t_train, "prototypes": t_proto, "validation": t_val},
            "validation_ran": do_validate, "is_best": is_best,
            "best_epoch_so_far": best_epoch_so_far,
            "timestamp_utc": _now_iso(),
        }

        rng_state = numerics.capture_rng_state()
        best_ref = ckpt_mod.commit_epoch(
            art, epoch=epoch, is_best=is_best, model_state=model.state_dict(),
            optimizer_state=opt.state_dict(), provenance=provenance, resolved_config=cfg,
            rng_state=rng_state, train_generator_state=train_gen.get_state(),
            metrics=epoch_metrics, canonical_history=canonical_history,
            history_row=history_row, previous_best=best_ref,
        )
        canonical_history = canonical_history + [history_row]

        run_manifest["status"] = "running"
        run_manifest["last_completed_epoch"] = epoch
        run_manifest["best"] = best_ref
        run_manifest["updated_at_utc"] = _now_iso()
        run_manifest_schema.validate_run_manifest(run_manifest, stage="epoch_committed")
        _write_json_atomic(run_manifest_path, run_manifest)

        if pause_after_epoch and epoch_number == pause_after_epoch:
            run_manifest["status"] = "paused_for_smoke"
            run_manifest["updated_at_utc"] = _now_iso()
            run_manifest_schema.validate_run_manifest(run_manifest, stage="epoch_committed")
            _write_json_atomic(run_manifest_path, run_manifest)
            print(f"[train] paused after epoch {epoch_number} (--pause-after-epoch) -- exiting "
                 f"without finalizing")
            return best_ref, canonical_history, "paused"

    return best_ref, canonical_history, "completed"


# ----------------------------------------------------------------- preflight
def run_preflight(args, cfg: dict) -> int:
    """Every data/hash/split/taxonomy/image-byte/output-safety/git check,
    with no model initialization, no downloads, and no writes. Returns a
    process exit code (0 = every check passed)."""
    problems: list[str] = []

    numerical_policy = numerics.apply_numerical_policy()
    print(f"[preflight] numerical policy applied: {numerical_policy}")

    env_info = collect_environment_info()
    print(f"[preflight] environment: {env_info}")

    git_commit, git_dirty = get_git_state()
    print(f"[preflight] git: commit={git_commit} dirty={git_dirty}")
    try:
        require_clean_git_state(git_commit, git_dirty)
        print("[preflight] PASS: git state (HEAD available, tree clean)")
    except SystemExit as e:
        problems.append(f"git state: {e}")
        print(f"[preflight] FAIL: git state: {e}")

    run_kind = "smoke" if args.limit_batches else "full"
    print(f"[preflight] run_kind={run_kind} limit_batches={args.limit_batches} "
         f"pause_after_epoch={args.pause_after_epoch} wandb_enabled={args.wandb_enabled} "
         f"validation_cadence={args.validation_cadence}")

    art = Path(args.artifacts_dir) if args.artifacts_dir else Path(cfg["artifacts_dir"])
    if not art.is_absolute():
        art = HERE / art
    print(f"[preflight] artifacts dir: {art}")
    try:
        check_output_dir_safety(art, args.resume)
        print("[preflight] PASS: output-directory safety")
    except SystemExit as e:
        problems.append(f"output-directory safety: {e}")
        print(f"[preflight] FAIL: output-directory safety: {e}")

    if args.resume and (art / "run_manifest.json").exists():
        try:
            existing_manifest = json.loads((art / "run_manifest.json").read_text())
            run_manifest_schema.validate_run_manifest(existing_manifest, stage="any")
            found_version = existing_manifest.get("run_manifest_schema_version")
            if found_version != run_manifest_schema.RUN_MANIFEST_SCHEMA_VERSION:
                raise SystemExit(
                    f"run_manifest_schema_version={found_version!r} cannot be resumed "
                    f"under this harness (only {run_manifest_schema.RUN_MANIFEST_SCHEMA_VERSION!r} "
                    f"is resumable)."
                )
            check_resumable_status(existing_manifest, art / "run_manifest.json")
            print(f"[preflight] PASS: run_manifest schema valid, status "
                 f"{existing_manifest.get('status')!r} is resumable")
        except (SystemExit, run_manifest_schema.RunManifestValidationError) as e:
            problems.append(f"resumable status: {e}")
            print(f"[preflight] FAIL: resumable status: {e}")

    samples = taxonomy = None
    manifest_sha256 = taxonomy_sha256 = None
    if args.manifest_csv:
        try:
            samples, taxonomy, manifest_sha256, taxonomy_sha256, _ = load_explicit_manifest_source(
                args.manifest_csv, args.local_data_dir, args.taxonomy_json,
                args.expected_manifest_sha256, args.expected_taxonomy_sha256,
                database_url=os.environ.get("DATABASE_URL"),
            )
            print(f"[preflight] PASS: explicit manifest source, hashes verified "
                 f"(manifest={manifest_sha256}, taxonomy={taxonomy_sha256})")
        except DataIntegrityError as e:
            problems.append(f"data source: {e}")
            print(f"[preflight] FAIL: data source: {e}")

        if samples is not None:
            try:
                n_train, n_val = assert_dataset_shape(samples, taxonomy)
                print(f"[preflight] PASS: dataset shape "
                     f"({len(samples)} samples, {len(taxonomy)} classes, "
                     f"{n_train} train, {n_val} val)")
            except DataIntegrityError as e:
                problems.append(f"dataset shape: {e}")
                print(f"[preflight] FAIL: dataset shape: {e}")

            try:
                image_stats = verify_image_bytes(args.manifest_csv, args.local_data_dir)
                print(f"[preflight] PASS: image bytes verified "
                     f"({image_stats['files_verified']} files, "
                     f"{image_stats['total_bytes']} bytes)")
            except DataIntegrityError as e:
                problems.append(f"image bytes: {e}")
                print(f"[preflight] FAIL: image bytes: {e}")
    else:
        print("[preflight] no --manifest-csv given: skipping explicit-source, dataset-shape, "
             "and image-byte checks (legacy data source, e.g. a smoke run)")

    val_split_sha256 = None
    if samples is not None:
        train_s, val_s = split_samples(samples, cfg["val_fraction"], cfg["seed"])
        try:
            split_record = build_val_split_record(cfg, train_s, val_s)
            val_split_bytes = serialize_val_split(split_record)
            val_split_sha256 = hashlib.sha256(val_split_bytes).hexdigest()
            print(f"[preflight] PASS: val_split would hash to {val_split_sha256} "
                 f"({split_record['n_train']} train / {split_record['n_val']} val keys, "
                 f"unique and disjoint)")
        except ValueError as e:
            problems.append(f"split record: {e}")
            print(f"[preflight] FAIL: split record: {e}")
            train_s = val_s = []

        per_class_counts: dict[str, dict[str, int]] = {}
        for s in train_s:
            per_class_counts.setdefault(s.slug, {"train": 0, "val": 0})["train"] += 1
        for s in val_s:
            per_class_counts.setdefault(s.slug, {"train": 0, "val": 0})["val"] += 1
        count_problems = check_per_class_counts(per_class_counts)
        if count_problems:
            problems.append("per-class counts: " + "; ".join(count_problems))
            print(f"[preflight] FAIL: per-class counts: {count_problems}")
        else:
            print(f"[preflight] PASS: per-class counts (all {len(NORTHEAST_NEW_SPECIES_SLUGS)} "
                 f"Northeast species 200/40, legacy species 158-160 train / nonempty val)")

    if args.resume:
        last_ckpt_path = art / "checkpoint_last.pth"
        if last_ckpt_path.exists():
            try:
                saved = ckpt_mod.load_resume_checkpoint(last_ckpt_path)  # always CPU -- see checkpoint.py
                if samples is not None:
                    current_provenance = build_provenance(
                        manifest_sha256=manifest_sha256, taxonomy_sha256=taxonomy_sha256,
                        val_split_sha256=val_split_sha256, cfg=cfg,
                        backbone=cfg["model"]["backbone"], num_classes=len(taxonomy),
                        git_commit=git_commit, numerical_policy=numerical_policy,
                        run_kind=run_kind, limit_batches=args.limit_batches,
                        wandb_enabled=args.wandb_enabled,
                        validation_cadence=args.validation_cadence,
                    )
                    mismatches = ckpt_mod.provenance_mismatches(current_provenance, saved["provenance"])
                    if mismatches:
                        problems.append("resume provenance: " + "; ".join(mismatches))
                        print("[preflight] FAIL: resume provenance mismatch:\n  "
                             + "\n  ".join(mismatches))
                    else:
                        print(f"[preflight] PASS: resume provenance matches "
                             f"checkpoint_last.pth (completed_epoch="
                             f"{saved.get('completed_epoch')})")
                try:
                    ckpt_mod.verify_referenced_best(art, saved)
                    print("[preflight] PASS: referenced best checkpoint verified")
                except ckpt_mod.CheckpointIntegrityError as e:
                    problems.append(f"referenced best: {e}")
                    print(f"[preflight] FAIL: referenced best: {e}")

                canonical_bytes = ckpt_mod.serialize_history_jsonl(saved.get("history", []))
                history_path = art / "history.jsonl"
                if history_path.exists() and history_path.read_bytes() == canonical_bytes:
                    print("[preflight] PASS: history.jsonl matches checkpoint_last's canonical history")
                else:
                    print("[preflight] NOTE: history.jsonl does not currently match "
                         "checkpoint_last's canonical history -- would be repaired on resume "
                         "(read-only check; preflight writes nothing)")

                orphans = ckpt_mod.list_orphan_best_files(
                    art, saved["best"]["filename"] if saved.get("best") else None)
                if orphans:
                    print(f"[preflight] NOTE: {len(orphans)} orphan best checkpoint file(s) "
                         f"(ignored, never treated as authoritative): "
                         f"{[p.name for p in orphans]}")
            except Exception as e:  # noqa: BLE001
                problems.append(f"resume checkpoint read: {e}")
                print(f"[preflight] FAIL: could not read {last_ckpt_path}: {e}")
        else:
            problems.append(f"resume: {last_ckpt_path} does not exist")
            print(f"[preflight] FAIL: --resume given but {last_ckpt_path} does not exist")

    est_classes = len(taxonomy) if taxonomy is not None else EXPECTED_CLASS_COUNT
    sizes = estimate_artifact_sizes(est_classes, cfg["model"]["embedding_dim"])
    print(f"[preflight] estimated artifact/checkpoint sizes (NOT measured): {sizes}")

    if problems:
        print(f"\n[preflight] FAILED: {len(problems)} problem(s) found.")
        return 1
    print("\n[preflight] PASS: all checks passed. No model initialized, nothing "
         "downloaded, nothing written.")
    return 0


# ----------------------------------------------------------------- main
def main() -> int:
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
    ap.add_argument("--limit-batches", type=int, dest="limit_batches", default=None,
                    help="Cap batches/epoch. Marks this run run_kind=smoke and binds "
                         "the value into provenance -- a smoke run can never resume as "
                         "a full run or vice versa.")

    ap.add_argument("--manifest-csv", type=Path, dest="manifest_csv", default=None,
                    help="Explicit manifest CSV path. Bypasses data.load_manifest()'s "
                         "DATABASE_URL/MANIFEST_CSV/bare-walk precedence entirely -- "
                         "DATABASE_URL present in the environment then becomes a hard "
                         "failure rather than being silently preferred or unset. "
                         "Requires --local-data-dir, --taxonomy-json, "
                         "--expected-manifest-sha256, and --expected-taxonomy-sha256 "
                         "together.")
    ap.add_argument("--local-data-dir", type=Path, dest="local_data_dir", default=None)
    ap.add_argument("--taxonomy-json", type=Path, dest="taxonomy_json", default=None,
                    help="Committed taxonomy JSON the manifest-derived taxonomy must "
                         "exactly equal (full object, not just genus/slug).")
    ap.add_argument("--expected-manifest-sha256", dest="expected_manifest_sha256", default=None)
    ap.add_argument("--expected-taxonomy-sha256", dest="expected_taxonomy_sha256", default=None)

    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint_last.pth in --artifacts-dir. Fails "
                         "closed unless manifest/taxonomy/split hashes, resolved "
                         "config, backbone/class count, git commit, numerical policy, "
                         "run_kind, limit_batches, and wandb_enabled all still match "
                         "the checkpoint's saved provenance.")
    ap.add_argument("--pause-after-epoch", type=int, dest="pause_after_epoch", default=None,
                    help="One-based epoch number to pause after (smoke-testing resume "
                         "only). Requires --limit-batches. NOT part of the immutable "
                         "provenance -- the resuming invocation may omit it. Pausing "
                         "only ever happens after a fully committed epoch, never "
                         "mid-epoch or mid-commit.")
    ap.add_argument("--wandb", action="store_true", dest="wandb_enabled",
                    help="Explicit opt-in for Weights & Biases logging. Disabled by "
                         "default regardless of WANDB_API_KEY.")
    ap.add_argument("--validation-cadence", type=int, dest="validation_cadence", default=1,
                    help="Validate at epoch 1, every Nth epoch, and always the final "
                         "epoch (e.g. cadence 3 over 30 epochs validates "
                         "1,3,6,...,27,30). Frozen at 3 for the Northeast B4 control "
                         "and the later EfficientNetV2-S comparison; default 1 "
                         "preserves the historical every-epoch behavior. Must be a "
                         "strict positive integer; bound into provenance, so a resume "
                         "with a different cadence is refused.")
    ap.add_argument("--preflight-only", action="store_true", dest="preflight_only",
                    help="Perform every data/hash/split/taxonomy/image-byte/output-"
                         "safety/git check and exit. Initializes no model, downloads "
                         "nothing, writes nothing. Exit code is nonzero on any problem.")
    args = ap.parse_args()

    explicit_group = (args.manifest_csv, args.local_data_dir, args.taxonomy_json,
                      args.expected_manifest_sha256, args.expected_taxonomy_sha256)
    if any(explicit_group) and not all(explicit_group):
        ap.error("--manifest-csv, --local-data-dir, --taxonomy-json, "
                 "--expected-manifest-sha256, and --expected-taxonomy-sha256 must all "
                 "be given together, or none of them.")
    if args.pause_after_epoch is not None:
        if not args.limit_batches:
            ap.error("--pause-after-epoch requires --limit-batches (smoke-testing only).")
        if args.pause_after_epoch < 1:
            ap.error("--pause-after-epoch is one-based and must be >= 1.")
    try:
        validate_validation_cadence(args.validation_cadence)
    except ValueError as e:
        ap.error(str(e))

    cfg = load_config(args.config, {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "image_size": args.image_size, "num_workers": args.num_workers,
    })
    if args.artifacts_dir:
        cfg["artifacts_dir"] = str(args.artifacts_dir)

    if args.preflight_only:
        return run_preflight(args, cfg)

    numerical_policy = numerics.apply_numerical_policy()
    print(f"[train] numerical policy: {numerical_policy}")
    env_info = collect_environment_info()
    print(f"[train] environment: {env_info}")

    device = pick_device()
    print(f"[train] device={device}")

    art = Path(cfg["artifacts_dir"])
    if not art.is_absolute():
        art = HERE / art
    check_output_dir_safety(art, args.resume)
    art.mkdir(parents=True, exist_ok=True)

    git_commit, git_dirty = get_git_state()
    require_clean_git_state(git_commit, git_dirty)

    run_kind = "smoke" if args.limit_batches else "full"
    run_manifest_path = art / "run_manifest.json"
    history_path = art / "history.jsonl"
    last_ckpt_path = art / "checkpoint_last.pth"
    val_split_path = art / "val_split.json"

    invocation_record = {
        "timestamp_utc": _now_iso(), "argv": sys.argv[1:], "resume": args.resume,
        "pause_after_epoch": args.pause_after_epoch,
    }

    # ---- run_manifest bootstrap (BEFORE data loading, so any subsequent
    # failure -- data loading, model init, DataLoader setup -- can update an
    # already-created run_manifest instead of leaving misleading state) ----
    run_manifest = bootstrap_run_manifest(
        run_manifest_path, resume=args.resume, invocation_record=invocation_record,
        run_kind=run_kind, git_commit=git_commit, git_dirty=git_dirty,
        validation_cadence=args.validation_cadence,
    )
    _write_json_atomic(run_manifest_path, run_manifest)

    use_wandb = False
    try:
        # ---- data ----
        if args.manifest_csv:
            try:
                samples, taxonomy, manifest_sha256, taxonomy_sha256, committed_taxonomy = (
                    load_explicit_manifest_source(
                        args.manifest_csv, args.local_data_dir, args.taxonomy_json,
                        args.expected_manifest_sha256, args.expected_taxonomy_sha256,
                        database_url=os.environ.get("DATABASE_URL"),
                    )
                )
                assert_dataset_shape(samples, taxonomy)
                image_stats = verify_image_bytes(args.manifest_csv, args.local_data_dir)
                print(f"[train] verified {image_stats['files_verified']} image files "
                     f"({image_stats['total_bytes']} bytes) against recorded sha256")
            except DataIntegrityError as e:
                raise SystemExit(str(e)) from e
        else:
            samples, taxonomy = load_manifest(cfg)
            manifest_sha256 = taxonomy_sha256 = None
            committed_taxonomy = None

        num_classes = len(taxonomy)
        training_seed = cfg["seed"]

        train_s, val_s = split_samples(samples, cfg["val_fraction"], training_seed)
        split_record = build_val_split_record(cfg, train_s, val_s)
        val_split_bytes = serialize_val_split(split_record)
        val_split_sha256 = hashlib.sha256(val_split_bytes).hexdigest()
        print(f"[train] {len(train_s)} train / {len(val_s)} val / {num_classes} classes "
             f"(val_split sha256 {val_split_sha256})")

        if args.manifest_csv:
            per_class_counts: dict[str, dict[str, int]] = {}
            for s in train_s:
                per_class_counts.setdefault(s.slug, {"train": 0, "val": 0})["train"] += 1
            for s in val_s:
                per_class_counts.setdefault(s.slug, {"train": 0, "val": 0})["val"] += 1
            count_problems = check_per_class_counts(per_class_counts)
            if count_problems:
                raise SystemExit("per-class count check failed: " + "; ".join(count_problems))

        provenance = build_provenance(
            manifest_sha256=manifest_sha256, taxonomy_sha256=taxonomy_sha256,
            val_split_sha256=val_split_sha256, cfg=cfg, backbone=cfg["model"]["backbone"],
            num_classes=num_classes, git_commit=git_commit, numerical_policy=numerical_policy,
            run_kind=run_kind, limit_batches=args.limit_batches, wandb_enabled=args.wandb_enabled,
            validation_cadence=args.validation_cadence,
        )

        # Persist (fresh) or verify-without-overwriting (resume) the exact
        # records evaluate.py needs to resolve this run's data source on its
        # own, later, without ever touching config.yaml/DATABASE_URL/env.
        manifest_record = ({"path": str(args.manifest_csv), "sha256": manifest_sha256,
                           "rows": len(samples)} if args.manifest_csv else None)
        taxonomy_record = ({"path": str(args.taxonomy_json), "sha256": taxonomy_sha256,
                           "num_classes": num_classes} if args.manifest_csv else None)
        val_split_record_for_manifest = {
            "path": str(val_split_path), "sha256": val_split_sha256,
            "n_train": split_record["n_train"], "n_val": split_record["n_val"],
            "n_total": split_record["n_total"],
        }
        persist_or_verify_data_provenance(
            run_manifest, resume=args.resume, manifest_record=manifest_record,
            taxonomy_record=taxonomy_record, val_split_record=val_split_record_for_manifest,
        )
        run_manifest["status"] = "running"
        run_manifest["updated_at_utc"] = _now_iso()
        _write_json_atomic(run_manifest_path, run_manifest)

        if args.resume:
            saved = ckpt_mod.load_resume_checkpoint(last_ckpt_path)  # always CPU -- see checkpoint.py
            mismatches = ckpt_mod.provenance_mismatches(provenance, saved["provenance"])
            if mismatches:
                raise SystemExit(
                    "Refusing to resume: provenance mismatch between this invocation and "
                    f"{last_ckpt_path}:\n  " + "\n  ".join(mismatches)
                )
            ckpt_mod.verify_referenced_best(art, saved)
            if ckpt_mod.repair_history_if_needed(history_path, saved.get("history", [])):
                print(f"[train] repaired {history_path} to match checkpoint_last's "
                     f"canonical history")
            orphans = ckpt_mod.list_orphan_best_files(
                art, saved["best"]["filename"] if saved.get("best") else None)
            if orphans:
                print(f"[train] WARNING: {len(orphans)} orphan best checkpoint file(s) "
                     f"found (ignored, never treated as authoritative): "
                     f"{[p.name for p in orphans]}")

            start_epoch = saved["completed_epoch"] + 1
            best_ref = saved["best"]
            canonical_history = saved.get("history", [])
            print(f"[train] resumed from {last_ckpt_path}: completed_epoch="
                 f"{saved['completed_epoch']}, best={best_ref}")

            # Corrected ordering: model (pretrained=False -- the checkpoint
            # supplies every weight; never consult or download pretrained
            # weights on resume) -> optimizer -> load state -> DataLoaders
            # -> RNG restore happens LAST, immediately before the epoch loop.
            model = AntIDModel(
                num_classes=num_classes, backbone=cfg["model"]["backbone"],
                pretrained=False, dropout=cfg["dropout"],
                embedding_dim=cfg["model"]["embedding_dim"],
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            train_gen = torch.Generator()  # placeholder; real state restored below
            pending_rng_restore = (saved["rng_state"], saved["train_generator_state"])
        else:
            if val_split_path.exists():
                raise SystemExit(
                    f"{val_split_path} already exists but --resume was not given."
                )
            ckpt_mod.atomic_write_bytes(val_split_path, val_split_bytes)

            start_epoch = 0
            best_ref = None
            canonical_history = []
            train_gen = numerics.seed_everything(training_seed)
            model = AntIDModel(
                num_classes=num_classes, backbone=cfg["model"]["backbone"],
                pretrained=cfg["model"]["pretrained"], dropout=cfg["dropout"],
                embedding_dim=cfg["model"]["embedding_dim"],
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            pending_rng_restore = None

        train_ds = AntDataset(train_s, cfg, train=True)
        val_ds = AntDataset(val_s, cfg, train=False)
        proto_ds = AntDataset(train_s, cfg, train=False)
        nw = cfg["num_workers"]
        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, drop_last=False, generator=train_gen,
                              worker_init_fn=numerics.worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=nw,
                            worker_init_fn=numerics.worker_init_fn)
        proto_dl = DataLoader(proto_ds, batch_size=cfg["batch_size"], num_workers=nw,
                              worker_init_fn=numerics.worker_init_fn)

        if pending_rng_restore is not None:
            rng_state, train_generator_state = pending_rng_restore
            numerics.restore_rng_state(rng_state)
            train_gen.set_state(train_generator_state)

        use_wandb = bool(args.wandb_enabled)  # explicit opt-in ONLY -- never from ambient WANDB_API_KEY
        if use_wandb:
            import wandb
            wandb.init(project="antid", config=cfg)

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **k):  # type: ignore
                return x

        best_ref, canonical_history, outcome = run_epochs(
            art=art, cfg=cfg, start_epoch=start_epoch, model=model, opt=opt,
            train_dl=train_dl, proto_dl=proto_dl, val_dl=val_dl, num_classes=num_classes,
            embedding_dim=cfg["model"]["embedding_dim"], taxonomy=taxonomy, device=device,
            provenance=provenance, canonical_history=canonical_history, best_ref=best_ref,
            train_gen=train_gen, run_manifest=run_manifest, run_manifest_path=run_manifest_path,
            limit_batches=args.limit_batches, pause_after_epoch=args.pause_after_epoch,
            validation_cadence=args.validation_cadence, use_wandb=use_wandb, tqdm_fn=tqdm,
        )

        if outcome == "paused":
            return 0

        # ---- finalize from the best checkpoint ----
        if best_ref is None:
            raise SystemExit("No epoch completed (cfg['epochs'] <= start_epoch) -- "
                             "nothing to finalize.")
        best_payload = ckpt_mod.verify_referenced_best(art, {"best": best_ref, "provenance": provenance})
        model.load_state_dict(best_payload["model"])
        model.eval()

        final_protos = compute_prototypes(model, proto_dl, num_classes, device,
                                          cfg["model"]["embedding_dim"])
        final_metrics = topk_accuracy(model, final_protos, val_dl, taxonomy, device)
        recorded = best_ref["metrics"]
        if (final_metrics["overall"]["top1"] != recorded["top1"]
                or final_metrics["overall"]["top3"] != recorded["top3"]):
            raise SystemExit(
                f"Finalization consistency check FAILED: recomputed top1/top3 "
                f"({final_metrics['overall']['top1']}/{final_metrics['overall']['top3']}) "
                f"do not exactly match the recorded best-epoch metrics "
                f"({recorded['top1']}/{recorded['top3']}) under the pinned deterministic "
                f"policy. Refusing to finalize."
            )
        print(f"[train] finalizing from best epoch {best_ref['epoch']}: "
             f"top1={final_metrics['overall']['top1']:.4f} "
             f"top3={final_metrics['overall']['top3']:.4f} (consistency check passed)")

        ckpt_mod.atomic_torch_save(
            {"model": best_payload["model"], "config": cfg, "provenance": provenance,
            "best_epoch": best_ref["epoch"]},
            art / "model.pth",
        )
        # Materialize the unversioned checkpoint_best.pth ONLY now, at
        # successful finalization, as a copy of the referenced best file.
        ckpt_mod.atomic_torch_save(best_payload, art / "checkpoint_best.pth")
        _atomic_np_save(art / "prototypes.npy", final_protos)

        taxonomy_out = {str(k): v for k, v in sorted(taxonomy.items())}
        if committed_taxonomy is not None and taxonomy_out != committed_taxonomy:
            raise SystemExit(
                "Emitted taxonomy no longer matches the committed taxonomy object -- "
                "refusing to write taxonomy.json."
            )
        _write_json_atomic(art / "taxonomy.json", taxonomy_out)

        geo_cfg = cfg.get("geo") or {}
        cell_size = float(geo_cfg.get("cell_size_deg", 1.0))
        cells = build_geo_index(train_s, taxonomy, cell_size, int(geo_cfg.get("min_obs_per_cell", 2)))
        write_geo_index_sidecar(art / "geo_index.json", cells, cell_size)

        _write_json_atomic(art / "eval.json", final_metrics)

        print("[train] exporting ONNX backbone…")
        export_backbone(model, art / "backbone.onnx", cfg["image_size"], device)

        final_hashes = {
            name: hashlib.sha256((art / name).read_bytes()).hexdigest()
            for name in ("model.pth", "prototypes.npy", "taxonomy.json", "geo_index.json",
                        "eval.json", "backbone.onnx", "val_split.json")
        }
        run_manifest["status"] = "completed"
        run_manifest["best"] = best_ref
        run_manifest["final_artifact_hashes"] = final_hashes
        run_manifest["finished_at_utc"] = _now_iso()
        run_manifest["updated_at_utc"] = _now_iso()
        run_manifest_schema.validate_run_manifest(run_manifest, stage="completed")
        _write_json_atomic(run_manifest_path, run_manifest)

    except BaseException as e:
        try:
            run_manifest["status"] = "failed"
            run_manifest["error"] = str(e)
            run_manifest["updated_at_utc"] = _now_iso()
            # Lenient ("any"): a failure can happen before data_verified is
            # even reached, so this must never itself raise and mask the
            # real error above.
            run_manifest_schema.validate_run_manifest(run_manifest, stage="any")
            _write_json_atomic(run_manifest_path, run_manifest)
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        if use_wandb:
            import wandb
            wandb.finish()

    print(f"[train] artifacts written to {art}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
