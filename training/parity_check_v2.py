#!/usr/bin/env python3
"""parity_check_v2.py — numerical correspondence between a completed
candidate run's model.pth (PyTorch/CUDA) and backbone.onnx (ONNX Runtime
CPU), on a deterministic sample drawn ONLY from the candidate's own pinned
development validation split.

Why not parity_check.py: that script (frozen, kept as historical 50-species
evidence -- see TODO.md's "Parity scripts: deferred audit items") hardcodes
N_PER_SPECIES*50=200 samples, the old calibration_v1 0.60 gate threshold as
a comparison point, and selects images from the raw manifest CSV with no
concept of a pinned train/val split at all (any resolvable image, train or
val, from data/manifest_all.csv -- the ORIGINAL 50-species manifest, not
the 65-species Northeast catalog). None of that is adaptable in place
without invalidating what it already froze, so this is a new, small,
purpose-built tool instead.

This tool:
  - derives species count, embedding dimension, model name, image size,
    normalization, and interpolation from the CANDIDATE'S OWN recorded
    run_manifest.json / model.pth config -- nothing is redeclared or
    assumed to still be 50/1792/380-only.
  - samples ONLY from the candidate's pinned "val" membership in
    val_split.json -- never train, never any frozen evaluation set.
  - verifies every artifact hash, the manifest/taxonomy-source hash pair,
    and every selected image's SHA-256 against the candidate's own
    committed provenance before running anything.
  - never imports, hardcodes, compares against, or reports a gate/threshold
    decision -- there is no confidence threshold for this catalog yet.
  - confirms CPU-only ONNX execution via ONNX Runtime's own profiling trace
    (per-node provider assignment), not merely session.get_providers().
  - reports numerical divergence and disagreement counts only -- no
    accuracy, no genus accuracy, no benchmark or calibration claim.

Label: "pinned development-split numerical parity analysis; not benchmark
evidence, not calibration evidence, and not a population-level equivalence
bound."

Usage:
    python parity_check_v2.py --run-dir artifacts/northeast_v1_b4_dev_v2 \\
        --local-data-dir ../data/clean \\
        --out reports/northeast_v1_b4_dev_v2_parity.json

    python parity_check_v2.py --run-dir artifacts/northeast_v1_b4_dev_v2 \\
        --local-data-dir ../data/clean --verify \\
        --existing reports/northeast_v1_b4_dev_v2_parity.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
API_DIR = REPO / "api"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(API_DIR))

import numerics
from report_dev_metrics import (
    build_model,
    load_completed_run_manifest,
    resolve_manifest_and_taxonomy,
    resolve_pinned_val_split,
    verify_final_artifact_hashes,
)
from evaluate import verify_run_manifest_matches_checkpoint_provenance
from data import build_transforms

LABEL = ("pinned development-split numerical parity analysis; not benchmark "
        "evidence, not calibration evidence, and not a population-level "
        "equivalence bound.")
REPORT_SCHEMA_VERSION = 2  # bumped: stable candidate/generator provenance fields added,
                          # ambient workspace git state moved to generation_metadata,
                          # preprocessing_parity.contract now sourced from api/inference.py
                          # (schema 2 has not yet been committed, so this stays 2 --
                          # not bumped again -- for the canonical-source-hash fix below)
GENERATOR_NAME = "parity_check_v2.py"
# 1.1.1: the generator binding (deterministic_content.generator.
# source_canonical_lf_sha256) now hashes this file's CANONICAL LF-normalized
# text, not its raw working-tree bytes. This repo has no .gitattributes and
# core.autocrlf=true, so the same committed source can check out as CRLF
# (Windows) or LF (Linux/mac) -- raw-byte hashing would make --verify fail
# on a clean checkout purely from line-ending translation, never from an
# actual change to the tool. The field is deliberately named
# "source_canonical_lf_sha256", not a bare "sha256", so this normalization
# is never mistaken for a raw-byte digest.
GENERATOR_VERSION = "1.1.1"

N_PER_SPECIES_DEFAULT = 4
DETERMINISM_REPEATS_DEFAULT = 5
ONNX_BATCH_SIZE_DEFAULT = 16


class ParityError(RuntimeError):
    """A required invariant did not hold. Always fail closed."""


# ------------------------------------------------------------------ utility
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_canonical_lf_sha256(path: Path) -> str:
    """SHA-256 of this file's own source, normalized to a line-ending-
    independent canonical form: decode as UTF-8 text, normalize CRLF and
    lone CR to LF, re-encode as UTF-8. This repo has no .gitattributes and
    `core.autocrlf` is true, so the exact same committed source can check
    out with CRLF (Windows) or LF (Linux/mac) line endings and still hash
    identically -- raw-byte hashing would otherwise make --verify fail on a
    clean checkout for a reason that has nothing to do with the tool
    actually changing."""
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _repo_relative_posix(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return resolved.name


def widen(x) -> float:
    """float32 scalar -> Python float, exact nearest-float64 representation
    of the float32 value -- never rounded or truncated."""
    return float(np.float32(x))


def is_finite(*arrays) -> bool:
    return all(np.all(np.isfinite(np.asarray(a))) for a in arrays)


def percentiles_with_min(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()), "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()), "n": int(arr.size),
    }


# --------------------------------------------------------- candidate integrity
def assert_contiguous_taxonomy(taxonomy: dict) -> None:
    keys = sorted(taxonomy)
    if keys != list(range(len(taxonomy))):
        raise ParityError(
            f"taxonomy keys are not contiguous 0..{len(taxonomy) - 1}: {keys}"
        )


def assert_prototype_shape(prototypes: np.ndarray, num_classes: int, embedding_dim: int) -> None:
    if tuple(prototypes.shape) != (num_classes, embedding_dim):
        raise ParityError(
            f"prototypes.npy shape {tuple(prototypes.shape)} != expected "
            f"({num_classes}, {embedding_dim})"
        )


# ------------------------------------------------------------ sample selection
def select_parity_samples(val_samples: list, taxonomy: dict, seed: int, n_per_species: int):
    """Deterministic n_per_species selection, drawn ONLY from `val_samples`
    (the candidate's pinned val membership -- callers must never pass train
    samples or anything from a frozen evaluation set here). Fails closed if
    any species has fewer than n_per_species available. The expected total
    is derived from len(taxonomy) * n_per_species -- never a literal
    constant.
    """
    by_slug: dict[str, list] = {}
    for s in val_samples:
        by_slug.setdefault(s.slug, []).append(s)

    slugs = sorted({v["slug"] for v in taxonomy.values()})
    expected_total = len(taxonomy) * n_per_species

    rng = random.Random(seed)
    selected: list = []
    shortfalls: list[str] = []
    for slug in slugs:
        cands = sorted(by_slug.get(slug, []), key=lambda s: Path(s.storage_path).stem)
        shuffled = cands[:]
        rng.shuffle(shuffled)
        picked = shuffled[:n_per_species]
        if len(picked) < n_per_species:
            shortfalls.append(
                f"{slug}: only {len(picked)}/{n_per_species} images available in the "
                f"pinned val split"
            )
            continue
        selected.extend(picked)

    if shortfalls:
        raise ParityError(
            "insufficient pinned-val images per species: " + "; ".join(shortfalls)
        )
    if len(selected) != expected_total:
        raise ParityError(
            f"expected {expected_total} selected images (derived from "
            f"{len(taxonomy)} classes x {n_per_species} per species), got {len(selected)}"
        )
    return selected, expected_total


def selection_record(selected: list, seed: int, n_per_species: int) -> dict:
    keys = sorted(f"{s.slug}/{Path(s.storage_path).stem}" for s in selected)
    selection_hash = _sha256_bytes("\n".join(keys).encode("utf-8"))
    return {
        "seed": seed, "n_per_species": n_per_species, "n_selected": len(selected),
        "selected_keys": keys, "selection_hash_sha256": selection_hash,
    }


def load_manifest_sha256_index(manifest_csv: Path) -> dict[tuple[str, str], str]:
    """(slug, photo_id) -> recorded sha256, read directly from the manifest
    CSV, for explicit per-selected-image verification (in addition to the
    bulk verify_image_bytes() check already performed over the whole
    manifest by resolve_manifest_and_taxonomy)."""
    index: dict[tuple[str, str], str] = {}
    with manifest_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            index[(row["slug"], row["photo_id"])] = (row.get("sha256") or "").strip()
    return index


def verify_selected_image_hashes(selected: list, manifest_sha256_index: dict) -> None:
    problems = []
    for s in selected:
        path = Path(s.storage_path)
        photo_id = path.stem
        expected = manifest_sha256_index.get((s.slug, photo_id))
        if not expected:
            problems.append(f"{s.slug}/{photo_id}: no recorded sha256 in manifest")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            problems.append(f"{s.slug}/{photo_id}: sha256 {actual} != manifest {expected}")
    if problems:
        raise ParityError(
            f"{len(problems)} selected-image hash mismatch(es): " + "; ".join(problems)
        )


# --------------------------------------------------------------- environment
def environment_recon() -> dict:
    import torch
    import torchvision
    import PIL
    import onnxruntime as ort

    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "pillow_version": PIL.__version__,
        "numpy_version": np.__version__,
        "onnxruntime_version": ort.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ort_available_providers": ort.get_available_providers(),
    }


def get_workspace_git_state() -> tuple[str | None, bool | None]:
    """The AMBIENT state of this workspace's git tree at report-BUILD time --
    NOT the candidate's own training provenance (see
    deterministic_content.candidate_training_git_head/_dirty, which come
    from the candidate's own run_manifest.json instead). Belongs only in
    generation_metadata, which --verify deliberately ignores: committing
    these very report/tool files changes this workspace's HEAD, and that
    must never invalidate a prior --verify run."""
    from train import get_git_state
    return get_git_state()


# ------------------------------------------------------------------- ORT node
def assert_preprocessing_contract_agreement(candidate_config: dict) -> dict:
    """Compare the candidate's own recorded config against api/inference.py's
    REAL preprocessing constants/PREPROCESSING_CONTRACT (imported, never
    reimplemented or hand-declared) -- BEFORE any preprocessing or model
    inference is run. Returns the record to embed in the report. Raises
    ParityError on any disagreement in a required field.
    """
    from inference import IMAGE_SIZE as api_image_size
    from inference import PREPROCESSING_CONTRACT as api_contract

    candidate_side = {
        "image_size": candidate_config["image_size"],
        "normalize_mean": list(candidate_config["normalize"]["mean"]),
        "normalize_std": list(candidate_config["normalize"]["std"]),
        "interpolation": candidate_config["interpolation_effective"],
    }
    api_side = {
        "image_size": api_image_size,
        "normalize_mean": list(api_contract["normalize_mean"]),
        "normalize_std": list(api_contract["normalize_std"]),
        "interpolation": api_contract["interpolation"],
    }

    disagreements = []
    if candidate_side["image_size"] != api_side["image_size"]:
        disagreements.append("image_size")
    if candidate_side["normalize_mean"] != api_side["normalize_mean"]:
        disagreements.append("normalize_mean")
    if candidate_side["normalize_std"] != api_side["normalize_std"]:
        disagreements.append("normalize_std")
    if candidate_side["interpolation"].lower() not in api_side["interpolation"].lower():
        disagreements.append("interpolation")

    record = {
        "candidate": candidate_side,
        "api_inference_py": api_side,
        "api_inference_py_raw_contract": dict(api_contract),
        "fields_agree": disagreements == [],
        "disagreeing_fields": disagreements,
    }
    if disagreements:
        raise ParityError(
            f"Candidate/API preprocessing-contract disagreement in {disagreements} "
            f"(candidate={candidate_side}, api={api_side}) -- refusing to run any "
            f"preprocessing or model inference."
        )
    return record


def confirm_cpu_only_via_profiling(onnx_path: Path, sample_tensors: list) -> dict:
    """Run every sample through a profiling-enabled, EXPLICITLY CPU-forced
    session and inspect the emitted trace's per-node provider field --
    session.get_providers() alone only proves which providers were
    REGISTERED, not which provider actually executed each node.

    The profile trace is written to a system temp directory and removed
    afterward -- never into the candidate's own (read-only, hash-verified)
    artifacts directory."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_profiling = True
    with tempfile.TemporaryDirectory() as tmp:
        so.profile_file_prefix = str(Path(tmp) / "parity_v2_profile")
        sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        registered_providers = sess.get_providers()
        if registered_providers != ["CPUExecutionProvider"]:
            raise ParityError(
                f"CPUExecutionProvider was not exclusively registered: {registered_providers}"
            )
        input_name = sess.get_inputs()[0].name
        for tensor in sample_tensors:
            sess.run(None, {input_name: tensor})
        profile_path = Path(sess.end_profiling())
        events = json.loads(profile_path.read_text())

    provider_counts: dict[str, int] = {}
    no_provider_field = 0
    non_cpu_nodes = []
    for e in events:
        if e.get("cat") != "Node":
            continue
        args_d = e.get("args", {})
        if "provider" not in args_d:
            no_provider_field += 1
            continue
        p = args_d["provider"]
        provider_counts[p] = provider_counts.get(p, 0) + 1
        if p != "CPUExecutionProvider":
            non_cpu_nodes.append({"name": e.get("name"), "provider": p})

    result = {
        "registered_providers": registered_providers,
        "node_event_count_by_provider": provider_counts,
        "nodes_without_provider_field": no_provider_field,
        "any_node_executed_outside_cpu": len(non_cpu_nodes) > 0,
        "non_cpu_nodes": non_cpu_nodes,
        "evidence_note": ("ORT profiling trace's per-Node-event args.provider field -- "
                          "not an inference from session.get_providers()."),
    }
    if result["any_node_executed_outside_cpu"]:
        raise ParityError(f"ONNX node(s) executed outside CPUExecutionProvider: {non_cpu_nodes}")
    return result


# --------------------------------------------------------------------- main
def run_parity(*, run_dir: Path, local_data_dir: Path, manifest_csv: Path,
              taxonomy_json: Path, database_url: str | None, device,
              seed: int = 42, n_per_species: int = N_PER_SPECIES_DEFAULT,
              onnx_batch_size: int = ONNX_BATCH_SIZE_DEFAULT,
              determinism_repeats: int = DETERMINISM_REPEATS_DEFAULT) -> dict:
    """Run the full parity pipeline and return the deterministic_content
    dict (the part that --verify compares byte-for-byte). Raises
    ParityError on any failure, before writing anything."""
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    from PIL import Image
    from inference import AntIdentifier

    numerical_policy = numerics.apply_numerical_policy()
    env = environment_recon()
    if not env["torch_cuda_available"]:
        raise ParityError(
            "CUDA is not available -- refusing to substitute CPU for the PyTorch "
            "reference path and silently call that parity."
        )

    # ---- 1. candidate integrity --------------------------------------------
    run_manifest, run_manifest_sha256 = load_completed_run_manifest(run_dir)
    final_hashes = verify_final_artifact_hashes(run_dir, run_manifest)

    samples, taxonomy, manifest_sha256, taxonomy_sha256 = resolve_manifest_and_taxonomy(
        run_manifest, local_data_dir, manifest_csv, taxonomy_json, database_url,
    )
    assert_contiguous_taxonomy(taxonomy)
    val_samples, split_record = resolve_pinned_val_split(run_dir, run_manifest, samples)
    num_classes = len(taxonomy)

    state = torch.load(run_dir / "model.pth", map_location=device, weights_only=False)
    cfg = state["config"]
    provenance = state["provenance"]
    verify_run_manifest_matches_checkpoint_provenance(run_manifest, provenance)

    embedding_dim = cfg["model"]["embedding_dim"]
    prototypes = np.load(run_dir / "prototypes.npy")
    assert_prototype_shape(prototypes, num_classes, embedding_dim)

    model = build_model(cfg, num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    candidate_config = {
        "backbone": cfg["model"]["backbone"],
        "embedding_dim": embedding_dim,
        "image_size": cfg["image_size"],
        "normalize": cfg["normalize"],
        "interpolation_effective": cfg.get("interpolation") or "bilinear",
    }

    # ---- preprocessing-contract agreement: BEFORE any preprocessing/inference --
    preprocessing_contract = assert_preprocessing_contract_agreement(candidate_config)

    # ---- 2. sample population -----------------------------------------------
    selected, expected_total = select_parity_samples(val_samples, taxonomy, seed, n_per_species)
    manifest_sha256_index = load_manifest_sha256_index(manifest_csv)
    verify_selected_image_hashes(selected, manifest_sha256_index)
    selection = selection_record(selected, seed, n_per_species)

    # ---- serving-side identifier (real production code), candidate dir only --
    identifier = AntIdentifier(artifacts_dir=run_dir)
    if identifier.runtime_providers != ["CPUExecutionProvider"]:
        raise ParityError(
            f"api/inference.py's own session did not resolve to CPU-only: "
            f"{identifier.runtime_providers}"
        )

    ref_transform = build_transforms(cfg, train=False)

    # ---- 5. preprocessing requirement: measure BEFORE any model inference ---
    preproc_diffs = []
    onnx_tensors = []
    for s in selected:
        path = Path(s.storage_path)
        img_ref = Image.open(path)
        img_ref.load()
        ref_tensor = ref_transform(img_ref.convert("RGB")).numpy()

        img_onnx = Image.open(path)
        img_onnx.load()
        onnx_tensor = identifier.preprocess(img_onnx)  # real production preprocess()

        if not is_finite(ref_tensor, onnx_tensor):
            raise ParityError(f"{s.slug}/{path.stem}: non-finite preprocessed tensor")

        diff = float(np.abs(ref_tensor - onnx_tensor[0]).max())
        preproc_diffs.append(diff)
        onnx_tensors.append(onnx_tensor)

    preprocessing_parity = {
        "n_images": len(selected),
        "max_abs_diff": max(preproc_diffs),
        "byte_identical": max(preproc_diffs) == 0.0,
        "contract": preprocessing_contract,
    }
    if preprocessing_parity["max_abs_diff"] != 0.0:
        raise ParityError(
            f"Preprocessing is NOT byte-identical between training/data.py's val "
            f"transform and api/inference.py's preprocess(): max abs diff = "
            f"{preprocessing_parity['max_abs_diff']!r}. Stopping before model parity "
            f"-- per requirement, this is surfaced, not silently tolerated or repaired."
        )

    # ---- CPU-forced session for the actual measurement (also identifier's own) --
    cpu_session = identifier.session
    cpu_input_name = identifier.input_name
    protos_torch = nn.functional.normalize(
        torch.as_tensor(prototypes, dtype=torch.float32, device=device), dim=1)
    protos_torch_np = protos_torch.cpu().numpy()
    protos_onnx_np = identifier.prototypes  # already L2-normalized by AntIdentifier

    # ---- 3/4. model parity over all selected images --------------------------
    per_image = []
    for s, onnx_tensor in zip(selected, onnx_tensors):
        path = Path(s.storage_path)
        img_ref = Image.open(path)
        img_ref.load()
        ref_tensor = ref_transform(img_ref.convert("RGB"))

        with torch.no_grad():
            ref_raw = model.embed(ref_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
        onnx_raw = cpu_session.run(None, {cpu_input_name: onnx_tensor})[0][0]

        if not is_finite(ref_raw, onnx_raw):
            raise ParityError(f"{s.slug}/{path.stem}: non-finite embedding")

        ref_norm = ref_raw / np.clip(np.linalg.norm(ref_raw), 1e-8, None)
        onnx_norm = onnx_raw / np.clip(np.linalg.norm(onnx_raw), 1e-8, None)

        ref_sims = protos_torch_np @ ref_norm
        onnx_sims = protos_onnx_np @ onnx_norm

        ref_top3 = [int(i) for i in np.argsort(-ref_sims, kind="stable")[:3]]
        onnx_top3 = [int(i) for i in np.argsort(-onnx_sims, kind="stable")[:3]]

        per_image.append({
            "slug": s.slug, "key": f"{s.slug}/{path.stem}",
            "raw_embedding_max_abs_diff": float(np.abs(ref_raw - onnx_raw).max()),
            "normalized_embedding_max_abs_diff": float(np.abs(ref_norm - onnx_norm).max()),
            "max_cosine_reference": widen(ref_sims.max()),
            "max_cosine_onnx_cpu": widen(onnx_sims.max()),
            "max_cosine_abs_diff": abs(widen(ref_sims.max()) - widen(onnx_sims.max())),
            "top1_reference": int(ref_top3[0]), "top1_onnx_cpu": int(onnx_top3[0]),
            "top1_agree": ref_top3[0] == onnx_top3[0],
            "top3_set_reference": sorted(int(i) for i in ref_top3),
            "top3_set_onnx_cpu": sorted(int(i) for i in onnx_top3),
            "top3_set_agree": set(ref_top3) == set(onnx_top3),
        })

    raw_diffs = [r["raw_embedding_max_abs_diff"] for r in per_image]
    norm_diffs = [r["normalized_embedding_max_abs_diff"] for r in per_image]
    cosine_diffs = [r["max_cosine_abs_diff"] for r in per_image]
    top1_disagree = sum(1 for r in per_image if not r["top1_agree"])
    top3_disagree = sum(1 for r in per_image if not r["top3_set_agree"])

    model_parity = {
        "n_images": len(per_image),
        "raw_embedding_max_abs_diff": max(raw_diffs),
        "normalized_embedding_max_abs_diff": max(norm_diffs),
        "max_cosine_abs_divergence": percentiles_with_min(cosine_diffs),
        "top1_disagreement_count": top1_disagree,
        "top1_disagreement_of": len(per_image),
        "top3_set_disagreement_count": top3_disagree,
        "top3_set_disagreement_of": len(per_image),
        "per_image": per_image,
    }

    # ---- ONNX batch=1 vs batch=N ----------------------------------------------
    stacked = np.concatenate([t for t in onnx_tensors], axis=0).astype(np.float32)
    n = stacked.shape[0]
    b1_raw = np.stack([cpu_session.run(None, {cpu_input_name: stacked[i:i + 1]})[0][0]
                       for i in range(n)], axis=0)
    batched_chunks = []
    for start in range(0, n, onnx_batch_size):
        chunk = stacked[start:start + onnx_batch_size]
        batched_chunks.append(cpu_session.run(None, {cpu_input_name: chunk})[0])
    b_batched_raw = np.concatenate(batched_chunks, axis=0)
    if not is_finite(b1_raw, b_batched_raw):
        raise ParityError("batch comparison produced non-finite embedding values")

    b1_norm = b1_raw / np.clip(np.linalg.norm(b1_raw, axis=1, keepdims=True), 1e-8, None)
    bb_norm = b_batched_raw / np.clip(np.linalg.norm(b_batched_raw, axis=1, keepdims=True), 1e-8, None)
    b1_sims = b1_norm @ protos_onnx_np.T
    bb_sims = bb_norm @ protos_onnx_np.T
    b1_top1 = np.argmax(b1_sims, axis=1)
    bb_top1 = np.argmax(bb_sims, axis=1)
    b1_top3 = np.argsort(-b1_sims, axis=1, kind="stable")[:, :3]
    bb_top3 = np.argsort(-bb_sims, axis=1, kind="stable")[:, :3]
    top3_set_disagree_batch = int(sum(
        1 for i in range(n) if set(b1_top3[i]) != set(bb_top3[i])
    ))

    onnx_batch_comparison = {
        "n_samples": int(n), "batch_size": onnx_batch_size,
        "raw_embedding_max_abs_diff": float(np.abs(b1_raw - b_batched_raw).max()),
        "normalized_embedding_max_abs_diff": float(np.abs(b1_norm - bb_norm).max()),
        "max_cosine_abs_divergence": percentiles_with_min(
            list(np.abs(np.array([widen(v) for v in b1_sims.max(axis=1)])
                       - np.array([widen(v) for v in bb_sims.max(axis=1)])))),
        "top1_disagreement_count": int(np.sum(b1_top1 != bb_top1)),
        "top1_disagreement_of": int(n),
        "top3_set_disagreement_count": top3_set_disagree_batch,
        "top3_set_disagreement_of": int(n),
    }

    # ---- repeated ONNX-CPU determinism checks --------------------------------
    repeat_outputs = []
    for _ in range(determinism_repeats):
        out = cpu_session.run(None, {cpu_input_name: stacked})[0]
        repeat_outputs.append(out)
    first = repeat_outputs[0]
    max_diff_across_repeats = max(
        float(np.abs(first - other).max()) for other in repeat_outputs[1:]
    ) if len(repeat_outputs) > 1 else 0.0
    onnx_determinism = {
        "n_repeats": determinism_repeats,
        "all_bit_identical": max_diff_across_repeats == 0.0,
        "max_abs_diff_across_repeats": max_diff_across_repeats,
    }

    # ---- ORT node-provider evidence (profiling) ------------------------------
    ort_evidence = confirm_cpu_only_via_profiling(run_dir / "backbone.onnx", onnx_tensors)

    deterministic_content = {
        "label": LABEL,
        "generator": {
            "name": GENERATOR_NAME, "version": GENERATOR_VERSION,
            "source_canonical_lf_sha256": source_canonical_lf_sha256(Path(__file__).resolve()),
        },
        "candidate": {
            "run_dir": _repo_relative_posix(run_dir),
            "run_manifest_sha256": run_manifest_sha256,
            "final_artifact_hashes": final_hashes,
            "manifest_sha256": manifest_sha256,
            "taxonomy_sha256": taxonomy_sha256,
            "val_split_sha256": run_manifest["val_split"]["sha256"],
            "num_classes": num_classes,
            "config": candidate_config,
            "resolved_config_sha256": provenance.get("resolved_config_sha256"),
        },
        # Stable measurement inputs: the CANDIDATE run's own recorded training
        # provenance -- never this workspace's ambient git HEAD, which changes
        # the moment these very files are committed and would otherwise
        # invalidate a --verify run for a reason unrelated to the candidate.
        "candidate_training_git_head": run_manifest.get("git_head"),
        "candidate_training_git_dirty": run_manifest.get("git_dirty"),
        "environment": {**env, "numerical_policy": numerical_policy},
        "sample_selection": selection,
        "preprocessing_parity": preprocessing_parity,
        "model_parity": model_parity,
        "onnx_batch_comparison": onnx_batch_comparison,
        "onnx_determinism": onnx_determinism,
        "ort_provider_evidence": ort_evidence,
    }
    return deterministic_content


def serialize_deterministic(deterministic_content: dict) -> bytes:
    text = json.dumps(deterministic_content, indent=2, sort_keys=True)
    return (text + "\n").replace("\r\n", "\n").encode("utf-8")


def build_full_report(deterministic_content: dict) -> dict:
    workspace_git_head, workspace_git_dirty = get_workspace_git_state()
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generation_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds").replace("+00:00", "Z"),
            "host": platform.node(),
            # Ambient workspace state at BUILD time -- excluded from
            # deterministic-content comparison on purpose (see
            # get_workspace_git_state's docstring). This is NOT the
            # candidate's own training commit; see
            # deterministic_content.candidate_training_git_head for that.
            "workspace_git_head": workspace_git_head,
            "workspace_git_dirty": workspace_git_dirty,
            "generator": {
                "name": GENERATOR_NAME, "version": GENERATOR_VERSION,
                "source_canonical_lf_sha256": source_canonical_lf_sha256(
                    Path(__file__).resolve()),
            },
        },
        "deterministic_content": deterministic_content,
    }


def validate_report_schema(report: dict, source_label: str) -> dict:
    """Minimal structural validation before ANY byte comparison is
    attempted -- a malformed or wrong-schema-version report must fail with a
    specific message, never a raw KeyError deep inside comparison logic."""
    if not isinstance(report, dict):
        raise ParityError(f"{source_label}: report root must be a JSON object.")
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ParityError(
            f"{source_label}: report_schema_version="
            f"{report.get('report_schema_version')!r}, expected {REPORT_SCHEMA_VERSION!r}."
        )
    if not isinstance(report.get("deterministic_content"), dict):
        raise ParityError(f"{source_label}: missing or invalid 'deterministic_content'.")
    return report["deterministic_content"]


def verify_byte_identical(regenerated: dict, existing_path: Path) -> None:
    if not existing_path.exists():
        raise ParityError(f"{existing_path} does not exist -- nothing to verify against.")
    try:
        existing_full = json.loads(existing_path.read_text())
    except json.JSONDecodeError as e:
        raise ParityError(f"{existing_path} is not valid JSON: {e}") from e
    existing_content = validate_report_schema(existing_full, str(existing_path))
    existing_bytes = serialize_deterministic(existing_content)
    regenerated_bytes = serialize_deterministic(regenerated)
    if regenerated_bytes != existing_bytes:
        raise ParityError(
            f"Regenerated deterministic_content does not byte-match {existing_path}. "
            f"regenerated_sha256={_sha256_bytes(regenerated_bytes)} "
            f"existing_sha256={_sha256_bytes(existing_bytes)}"
        )


# ----------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--local-data-dir", type=Path, required=True)
    ap.add_argument("--manifest-csv", type=Path, default=None)
    ap.add_argument("--taxonomy-json", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-species", type=int, default=N_PER_SPECIES_DEFAULT)
    ap.add_argument("--onnx-batch-size", type=int, default=ONNX_BATCH_SIZE_DEFAULT)
    ap.add_argument("--determinism-repeats", type=int, default=DETERMINISM_REPEATS_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--existing", type=Path, default=None)
    args = ap.parse_args()

    if args.verify and args.existing is None:
        ap.error("--verify requires --existing.")
    if not args.verify and args.out is None:
        ap.error("--out is required unless --verify is given.")

    database_url = os.environ.get("DATABASE_URL")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.run_dir.resolve()
    manifest_csv = args.manifest_csv
    taxonomy_json = args.taxonomy_json
    rm_probe = run_dir / "run_manifest.json"
    if (manifest_csv is None or taxonomy_json is None) and rm_probe.exists():
        rm = json.loads(rm_probe.read_text())
        if manifest_csv is None and rm.get("manifest"):
            manifest_csv = Path(rm["manifest"]["path"])
        if taxonomy_json is None and rm.get("taxonomy_source"):
            taxonomy_json = Path(rm["taxonomy_source"]["path"])
    if manifest_csv is None or taxonomy_json is None:
        ap.error("--manifest-csv/--taxonomy-json could not be inferred; pass explicitly.")

    try:
        deterministic_content = run_parity(
            run_dir=run_dir, local_data_dir=args.local_data_dir,
            manifest_csv=manifest_csv, taxonomy_json=taxonomy_json,
            database_url=database_url, device=device, seed=args.seed,
            n_per_species=args.n_per_species, onnx_batch_size=args.onnx_batch_size,
            determinism_repeats=args.determinism_repeats,
        )
    except ParityError as e:
        print(f"[parity_check_v2] FAILED: {e}")
        return 1

    if args.verify:
        try:
            verify_byte_identical(deterministic_content, args.existing)
        except ParityError as e:
            print(f"[parity_check_v2] VERIFY FAILED: {e}")
            return 1
        print(f"[parity_check_v2] VERIFY PASSED: regenerated deterministic content is "
             f"byte-identical to {args.existing}")
        return 0

    report = build_full_report(deterministic_content)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / f"{args.out.stem}.tmp{os.getpid()}{args.out.suffix}"
    tmp.write_bytes(payload)
    os.replace(tmp, args.out)
    print(f"[parity_check_v2] wrote {args.out} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
