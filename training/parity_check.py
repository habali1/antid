#!/usr/bin/env python3
"""parity_check.py — measure, don't assume, that training and serving agree.

Three independent comparisons, none of them a retrain or a re-export:

  Stage A (graph/runtime parity only): a synthetic, seeded input tensor run
  through the PyTorch checkpoint and the exported ONNX backbone. This tests
  whether the two graphs compute the same function -- it says nothing about
  real preprocessing, since the input is not a real image.

  Stage B (real end-to-end parity): 200 real cached images (4 per species,
  deterministically selected) run through two COMPLETE pipelines --
  training/data.py's val transform + the PyTorch checkpoint on CUDA, versus
  api/inference.py's actual preprocess() + the ONNX backbone on an explicitly
  CPU-forced session -- reusing the real production code on both sides, not
  a reimplementation of it.

  Batch comparison: the same 200 preprocessed tensors run through the same
  CPU ONNX session at batch=1 versus batch=16 (dynamic batch dimension),
  to check ONNX Runtime's own batching doesn't perturb results.

This script reports divergence and disagreement counts. It does not decide
what divergence is acceptable, does not compute or report accuracy against
any ground truth, and does not modify api/, mobile/, or any frozen artifact.
Every number in the JSON report is unrounded float; terminal output may round
for display only.

Usage:
    python parity_check.py
    python parity_check.py --out artifacts/parity_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent / "api"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(API_DIR))

SEED = 42                    # matches config.yaml's training seed, reused here for
                              # the deterministic image-selection RNG (unrelated use)
N_PER_SPECIES = 4
N_SAMPLES = 50 * N_PER_SPECIES
BATCH_SIZE = 16
GATE_THRESHOLD = 0.60         # the frozen candidate value, calibration_v1.json's
                              # frozen_candidate_abstention_threshold -- used here only
                              # as the comparison point for reporting agreement/
                              # disagreement between implementations, not re-derived,
                              # not applied to any real decision.
IMG_EXTS_ORDER = (".jpg", ".jpeg", ".png", ".webp")  # data.py's EXT_PROBE_ORDER


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- env recon
def environment_recon() -> dict:
    import torch
    import torchvision
    import PIL
    import onnxruntime as ort

    info = {
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
    return info


# ------------------------------------------------------------- deterministic selection
def select_samples(manifest_path: Path, clean_dir: Path, taxonomy: dict, seed: int, n_per_species: int):
    """4 (default) resolvable images per known species, deterministic given seed.

    Stable ordering: candidates for each species are sorted by photo_id before
    a seeded shuffle, so re-running with the same seed reproduces the same
    selection regardless of manifest row order. "Resolvable" means the file
    actually exists locally under clean_dir/{slug}/{photo_id}.{ext} for one of
    EXT_PROBE_ORDER's extensions, tried in that fixed order -- rows whose file
    is missing are skipped (deterministically, not silently reordered).
    """
    import csv

    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    by_slug: dict[str, list[dict]] = {}
    for r in rows:
        by_slug.setdefault(r["slug"], []).append(r)

    def resolve(slug: str, photo_id: str) -> Path | None:
        for ext in IMG_EXTS_ORDER:
            p = clean_dir / slug / f"{photo_id}{ext}"
            if p.exists():
                return p
        return None

    slugs = sorted({v["slug"] for v in taxonomy.values()})
    rng = random.Random(seed)
    selected: list[dict] = []
    shortfalls: list[str] = []

    for slug in slugs:
        cands = sorted(by_slug.get(slug, []), key=lambda r: r["photo_id"])
        shuffled = cands[:]
        rng.shuffle(shuffled)
        picked = []
        for r in shuffled:
            path = resolve(slug, r["photo_id"])
            if path is None:
                continue
            picked.append({"slug": slug, "photo_id": r["photo_id"], "species": r["species"],
                          "path": str(path)})
            if len(picked) >= n_per_species:
                break
        if len(picked) < n_per_species:
            shortfalls.append(f"{slug}: {len(picked)}/{n_per_species} resolvable")
        selected.extend(picked)

    return selected, shortfalls


# --------------------------------------------------------------------- comparison helpers
def widen(x) -> float:
    """float32 scalar -> Python float, preserving the exact float32 value
    (its nearest float64 representation), never rounding or truncating."""
    return float(np.float32(x))


def gate_reject(max_sim: float, threshold: float = GATE_THRESHOLD) -> bool:
    """Strict `<`, threshold never cast to float32, max_sim already widened."""
    return max_sim < threshold


def is_finite(*arrays) -> bool:
    return all(np.all(np.isfinite(a)) for a in arrays)


def percentiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "max": float(arr.max()), "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()), "n": int(arr.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--manifest", type=Path, default=HERE.parent / "data" / "manifest_all.csv")
    ap.add_argument("--clean-dir", type=Path, default=HERE.parent / "data" / "clean")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "parity_report.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-per-species", type=int, default=N_PER_SPECIES)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "script": "training/parity_check.py",
    }
    failures: list[str] = []

    # ---- environment recon --------------------------------------------------
    log("=== environment recon ===")
    env = environment_recon()
    for k, v in env.items():
        log(f"  {k}: {v}")
    report["environment"] = env

    if not env["torch_cuda_available"]:
        msg = ("CUDA is not available in this environment. calibration_v1/Phase C ran "
              "on CUDA; this run cannot fully reproduce that calibration-to-serving "
              "comparison. Stopping rather than silently substituting CPU for the "
              "PyTorch reference and declaring parity.")
        log(f"FATAL: {msg}")
        report["fatal_error"] = msg
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        return 1

    import torch
    import torch.nn as nn
    import onnxruntime as ort
    import yaml
    from PIL import Image
    from data import build_transforms
    from model import AntIDModel
    from inference import AntIdentifier

    device = torch.device("cuda")

    # An API-style session using the same provider selection api/inference.py
    # uses, purely to report what it *actually* resolves to in this
    # environment -- not used for any measurement below.
    _probe_sess = ort.InferenceSession(str(args.artifacts / "backbone.onnx"),
                                       providers=ort.get_available_providers())
    api_style_providers = _probe_sess.get_providers()
    log(f"  api-style session providers (unforced): {api_style_providers}")
    report["environment"]["api_style_session_providers"] = api_style_providers
    del _probe_sess

    # ---- artifact/config hashes ----------------------------------------------
    hashes = {
        "model_pth": sha256_file(args.artifacts / "model.pth"),
        "backbone_onnx": sha256_file(args.artifacts / "backbone.onnx"),
        "prototypes_npy": sha256_file(args.artifacts / "prototypes.npy"),
        "taxonomy_json": sha256_file(args.artifacts / "taxonomy.json"),
        "config_yaml": sha256_file(args.config),
    }
    report["artifact_hashes"] = hashes
    log("=== artifact hashes ===")
    for k, v in hashes.items():
        log(f"  {k}: {v[:16]}...")

    # ---- normalization recon ---------------------------------------------------
    log("=== normalization recon ===")
    protos_np = np.load(args.artifacts / "prototypes.npy")
    row_norms = np.linalg.norm(protos_np, axis=1)
    norm_stats = {
        "shape": list(protos_np.shape),
        "row_norm_min": float(row_norms.min()),
        "row_norm_max": float(row_norms.max()),
        "row_norm_mean": float(row_norms.mean()),
        "row_norm_max_abs_deviation_from_1": float(np.abs(row_norms - 1.0).max()),
    }
    for k, v in norm_stats.items():
        log(f"  {k}: {v}")
    report["prototype_norm_stats"] = norm_stats

    # ---- load config, taxonomy, model ------------------------------------------
    cfg = yaml.safe_load(args.config.read_text())
    taxonomy_raw = json.loads((args.artifacts / "taxonomy.json").read_text())
    taxonomy = {int(k): v for k, v in taxonomy_raw.items()}
    n_classes = len(taxonomy)

    ckpt_path = args.artifacts / "model.pth"
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model = AntIDModel(num_classes=n_classes, backbone=cfg["model"]["backbone"],
                       pretrained=False, dropout=cfg["dropout"],
                       embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    model.load_state_dict(sd)
    model.eval()

    protos_torch = nn.functional.normalize(
        torch.as_tensor(protos_np, dtype=torch.float32, device=device), dim=1)
    protos_torch_np = protos_torch.cpu().numpy()

    # ---- serving-side identifier (real production code) -----------------------
    identifier = AntIdentifier(artifacts_dir=args.artifacts)

    # Explicitly CPU-forced session for the actual parity measurement --
    # api/inference.py's own session uses ort.get_available_providers(), which
    # is NOT forced CPU; identifier.session is used only for its preprocess()
    # method and its already-normalized `prototypes` array below, never for
    # the timed/measured ONNX execution.
    cpu_session = ort.InferenceSession(str(args.artifacts / "backbone.onnx"),
                                       providers=["CPUExecutionProvider"])
    cpu_providers = cpu_session.get_providers()
    log(f"  parity-measurement ONNX providers (forced): {cpu_providers}")
    report["environment"]["parity_onnx_providers_forced"] = cpu_providers
    if cpu_providers != ["CPUExecutionProvider"]:
        failures.append(f"CPUExecutionProvider was not exclusively selected: got {cpu_providers}")
    cpu_input_name = cpu_session.get_inputs()[0].name

    image_size = cfg["image_size"]

    # ================================================================= STAGE A
    log("\n=== Stage A: graph/runtime parity only (synthetic input, NOT a real image) ===")
    gen = torch.Generator().manual_seed(args.seed)
    synth = torch.randn(1, 3, image_size, image_size, generator=gen)  # ~N(0,1), stand-in
                                                                       # for a normalized
                                                                       # tensor's statistics

    with torch.no_grad():
        ref_raw = model.embed(synth.to(device)).cpu().numpy()[0]
    onnx_raw = cpu_session.run(None, {cpu_input_name: synth.numpy()})[0][0]

    if not is_finite(ref_raw, onnx_raw):
        failures.append("Stage A produced non-finite embedding values")

    ref_norm = ref_raw / np.clip(np.linalg.norm(ref_raw), 1e-8, None)
    onnx_norm = onnx_raw / np.clip(np.linalg.norm(onnx_raw), 1e-8, None)

    protos_np_norm = protos_np / np.clip(np.linalg.norm(protos_np, axis=1, keepdims=True), 1e-8, None)
    ref_sims = protos_np_norm @ ref_norm
    onnx_sims = protos_np_norm @ onnx_norm

    stage_a = {
        "label": "graph/runtime parity only -- synthetic seeded input, not a real image "
                "or real preprocessing pipeline",
        "input_seed": args.seed,
        "input_shape": list(synth.shape),
        "raw_embedding_max_abs_diff": float(np.abs(ref_raw - onnx_raw).max()),
        "normalized_embedding_max_abs_diff": float(np.abs(ref_norm - onnx_norm).max()),
        "score_vector_max_abs_diff": float(np.abs(ref_sims - onnx_sims).max()),
        "max_similarity_reference": widen(ref_sims.max()),
        "max_similarity_onnx_cpu": widen(onnx_sims.max()),
        "max_similarity_abs_diff": abs(widen(ref_sims.max()) - widen(onnx_sims.max())),
        "top1_agree": bool(int(np.argmax(ref_sims)) == int(np.argmax(onnx_sims))),
    }
    report["stage_a_graph_runtime_parity"] = stage_a
    for k, v in stage_a.items():
        log(f"  {k}: {v}")

    # ================================================================= STAGE B
    log("\n=== Stage B: real end-to-end preprocessing + runtime parity ===")
    samples, shortfalls = select_samples(args.manifest, args.clean_dir, taxonomy,
                                        args.seed, args.n_per_species)
    if shortfalls:
        log(f"  shortfalls: {shortfalls}")
        failures.append(f"{len(shortfalls)} species short of {args.n_per_species} "
                        f"resolvable images: {shortfalls}")
    if len(samples) != N_SAMPLES:
        failures.append(f"expected {N_SAMPLES} samples, selected {len(samples)}")

    log(f"  selected {len(samples)} samples across {len(set(s['slug'] for s in samples))} species")

    ref_transform = build_transforms(cfg, train=False)

    decode_failures: list[str] = []
    per_image = []
    onnx_preproc_batch = []  # collect for the batch-comparison stage
    for s in samples:
        path = Path(s["path"])
        file_hash = sha256_file(path)
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:  # noqa: BLE001
            decode_failures.append(f"{s['slug']}/{s['photo_id']}: {e!r}")
            continue

        # reference: training/data.py's val transform, exact production call
        ref_tensor = ref_transform(img.convert("RGB"))  # (3,H,W) torch tensor

        # serving: api/inference.py's actual preprocess(), exact production call
        img2 = Image.open(path)
        img2.load()
        onnx_tensor = identifier.preprocess(img2)  # (1,3,H,W) numpy

        preproc_diff = float(np.abs(ref_tensor.numpy() - onnx_tensor[0]).max())

        with torch.no_grad():
            ref_raw_e = model.embed(ref_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
        onnx_raw_e = cpu_session.run(None, {cpu_input_name: onnx_tensor})[0][0]

        if not is_finite(ref_raw_e, onnx_raw_e):
            decode_failures.append(f"{s['slug']}/{s['photo_id']}: non-finite embedding")
            continue

        ref_norm_e = ref_raw_e / np.clip(np.linalg.norm(ref_raw_e), 1e-8, None)
        onnx_norm_e = onnx_raw_e / np.clip(np.linalg.norm(onnx_raw_e), 1e-8, None)

        ref_sims_e = (protos_torch_np @ ref_norm_e)
        onnx_sims_e = (identifier.prototypes @ onnx_norm_e)

        ref_max = widen(ref_sims_e.max())
        onnx_max = widen(onnx_sims_e.max())
        ref_top1 = int(np.argmax(ref_sims_e))
        onnx_top1 = int(np.argmax(onnx_sims_e))
        ref_reject = gate_reject(ref_max)
        onnx_reject = gate_reject(onnx_max)

        onnx_preproc_batch.append(onnx_tensor[0])

        per_image.append({
            "slug": s["slug"], "photo_id": s["photo_id"], "file_sha256": file_hash,
            "preprocessing_tensor_max_abs_diff": preproc_diff,
            "raw_embedding_max_abs_diff": float(np.abs(ref_raw_e - onnx_raw_e).max()),
            "normalized_embedding_max_abs_diff": float(np.abs(ref_norm_e - onnx_norm_e).max()),
            "score_vector_max_abs_diff": float(np.abs(ref_sims_e - onnx_sims_e).max()),
            "max_similarity_reference": ref_max,
            "max_similarity_onnx_cpu": onnx_max,
            "max_similarity_abs_diff": abs(ref_max - onnx_max),
            "top1_reference": ref_top1, "top1_onnx_cpu": onnx_top1,
            "top1_agree": ref_top1 == onnx_top1,
            "gate_reject_reference": ref_reject, "gate_reject_onnx_cpu": onnx_reject,
            "gate_disagree": ref_reject != onnx_reject,
        })

    if decode_failures:
        failures.append(f"{len(decode_failures)} decode/finite failures: {decode_failures}")
    if len(per_image) < N_SAMPLES:
        failures.append(f"only {len(per_image)}/{N_SAMPLES} images produced valid results")

    # provenance hashes -- not verification against any authoritative manifest
    ordered_list_str = "\n".join(f"{r['slug']},{r['photo_id']},{r['file_sha256']}"
                                 for r in per_image)
    sample_list_hash = sha256_bytes(ordered_list_str.encode("utf-8"))

    if per_image:
        preproc_diffs = [r["preprocessing_tensor_max_abs_diff"] for r in per_image]
        raw_diffs = [r["raw_embedding_max_abs_diff"] for r in per_image]
        norm_diffs = [r["normalized_embedding_max_abs_diff"] for r in per_image]
        score_diffs = [r["score_vector_max_abs_diff"] for r in per_image]
        max_sim_diffs = [r["max_similarity_abs_diff"] for r in per_image]
        top1_agree_count = sum(1 for r in per_image if r["top1_agree"])
        gate_disagree_count = sum(1 for r in per_image if r["gate_disagree"])

        # images closest to the gate on either path
        def dist_to_gate(r):
            return min(abs(r["max_similarity_reference"] - GATE_THRESHOLD),
                      abs(r["max_similarity_onnx_cpu"] - GATE_THRESHOLD))
        closest = sorted(per_image, key=dist_to_gate)[:5]
        closest_report = [{
            "slug": r["slug"], "photo_id": r["photo_id"], "file_sha256": r["file_sha256"],
            "max_similarity_reference": r["max_similarity_reference"],
            "max_similarity_onnx_cpu": r["max_similarity_onnx_cpu"],
            "gate_reject_reference": r["gate_reject_reference"],
            "gate_reject_onnx_cpu": r["gate_reject_onnx_cpu"],
        } for r in closest]

        stage_b = {
            "label": "real end-to-end preprocessing + runtime parity",
            "seed": args.seed, "n_per_species": args.n_per_species,
            "n_samples_selected": len(samples), "n_samples_valid": len(per_image),
            "sample_list_hash_sha256": sample_list_hash,
            "shortfalls": shortfalls, "decode_failures": decode_failures,
            "preprocessing_tensor_max_abs_diff": max(preproc_diffs),
            "raw_embedding_max_abs_diff": max(raw_diffs),
            "normalized_embedding_max_abs_diff": max(norm_diffs),
            "score_vector_max_abs_diff": max(score_diffs),
            "max_similarity_divergence": percentiles(max_sim_diffs),
            "top1_agreement_count": top1_agree_count,
            "top1_agreement_of": len(per_image),
            "gate_disagreement_count": gate_disagree_count,
            "gate_threshold_used_for_reporting": GATE_THRESHOLD,
            "images_closest_to_gate": closest_report,
            "per_image": per_image,
        }
    else:
        stage_b = {"label": "real end-to-end preprocessing + runtime parity",
                  "error": "no valid images", "shortfalls": shortfalls,
                  "decode_failures": decode_failures}
        failures.append("Stage B produced zero valid image comparisons")

    report["stage_b_real_end_to_end_parity"] = stage_b
    for k, v in stage_b.items():
        if k != "per_image":
            log(f"  {k}: {v}")

    # ================================================================= BATCH CMP
    log(f"\n=== Batch comparison: ONNX CPU batch=1 vs batch={args.batch_size} ===")
    if len(onnx_preproc_batch) >= 1:
        stacked = np.stack(onnx_preproc_batch, axis=0).astype(np.float32)  # (N,3,H,W)
        n = stacked.shape[0]

        # batch=1 embeddings (recomputed explicitly for this comparison, rather
        # than reusing Stage B's, to keep this stage self-contained/independent)
        b1_raw = np.stack([cpu_session.run(None, {cpu_input_name: stacked[i:i+1]})[0][0]
                          for i in range(n)], axis=0)

        # batched embeddings, including the final partial batch
        batched_chunks = []
        for start in range(0, n, args.batch_size):
            chunk = stacked[start:start + args.batch_size]
            out = cpu_session.run(None, {cpu_input_name: chunk})[0]
            batched_chunks.append(out)
        b_batched_raw = np.concatenate(batched_chunks, axis=0)

        if not is_finite(b1_raw, b_batched_raw):
            failures.append("batch comparison produced non-finite embedding values")

        b1_norm = b1_raw / np.clip(np.linalg.norm(b1_raw, axis=1, keepdims=True), 1e-8, None)
        bb_norm = b_batched_raw / np.clip(np.linalg.norm(b_batched_raw, axis=1, keepdims=True), 1e-8, None)

        b1_sims = b1_norm @ identifier.prototypes.T
        bb_sims = bb_norm @ identifier.prototypes.T

        b1_max = np.array([widen(v) for v in b1_sims.max(axis=1)])
        bb_max = np.array([widen(v) for v in bb_sims.max(axis=1)])
        b1_top1 = np.argmax(b1_sims, axis=1)
        bb_top1 = np.argmax(bb_sims, axis=1)
        b1_reject = b1_max < GATE_THRESHOLD
        bb_reject = bb_max < GATE_THRESHOLD

        n_full_batches = n // args.batch_size
        final_batch_size = n - n_full_batches * args.batch_size

        batch_cmp = {
            "n_samples": int(n),
            "batch_size": args.batch_size,
            "n_full_batches": n_full_batches,
            "final_partial_batch_size": final_batch_size if final_batch_size else args.batch_size,
            "raw_embedding_max_abs_diff": float(np.abs(b1_raw - b_batched_raw).max()),
            "normalized_embedding_max_abs_diff": float(np.abs(b1_norm - bb_norm).max()),
            "score_vector_max_abs_diff": float(np.abs(b1_sims - bb_sims).max()),
            "max_similarity_divergence": percentiles(list(np.abs(b1_max - bb_max))),
            "top1_agreement_count": int(np.sum(b1_top1 == bb_top1)),
            "top1_agreement_of": int(n),
            "gate_disagreement_count": int(np.sum(b1_reject != bb_reject)),
        }
    else:
        batch_cmp = {"error": "no preprocessed tensors available from Stage B"}
        failures.append("batch comparison had no input tensors (Stage B produced none)")

    report["batch_comparison_onnx_cpu_batch1_vs_batched"] = batch_cmp
    for k, v in batch_cmp.items():
        log(f"  {k}: {v}")

    # ================================================================= WRAP UP
    report["structural_verification"] = {
        "n_samples_required": N_SAMPLES,
        "n_samples_valid": len(per_image) if per_image else 0,
        "cpu_execution_provider_confirmed": cpu_providers == ["CPUExecutionProvider"],
        "cuda_available": env["torch_cuda_available"],
        "failures": failures,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log(f"\nwrote {args.out}")

    if failures:
        log(f"\nFAILED structural checks ({len(failures)}):")
        for f in failures:
            log(f"  ! {f}")
        return 1

    log("\nAll structural checks passed. No acceptance threshold applied -- "
       "see the reported divergence and disagreement numbers above for review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
