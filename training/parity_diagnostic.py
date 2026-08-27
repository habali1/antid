#!/usr/bin/env python3
"""parity_diagnostic.py — Step 4a: isolate the source of the single top-1
disagreement found in training/artifacts/parity_report.json
(pseudomyrmex-gracilis/671332727, raw-embedding max|diff| 9.78 against a
population where every other image is 0.001-0.009).

This is diagnostic only. It does not propose a fix, does not touch
parity_report.json, and does not change api/, mobile/, or any frozen
artifact. It writes its own report to
training/artifacts/parity_diagnostic.json.

Correction carried in from the planning discussion: max|raw_a-raw_b| /
max|normalized_a-normalized_b| is NOT a valid lower bound on either raw
embedding's norm (the two vectors can have different norms, and the two
maxima can occur at different indices). This script does not use that
ratio anywhere; raw L2 norm and RMS magnitude are measured directly, per
image, per runtime path.

Sections:
  1. Input/embedding measurement for 4 images (the outlier + 3 controls
     selected from the existing parity sample): decode stats, preprocessing
     parity, raw embedding norm/RMS on every runtime path, divergent-channel
     evidence.
  2. A 7-path runtime/device matrix (PyTorch CUDA/CPU; ONNX CPU at four
     optimization/threading configurations; the unforced API-style ONNX
     config) with the 5 isolating pairwise comparisons the plan specifies.
  3. ORT provider evidence via SessionOptions.enable_profiling -- reports
     actual per-node provider assignment from the emitted trace, not an
     inference from session.get_providers().
  4. Determinism: >=20 repeats within this process for PyTorch CUDA and the
     relevant ONNX CPU arms; >=5 repeats of the outlier in fresh interpreter
     subprocesses for PyTorch CUDA and ONNX CPU default.
  5. CUDA numerical controls: record cudnn/TF32/matmul-precision/autocast
     state, rerun under maximally-deterministic settings, restore afterward.

Usage:
    python parity_diagnostic.py
    python parity_diagnostic.py --subprocess-embed TENSOR.npy MODE OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent / "api"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(API_DIR))

GATE_THRESHOLD = 0.60  # comparison point only, same convention as parity_check.py

OUTLIER = {"slug": "pseudomyrmex-gracilis", "photo_id": "671332727",
          "expected_sha256": "d1b5161f93b7ada9b535a93b6dcd290630676814d8441d3627d03f16fa802932"}


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


def widen(x) -> float:
    return float(np.float32(x))


def l2_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v.astype(np.float64)))


def rms(v: np.ndarray) -> float:
    v64 = v.astype(np.float64)
    return float(np.sqrt(np.mean(v64 * v64)))


def top_k_divergent_channels(a: np.ndarray, b: np.ndarray, k: int = 10) -> list[dict]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    order = np.argsort(-diff)[:k]
    return [{"index": int(i), "abs_diff": float(diff[i]),
             "value_a": float(a[i]), "value_b": float(b[i])} for i in order]


# ============================================================= model/session loading
def load_pytorch_model(artifacts: Path, config: dict, device: "torch.device"):
    import torch
    from model import AntIDModel
    ckpt = torch.load(artifacts / "model.pth", map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    taxonomy = json.loads((artifacts / "taxonomy.json").read_text())
    model = AntIDModel(num_classes=len(taxonomy), backbone=config["model"]["backbone"],
                       pretrained=False, dropout=config["dropout"],
                       embedding_dim=config["model"]["embedding_dim"]).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def build_onnx_session(onnx_path: Path, providers: list[str],
                       graph_opt=None, intra_threads: int | None = None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if graph_opt is not None:
        so.graph_optimization_level = graph_opt
    if intra_threads is not None:
        so.intra_op_num_threads = intra_threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    return sess


def onnx_embed(sess, x: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0][0]


def pytorch_embed(model, x, device) -> np.ndarray:
    import torch
    with torch.no_grad():
        t = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model.embed(t).cpu().numpy()[0]


# ============================================================== subprocess entry point
def subprocess_embed_main(tensor_path: Path, mode: str, out_path: Path) -> int:
    """Runs in a FRESH interpreter process. Loads artifacts, computes one raw
    embedding via `mode`, writes {embedding, l2_norm, sha256} to out_path."""
    import yaml
    x = np.load(tensor_path)
    artifacts = HERE / "artifacts"
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())

    if mode == "pytorch_cuda":
        import torch
        device = torch.device("cuda")
        model = load_pytorch_model(artifacts, cfg, device)
        emb = pytorch_embed(model, x, device)
    elif mode == "onnx_cpu":
        sess = build_onnx_session(artifacts / "backbone.onnx", ["CPUExecutionProvider"])
        emb = onnx_embed(sess, x)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    payload = {
        "mode": mode,
        "l2_norm": l2_norm(emb),
        "embedding_sha256": sha256_bytes(emb.astype(np.float32).tobytes()),
        "embedding": emb.astype(np.float64).tolist(),
    }
    out_path.write_text(json.dumps(payload))
    return 0


# ==================================================================== image loading
def load_diagnostic_image(slug: str, photo_id: str, clean_dir: Path):
    from PIL import Image
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = clean_dir / slug / f"{photo_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{slug}/{photo_id} not found under {clean_dir}")


def exif_orientation(img) -> int | None:
    try:
        exif = img.getexif()
        return exif.get(0x0112)  # Orientation tag
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subprocess-embed", nargs=3, metavar=("TENSOR_NPY", "MODE", "OUT_JSON"))
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--clean-dir", type=Path, default=HERE.parent / "data" / "clean")
    ap.add_argument("--parity-report", type=Path, default=HERE / "artifacts" / "parity_report.json")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "parity_diagnostic.json")
    ap.add_argument("--within-process-repeats", type=int, default=20)
    ap.add_argument("--fresh-process-repeats", type=int, default=5)
    args = ap.parse_args()

    if args.subprocess_embed:
        tensor_path, mode, out_json = args.subprocess_embed
        return subprocess_embed_main(Path(tensor_path), mode, Path(out_json))

    # ---- normal (main) run ---------------------------------------------------
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    import yaml
    from PIL import Image

    from data import build_transforms
    from inference import AntIdentifier

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "script": "training/parity_diagnostic.py",
        "correction_from_planning_discussion": (
            "max|raw_a-raw_b| / max|normalized_a-normalized_b| is NOT a valid lower "
            "bound on either raw embedding's norm -- the vectors can have different "
            "norms and the two maxima can occur at different indices. This script "
            "measures L2 norm and RMS magnitude directly per image per runtime path, "
            "and does not use that ratio anywhere."
        ),
    }
    failures: list[str] = []

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        msg = "CUDA unavailable in this environment; cannot run the CUDA arms of this diagnostic."
        log(f"FATAL: {msg}")
        report["fatal_error"] = msg
        args.out.write_text(json.dumps(report, indent=2))
        return 1

    model = load_pytorch_model(args.artifacts, cfg, device)
    identifier = AntIdentifier(artifacts_dir=args.artifacts)  # real preprocess() + prototypes
    ref_transform = build_transforms(cfg, train=False)
    onnx_path = args.artifacts / "backbone.onnx"

    hashes = {name: sha256_file(args.artifacts / name)
             for name in ("model.pth", "backbone.onnx", "prototypes.npy", "taxonomy.json")}
    hashes["config_yaml"] = sha256_file(args.config)
    report["artifact_hashes"] = hashes

    # ---- select the 4 diagnostic images from the EXISTING parity sample --------
    parity = json.loads(args.parity_report.read_text())
    per_image = parity["stage_b_real_end_to_end_parity"]["per_image"]
    outlier_rec = next((r for r in per_image if r["slug"] == OUTLIER["slug"]
                        and r["photo_id"] == OUTLIER["photo_id"]), None)
    if outlier_rec is None:
        failures.append("outlier not found in parity_report.json's per_image list")
        outlier_rec = OUTLIER
    if outlier_rec.get("file_sha256") != OUTLIER["expected_sha256"]:
        failures.append(f"outlier sha256 mismatch: expected {OUTLIER['expected_sha256']}, "
                        f"got {outlier_rec.get('file_sha256')}")

    others = [r for r in per_image if not (r["slug"] == OUTLIER["slug"] and r["photo_id"] == OUTLIER["photo_id"])]
    close_to_gate = min(others, key=lambda r: abs(r["max_similarity_reference"] - GATE_THRESHOLD))
    sims_sorted = sorted(others, key=lambda r: r["max_similarity_reference"])
    typical_low = sims_sorted[len(sims_sorted) // 10]
    typical_high = sims_sorted[-1 - len(sims_sorted) // 10]

    diagnostic_images = [
        {"role": "outlier", **{k: outlier_rec[k] for k in ("slug", "photo_id", "file_sha256")}},
        {"role": "control_close_to_0.60", **{k: close_to_gate[k] for k in ("slug", "photo_id", "file_sha256")}},
        {"role": "control_typical_low", **{k: typical_low[k] for k in ("slug", "photo_id", "file_sha256")}},
        {"role": "control_typical_high", **{k: typical_high[k] for k in ("slug", "photo_id", "file_sha256")}},
    ]
    log("=== diagnostic images ===")
    for d in diagnostic_images:
        log(f"  {d['role']}: {d['slug']}/{d['photo_id']}  sha256={d['file_sha256'][:16]}...")
    report["diagnostic_images"] = diagnostic_images

    protos_np = np.load(args.artifacts / "prototypes.npy")
    protos_np_norm = protos_np / np.clip(np.linalg.norm(protos_np, axis=1, keepdims=True), 1e-8, None)

    # ---- build the 7 ONNX/PyTorch execution paths -------------------------------
    sess_default = build_onnx_session(onnx_path, ["CPUExecutionProvider"])
    sess_noopt = build_onnx_session(onnx_path, ["CPUExecutionProvider"],
                                    graph_opt=ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    sess_1thread = build_onnx_session(onnx_path, ["CPUExecutionProvider"], intra_threads=1)
    sess_noopt_1thread = build_onnx_session(onnx_path, ["CPUExecutionProvider"],
                                            graph_opt=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                                            intra_threads=1)
    sess_api_style = build_onnx_session(onnx_path, ort.get_available_providers())

    path_names = ["pytorch_cuda", "pytorch_cpu", "onnx_cpu_default", "onnx_cpu_noopt",
                 "onnx_cpu_1thread", "onnx_cpu_noopt_1thread", "onnx_api_style"]
    report["runtime_matrix_paths"] = path_names
    report["onnx_api_style_providers"] = sess_api_style.get_providers()
    for sname, sess in (("default", sess_default), ("noopt", sess_noopt), ("1thread", sess_1thread),
                        ("noopt_1thread", sess_noopt_1thread)):
        provs = sess.get_providers()
        if provs != ["CPUExecutionProvider"]:
            failures.append(f"onnx_cpu_{sname} session providers != ['CPUExecutionProvider']: {provs}")

    model_cpu = load_pytorch_model(args.artifacts, cfg, torch.device("cpu"))

    def compute_all_paths(x_np: np.ndarray) -> dict[str, np.ndarray]:
        """x_np: (1,3,H,W) float32 preprocessed tensor (identical across all paths)."""
        return {
            "pytorch_cuda": pytorch_embed(model, x_np, device),
            "pytorch_cpu": pytorch_embed(model_cpu, x_np, torch.device("cpu")),
            "onnx_cpu_default": onnx_embed(sess_default, x_np),
            "onnx_cpu_noopt": onnx_embed(sess_noopt, x_np),
            "onnx_cpu_1thread": onnx_embed(sess_1thread, x_np),
            "onnx_cpu_noopt_1thread": onnx_embed(sess_noopt_1thread, x_np),
            "onnx_api_style": onnx_embed(sess_api_style, x_np),
        }

    ISOLATIONS = [
        ("pytorch_cpu", "onnx_cpu_default", "framework/export difference, device held constant (both CPU)"),
        ("pytorch_cuda", "pytorch_cpu", "device/kernel difference, framework held constant (both PyTorch)"),
        ("onnx_cpu_default", "onnx_cpu_noopt", "graph-optimization/fusion effect"),
        ("onnx_cpu_default", "onnx_cpu_1thread", "threading/reduction-order effect"),
        ("onnx_api_style", "onnx_cpu_default", "actual serving-provider effect (unforced vs forced CPU)"),
    ]

    # ============================================================== SECTION 1 + 2
    log("\n=== Section 1+2: per-image measurement + runtime/device matrix ===")
    section1_records = []
    section2_records = []

    for d in diagnostic_images:
        path = load_diagnostic_image(d["slug"], d["photo_id"], args.clean_dir)
        file_hash = sha256_file(path)
        if file_hash != d["file_sha256"]:
            failures.append(f"{d['slug']}/{d['photo_id']}: local file hash changed since "
                            f"parity_report.json ({file_hash} vs {d['file_sha256']})")

        img_raw = Image.open(path)
        img_raw.load()
        orientation = exif_orientation(img_raw)
        dims = img_raw.size
        mode_str = img_raw.mode
        pixel_arr = np.asarray(img_raw.convert("RGB"), dtype=np.float64)

        # reference (training) preprocessing
        img_ref = Image.open(path).convert("RGB")
        ref_tensor = ref_transform(img_ref).numpy()  # (3,H,W)

        # serving (API) preprocessing
        img_api = Image.open(path)
        img_api.load()
        onnx_tensor = identifier.preprocess(img_api)  # (1,3,H,W)

        preproc_diff = float(np.abs(ref_tensor - onnx_tensor[0]).max())
        finite_ok = bool(np.all(np.isfinite(ref_tensor)) and np.all(np.isfinite(onnx_tensor)))

        embeddings = compute_all_paths(onnx_tensor.astype(np.float32))
        if not all(np.all(np.isfinite(e)) for e in embeddings.values()):
            failures.append(f"{d['slug']}/{d['photo_id']}: non-finite embedding on some path")

        norms = {k: l2_norm(v) for k, v in embeddings.items()}
        rmss = {k: rms(v) for k, v in embeddings.items()}
        normed = {k: v / np.clip(np.linalg.norm(v), 1e-8, None) for k, v in embeddings.items()}
        normed_l2 = {k: l2_norm(v) for k, v in normed.items()}

        # divergent-channel evidence: PyTorch CUDA vs ONNX CPU default (the pairing
        # that originally surfaced the outlier in parity_check.py's Stage B)
        top10 = top_k_divergent_channels(embeddings["pytorch_cuda"], embeddings["onnx_cpu_default"])

        rec1 = {
            "role": d["role"], "slug": d["slug"], "photo_id": d["photo_id"],
            "file_sha256": file_hash,
            "dimensions": list(dims), "mode": mode_str, "exif_orientation": orientation,
            "decoded_pixel_stats": {
                "min": float(pixel_arr.min()), "max": float(pixel_arr.max()),
                "mean": float(pixel_arr.mean()), "std": float(pixel_arr.std()),
            },
            "normalized_input_stats_reference": {
                "min": float(ref_tensor.min()), "max": float(ref_tensor.max()),
                "mean": float(ref_tensor.mean()), "std": float(ref_tensor.std()),
            },
            "normalized_input_stats_serving": {
                "min": float(onnx_tensor.min()), "max": float(onnx_tensor.max()),
                "mean": float(onnx_tensor.mean()), "std": float(onnx_tensor.std()),
            },
            "preprocessing_max_abs_diff_training_vs_api": preproc_diff,
            "finite_check_passed": finite_ok,
            "raw_embedding_l2_norm_by_path": norms,
            "raw_embedding_rms_by_path": rmss,
            "normalized_embedding_l2_norm_by_path": normed_l2,
            "top10_divergent_channels_pytorch_cuda_vs_onnx_cpu_default": top10,
        }
        section1_records.append(rec1)
        log(f"  [{d['role']}] {d['slug']}/{d['photo_id']}: preproc_diff={preproc_diff}  "
           f"norms(pytorch_cuda={norms['pytorch_cuda']:.4f}, onnx_cpu_default={norms['onnx_cpu_default']:.4f})  "
           f"top divergent channel idx={top10[0]['index']} diff={top10[0]['abs_diff']:.4f}")

        # ---- Section 2: pairwise isolation metrics for this image --------------
        pair_results = {}
        for a, b, label in ISOLATIONS:
            ea, eb = embeddings[a], embeddings[b]
            na = ea / np.clip(np.linalg.norm(ea), 1e-8, None)
            nb = eb / np.clip(np.linalg.norm(eb), 1e-8, None)
            sims_a = protos_np_norm @ na
            sims_b = protos_np_norm @ nb
            max_a, max_b = widen(sims_a.max()), widen(sims_b.max())
            pair_results[f"{a}__vs__{b}"] = {
                "isolation": label,
                "raw_embedding_max_abs_diff": float(np.abs(ea - eb).max()),
                "normalized_embedding_max_abs_diff": float(np.abs(na - nb).max()),
                "score_vector_max_abs_diff": float(np.abs(sims_a - sims_b).max()),
                "max_similarity_a": max_a, "max_similarity_b": max_b,
                "max_similarity_abs_diff": abs(max_a - max_b),
                "top1_a": int(np.argmax(sims_a)), "top1_b": int(np.argmax(sims_b)),
                "top1_agree": int(np.argmax(sims_a)) == int(np.argmax(sims_b)),
                "gate_reject_a": max_a < GATE_THRESHOLD, "gate_reject_b": max_b < GATE_THRESHOLD,
            }
            pair_results[f"{a}__vs__{b}"]["gate_agree"] = (
                pair_results[f"{a}__vs__{b}"]["gate_reject_a"] == pair_results[f"{a}__vs__{b}"]["gate_reject_b"])

        # also record raw max-sim + top1 per individual path for completeness
        per_path_summary = {}
        for pname, emb in embeddings.items():
            n = emb / np.clip(np.linalg.norm(emb), 1e-8, None)
            s = protos_np_norm @ n
            m = widen(s.max())
            per_path_summary[pname] = {"max_similarity": m, "top1": int(np.argmax(s)),
                                       "gate_reject": m < GATE_THRESHOLD}

        section2_records.append({
            "role": d["role"], "slug": d["slug"], "photo_id": d["photo_id"],
            "per_path_summary": per_path_summary,
            "isolations": pair_results,
        })

    report["section1_input_embedding_measurements"] = section1_records
    report["section2_runtime_device_matrix"] = section2_records

    # channel-consistency evidence across images (outlier vs controls)
    outlier_top_idx = section1_records[0]["top10_divergent_channels_pytorch_cuda_vs_onnx_cpu_default"][0]["index"]
    control_top_idxs = [r["top10_divergent_channels_pytorch_cuda_vs_onnx_cpu_default"][0]["index"]
                        for r in section1_records[1:]]
    channel_overlap = [idx for idx in control_top_idxs if idx == outlier_top_idx]
    report["channel_consistency_evidence"] = {
        "outlier_top_divergent_channel": outlier_top_idx,
        "control_top_divergent_channels": control_top_idxs,
        "outlier_channel_also_top_for_n_controls": len(channel_overlap),
        "note": "Reported as evidence only. A single index matching (or not matching) across "
               "4 images does not by itself establish an image-driven or channel-driven cause; "
               "see Section 2 isolations and Section 4/5 for the determinism and CUDA-control "
               "evidence needed to narrow further.",
    }
    log(f"\n  outlier's top divergent channel: {outlier_top_idx}; "
       f"appears as controls' top channel in {len(channel_overlap)}/3 controls")

    # ============================================================== SECTION 3
    log("\n=== Section 3: ORT provider evidence (profiling, not get_providers()) ===")
    prof_dir = args.artifacts
    so_prof = ort.SessionOptions()
    so_prof.enable_profiling = True
    so_prof.profile_file_prefix = str(prof_dir / "parity_diag_profile")
    sess_prof = ort.InferenceSession(str(onnx_path), sess_options=so_prof,
                                     providers=ort.get_available_providers())
    registered_providers = sess_prof.get_providers()
    name = sess_prof.get_inputs()[0].name
    for d in diagnostic_images:
        path = load_diagnostic_image(d["slug"], d["photo_id"], args.clean_dir)
        img = Image.open(path)
        img.load()
        x = identifier.preprocess(img)
        sess_prof.run(None, {name: x})
    profile_path = sess_prof.end_profiling()
    profile_events = json.loads(Path(profile_path).read_text())

    provider_counts: dict[str, int] = {}
    no_provider_field = 0
    non_cpu_nodes = []
    for e in profile_events:
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

    section3 = {
        "registered_providers": registered_providers,
        "node_event_count_by_provider": provider_counts,
        "nodes_without_provider_field": no_provider_field,
        "any_node_executed_outside_cpu": len(non_cpu_nodes) > 0,
        "non_cpu_nodes": non_cpu_nodes,
        "profile_file_path": str(profile_path),
        "schema_note": "ORT's profiling trace exposes args.provider per Node-category event; "
                      "this was usable and is not an inference from session.get_providers().",
    }
    report["section3_ort_provider_evidence"] = section3
    for k, v in section3.items():
        log(f"  {k}: {v}")

    # ============================================================== SECTION 4
    log(f"\n=== Section 4: determinism (within-process x{args.within_process_repeats}, "
       f"fresh-process x{args.fresh_process_repeats}) ===")

    outlier_path = load_diagnostic_image(OUTLIER["slug"], OUTLIER["photo_id"], args.clean_dir)
    img = Image.open(outlier_path)
    img.load()
    outlier_x = identifier.preprocess(img).astype(np.float32)

    def within_process_repeat(fn, n):
        embs = [fn() for _ in range(n)]
        stacked = np.stack(embs, axis=0)
        max_within = float(np.abs(stacked - stacked[0]).max())
        return max_within

    within_pytorch_cuda = within_process_repeat(
        lambda: pytorch_embed(model, outlier_x, device), args.within_process_repeats)
    within_onnx_default = within_process_repeat(
        lambda: onnx_embed(sess_default, outlier_x), args.within_process_repeats)
    within_onnx_noopt = within_process_repeat(
        lambda: onnx_embed(sess_noopt, outlier_x), args.within_process_repeats)
    within_onnx_1thread = within_process_repeat(
        lambda: onnx_embed(sess_1thread, outlier_x), args.within_process_repeats)

    section4_within = {
        "repeats": args.within_process_repeats,
        "max_within_process_divergence": {
            "pytorch_cuda": within_pytorch_cuda,
            "onnx_cpu_default": within_onnx_default,
            "onnx_cpu_noopt": within_onnx_noopt,
            "onnx_cpu_1thread": within_onnx_1thread,
        },
    }
    log(f"  within-process max divergence: {section4_within['max_within_process_divergence']}")

    # fresh-process determinism
    scratch_dir = args.artifacts / "_parity_diag_scratch"
    scratch_dir.mkdir(exist_ok=True)
    tensor_path = scratch_dir / "outlier_tensor.npy"
    np.save(tensor_path, outlier_x)

    fresh_results: dict[str, list[dict]] = {"pytorch_cuda": [], "onnx_cpu": []}
    for mode in ("pytorch_cuda", "onnx_cpu"):
        for i in range(args.fresh_process_repeats):
            out_json = scratch_dir / f"fresh_{mode}_{i}.json"
            cmd = [sys.executable, str(HERE / "parity_diagnostic.py"),
                  "--subprocess-embed", str(tensor_path), mode, str(out_json)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                failures.append(f"fresh-process run failed ({mode} #{i}): {r.stderr[-500:]}")
                continue
            payload = json.loads(out_json.read_text())
            emb = np.array(payload["embedding"], dtype=np.float64)
            n = emb / np.clip(np.linalg.norm(emb), 1e-8, None)
            s = protos_np_norm @ n
            m = widen(s.max())
            fresh_results[mode].append({
                "embedding_sha256": payload["embedding_sha256"],
                "l2_norm": payload["l2_norm"],
                "max_similarity": m, "top1": int(np.argmax(s)),
                "gate_reject": m < GATE_THRESHOLD,
            })

    def fresh_summary(results: list[dict]) -> dict:
        if not results:
            return {"error": "no successful runs"}
        hashes_ = {r["embedding_sha256"] for r in results}
        max_sims = [r["max_similarity"] for r in results]
        top1s = {r["top1"] for r in results}
        gates = {r["gate_reject"] for r in results}
        return {
            "n_runs": len(results),
            "identical_embedding_bytes_across_runs": len(hashes_) == 1,
            "n_distinct_embedding_hashes": len(hashes_),
            "max_similarity_spread": max(max_sims) - min(max_sims),
            "identical_top1_across_runs": len(top1s) == 1,
            "distinct_top1_values": sorted(top1s),
            "identical_gate_decision_across_runs": len(gates) == 1,
        }

    section4_fresh = {mode: fresh_summary(fresh_results[mode]) for mode in fresh_results}
    report["section4_determinism"] = {**section4_within, "fresh_process": section4_fresh}
    log(f"  fresh-process summary: {json.dumps(section4_fresh, indent=2)}")

    # ============================================================== SECTION 5
    log("\n=== Section 5: CUDA numerical controls ===")
    before = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled_cuda": torch.is_autocast_enabled("cuda"),
    }
    log(f"  before: {before}")

    default_cuda_embs = {d["role"]: section1_records[i]["raw_embedding_l2_norm_by_path"]["pytorch_cuda"]
                        for i, d in enumerate(diagnostic_images)}

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        controlled = {}
        controlled_embs: dict[str, np.ndarray] = {}
        for d in diagnostic_images:
            path = load_diagnostic_image(d["slug"], d["photo_id"], args.clean_dir)
            img = Image.open(path)
            img.load()
            x = identifier.preprocess(img).astype(np.float32)
            emb_ctrl = pytorch_embed(model, x, device)
            emb_cpu = pytorch_embed(model_cpu, x, torch.device("cpu"))
            controlled_embs[d["role"]] = emb_ctrl
            controlled[d["role"]] = {
                "l2_norm_controlled_cuda": l2_norm(emb_ctrl),
                "l2_norm_pytorch_cpu": l2_norm(emb_cpu),
                "raw_abs_diff_controlled_cuda_vs_pytorch_cpu": float(np.abs(emb_ctrl - emb_cpu).max()),
            }
    finally:
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
        torch.set_float32_matmul_precision(before["float32_matmul_precision"])

    after = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled_cuda": torch.is_autocast_enabled("cuda"),
    }
    restored_ok = after == before
    log(f"  after (restored): {after}  matches before: {restored_ok}")
    if not restored_ok:
        failures.append(f"CUDA global settings not fully restored: before={before} after={after}")

    # now recompute default-settings CUDA embeddings (settings are restored) to
    # compare against the controlled-settings embeddings captured above
    for i, d in enumerate(diagnostic_images):
        path = load_diagnostic_image(d["slug"], d["photo_id"], args.clean_dir)
        img = Image.open(path)
        img.load()
        x = identifier.preprocess(img).astype(np.float32)
        emb_default = pytorch_embed(model, x, device)
        controlled[d["role"]]["l2_norm_default_cuda_recomputed"] = l2_norm(emb_default)
        controlled[d["role"]]["raw_abs_diff_controlled_cuda_vs_default_cuda_recomputed"] = \
            float(np.abs(controlled_embs[d["role"]] - emb_default).max())

    section5 = {
        "settings_before": before,
        "settings_after_restore": after,
        "settings_restored_correctly": restored_ok,
        "controlled_settings_applied": {
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "cudnn_allow_tf32": False, "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest", "autocast": "disabled (was already disabled)",
        },
        "per_image": controlled,
        "note": "Default-CUDA embeddings for this comparison were recomputed AFTER restoring "
               "global settings (identical config to the original Stage A/B run), since PyTorch "
               "global backend flags cannot be un-applied retroactively to an embedding already "
               "computed under different settings. Section 4's within-process repeat already "
               "characterizes default-settings run-to-run stability independently.",
    }
    report["section5_cuda_numerical_controls"] = section5
    log(f"  controlled-vs-pytorch_cpu per image: "
       f"{ {k: v['raw_abs_diff_controlled_cuda_vs_pytorch_cpu'] for k, v in controlled.items()} }")

    # ============================================================== WRAP UP
    forced_sessions = {"onnx_cpu_default": sess_default, "onnx_cpu_noopt": sess_noopt,
                      "onnx_cpu_1thread": sess_1thread, "onnx_cpu_noopt_1thread": sess_noopt_1thread}
    report["structural_verification"] = {
        "n_diagnostic_images": len(diagnostic_images),
        "cpu_execution_provider_confirmed_all_forced_sessions": all(
            s.get_providers() == ["CPUExecutionProvider"] for s in forced_sessions.values()),
        "cuda_available": True,
        "failures": failures,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log(f"\nwrote {args.out}")

    if failures:
        log(f"\nFAILED checks ({len(failures)}):")
        for f in failures:
            log(f"  ! {f}")
        return 1
    log("\nDiagnostic run complete. No cause is declared here beyond what the "
       "matrix/determinism/profiling evidence directly shows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
