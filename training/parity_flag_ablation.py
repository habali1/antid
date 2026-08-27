#!/usr/bin/env python3
"""parity_flag_ablation.py — Step 4a continuation: four-arm CUDA flag
ablation (Part A) and an exact 200-image execution-path supplement (Part B).

Diagnostic only. Does not implement, ship, or adjust any gate. Does not
retrain or re-export. Does not touch parity_report.json or
parity_diagnostic.json (both are read-only inputs here; their hashes are
recorded before and after this run to prove that).

Correction carried forward: max|raw_a-raw_b| / max|normalized_a-normalized_b|
is never used as a norm estimate anywhere in this script. All norms and RMS
values are measured directly.

Scope note: the 200-image Stage B sample is a deterministic, STRATIFIED
DIAGNOSTIC sample (4 images/species x 50 species) -- not a random production
or calibration sample. Counts in this script (e.g. "N/200 in the boundary
band") are reported as exactly that: counts within this fixed diagnostic
sample. They are not prevalence estimates, are not projected onto
calibration_v1's 1,005 images or any other population, and are not used to
retune or reselect the 0.60 threshold, which is loaded read-only from
calibration_v1.json's machine_readable_rule (this is an evaluation-side tool;
the future API must read inference_policy.json only and must never read
calibration_v1.json -- that boundary is unaffected by this script).

Part A: cudnn.allow_tf32 x cudnn.deterministic, 2x2 design (arms A-D), each
run in >=5 fresh OS processes, on the outlier + 3 controls from
parity_diagnostic.json.

Part B: 5 execution paths across the exact frozen 200-image Stage B sample --
a CUDA calibration-code mirror (batch=32, Torch-side normalize+matmul), a
CUDA batch=1 Torch-side path, the original batch=1 NumPy-side path (the old
Stage B reference), a CPU batch=32 Torch-side path, and the production ONNX
CPU batch=1 path -- to isolate batch-size, scoring-backend, device, and
export effects, with calibration_mirror_cuda_batch32 vs
production_onnx_cpu_batch1 as the primary comparison.

Usage:
    python parity_flag_ablation.py
    python parity_flag_ablation.py --subprocess-arm A tensors_dir out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent / "api"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(API_DIR))

EXPECTED_SEED = 42
EXPECTED_N_PER_SPECIES = 4
EXPECTED_N_VALID = 200
EXPECTED_SAMPLE_HASH = "c6a3f2b0b1ed2bc8887eadf5cac5475c2195ac8abc4887e1315adb07075e6887"
OUTLIER = {"slug": "pseudomyrmex-gracilis", "photo_id": "671332727"}
BATCH_SIZE = 32
BOUNDARY_LO, BOUNDARY_HI = 0.55, 0.65

ARM_FLAGS = {
    "A": {"allow_tf32": True, "deterministic": False},
    "B": {"allow_tf32": False, "deterministic": False},
    "C": {"allow_tf32": True, "deterministic": True},
    "D": {"allow_tf32": False, "deterministic": True},
}


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


def percentiles(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(arr.min()), "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)), "max": float(arr.max())}


def spearman(x, y) -> float:
    """Rank-based Pearson, ties broken by average rank. scipy is not
    installed in this environment; implemented directly rather than adding a
    dependency. Exploratory use only -- callers must label it as such."""
    def rank(a):
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a))
        ranks[order] = np.arange(1, len(a) + 1)
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
            i = j + 1
        return ranks
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ============================================================= model/session loading
def load_pytorch_model(artifacts: Path, config: dict, device):
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


def load_gate_threshold(calibration_json_path: Path) -> float:
    d = json.loads(calibration_json_path.read_text())
    rule = d["frozen_candidate_abstention_threshold"]["machine_readable_rule"]
    if rule["operator"] != "max_sim < value":
        raise SystemExit(f"unsupported operator {rule['operator']!r}")
    v = float(rule["value"])
    if not (0.0 < v < 1.0):
        raise SystemExit(f"threshold value {v} outside (0,1)")
    return v


# ============================================================ Part A subprocess
def part_a_subprocess_main(arm: str, tensor_dir: Path, out_path: Path) -> int:
    """Runs in a FRESH interpreter process. Flags are set before any CUDA
    operation, including model construction/`.to(device)`."""
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    flags = ARM_FLAGS[arm]
    torch.backends.cudnn.allow_tf32 = flags["allow_tf32"]
    torch.backends.cudnn.deterministic = flags["deterministic"]
    # autocast is simply never entered -- nothing to globally disable beyond that.

    import yaml
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    artifacts = HERE / "artifacts"
    device = torch.device("cuda")
    model_cuda = load_pytorch_model(artifacts, cfg, device)   # first CUDA op is inside here
    model_cpu = load_pytorch_model(artifacts, cfg, torch.device("cpu"))

    actual_flags = {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled_cuda": torch.is_autocast_enabled("cuda"),
    }

    manifest = json.loads((tensor_dir / "manifest.json").read_text())
    results = {}
    with torch.no_grad():
        for role, fname in manifest.items():
            x = np.load(tensor_dir / fname)
            t = torch.as_tensor(x, dtype=torch.float32)
            emb_cuda = model_cuda.embed(t.to(device)).cpu().numpy()[0]
            emb_cpu = model_cpu.embed(t).cpu().numpy()[0]
            results[role] = {
                "raw_embedding_cuda": emb_cuda.astype(np.float64).tolist(),
                "raw_embedding_cpu": emb_cpu.astype(np.float64).tolist(),
            }

    payload = {
        "arm": arm, "actual_flags": actual_flags,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "results": results,
    }
    out_path.write_text(json.dumps(payload))
    return 0


# ============================================================ image utilities
def resolve_local_path(slug: str, photo_id: str, clean_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = clean_dir / slug / f"{photo_id}{ext}"
        if p.exists():
            return p
    return None


def exif_orientation(img) -> int | None:
    try:
        return img.getexif().get(0x0112)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subprocess-arm", nargs=3, metavar=("ARM", "TENSOR_DIR", "OUT_JSON"))
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--clean-dir", type=Path, default=HERE.parent / "data" / "clean")
    ap.add_argument("--parity-report", type=Path, default=HERE / "artifacts" / "parity_report.json")
    ap.add_argument("--parity-diagnostic", type=Path, default=HERE / "artifacts" / "parity_diagnostic.json")
    ap.add_argument("--calibration-json", type=Path,
                    default=HERE.parent / "data" / "calibration_v1" / "calibration_v1.json")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "parity_flag_ablation.json")
    ap.add_argument("--arm-repeats", type=int, default=5)
    args = ap.parse_args()

    if args.subprocess_arm:
        arm, tensor_dir, out_json = args.subprocess_arm
        return part_a_subprocess_main(arm, Path(tensor_dir), Path(out_json))

    # ================================================================ SETUP
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    import yaml
    from PIL import Image

    from data import build_transforms  # noqa: F401  (parity confirmed already; not reused directly)
    from inference import AntIdentifier

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "script": "training/parity_flag_ablation.py",
        "correction_from_planning_discussion": (
            "max|raw_a-raw_b| / max|normalized_a-normalized_b| is not used anywhere in this "
            "script as a norm estimate. L2 norm and RMS are measured directly, per image, "
            "per execution path."
        ),
        "scope_note": (
            "The 200-image Stage B sample is a deterministic, stratified DIAGNOSTIC sample "
            "(4 images/species x 50 species), not a random production or calibration sample. "
            "Counts below are exact counts within this fixed sample only -- not prevalence "
            "estimates, not projected onto calibration_v1 or any other population, and not "
            "used to retune or reselect the 0.60 threshold."
        ),
    }
    failures: list[str] = []
    warnings: list[str] = []

    if not torch.cuda.is_available():
        msg = "CUDA unavailable; cannot run this diagnostic's CUDA arms."
        report["fatal_error"] = msg
        args.out.write_text(json.dumps(report, indent=2))
        log(f"FATAL: {msg}")
        return 1
    device = torch.device("cuda")

    # ---- pre-run hashes of the two read-only prior reports ---------------------
    parity_report_hash_before = sha256_file(args.parity_report)
    parity_diagnostic_hash_before = sha256_file(args.parity_diagnostic)

    # ---- artifact/config hashes --------------------------------------------------
    art_hashes = {name: sha256_file(args.artifacts / name)
                 for name in ("model.pth", "backbone.onnx", "prototypes.npy", "taxonomy.json")}
    art_hashes["config_yaml"] = sha256_file(args.config)
    art_hashes["calibration_v1_json"] = sha256_file(args.calibration_json)
    report["artifact_hashes"] = art_hashes
    report["input_report_hashes_before"] = {
        "parity_report_json": parity_report_hash_before,
        "parity_diagnostic_json": parity_diagnostic_hash_before,
    }

    gate_threshold = load_gate_threshold(args.calibration_json)
    report["gate_threshold_loaded_from_calibration_v1_json"] = gate_threshold
    log(f"gate threshold (loaded, not hardcoded): {gate_threshold}")

    # ================================================================ INTEGRITY GATE
    log("\n=== input integrity gate ===")
    parity = json.loads(args.parity_report.read_text())
    sb = parity["stage_b_real_end_to_end_parity"]
    integrity = {
        "seed": sb["seed"], "seed_ok": sb["seed"] == EXPECTED_SEED,
        "n_per_species": sb["n_per_species"], "n_per_species_ok": sb["n_per_species"] == EXPECTED_N_PER_SPECIES,
        "n_samples_valid": sb["n_samples_valid"], "n_samples_valid_ok": sb["n_samples_valid"] == EXPECTED_N_VALID,
        "sample_list_hash_sha256": sb["sample_list_hash_sha256"],
        "sample_list_hash_ok": sb["sample_list_hash_sha256"] == EXPECTED_SAMPLE_HASH,
    }
    rows = sb["per_image"]
    keys = [(r["slug"], r["photo_id"]) for r in rows]
    file_hashes = [r["file_sha256"] for r in rows]
    integrity["n_rows"] = len(rows)
    integrity["duplicate_row_keys"] = len(keys) - len(set(keys))
    integrity["duplicate_file_hashes"] = len(file_hashes) - len(set(file_hashes))

    verify_problems = []
    resolved_paths = {}
    for r in rows:
        p = resolve_local_path(r["slug"], r["photo_id"], args.clean_dir)
        if p is None:
            verify_problems.append(f"{r['slug']}/{r['photo_id']}: file missing")
            continue
        digest = sha256_file(p)
        if digest != r["file_sha256"]:
            verify_problems.append(f"{r['slug']}/{r['photo_id']}: hash mismatch")
            continue
        resolved_paths[(r["slug"], r["photo_id"])] = p
    integrity["file_verification_problems"] = verify_problems
    report["integrity_gate"] = integrity
    for k, v in integrity.items():
        log(f"  {k}: {v}")

    gate_ok = (integrity["seed_ok"] and integrity["n_per_species_ok"] and integrity["n_samples_valid_ok"]
              and integrity["sample_list_hash_ok"] and integrity["duplicate_row_keys"] == 0
              and integrity["duplicate_file_hashes"] == 0 and len(verify_problems) == 0)
    if not gate_ok:
        msg = "Input integrity gate FAILED -- aborting before any inference."
        log(f"FATAL: {msg}")
        report["fatal_error"] = msg
        args.out.write_text(json.dumps(report, indent=2))
        return 1
    log("  integrity gate PASSED -- proceeding to inference.")

    # ================================================================ COMMON SETUP
    cfg = yaml.safe_load(args.config.read_text())
    identifier = AntIdentifier(artifacts_dir=args.artifacts)
    model_cuda_default = load_pytorch_model(args.artifacts, cfg, device)
    model_cpu_default = load_pytorch_model(args.artifacts, cfg, torch.device("cpu"))
    sess_cpu = ort.InferenceSession(str(args.artifacts / "backbone.onnx"),
                                    providers=["CPUExecutionProvider"])
    if sess_cpu.get_providers() != ["CPUExecutionProvider"]:
        failures.append(f"ONNX CPU session providers != forced CPU: {sess_cpu.get_providers()}")
    onnx_input_name = sess_cpu.get_inputs()[0].name

    protos_np = np.load(args.artifacts / "prototypes.npy")
    protos_np_norm = identifier.prototypes  # production normalization, numpy
    protos_torch_cuda = nn.functional.normalize(
        torch.as_tensor(protos_np, dtype=torch.float32, device=device), dim=1)
    protos_torch_cpu = nn.functional.normalize(
        torch.as_tensor(protos_np, dtype=torch.float32, device="cpu"), dim=1)

    # ============================================================================
    # PART A: four-arm CUDA flag ablation
    # ============================================================================
    log("\n=== PART A: four-arm CUDA flag ablation ===")

    diag = json.loads(args.parity_diagnostic.read_text())
    diag_images = diag["diagnostic_images"]  # outlier + 3 controls, already verified in Step 4a

    scratch = args.artifacts / "_flag_ablation_scratch"
    scratch.mkdir(exist_ok=True)
    manifest = {}
    for d in diag_images:
        p = resolve_local_path(d["slug"], d["photo_id"], args.clean_dir)
        if p is None or sha256_file(p) != d["file_sha256"]:
            failures.append(f"Part A image {d['slug']}/{d['photo_id']} failed re-verification")
            continue
        img = Image.open(p)
        img.load()
        x = identifier.preprocess(img).astype(np.float32)
        fname = f"{d['role']}.npy"
        np.save(scratch / fname, x)
        manifest[d["role"]] = fname
    (scratch / "manifest.json").write_text(json.dumps(manifest))

    part_a_raw: dict[str, list[dict]] = {arm: [] for arm in "ABCD"}
    for arm in "ABCD":
        for i in range(args.arm_repeats):
            out_json = scratch / f"arm_{arm}_{i}.json"
            cmd = [sys.executable, str(HERE / "parity_flag_ablation.py"),
                  "--subprocess-arm", arm, str(scratch), str(out_json)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                failures.append(f"Part A arm {arm} repeat {i} failed: {r.stderr[-500:]}")
                continue
            part_a_raw[arm].append(json.loads(out_json.read_text()))
        log(f"  arm {arm}: {len(part_a_raw[arm])}/{args.arm_repeats} subprocess runs succeeded")

    # ---- within-arm determinism + representative embedding per arm/image -------
    part_a_determinism = {}
    part_a_repr: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in "ABCD"}
    part_a_cpu_repr: dict[str, np.ndarray] = {}  # CPU embedding is flag-independent; sanity-check equal across arms
    for arm in "ABCD":
        runs = part_a_raw[arm]
        if not runs:
            continue
        for role in manifest:
            cuda_embs = [np.array(r["results"][role]["raw_embedding_cuda"]) for r in runs]
            cpu_embs = [np.array(r["results"][role]["raw_embedding_cpu"]) for r in runs]
            stacked_cuda = np.stack(cuda_embs, axis=0)
            stacked_cpu = np.stack(cpu_embs, axis=0)
            max_within_cuda = float(np.abs(stacked_cuda - stacked_cuda[0]).max())
            max_within_cpu = float(np.abs(stacked_cpu - stacked_cpu[0]).max())
            part_a_determinism[f"{arm}_{role}"] = {
                "n_runs": len(runs), "max_within_arm_cuda_divergence": max_within_cuda,
                "max_within_arm_cpu_divergence": max_within_cpu,
            }
            part_a_repr[arm][role] = stacked_cuda[0]
            part_a_cpu_repr[role] = stacked_cpu[0]  # last arm wins; checked for cross-arm agreement below

    # cross-arm CPU agreement (CPU embedding should not depend on the CUDA flags at all)
    cpu_cross_arm = {}
    for role in manifest:
        all_cpu = [np.array(part_a_raw[arm][0]["results"][role]["raw_embedding_cpu"])
                  for arm in "ABCD" if part_a_raw[arm]]
        if len(all_cpu) > 1:
            cpu_cross_arm[role] = float(np.abs(np.stack(all_cpu) - all_cpu[0]).max())

    report["part_a_within_arm_determinism"] = part_a_determinism
    report["part_a_cpu_embedding_cross_arm_agreement"] = cpu_cross_arm
    log(f"  within-arm determinism (max divergence, should be 0.0 if stable): "
       f"{ {k: v['max_within_arm_cuda_divergence'] for k, v in part_a_determinism.items()} }")
    log(f"  CPU embedding cross-arm agreement (should be ~0.0, CPU is flag-independent): {cpu_cross_arm}")

    # ---- reproduction check: does Arm A reproduce the original default-CUDA result? ----
    outlier_role = next(d["role"] for d in diag_images if d["slug"] == OUTLIER["slug"]
                        and d["photo_id"] == OUTLIER["photo_id"])
    arm_a_outlier_norm = l2_norm(part_a_repr["A"][outlier_role]) if part_a_repr["A"] else None
    arm_a_outlier_top1 = int(np.argmax(protos_np_norm @ (
        part_a_repr["A"][outlier_role] / np.clip(np.linalg.norm(part_a_repr["A"][outlier_role]), 1e-8, None)
    ))) if part_a_repr["A"] else None
    EXPECTED_ARM_A_NORM = 70.98
    EXPECTED_ARM_A_TOP1 = 4
    reproduction_ok = (arm_a_outlier_norm is not None
                      and abs(arm_a_outlier_norm - EXPECTED_ARM_A_NORM) < 0.5
                      and arm_a_outlier_top1 == EXPECTED_ARM_A_TOP1)
    report["part_a_reproduction_check"] = {
        "expected_norm_approx": EXPECTED_ARM_A_NORM, "expected_top1": EXPECTED_ARM_A_TOP1,
        "arm_a_outlier_norm": arm_a_outlier_norm, "arm_a_outlier_top1": arm_a_outlier_top1,
        "reproduction_confirmed": reproduction_ok,
    }
    log(f"  Arm A reproduction check: norm={arm_a_outlier_norm} (expect ~{EXPECTED_ARM_A_NORM}), "
       f"top1={arm_a_outlier_top1} (expect {EXPECTED_ARM_A_TOP1}) -> confirmed={reproduction_ok}")
    if not reproduction_ok:
        warnings.append("Arm A did not reproduce the original default-CUDA behavior for the "
                        "outlier -- causal interpretation of the A-D matrix below is suspended; "
                        "raw data is still reported.")

    # ---- per-arm, per-image metrics (vs PyTorch CPU) ----------------------------
    part_a_metrics = {}
    for arm in "ABCD":
        if not part_a_repr[arm]:
            continue
        part_a_metrics[arm] = {}
        actual_flags = part_a_raw[arm][0]["actual_flags"]
        versions = {k: part_a_raw[arm][0][k] for k in
                   ("torch_version", "torch_cuda_version", "cudnn_version", "gpu_name")}
        for role in manifest:
            emb_cuda = part_a_repr[arm][role]
            emb_cpu = part_a_cpu_repr[role]
            n_cuda = emb_cuda / np.clip(np.linalg.norm(emb_cuda), 1e-8, None)
            n_cpu = emb_cpu / np.clip(np.linalg.norm(emb_cpu), 1e-8, None)
            s_cuda = protos_np_norm @ n_cuda
            s_cpu = protos_np_norm @ n_cpu
            m_cuda, m_cpu = widen(s_cuda.max()), widen(s_cpu.max())
            part_a_metrics[arm][role] = {
                "actual_flags": actual_flags, "versions": versions,
                "raw_l2_norm": l2_norm(emb_cuda), "raw_rms": rms(emb_cuda),
                "raw_abs_diff_vs_pytorch_cpu": float(np.abs(emb_cuda - emb_cpu).max()),
                "normalized_abs_diff_vs_pytorch_cpu": float(np.abs(n_cuda - n_cpu).max()),
                "score_vector_max_abs_diff_vs_pytorch_cpu": float(np.abs(s_cuda - s_cpu).max()),
                "max_similarity": m_cuda, "top1": int(np.argmax(s_cuda)),
                "gate_reject": m_cuda < gate_threshold,
            }
    report["part_a_per_arm_metrics"] = part_a_metrics

    # ---- factorial comparisons --------------------------------------------------
    factorial = {}
    if reproduction_ok:
        pairs = [("A", "B", "TF32 effect while deterministic=False"),
                ("C", "D", "TF32 effect while deterministic=True"),
                ("A", "C", "deterministic-algorithm effect while TF32=True"),
                ("B", "D", "deterministic-algorithm effect while TF32=False")]
        for a1, a2, label in pairs:
            if not (part_a_repr[a1] and part_a_repr[a2]):
                continue
            per_image = {}
            for role in manifest:
                e1, e2 = part_a_repr[a1][role], part_a_repr[a2][role]
                per_image[role] = {
                    "raw_abs_diff": float(np.abs(e1 - e2).max()),
                    "raw_l2_norm_a": l2_norm(e1), "raw_l2_norm_b": l2_norm(e2),
                }
            factorial[f"{a1}_vs_{a2}"] = {"label": label, "per_image": per_image}
        report["part_a_factorial_comparisons"] = factorial
        log(f"  factorial comparisons computed: {list(factorial.keys())}")
    else:
        report["part_a_factorial_comparisons"] = {
            "skipped": "reproduction check failed; see part_a_reproduction_check"}

    # ============================================================================
    # PART B: exact 200-image execution-path supplement
    # ============================================================================
    log("\n=== PART B: exact 200-image execution-path supplement ===")

    # preprocess all 200 in fixed order once (training vs API preprocessing already
    # proven bit-identical for these exact images in parity_report.json's Stage B;
    # reused here as the single canonical input rather than recomputed twice)
    ordered_tensors = []
    ordered_meta = []
    for r in rows:
        p = resolved_paths[(r["slug"], r["photo_id"])]
        img = Image.open(p)
        img.load()
        x = identifier.preprocess(img).astype(np.float32)  # (1,3,H,W)
        ordered_tensors.append(x[0])
        ordered_meta.append({"slug": r["slug"], "photo_id": r["photo_id"], "file_sha256": r["file_sha256"]})
    stacked = np.stack(ordered_tensors, axis=0)  # (200,3,H,W)
    n = stacked.shape[0]
    assert n == EXPECTED_N_VALID

    def batched_torch_embed(model, device_, batch_size):
        embs = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                chunk = torch.as_tensor(stacked[start:start + batch_size], dtype=torch.float32, device=device_)
                embs.append(model.embed(chunk).cpu().numpy())
        return np.concatenate(embs, axis=0)

    def batch1_torch_embed(model, device_):
        embs = []
        with torch.no_grad():
            for i in range(n):
                t = torch.as_tensor(stacked[i:i + 1], dtype=torch.float32, device=device_)
                embs.append(model.embed(t).cpu().numpy()[0])
        return np.stack(embs, axis=0)

    def batch1_onnx_embed():
        embs = []
        for i in range(n):
            embs.append(sess_cpu.run(None, {onnx_input_name: stacked[i:i + 1]})[0][0])
        return np.stack(embs, axis=0)

    log("  computing calibration_mirror_cuda_batch32 ...")
    raw_calib_mirror = batched_torch_embed(model_cuda_default, device, BATCH_SIZE)
    calib_mirror_flags = {
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

    log("  computing pytorch_cuda_batch1 (shared raw embeddings for torch- and numpy-scoring paths) ...")
    raw_cuda_batch1 = batch1_torch_embed(model_cuda_default, device)

    log("  computing pytorch_cpu_batch32_torch ...")
    raw_cpu_batch32 = batched_torch_embed(model_cpu_default, torch.device("cpu"), BATCH_SIZE)

    log("  computing production_onnx_cpu_batch1 ...")
    raw_onnx_batch1 = batch1_onnx_embed()

    if not all(np.all(np.isfinite(a)) for a in
              (raw_calib_mirror, raw_cuda_batch1, raw_cpu_batch32, raw_onnx_batch1)):
        failures.append("non-finite embedding values in one or more Part B paths")

    def score_numpy(raw: np.ndarray):
        norm = raw / np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8, None)
        sims = norm @ protos_np_norm.T
        max_sim = np.array([widen(v) for v in sims.max(axis=1)])
        top1 = np.argmax(sims, axis=1)
        return norm, sims, max_sim, top1

    def score_torch(raw: np.ndarray, protos_t, device_):
        t = torch.as_tensor(raw, dtype=torch.float32, device=device_)
        norm_t = nn.functional.normalize(t, dim=1)
        sims_t = norm_t @ protos_t.T
        norm = norm_t.cpu().numpy()
        sims = sims_t.cpu().numpy()
        max_sim = np.array([widen(v) for v in sims.max(axis=1)])
        top1 = np.argmax(sims, axis=1)
        return norm, sims, max_sim, top1

    n_calib, s_calib, m_calib, t1_calib = score_torch(raw_calib_mirror, protos_torch_cuda, device)
    n_cuda1_t, s_cuda1_t, m_cuda1_t, t1_cuda1_t = score_torch(raw_cuda_batch1, protos_torch_cuda, device)
    n_cuda1_np, s_cuda1_np, m_cuda1_np, t1_cuda1_np = score_numpy(raw_cuda_batch1)
    n_cpu32, s_cpu32, m_cpu32, t1_cpu32 = score_torch(raw_cpu_batch32, protos_torch_cpu, torch.device("cpu"))
    n_onnx1, s_onnx1, m_onnx1, t1_onnx1 = score_numpy(raw_onnx_batch1)

    PATHS = {
        "calibration_mirror_cuda_batch32": (raw_calib_mirror, n_calib, s_calib, m_calib, t1_calib),
        "pytorch_cuda_batch1_torch": (raw_cuda_batch1, n_cuda1_t, s_cuda1_t, m_cuda1_t, t1_cuda1_t),
        "pytorch_cuda_batch1_numpy": (raw_cuda_batch1, n_cuda1_np, s_cuda1_np, m_cuda1_np, t1_cuda1_np),
        "pytorch_cpu_batch32_torch": (raw_cpu_batch32, n_cpu32, s_cpu32, m_cpu32, t1_cpu32),
        "production_onnx_cpu_batch1": (raw_onnx_batch1, n_onnx1, s_onnx1, m_onnx1, t1_onnx1),
    }

    gate_by_path = {name: (m < gate_threshold) for name, (_, _, _, m, _) in PATHS.items()}

    # ---- per-image records -------------------------------------------------------
    per_image_records = []
    for i, meta in enumerate(ordered_meta):
        rec = {**meta}
        for name, (raw, norm, sims, max_sim, top1) in PATHS.items():
            rec[name] = {
                "raw_l2_norm": l2_norm(raw[i]), "raw_rms": rms(raw[i]),
                "normalized_l2_norm_sanity": l2_norm(norm[i]),
                "max_similarity": float(max_sim[i]), "top1": int(top1[i]),
                "gate_reject": bool(gate_by_path[name][i]),
            }
        per_image_records.append(rec)
    report["part_b_per_image"] = per_image_records

    # cross-check against parity_report.json's stored old-reference/onnx values
    old_ref = {(r["slug"], r["photo_id"]): r for r in rows}
    cross_check_diffs = []
    for i, meta in enumerate(ordered_meta):
        key = (meta["slug"], meta["photo_id"])
        old = old_ref[key]
        new_ref = float(m_cuda1_np[i])
        new_onnx = float(m_onnx1[i])
        cross_check_diffs.append({
            "slug": meta["slug"], "photo_id": meta["photo_id"],
            "old_reference_max_sim": old["max_similarity_reference"],
            "new_pytorch_cuda_batch1_numpy_max_sim": new_ref,
            "diff_reference": abs(old["max_similarity_reference"] - new_ref),
            "old_onnx_max_sim": old["max_similarity_onnx_cpu"],
            "new_production_onnx_max_sim": new_onnx,
            "diff_onnx": abs(old["max_similarity_onnx_cpu"] - new_onnx),
        })
    max_cross_check_diff_ref = max(c["diff_reference"] for c in cross_check_diffs)
    max_cross_check_diff_onnx = max(c["diff_onnx"] for c in cross_check_diffs)
    report["part_b_cross_check_vs_parity_report"] = {
        "max_diff_reference_path": max_cross_check_diff_ref,
        "max_diff_onnx_path": max_cross_check_diff_onnx,
        "note": "pytorch_cuda_batch1_numpy and production_onnx_cpu_batch1 recomputed fresh in "
               "this script should reproduce parity_report.json's stored reference/onnx_cpu "
               "values exactly (same code paths, same images, same artifacts); this checks that.",
    }
    log(f"  cross-check vs parity_report.json: max diff (reference)={max_cross_check_diff_ref}, "
       f"(onnx)={max_cross_check_diff_onnx}")

    # ---- pairwise path distributions ---------------------------------------------
    KEY_PAIRS = [
        ("calibration_mirror_cuda_batch32", "production_onnx_cpu_batch1"),
        ("calibration_mirror_cuda_batch32", "pytorch_cuda_batch1_torch"),
        ("pytorch_cuda_batch1_torch", "pytorch_cuda_batch1_numpy"),
        ("pytorch_cpu_batch32_torch", "production_onnx_cpu_batch1"),
        ("calibration_mirror_cuda_batch32", "pytorch_cpu_batch32_torch"),
    ]
    pairwise = {}
    for a, b in KEY_PAIRS:
        raw_a, norm_a, sims_a, max_a, top1_a = PATHS[a]
        raw_b, norm_b, sims_b, max_b, top1_b = PATHS[b]
        raw_diff = np.abs(raw_a - raw_b).max(axis=1)
        norm_diff = np.abs(norm_a - norm_b).max(axis=1)
        score_diff = np.abs(sims_a - sims_b).max(axis=1)
        maxsim_diff = np.abs(max_a - max_b)
        top1_disagree = int(np.sum(top1_a != top1_b))
        gate_a = max_a < gate_threshold
        gate_b = max_b < gate_threshold
        gate_disagree = int(np.sum(gate_a != gate_b))
        pairwise[f"{a}__vs__{b}"] = {
            "raw_embedding_max_abs_diff": percentiles(raw_diff),
            "normalized_embedding_max_abs_diff": percentiles(norm_diff),
            "score_vector_max_abs_diff": percentiles(score_diff),
            "max_similarity_abs_diff": percentiles(maxsim_diff),
            "top1_disagreement_count": top1_disagree,
            "gate_disagreement_count": gate_disagree,
        }
    report["part_b_pairwise_distributions"] = pairwise
    log(f"  primary comparison (calib_mirror vs production_onnx) max_sim_diff: "
       f"{pairwise['calibration_mirror_cuda_batch32__vs__production_onnx_cpu_batch1']['max_similarity_abs_diff']}")
    log(f"  primary comparison top1 disagreements: "
       f"{pairwise['calibration_mirror_cuda_batch32__vs__production_onnx_cpu_batch1']['top1_disagreement_count']}  "
       f"gate disagreements: "
       f"{pairwise['calibration_mirror_cuda_batch32__vs__production_onnx_cpu_batch1']['gate_disagreement_count']}")

    # ---- Task 2c data collection (recorded, not declared approved) --------------
    report["task_2c_data_collected_not_approved"] = {
        "note": "PyTorch-CPU <-> ONNX-CPU population measurements across all 200 images are "
               "collected here as pytorch_cpu_batch32_torch vs production_onnx_cpu_batch1 -- "
               "recorded for review, Task 2c itself is NOT declared approved by this script.",
        "distribution": pairwise["pytorch_cpu_batch32_torch__vs__production_onnx_cpu_batch1"],
    }

    # ============================================================ POPULATION SUMMARIES
    log("\n=== population summaries (descriptive, n=200 diagnostic sample only) ===")
    pop_summary = {}
    for name, (raw, norm, sims, max_sim, top1) in PATHS.items():
        norms = np.array([l2_norm(raw[i]) for i in range(n)])
        rmss = np.array([rms(raw[i]) for i in range(n)])
        pop_summary[name] = {"raw_l2_norm": percentiles(norms), "raw_rms": percentiles(rmss)}
    report["part_b_population_summaries"] = pop_summary

    onnx_norms = np.array([l2_norm(raw_onnx_batch1[i]) for i in range(n)])
    cuda1_norms = np.array([l2_norm(raw_cuda_batch1[i]) for i in range(n)])
    calib_norms = np.array([l2_norm(raw_calib_mirror[i]) for i in range(n)])

    top10_onnx_norm = sorted(range(n), key=lambda i: -onnx_norms[i])[:10]
    ratio_discrepancy = np.abs(cuda1_norms - onnx_norms) / np.clip(onnx_norms, 1e-8, None)
    top10_ratio = sorted(range(n), key=lambda i: -ratio_discrepancy[i])[:10]
    calib_prod_divergence = np.abs(m_calib - m_onnx1)
    top10_divergence = sorted(range(n), key=lambda i: -calib_prod_divergence[i])[:10]

    def image_summary(i):
        return {"slug": ordered_meta[i]["slug"], "photo_id": ordered_meta[i]["photo_id"],
               "file_sha256": ordered_meta[i]["file_sha256"]}

    report["part_b_top10_lists"] = {
        "by_onnx_cpu_raw_norm": [{"rank": r + 1, **image_summary(i), "onnx_raw_norm": float(onnx_norms[i])}
                                 for r, i in enumerate(top10_onnx_norm)],
        "by_cuda_onnx_norm_ratio_discrepancy": [
            {"rank": r + 1, **image_summary(i), "ratio_discrepancy": float(ratio_discrepancy[i]),
             "cuda_norm": float(cuda1_norms[i]), "onnx_norm": float(onnx_norms[i])}
            for r, i in enumerate(top10_ratio)],
        "by_calibration_mirror_vs_production_max_sim_divergence": [
            {"rank": r + 1, **image_summary(i), "divergence": float(calib_prod_divergence[i]),
             "calib_mirror_max_sim": float(m_calib[i]), "production_max_sim": float(m_onnx1[i])}
            for r, i in enumerate(top10_divergence)],
    }
    log(f"  top divergence (calib-mirror vs production): "
       f"{report['part_b_top10_lists']['by_calibration_mirror_vs_production_max_sim_divergence'][0]}")

    median_by_path = {name: float(np.median([l2_norm(raw[i]) for i in range(n)]))
                      for name, (raw, *_ ) in PATHS.items()}
    multiplier_counts = {}
    for name, (raw, *_ ) in PATHS.items():
        norms_ = np.array([l2_norm(raw[i]) for i in range(n)])
        med = median_by_path[name]
        multiplier_counts[name] = {
            "median": med,
            "count_above_2x_median": int(np.sum(norms_ > 2 * med)),
            "count_above_3x_median": int(np.sum(norms_ > 3 * med)),
            "count_above_5x_median": int(np.sum(norms_ > 5 * med)),
        }
    report["part_b_multiplier_band_counts"] = {
        "note": "Descriptive diagnostics on this fixed 200-image sample only -- NOT learned "
               "OOD thresholds, not applied to any decision.",
        "by_path": multiplier_counts,
    }
    log(f"  multiplier bands (production_onnx_cpu_batch1): "
       f"{multiplier_counts['production_onnx_cpu_batch1']}")

    # ============================================================ BOUNDARY ANALYSIS
    log("\n=== boundary analysis: unrounded max_sim in [0.55, 0.65] ===")

    in_band_old_ref = np.array([BOUNDARY_LO <= r["max_similarity_reference"] <= BOUNDARY_HI for r in rows])
    in_band_old_onnx = np.array([BOUNDARY_LO <= r["max_similarity_onnx_cpu"] <= BOUNDARY_HI for r in rows])
    in_band_calib = (m_calib >= BOUNDARY_LO) & (m_calib <= BOUNDARY_HI)
    in_band_prod = (m_onnx1 >= BOUNDARY_LO) & (m_onnx1 <= BOUNDARY_HI)

    boundary_summary = {
        "band": [BOUNDARY_LO, BOUNDARY_HI],
        "existing_cuda_batch1_numpy_path_count": int(in_band_old_ref.sum()),
        "existing_onnx_path_count": int(in_band_old_onnx.sum()),
        "existing_union_count": int((in_band_old_ref | in_band_old_onnx).sum()),
        "calibration_mirror_count": int(in_band_calib.sum()),
        "production_onnx_count": int(in_band_prod.sum()),
        "calib_prod_intersection_count": int((in_band_calib & in_band_prod).sum()),
        "calib_prod_union_count": int((in_band_calib | in_band_prod).sum()),
    }
    report["boundary_analysis_summary"] = boundary_summary
    for k, v in boundary_summary.items():
        log(f"  {k}: {v}")

    if boundary_summary["calibration_mirror_count"] != boundary_summary["existing_cuda_batch1_numpy_path_count"]:
        boundary_summary["reconciliation_note"] = (
            "calibration_mirror_cuda_batch32 uses Torch-side batch=32 normalization/scoring on "
            "CUDA with current default flags, while the existing parity_report.json reference "
            "path used Torch CUDA batch=1 embeddings scored in NumPy -- a different scoring "
            "backend AND batch size. A different in-band count is expected if any image's score "
            "sits close enough to the band edge for these small numeric differences to cross "
            "0.55 or 0.65; this is not an error, and no image outside typical parity-level "
            "differences (see pairwise distributions above) should be involved."
        )
        log(f"  reconciliation: {boundary_summary['reconciliation_note']}")

    union_idx = sorted(set(np.where(in_band_old_ref | in_band_old_onnx)[0].tolist())
                       | set(np.where(in_band_calib | in_band_prod)[0].tolist()))
    boundary_rows = []
    for i in union_idx:
        old = old_ref[(ordered_meta[i]["slug"], ordered_meta[i]["photo_id"])]
        row = {
            **image_summary(i),
            "old_reference_max_sim": old["max_similarity_reference"],
            "old_onnx_max_sim": old["max_similarity_onnx_cpu"],
            "calibration_mirror_max_sim": float(m_calib[i]),
            "pytorch_cuda_batch1_torch_max_sim": float(m_cuda1_t[i]),
            "pytorch_cuda_batch1_numpy_max_sim": float(m_cuda1_np[i]),
            "pytorch_cpu_batch32_torch_max_sim": float(m_cpu32[i]),
            "production_onnx_max_sim": float(m_onnx1[i]),
            "distance_from_0.60": {
                "calibration_mirror": abs(float(m_calib[i]) - gate_threshold),
                "production_onnx": abs(float(m_onnx1[i]) - gate_threshold),
            },
            "cross_path_shift_calib_to_production": abs(float(m_calib[i]) - float(m_onnx1[i])),
            "top1": {
                "calibration_mirror": int(t1_calib[i]), "production_onnx": int(t1_onnx1[i]),
            },
            "gate_reject": {
                "calibration_mirror": bool(gate_by_path["calibration_mirror_cuda_batch32"][i]),
                "production_onnx": bool(gate_by_path["production_onnx_cpu_batch1"][i]),
            },
            "raw_norms": {
                "calibration_mirror": float(calib_norms[i]),
                "pytorch_cuda_batch1": float(cuda1_norms[i]),
                "production_onnx": float(onnx_norms[i]),
            },
        }
        boundary_rows.append(row)
    report["boundary_band_rows"] = boundary_rows
    log(f"  {len(boundary_rows)} rows in the union boundary band")

    exact_gate_disagreements = [r for r in boundary_rows
                               if r["gate_reject"]["calibration_mirror"] != r["gate_reject"]["production_onnx"]]
    report["boundary_gate_disagreements"] = exact_gate_disagreements
    log(f"  exact gate disagreements (calib_mirror vs production) in boundary band: {len(exact_gate_disagreements)}")

    shift_ge_distance_rows = [r for r in boundary_rows
                             if r["cross_path_shift_calib_to_production"] >=
                             min(r["distance_from_0.60"]["calibration_mirror"],
                                 r["distance_from_0.60"]["production_onnx"])]
    report["boundary_rows_shift_ge_distance_to_gate"] = shift_ge_distance_rows
    log(f"  rows where cross-path shift >= distance to 0.60 on at least one path: {len(shift_ge_distance_rows)}")

    # cross-tab boundary membership x multiplier bands (production ONNX norm reference)
    prod_median = median_by_path["production_onnx_cpu_batch1"]
    cross_tab = {"2x": 0, "3x": 0, "5x": 0, "boundary_band_n": len(boundary_rows)}
    for r in boundary_rows:
        norm_ = r["raw_norms"]["production_onnx"]
        if norm_ > 2 * prod_median:
            cross_tab["2x"] += 1
        if norm_ > 3 * prod_median:
            cross_tab["3x"] += 1
        if norm_ > 5 * prod_median:
            cross_tab["5x"] += 1
    report["boundary_x_multiplier_band_crosstab"] = cross_tab
    log(f"  boundary x multiplier-band cross-tab (production ONNX norm reference): {cross_tab}")

    # exploratory Spearman: raw norm (production ONNX) vs cross-path divergence (calib vs prod)
    spearman_rho = spearman(onnx_norms, calib_prod_divergence)
    report["exploratory_spearman_norm_vs_divergence"] = {
        "label": "EXPLORATORY ONLY -- not a causal or population claim",
        "variable_1": "production_onnx_cpu_batch1 raw embedding L2 norm (n=200)",
        "variable_2": "abs(calibration_mirror_max_sim - production_onnx_max_sim) (n=200)",
        "spearman_rho": spearman_rho,
    }
    log(f"  exploratory Spearman(norm, cross-path divergence) = {spearman_rho:.4f} (n=200)")

    # ============================================================ OUTLIER METADATA
    log("\n=== outlier metadata verification ===")
    sec1 = diag["section1_input_embedding_measurements"]
    outlier_diag = next(r for r in sec1 if r["role"] == "outlier")
    expected = {
        "dimensions": [500, 375], "mode": "RGB", "exif_orientation": None,
        "pixel_min": 0.0, "pixel_max": 255.0, "pixel_mean_approx": 247.3766, "pixel_std_approx": 31.1306,
        "norm_input_mean_approx": 2.30669, "norm_input_std_approx": 0.54903,
        "onnx_norm_approx": 121, "default_cuda_norm_approx": 70.98,
    }
    actual = {
        "dimensions": outlier_diag["dimensions"], "mode": outlier_diag["mode"],
        "exif_orientation": outlier_diag["exif_orientation"],
        "pixel_min": outlier_diag["decoded_pixel_stats"]["min"],
        "pixel_max": outlier_diag["decoded_pixel_stats"]["max"],
        "pixel_mean": outlier_diag["decoded_pixel_stats"]["mean"],
        "pixel_std": outlier_diag["decoded_pixel_stats"]["std"],
        "norm_input_mean": outlier_diag["normalized_input_stats_reference"]["mean"],
        "norm_input_std": outlier_diag["normalized_input_stats_reference"]["std"],
        "onnx_norm": outlier_diag["raw_embedding_l2_norm_by_path"]["onnx_cpu_default"],
        "default_cuda_norm": outlier_diag["raw_embedding_l2_norm_by_path"]["pytorch_cuda"],
    }
    outlier_idx = next(i for i, m in enumerate(ordered_meta)
                       if m["slug"] == OUTLIER["slug"] and m["photo_id"] == OUTLIER["photo_id"])
    pixel_means = np.array([np.asarray(Image.open(resolved_paths[(m["slug"], m["photo_id"])]).convert("RGB"),
                                       dtype=np.float64).mean() for m in ordered_meta])
    pixel_stds = np.array([np.asarray(Image.open(resolved_paths[(m["slug"], m["photo_id"])]).convert("RGB"),
                                      dtype=np.float64).std() for m in ordered_meta])

    def percentile_rank(arr, value):
        return float((arr < value).mean() * 100)

    outlier_percentiles = {
        "decoded_pixel_mean_percentile": percentile_rank(pixel_means, pixel_means[outlier_idx]),
        "decoded_pixel_std_percentile": percentile_rank(pixel_stds, pixel_stds[outlier_idx]),
        "production_onnx_raw_norm_percentile": percentile_rank(onnx_norms, onnx_norms[outlier_idx]),
    }
    report["outlier_metadata_verification"] = {
        "expected": expected, "actual": actual,
        "matches_expected": (
            actual["dimensions"] == expected["dimensions"] and actual["mode"] == expected["mode"]
            and actual["exif_orientation"] == expected["exif_orientation"]
            and actual["pixel_min"] == expected["pixel_min"] and actual["pixel_max"] == expected["pixel_max"]
            and abs(actual["pixel_mean"] - expected["pixel_mean_approx"]) < 0.01
            and abs(actual["pixel_std"] - expected["pixel_std_approx"]) < 0.01
            and abs(actual["norm_input_mean"] - expected["norm_input_mean_approx"]) < 0.01
            and abs(actual["norm_input_std"] - expected["norm_input_std_approx"]) < 0.01
        ),
        "percentiles_within_200_diagnostic_sample": outlier_percentiles,
        "note": "Percentiles are within this fixed 200-image diagnostic sample only. Not a "
               "population/production prevalence claim; sample is stratified (4/species), not random.",
    }
    log(f"  outlier metadata matches expected: {report['outlier_metadata_verification']['matches_expected']}")
    log(f"  outlier percentiles within n=200: {outlier_percentiles}")

    # ============================================================ WRAP UP
    parity_report_hash_after = sha256_file(args.parity_report)
    parity_diagnostic_hash_after = sha256_file(args.parity_diagnostic)
    report["input_report_hashes_after"] = {
        "parity_report_json": parity_report_hash_after,
        "parity_diagnostic_json": parity_diagnostic_hash_after,
    }
    if parity_report_hash_after != parity_report_hash_before:
        failures.append("parity_report.json changed during this run")
    if parity_diagnostic_hash_after != parity_diagnostic_hash_before:
        failures.append("parity_diagnostic.json changed during this run")

    report["warnings"] = warnings
    report["structural_verification"] = {
        "n_diagnostic_images_part_a": len(manifest),
        "n_images_part_b": n,
        "cuda_available": True,
        "onnx_cpu_provider_confirmed": sess_cpu.get_providers() == ["CPUExecutionProvider"],
        "prior_reports_unchanged": (parity_report_hash_after == parity_report_hash_before
                                    and parity_diagnostic_hash_after == parity_diagnostic_hash_before),
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
    log("\nRun complete. See report for factorial/boundary evidence; no cause is asserted "
       "beyond what the matrix directly shows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
