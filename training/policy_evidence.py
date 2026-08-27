"""Evidence loading + cross-checking for inference_policy_generator.py.

Generator-only logic (unlike policy_schema.py, which is shared with the
future Task 4 loader). Every check here raises EvidenceFailure on the
first problem found -- the generator catches it once at the top level
and fails closed (writes nothing).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

EXPECTED_PHASE_C_STATUS = (
    "INDEPENDENTLY VALIDATED as a selective confidence/abstention gate; "
    "NOT VALIDATED as an unknown-species detector"
)
EXPECTED_RULE_SIGNAL = "raw max cosine similarity before geo re-ranking"
THRESHOLD_SOURCE_TOKEN = "calibration_v1.json:frozen_candidate_abstention_threshold"
MANDATORY_EVAL_KEYS = {"checkpoint": "model.pth", "prototypes_npy": "prototypes.npy",
                        "taxonomy_json": "taxonomy.json", "config_yaml": "config.yaml"}
PROVENANCE_SENTENCE = (
    "The calibration and Phase C evaluation artifacts evaluated model.pth and did not record "
    "backbone.onnx. ONNX correspondence is not established by derivation from the checkpoint; it is "
    "established empirically by the parity reports, which recorded the current backbone.onnx hash "
    "alongside the same model.pth, prototypes.npy, taxonomy.json, and config.yaml hashes used by "
    "Phase C, and measured max-sim divergence between the two paths on a fixed 200-image stratified "
    "diagnostic sample."
)


class EvidenceFailure(Exception):
    pass


def fail(msg: str) -> None:
    raise EvidenceFailure(msg)


def read_bytes_and_hash(path: Path):
    if not path.exists():
        fail(f"required file missing: {path}")
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def read_json_and_hash(path: Path):
    raw, digest = read_bytes_and_hash(path)
    try:
        return json.loads(raw), digest
    except json.JSONDecodeError as e:
        fail(f"{path} is not valid JSON: {e}")


def read_yaml_and_hash(path: Path):
    import yaml
    raw, digest = read_bytes_and_hash(path)
    return yaml.safe_load(raw), digest


def hash_streaming(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report_hash(label: str, report: dict, *names: str):
    h = report.get("artifact_hashes", {})
    found = {n: h[n] for n in names if n in h}
    if len(set(found.values())) > 1:
        fail(f"{label} has conflicting alias hashes {list(found)}: {found}")
    return next(iter(found.values()), None)


def close3(eval_val, calib_val) -> bool:
    return eval_val is not None and calib_val is not None and abs(round(eval_val, 3) - calib_val) < 1e-9


def derive_count(n, rate, label: str) -> dict:
    if not isinstance(n, int) or not isinstance(rate, (int, float)):
        fail(f"{label}: missing/invalid n or rate (n={n!r}, rate={rate!r})")
    raw = rate * n
    count = round(raw)
    if abs(raw - count) > 1e-6:
        fail(f"{label}: rate*n does not resolve to an integer count (rate={rate} n={n} raw={raw})")
    return {"count": count, "of": n, "rate": rate}


# ---------------------------------------------------------------- Sec 1/3
def check_threshold_lineage(calib: dict, calib_hash: str, unk_eval: dict):
    fca = calib.get("frozen_candidate_abstention_threshold")
    if not isinstance(fca, dict):
        fail("frozen_candidate_abstention_threshold missing")
    rule = fca.get("machine_readable_rule")
    if not isinstance(rule, dict):
        fail("machine_readable_rule missing")
    operator, signal, value = rule.get("operator"), rule.get("signal"), rule.get("value")
    if operator != "max_sim < value":
        fail(f"unsupported operator {operator!r}")
    if signal != EXPECTED_RULE_SIGNAL:
        fail(f"unexpected rule signal {signal!r}; expected {EXPECTED_RULE_SIGNAL!r} exactly")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 < value < 1.0):
        fail(f"invalid threshold value {value!r}")

    phase_c = calib.get("phase_c_validation")
    if not isinstance(phase_c, dict):
        fail("phase_c_validation missing")
    status = phase_c.get("status")
    if status != EXPECTED_PHASE_C_STATUS:
        fail(f"phase_c_validation.status is not the currently supported status (exact match): {status!r}")

    if unk_eval.get("frozen_threshold") != value:
        fail(f"unknown_test_v1_eval.json.frozen_threshold {unk_eval.get('frozen_threshold')!r} != "
             f"calibration rule value {value!r}")
    if THRESHOLD_SOURCE_TOKEN not in (unk_eval.get("threshold_source") or ""):
        fail(f"unknown_test_v1_eval.json.threshold_source does not name {THRESHOLD_SOURCE_TOKEN!r}")
    uev_hashes = unk_eval.get("hashes") or {}
    if uev_hashes.get("calibration_v1_json") != calib_hash:
        fail("unknown_test_v1_eval.json.hashes.calibration_v1_json missing or disagrees with the "
             "current calibration_v1.json")
    return fca, rule, phase_c, operator, signal, value, status


# -------------------------------------------------------------------- Sec 3
def build_phase_c_block(fca: dict, phase_c: dict, status: str, unk_eval: dict, unk_eval_hash: str):
    sc = unk_eval.get("selective_classification", {})
    fac = unk_eval.get("false_acceptance_by_category", {})
    kpr = unk_eval.get("known_photo_rejection", {})
    n_known = kpr.get("n")

    metrics = {
        "known_photos_rejected": derive_count(n_known, kpr.get("rate"), "known_photos_rejected"),
        "accuracy_before_abstention": derive_count(n_known, sc.get("accuracy_before_abstention"),
                                                    "accuracy_before_abstention"),
        "correct_predictions_rejected": derive_count(
            sc.get("correct_predictions_rejected", {}).get("n"),
            sc.get("correct_predictions_rejected", {}).get("rate"), "correct_predictions_rejected"),
        "incorrect_predictions_rejected": derive_count(
            sc.get("incorrect_predictions_rejected", {}).get("n"),
            sc.get("incorrect_predictions_rejected", {}).get("rate"), "incorrect_predictions_rejected"),
        "accuracy_among_accepted": derive_count(
            sc.get("accuracy_among_accepted", {}).get("n"),
            sc.get("accuracy_among_accepted", {}).get("rate"), "accuracy_among_accepted"),
        "out_of_scope_ant_false_acceptance": derive_count(
            fac.get("out_of_scope_ant", {}).get("n"),
            fac.get("out_of_scope_ant", {}).get("false_acceptance_rate"), "out_of_scope_ant_false_acceptance"),
        "non_ant_insect_false_acceptance": derive_count(
            fac.get("non_ant_insect", {}).get("n"),
            fac.get("non_ant_insect", {}).get("false_acceptance_rate"), "non_ant_insect_false_acceptance"),
        "unrelated_false_acceptance": derive_count(
            fac.get("unrelated", {}).get("n"),
            fac.get("unrelated", {}).get("false_acceptance_rate"), "unrelated_false_acceptance"),
    }

    results_0_60 = phase_c.get("results_at_0.60", {})
    crosscheck_pairs = [
        ("known_photo_rejection_rate", kpr.get("rate")),
        ("accuracy_before_abstention", sc.get("accuracy_before_abstention")),
        ("correct_predictions_rejected", sc.get("correct_predictions_rejected", {}).get("rate")),
        ("incorrect_predictions_rejected", sc.get("incorrect_predictions_rejected", {}).get("rate")),
        ("accuracy_among_accepted", sc.get("accuracy_among_accepted", {}).get("rate")),
        ("out_of_scope_ant_false_acceptance_rate", fac.get("out_of_scope_ant", {}).get("false_acceptance_rate")),
    ]
    for key, eval_val in crosscheck_pairs:
        calib_val = results_0_60.get(key)
        if not close3(eval_val, calib_val):
            fail(f"phase_c_validation.results_at_0.60.{key}={calib_val!r} disagrees with "
                 f"unknown_test_v1_eval.json (={eval_val!r})")

    return {
        "status": status,
        "not_a_photo_quality_detector": True,
        "historical_candidate_status": {
            "value": fca.get("status"), "selected_at": fca.get("selected_at"),
            "note": "Describes the threshold at selection time, before independent evaluation. The "
                    "phase_c_validation status above is the later, independent evaluation on "
                    "unknown_test_v1 that superseded this candidate status.",
        },
        "metrics": metrics,
        "cross_checked_against_calibration_v1_results_at_0_60": "6 of 8 metrics (non_ant_insect and "
            "unrelated false-acceptance rates have no calibration_v1.json counterpart to cross-check)",
        "unknown_test_v1_eval_json_sha256": unk_eval_hash,
        "known_holdout_species_coverage": phase_c.get("known_holdout_species_coverage"),
    }


# -------------------------------------------------------------------- Sec 2
def check_mandatory_eval_hashes(calib_scores: dict, unk_eval: dict, current: dict, notes: list):
    def check_one(label, ev):
        h = ev.get("hashes") or {}
        for key, artifact in MANDATORY_EVAL_KEYS.items():
            v = h.get(key)
            if v is None:
                fail(f"{label} is missing required hash: hashes.{key}")
            if v != current[artifact]:
                fail(f"{label} hashes.{key} disagrees with current {artifact}")
        onnx_h = h.get("backbone_onnx")
        if onnx_h is not None and onnx_h != current["backbone.onnx"]:
            fail(f"{label} hashes.backbone_onnx disagrees with current backbone.onnx")
        if onnx_h is None:
            notes.append(f"{label} does not record a hash for backbone.onnx (expected optional omission)")
        return h

    h_calib = check_one("calibration_v1_scores.json", calib_scores)
    h_unk = check_one("unknown_test_v1_eval.json", unk_eval)
    for key in MANDATORY_EVAL_KEYS:
        if h_calib[key] != h_unk[key]:
            fail(f"calibration_v1_scores.json and unknown_test_v1_eval.json disagree on hashes.{key}")


# -------------------------------------------------------------------- Sec 5
def check_parity_binding(parity: dict, current: dict):
    """A report qualifies for onnx_bound_by only if all FIVE hashes
    (onnx, model, prototypes, taxonomy, config) match current; a missing
    config_yaml hash disqualifies rather than being treated as silent."""
    onnx_bound_by = []
    for label, prep in parity.items():
        onnx_h = report_hash(label, prep, "backbone_onnx", "backbone.onnx")
        model_h = report_hash(label, prep, "model_pth", "model.pth")
        proto_h = report_hash(label, prep, "prototypes_npy", "prototypes.npy")
        tax_h = report_hash(label, prep, "taxonomy_json", "taxonomy.json")
        cfg_h = report_hash(label, prep, "config_yaml")
        for name, h in (("backbone.onnx", onnx_h), ("model.pth", model_h), ("prototypes.npy", proto_h),
                        ("taxonomy.json", tax_h), ("config.yaml", cfg_h)):
            if h is not None and h != current[name]:
                fail(f"{label} records a {name} hash that disagrees with the current artifact")
        if (onnx_h == current["backbone.onnx"] and model_h == current["model.pth"]
                and proto_h == current["prototypes.npy"] and tax_h == current["taxonomy.json"]
                and cfg_h == current["config.yaml"]):
            onnx_bound_by.append(label)
    if not onnx_bound_by:
        fail("no evidence report binds the current backbone.onnx alongside matching "
             "model.pth/prototypes.npy/taxonomy.json/config.yaml -- unsatisfiable")
    return onnx_bound_by


def check_parity_integrity(pr: dict, pd: dict, fa: dict, pr_hash: str, pd_hash: str):
    sv1, sv2, sv3 = (pr.get("structural_verification", {}), pd.get("structural_verification", {}),
                     fa.get("structural_verification", {}))
    if sv1.get("failures"):
        fail(f"parity_report.json structural_verification.failures non-empty: {sv1['failures']}")
    if not sv1.get("cpu_execution_provider_confirmed"):
        fail("parity_report.json did not confirm exclusive CPU provider")
    if sv2.get("failures"):
        fail(f"parity_diagnostic.json structural_verification.failures non-empty: {sv2['failures']}")
    if not sv2.get("cpu_execution_provider_confirmed_all_forced_sessions"):
        fail("parity_diagnostic.json did not confirm CPU provider for all forced sessions")
    if sv3.get("failures"):
        fail(f"parity_flag_ablation.json structural_verification.failures non-empty: {sv3['failures']}")
    if not sv3.get("onnx_cpu_provider_confirmed"):
        fail("parity_flag_ablation.json did not confirm ONNX CPU provider")
    if not sv3.get("prior_reports_unchanged"):
        fail("parity_flag_ablation.json reports prior_reports_unchanged=False")

    before, after = fa.get("input_report_hashes_before", {}), fa.get("input_report_hashes_after", {})
    live = {"parity_report_json": pr_hash, "parity_diagnostic_json": pd_hash}
    for label, recorded in (("input_report_hashes_before", before), ("input_report_hashes_after", after)):
        for key, live_hash in live.items():
            if recorded.get(key) != live_hash:
                fail(f"parity_flag_ablation.json.{label}.{key} does not match the current live hash "
                     f"of that file -- evidence chain is stale or the file changed since flag_ablation ran")

    ig = fa.get("integrity_gate", {})
    for k in ("seed_ok", "n_per_species_ok", "n_samples_valid_ok", "sample_list_hash_ok"):
        if not ig.get(k):
            fail(f"parity_flag_ablation.json integrity_gate.{k} is not True")
    if ig.get("duplicate_row_keys", 0) != 0 or ig.get("duplicate_file_hashes", 0) != 0:
        fail(f"parity_flag_ablation.json integrity_gate reports duplicate rows/hashes: "
             f"duplicate_row_keys={ig.get('duplicate_row_keys')} duplicate_file_hashes={ig.get('duplicate_file_hashes')}")
    if ig.get("file_verification_problems"):
        fail("parity_flag_ablation.json integrity_gate.file_verification_problems non-empty")

    sb = pr.get("stage_b_real_end_to_end_parity", {})
    if sb.get("sample_list_hash_sha256") != ig.get("sample_list_hash_sha256"):
        fail("fixed sample identity disagrees between parity_report.json and parity_flag_ablation.json")
    if sb.get("n_samples_valid") != ig.get("n_samples_valid"):
        fail("fixed sample count disagrees between parity_report.json and parity_flag_ablation.json")
    if sb.get("preprocessing_tensor_max_abs_diff") != 0.0:
        fail("parity_report.json's real-image preprocessing divergence is not the recorded zero result")

    pw = fa["part_b_pairwise_distributions"]["calibration_mirror_cuda_batch32__vs__production_onnx_cpu_batch1"]
    if pw.get("top1_disagreement_count", -1) != 0:
        fail(f"calibration-mirror vs production top1_disagreement_count is nonzero: {pw.get('top1_disagreement_count')}")
    if pw.get("gate_disagreement_count", -1) != 0:
        fail(f"calibration-mirror vs production gate_disagreement_count is nonzero: {pw.get('gate_disagreement_count')}")
    return sb, ig, pw


def build_measured_parity(fa: dict, sb: dict, ig: dict, pw: dict):
    bnd, n = fa["boundary_analysis_summary"], ig["n_samples_valid"]
    msd = pw["max_similarity_abs_diff"]
    cross_check = fa.get("part_b_cross_check_vs_parity_report", {})
    return {
        "sample": f"fixed stratified {n}-image diagnostic sample (seed={ig['seed']}, "
                  f"{ig['n_per_species']} images/species), not a random production sample "
                  f"(sample_list_hash_sha256={ig['sample_list_hash_sha256']})",
        "cuda_batch32_vs_onnx_cpu_max_sim_divergence_percentiles": {k: msd[k] for k in ("p50", "p95", "p99", "max")},
        "top1_disagreements": f"{pw['top1_disagreement_count']}/{n}",
        "gate_disagreements": f"{pw['gate_disagreement_count']}/{n}",
        "boundary_band": bnd["band"],
        "boundary_band_count": f"{bnd['calib_prod_union_count']}/{n}",
        "boundary_band_gate_shift_crossings": len(fa.get("boundary_gate_disagreements", [])),
        "preprocessing_tensor_max_abs_diff_on_200_real_images": sb["preprocessing_tensor_max_abs_diff"],
        "historical_replay_caveat": f"parity_flag_ablation.part_b_cross_check_vs_parity_report."
            f"max_diff_reference_path is {cross_check.get('max_diff_reference_path')!r} (~0.0401539), even "
            "though that report's own note expected an exact replay. Not a shipping blocker: the "
            "production-relevant result is described only as a fixed batch-32 embedding diagnostic "
            "comparison, not an exact historical-evaluator replay.",
        "calibration_mirror_scope_limitation": "This comparison path ('calibration_mirror_cuda_batch32') "
            "is a batch-32 embedding diagnostic reproduction, not a byte-for-byte replay of the "
            "historical calibration/Phase C evaluator's entire scoring execution.",
        "eval_batch_size_32_confirmed": "eval_calibration.py, eval_unknown_test.py, evaluate.py, and "
            "eval_benchmark.py all read batch_size from config.yaml (=32) via "
            "DataLoader(..., batch_size=cfg['batch_size']); confirmed by source inspection, not rerun",
        "limitation": "Sufficient engineering evidence for this personal/local app; not a population "
                     "bound and not proof of perfect parity.",
    }


# -------------------------------------------------------------------- Sec 4
def preprocessing_parity_check(here: Path):
    import numpy as np
    import yaml
    from PIL import Image
    cfg = yaml.safe_load((here / "config.yaml").read_text())
    from data import build_transforms
    import inference as api_inference

    rng = random.Random(20260823)
    w, h = 517, 241
    img = Image.frombytes("RGBA", (w, h), bytes(rng.randrange(256) for _ in range(w * h * 4)))
    train_arr = build_transforms(cfg, train=False)(img.convert("RGB")).numpy()
    serve_arr = api_inference.AntIdentifier.preprocess(None, img)[0]

    if train_arr.shape != serve_arr.shape:
        fail(f"preprocessing shape mismatch: training={train_arr.shape} serving={serve_arr.shape}")
    if str(train_arr.dtype) != "float32" or str(serve_arr.dtype) != "float32":
        fail(f"preprocessing dtype mismatch: training={train_arr.dtype} serving={serve_arr.dtype}")
    if not (np.isfinite(train_arr).all() and np.isfinite(serve_arr).all()):
        fail("preprocessing produced non-finite values")
    max_diff = float(np.abs(train_arr.astype("float64") - serve_arr.astype("float64")).max())
    if max_diff != 0.0:
        fail(f"executable preprocessing check found nonzero divergence: max_abs_diff={max_diff}")
    return {
        "method": "in-memory synthetic PIL image (non-square 517x241, seeded random RGBA pixels) run "
                  "through training's build_transforms(train=False) [incl. .convert('RGB')] and "
                  "api.AntIdentifier.preprocess() independently at generation time, no model inference",
        "shape": list(train_arr.shape), "dtype": "float32", "all_finite": True, "max_abs_diff": max_diff,
        "covers": "resize geometry/interpolation, /255 scaling, normalization, channel order, layout",
        "antialias_note": "Both paths receive PIL Images and use Pillow bilinear resampling for the "
                          "resize; torchvision's antialias option only affects its separate tensor-input "
                          "code path and has no effect here -- confirmed empirically by max_abs_diff=0.0.",
    }


PREPROCESSING_CONTRACT = {
    "rgb_conversion": "img.convert('RGB')", "resize": "squish to fixed 380x380 (both dims set, no crop)",
    "interpolation": "Pillow bilinear", "scale_divisor": 255.0,
    "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
    "dtype": "float32", "channel_layout": "RGB -> CHW, batched to NCHW",
}


# -------------------------------------------------------------------- Sec 5 (timm)
def timm_resolution_note(here: Path, cfg: dict):
    import timm
    from timm.data import resolve_model_data_config
    name = cfg["model"]["backbone"]
    model = timm.create_model(name, pretrained=False)
    dc = resolve_model_data_config(model)
    pcfg = getattr(model, "pretrained_cfg", None) or {}
    req_line = next((l.strip() for l in (here / "requirements.txt").read_text().splitlines()
                      if l.strip().lower().startswith("timm")), None)
    return {
        "resolved_in_timm_version": timm.__version__,
        "resolved_tag": f"{name}/{pcfg.get('tag')}" if pcfg.get("tag") else name,
        "mean": list(dc.get("mean", ())), "std": list(dc.get("std", ())),
        "interpolation": dc.get("interpolation"), "input_size": list(dc.get("input_size", ())),
        "crop_pct": dc.get("crop_pct"), "crop_mode": dc.get("crop_mode"),
        "project_actually_uses": {"interpolation": "bilinear", "crop": "none (squish resize)"},
        "note": (f"Resolved pretrained data configuration in the currently installed timm version. Not "
                 f"proof of the historical training environment -- requirements.txt pins timm as "
                 f"{req_line!r}. timm resolves interpolation={dc.get('interpolation')!r} for this "
                 "backbone; the project's own pipeline uses PIL bilinear, validated empirically by the "
                 "executable preprocessing check above. Informational only -- changes nothing."),
    }
