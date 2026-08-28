#!/usr/bin/env python3
"""Emits training/artifacts/inference_policy.json.

Thin orchestrator: evidence loading/cross-checking lives in
policy_evidence.py, schema/decision semantics live in policy_schema.py
(duplicated byte-for-byte in the API-local runtime loader). Fails closed -- on any problem,
nothing is written -- and self-validates before and after writing.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "api"))
sys.path.insert(0, str(HERE))  # training-local policy_schema must win

import policy_schema as schema
import policy_evidence as ev

GENERATOR_VERSION = "4.0.0"
SEQUENCING_NOTE = (
    "api/inference.py's source hash recorded in preprocessing.source_hashes is diagnostic provenance "
    "only, not a binding constraint. The API now ships the confidence gate and reads this policy from "
    "the artifact directory. Regenerate the policy after serving-code or bound-artifact changes so "
    "the recorded provenance describes the runtime being shipped."
)


def main() -> int:
    artifacts = Path(os.environ.get("POLICY_GEN_ARTIFACTS_DIR", HERE / "artifacts"))
    data = Path(os.environ.get("POLICY_GEN_DATA_DIR", HERE.parent / "data"))
    notes: list[str] = []

    calib, calib_hash = ev.read_json_and_hash(data / "calibration_v1" / "calibration_v1.json")
    unk_eval, unk_eval_hash = ev.read_json_and_hash(data / "unknown_test_v1" / "unknown_test_v1_eval.json")
    fca, rule, phase_c, operator, signal, value, status = ev.check_threshold_lineage(calib, calib_hash, unk_eval)
    phase_c_block = ev.build_phase_c_block(fca, phase_c, status, unk_eval, unk_eval_hash)

    current = {
        "backbone.onnx": ev.hash_streaming(artifacts / "backbone.onnx"),
        "prototypes.npy": ev.hash_streaming(artifacts / "prototypes.npy"),
        "taxonomy.json": ev.hash_streaming(artifacts / "taxonomy.json"),
        "model.pth": ev.hash_streaming(artifacts / "model.pth"),
    }
    cfg, config_hash = ev.read_yaml_and_hash(HERE / "config.yaml")
    current["config.yaml"] = config_hash
    for k, v in current.items():
        if v is None:
            ev.fail(f"required artifact/source missing: {k}")

    calib_scores, calib_scores_hash = ev.read_json_and_hash(data / "calibration_v1" / "calibration_v1_scores.json")
    ev.check_mandatory_eval_hashes(calib_scores, unk_eval, current, notes)

    parity_paths = {name: artifacts / name for name in
                     ("parity_report.json", "parity_diagnostic.json", "parity_flag_ablation.json")}
    parity, parity_hashes = {}, {}
    for name, path in parity_paths.items():
        parity[name], parity_hashes[name] = ev.read_json_and_hash(path)
    onnx_bound_by = ev.check_parity_binding(parity, current)
    pr, pd, fa = parity["parity_report.json"], parity["parity_diagnostic.json"], parity["parity_flag_ablation.json"]
    sb, ig, pw = ev.check_parity_integrity(pr, pd, fa, parity_hashes["parity_report.json"],
                                            parity_hashes["parity_diagnostic.json"])
    measured_parity = ev.build_measured_parity(fa, sb, ig, pw)

    exec_check = ev.preprocessing_parity_check(HERE)
    preprocessing = {
        "contract": ev.PREPROCESSING_CONTRACT,
        "executable_check": exec_check,
        "source_hashes": {"training/data.py": ev.hash_streaming(HERE / "data.py"),
                          "api/inference.py": ev.hash_streaming(HERE.parent / "api" / "inference.py")},
        "prior_real_image_diagnostic": {
            "n_images": sb["n_samples_valid"], "preprocessing_tensor_max_abs_diff": sb["preprocessing_tensor_max_abs_diff"],
            "sample_list_hash_sha256": sb["sample_list_hash_sha256"],
            "source": "training/artifacts/parity_report.json:stage_b_real_end_to_end_parity",
        },
        "sequencing_note": SEQUENCING_NOTE,
    }
    timm_info = ev.timm_resolution_note(HERE, cfg)

    import onnx as onnx_lib
    import onnxruntime as ort
    import numpy as np
    import torch
    import torchvision
    import PIL
    sess = ort.InferenceSession(str(artifacts / "backbone.onnx"), providers=["CPUExecutionProvider"])
    if sess.get_providers() != ["CPUExecutionProvider"]:
        ev.fail(f"CPUExecutionProvider was not exclusively selected: {sess.get_providers()}")
    opset = next((oi.version for oi in onnx_lib.load(str(artifacts / "backbone.onnx")).opset_import
                  if oi.domain == ""), None)

    evidence_environment = pr.get("environment", {})
    generation_environment = {
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "pillow_version": PIL.__version__,
        "numpy_version": np.__version__, "onnxruntime_version": ort.__version__,
        "onnx_opset": opset, "graph_optimization_level": str(ort.SessionOptions().graph_optimization_level),
    }
    shared_keys = evidence_environment.keys() & generation_environment.keys()
    environment_diff = {k: {"evidence": evidence_environment[k], "generation": generation_environment[k]}
                        for k in shared_keys if evidence_environment[k] != generation_environment[k]}
    validated_environment = {
        "evidence_environment": evidence_environment, "generation_environment": generation_environment,
        "differences": environment_diff,
        "note": "Both blocks are informational context, not binding. A difference here does not mean "
                "the parity evidence is invalid, but it does mean the evidence was NOT measured under "
                "the environment recorded in generation_environment -- a newly upgraded runtime is not "
                "retroactively covered by historical parity evidence just because this generator ran "
                "successfully under it. CPUExecutionProvider exclusivity (provider_policy) is the only "
                "binding runtime requirement.",
    }

    not_validated_for = [
        "CUDA or any non-CPU execution provider",
        "retrained or re-exported artifacts (any artifact_hashes change invalidates this policy)",
        "another backbone architecture, including EfficientNetV2",
        "expanded or reordered taxonomy, or new prototype rows",
        "unknown-species detection (this is a confidence/quality gate, not an out-of-catalog detector)",
        "diagnosing photo quality",
        "reliably distinguishing a blurry known ant from an unsupported ant",
        "post-geo or rounded similarity scores (gate applies to raw, unrounded, pre-geo max cosine)",
        "per-species thresholds (a single global threshold only)",
        "population claims from the 200-image parity diagnostic sample",
        f"the species absent from Phase C known_holdout: "
        f"{phase_c.get('known_holdout_species_coverage', {}).get('absent_species_unmeasured')}",
    ]

    decision = {
        "low_confidence_if": {"signal": "raw_pre_geo_max_cosine", "comparison": "strict_less_than", "threshold": value},
        "normal_results_if": {"comparison": "greater_than_or_equal", "threshold": value},
        "equal_threshold_action": "normal_results",
        "non_finite_action": "request_error",
        "float32_note": "No float32 value widens to exactly this float64 threshold (verified: "
                        "float(np.float32(0.6))=0.6000000238418579 != 0.6 exactly), so "
                        "equal_threshold_action documents an action for a boundary that a float32 "
                        "similarity score can never exactly hit in practice.",
    }
    enc_errors = schema.decision_encodings_agree(decision)
    if enc_errors:
        ev.fail("generator built an internally inconsistent decision block: " + "; ".join(enc_errors))
    for probe, expected in ((0.5999, True), (0.6000, False), (0.6001, False)):
        if schema.should_abstain(probe, decision) != expected:
            ev.fail(f"decision boundary self-check failed for should_abstain({probe})")

    content = {
        "rule": {"operator_verbatim": operator, "operator_normalized": {"comparison": "strict_less_than"},
                 "value": value, "signal": signal,
                 "evaluated": "raw pre-geo max cosine similarity, strict <, before any rounding"},
        "decision": decision,
        "artifact_hashes": {k: current[k] for k in schema.REQUIRED_ARTIFACT_HASH_KEYS},
        "provider_policy": {"providers": ["CPUExecutionProvider"], "exclusive": True},
        "preprocessing": preprocessing,
        "phase_c": phase_c_block,
        "validation": {
            "onnx_bound_by": onnx_bound_by,
            "provenance_statement": ev.PROVENANCE_SENTENCE,
            "measured_parity": measured_parity,
            "evidence_notes": notes,
            "provenance_hashes": {
                "calibration_v1_json": calib_hash, "calibration_v1_scores_json": calib_scores_hash,
                "unknown_test_v1_eval_json": unk_eval_hash,
                "parity_report_json": parity_hashes["parity_report.json"],
                "parity_diagnostic_json": parity_hashes["parity_diagnostic.json"],
                "parity_flag_ablation_json": parity_hashes["parity_flag_ablation.json"],
                "config_yaml": current["config.yaml"],
            },
        },
        "runtime_comparison_semantics": {
            "widening": "raw float32 max similarity is widened to float64 before comparison, without rounding",
            "non_finite_handling": "non-finite similarity (NaN/Inf) is a request error; see decision.non_finite_action",
            "implementation": "Implemented by the API on the raw, unrounded, pre-geo maximum cosine. "
                              "If this optional policy is absent or invalid, the API keeps closest-match "
                              "inference available with gate_active=false and low_confidence=null.",
        },
        "validated_environment": validated_environment,
        "not_validated_for": not_validated_for,
        "timm_pretrained_resolution": timm_info,
        "not_in_scope": "No per-species exceptions. No geo dependency.",
    }

    content_sha256 = schema.compute_content_sha256(schema.SCHEMA_VERSION, content)
    policy = {"policy_schema_version": schema.SCHEMA_VERSION, "content": content,
              "content_sha256": content_sha256,
              "generation": {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
                                             .replace("+00:00", "Z"),
                            "generator_version": GENERATOR_VERSION}}
    errors = schema.validate(policy)
    if errors:
        ev.fail("generated policy failed schema/semantic self-validation: " + "; ".join(errors))

    out = artifacts / "inference_policy.json"
    tmp = artifacts / f".inference_policy.json.tmp{os.getpid()}"
    try:
        tmp.write_text(json.dumps(policy, sort_keys=True, indent=2) + "\n", newline="\n")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()

    final_policy = json.loads(out.read_text())
    final_errors = schema.validate(final_policy)
    if final_errors:
        ev.fail(f"final written {out} failed readback self-validation: " + "; ".join(final_errors))
    if final_policy["content_sha256"] != content_sha256:
        ev.fail("final written file's content_sha256 does not match the in-memory value")

    print(f"wrote {out}")
    print(f"content_sha256: {content_sha256}")
    print(f"threshold: {value} ({operator})  onnx_bound_by: {onnx_bound_by}")
    if notes:
        print(f"evidence notes ({len(notes)}): {notes}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ev.EvidenceFailure as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
