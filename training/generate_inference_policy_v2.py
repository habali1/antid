#!/usr/bin/env python3
"""generate_inference_policy_v2.py — prepares (and, in a separately
authorized later phase, writes) a schema-v2 inference_policy.json candidate
for the independently-validated Gate v2 threshold (0.61), bound to an
explicit candidate artifact directory.

This is a SEPARATE generator from training/inference_policy_generator.py
(V1) -- it never repurposes or duplicates that generator's evidence-reading
logic, and it targets an explicit --artifacts-dir that must never be the
live training/artifacts root, so the current live V1 policy is completely
unaffected until a separately authorized atomic-promotion step (Phase 5E2).

Reads Gate v2 evidence through the ALREADY-COMMITTED shared validators
(gate_v2_contract.py, gate_v2_evaluation_contract.py,
select_gate_v2_threshold.py) rather than re-implementing their checks:
  - gate_v2_selection_contract.json   (gc.load_and_verify_contract)
  - calibration_v2_scores.json        (gc.load_and_verify_score_file)
  - calibration_v2_selection.json     (loaded + hash-verified; its `result`
                                        is mechanically RECOMPUTED via
                                        select_gate_v2_threshold.select_threshold
                                        and required to match exactly)
  - gate_v2_evaluation_contract.json  (ec.load_and_verify_evaluation_contract)
  - unknown_test_v2_eval.json AND
    unknown_test_v2_evaluation_attempt.json
                                       (ec.load_and_verify_eval_file, which
                                        fully re-derives the stored
                                        validation block from records and
                                        cross-checks the attempt marker's
                                        own byte hash -- both in one call)

Also validated here (generator-specific, not duplicated from the shared
validators above):
  - the current candidate's backbone.onnx/prototypes.npy/taxonomy.json
    hashes, bound both in the Gate v2 evaluation contract's own bindings
    AND independently re-hashed here from --artifacts-dir;
  - the ACTUAL calibration_v2_scores.json / calibration_v2_selection.json
    byte+content hashes, cross-checked against the evaluation contract's
    OWN scores_binding/selection_binding (defense in depth beyond the
    shared validators' frozen-constant checks) -- malformed JSON or a
    missing/wrong binding field fails closed with a controlled error, not
    an uncaught traceback;
  - training/reports/northeast_v1_b4_dev_v2_parity.json: hash-bound and
    checked for 260 images / 0 top-1 disagreements / 0 top-3-set
    disagreements / the exact recorded max cosine divergence -- NEVER
    rerun or rewritten, and its generation_metadata.workspace_git_dirty is
    carried through verbatim into the policy, never concealed;
  - the in-memory preprocessing parity check (reused from
    policy_evidence.py -- identical contract regardless of species count);
  - construction of a CPUExecutionProvider-exclusive ONNX Runtime session
    against --artifacts-dir/backbone.onnx;
  - git HEAD and the canonical-LF source hash of six implementation files
    (this generator, both policy_schema.py copies -- asserted
    byte-identical -- api/inference_policy.py, api/inference.py,
    training/data.py), recorded into the policy's content.provenance as
    DIAGNOSTIC provenance only: the API loader never reads this block, so
    an edit to any of these files after generation does not retroactively
    deactivate an already-active gate, but it does mean the policy's
    provenance is stale relative to the current tree and the policy should
    be regenerated before deployment. This applies equally to
    api/inference.py.

Every field of schema v2's validation_evidence block is checked for EXACT
equality against policy_schema.py's EXPECTED_V2_VALIDATION_EVIDENCE, not
merely format -- a rehashed policy that swaps in a different, individually
well-formed hash/status/FAR value in any evidence field is rejected.

Three modes:
  --preflight  validates everything above; writes ZERO bytes. Does not
               require a clean tree (a dirty tree is faithfully recorded
               via the parity report's own workspace_git_dirty flag).
  --write      (reserved; NOT run in Phase 5E1) requires a clean tracked
               tree, then pre-validates the built candidate policy through
               the REAL API loader in a TEMPORARY, ISOLATED artifact
               directory (hard-linked artifact files) -- every mismatch
               must fail THERE, before any publication is attempted. Only
               after that passes does it atomically and exclusively write
               --artifacts-dir/inference_policy.json -- a schema-v2
               CANDIDATE policy, colocated with the candidate artifacts,
               never the live training/artifacts root -- refusing to
               overwrite an existing file there. A final readback + schema
               + API-loader check against the real published file follows.
  --check      re-derives authority for a previously-written candidate
               policy: reruns the complete read-only preflight, rebuilds
               the expected deterministic policy content from it, and
               requires the stored schema version/content/content_sha256
               to match EXACTLY (only `generation` may differ) before
               verifying through the API loader. Writes nothing.

--artifacts-dir is always required and is refused outright if it resolves
to the live training/artifacts root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "api"))
sys.path.insert(0, str(HERE))  # training-local policy_schema/gate_v2_* win

import gate_v2_contract as gc  # noqa: E402
import gate_v2_evaluation_contract as ec  # noqa: E402
import select_gate_v2_threshold as sel  # noqa: E402
import policy_schema as schema  # noqa: E402
import policy_evidence as ev  # noqa: E402

# The ONE frozen source of truth for this version string lives in
# policy_schema.py (schema.V2_GENERATOR_VERSION), which enforces it via
# schema-v2's envelope check -- referenced here, never redefined, so this
# generator and that check can never silently drift apart.
GENERATOR_VERSION = schema.V2_GENERATOR_VERSION
LIVE_ARTIFACTS_ROOT = (REPO / "training" / "artifacts").resolve()

FROZEN_PARITY_REPORT_PATH = "training/reports/northeast_v1_b4_dev_v2_parity.json"
FROZEN_PARITY_REPORT_BYTE_SHA256 = "bec08235a48e9583f2b40556c9c8d860c41904729bcded9d6f91c37716c8297e"
FROZEN_PARITY_N_IMAGES = 260
FROZEN_PARITY_MAX_COSINE_DIVERGENCE = 1.9669532775878906e-06

# The provenance sources whose canonical-LF sha256 is recorded (never
# enforced as a runtime binding -- the API loader never reads this block)
# into every generated v2 policy's content.provenance. Diagnostic only: an
# edit to any of these files after generation does not retroactively
# deactivate an already-active gate, but it means the policy's recorded
# provenance is stale relative to the current tree and the policy should be
# regenerated before deployment -- this applies equally to api/inference.py,
# which is otherwise not part of the API loader's own runtime hash-binding
# (that binding is the preprocessing.contract/executable_check fields).
PROVENANCE_SOURCE_PATHS = {
    "generate_inference_policy_v2": "training/generate_inference_policy_v2.py",
    "training_policy_schema": "training/policy_schema.py",
    "api_policy_schema": "api/policy_schema.py",
    "api_inference_policy": "api/inference_policy.py",
    "api_inference": "api/inference.py",
    "training_data": "training/data.py",
}

NOT_VALIDATED_FOR = [
    "CUDA or any non-CPU execution provider",
    "retrained or re-exported artifacts (any artifact_hashes change invalidates this policy)",
    "unknown-species detection (this is a selective confidence/abstention gate, not an out-of-catalog detector)",
    "per-species thresholds (a single global threshold only)",
    "population-level claims from the 260-image parity sample or the 65 x 6-row per-species "
    "independent-evaluation table",
    "post-geo or rounded similarity scores (gate applies to raw, unrounded, pre-geo max cosine)",
]


class GeneratorV2Error(RuntimeError):
    """A source-evidence or generator problem. Always fails closed -- on
    any problem, nothing is written."""


def _fail(msg: str) -> None:
    raise GeneratorV2Error(msg)


def require_explicit_non_live_artifacts_dir(repo: Path, artifacts_dir: Path) -> Path:
    resolved = Path(artifacts_dir).resolve()
    live_root = (Path(repo) / "training" / "artifacts").resolve()
    if resolved == live_root:
        _fail(f"--artifacts-dir must not be the live serving root {live_root} -- point it at "
              f"the specific candidate directory instead "
              f"(e.g. training/artifacts/northeast_v1_b4_dev_v2)")
    return resolved


def compute_source_provenance(repo: Path) -> dict[str, str]:
    """Canonical-LF sha256 of every file in PROVENANCE_SOURCE_PATHS, keyed
    the same way. Fails closed (a controlled GeneratorV2Error, never an
    uncaught exception) if any listed file is missing. Recorded into the
    policy for informational provenance only -- NOT the synchronization
    gate below, which deliberately does not normalize line endings."""
    hashes: dict[str, str] = {}
    for name, rel in PROVENANCE_SOURCE_PATHS.items():
        path = Path(repo) / rel
        if not path.exists():
            _fail(f"provenance source file missing: {path} (binding {name!r})")
        hashes[name] = gc.canonical_lf_sha256_file(path)
    return hashes


def _require_schema_copies_byte_identical(repo: Path) -> None:
    """The required synchronization gate: training/policy_schema.py and
    api/policy_schema.py must be byte-for-byte IDENTICAL, checked via raw
    read_bytes() equality -- deliberately NOT a hash and NOT canonical-LF
    normalized. A CRLF-vs-LF-only divergence between the two copies is
    exactly the drift this must catch; canonical-LF normalization (used
    elsewhere, for provenance content-identity) would silently treat two
    copies with different line endings as equal, which is wrong here."""
    training_path = Path(repo) / "training/policy_schema.py"
    api_path = Path(repo) / "api/policy_schema.py"
    if not training_path.exists():
        _fail(f"provenance source file missing: {training_path}")
    if not api_path.exists():
        _fail(f"provenance source file missing: {api_path}")
    if training_path.read_bytes() != api_path.read_bytes():
        _fail("training/policy_schema.py and api/policy_schema.py are not byte-identical "
              "(raw byte comparison, including line endings) -- the two copies must be kept in sync")


# --------------------------------------------------------------- preflight
def run_preflight(args) -> dict[str, Any]:
    """Validates every piece of Gate v2 evidence, the candidate artifacts,
    parity, preprocessing, and the CPU-only ONNX session. Writes nothing."""
    artifacts_dir = require_explicit_non_live_artifacts_dir(args.repo, args.artifacts_dir)
    if not artifacts_dir.is_dir():
        _fail(f"--artifacts-dir does not exist or is not a directory: {artifacts_dir}")

    # ---- calibration selection contract + scores/selection, via the
    # already-committed shared validators -----------------------------
    selection_contract = gc.load_and_verify_contract(args.repo / "training/gate_v2_selection_contract.json")
    scores_path = args.repo / "data/calibration_v2/calibration_v2_scores.json"
    scores = gc.load_and_verify_score_file(scores_path, selection_contract)

    selection_path = args.repo / "data/calibration_v2/calibration_v2_selection.json"
    if not selection_path.exists():
        _fail(f"calibration_v2_selection.json does not exist: {selection_path}")
    try:
        selection_doc = json.loads(selection_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail(f"{selection_path} is not valid JSON: {exc}")
    if not isinstance(selection_doc, dict):
        _fail(f"{selection_path} is not a JSON object")

    # ---- load the frozen evaluation contract EARLY -- needed immediately
    # below to bind the selection file's BYTE identity before any of its
    # claimed content (status/result/diagnostics) is trusted enough to
    # index into. The shared validator itself fully recomputes validation
    # from records AND cross-checks the attempt marker's own byte hash. ----
    eval_contract_path = args.repo / "training/gate_v2_evaluation_contract.json"
    eval_contract = ec.load_and_verify_evaluation_contract(eval_contract_path)
    eval_content = eval_contract.get("content") if isinstance(eval_contract, dict) else None
    eval_content = eval_content if isinstance(eval_content, dict) else {}

    # ---- scores/selection BYTE+content hashes vs the evaluation
    # contract's OWN scores_binding/selection_binding -- defense in depth
    # beyond the shared score-file validator's frozen-constant check: this
    # confirms the files this run just read are the SAME files the
    # evaluation contract itself expects. The SELECTION file's check runs
    # BEFORE any status/result/diagnostics field of it is ever indexed --
    # a byte-tampered selection file is caught by identity alone, before
    # its claimed content is trusted at all. Missing/malformed binding
    # fields fail closed with a controlled error, never an uncaught
    # KeyError/TypeError. ---------------------------------------------------
    scores_binding = eval_content.get("scores_binding")
    if not isinstance(scores_binding, dict) or not gc.SHA256_HEX_RE.match(str(scores_binding.get("byte_sha256"))) \
            or not gc.SHA256_HEX_RE.match(str(scores_binding.get("content_sha256"))):
        _fail(f"gate_v2_evaluation_contract.json content.scores_binding is missing or malformed: "
              f"{scores_binding!r}")
    actual_scores_byte_hash = gc.sha256_file(scores_path)
    actual_scores_content_hash = scores.get("content_sha256") if isinstance(scores, dict) else None
    if (actual_scores_byte_hash != scores_binding["byte_sha256"]
            or actual_scores_content_hash != scores_binding["content_sha256"]):
        _fail(f"calibration_v2_scores.json hashes (byte={actual_scores_byte_hash}, "
              f"content={actual_scores_content_hash!r}) do not exactly match evaluation_contract "
              f"content.scores_binding ({scores_binding!r})")

    selection_binding = eval_content.get("selection_binding")
    if not isinstance(selection_binding, dict) \
            or not gc.SHA256_HEX_RE.match(str(selection_binding.get("byte_sha256"))) \
            or not gc.SHA256_HEX_RE.match(str(selection_binding.get("content_sha256"))):
        _fail(f"gate_v2_evaluation_contract.json content.selection_binding is missing or malformed: "
              f"{selection_binding!r}")
    actual_selection_byte_hash = gc.sha256_file(selection_path)
    if actual_selection_byte_hash != selection_binding["byte_sha256"]:
        _fail(f"calibration_v2_selection.json byte sha256 {actual_selection_byte_hash} != "
              f"evaluation_contract content.selection_binding.byte_sha256 "
              f"{selection_binding['byte_sha256']!r}")
    actual_selection_content_hash = selection_doc.get("content_sha256")
    if actual_selection_content_hash != selection_binding["content_sha256"]:
        _fail(f"calibration_v2_selection.json content_sha256 {actual_selection_content_hash!r} != "
              f"evaluation_contract content.selection_binding.content_sha256 "
              f"{selection_binding['content_sha256']!r}")

    # ---- ONLY NOW: validate the required selection fields/types, entirely
    # via .get() -- no direct indexing that could raise KeyError on
    # malformed input. Missing/malformed status, result, diagnostics,
    # schema_version, content_sha256, or generation each produce a
    # controlled GeneratorV2Error, never a traceback. -----------------------
    if not gc.is_strict_int(selection_doc.get("schema_version")):
        _fail(f"{selection_path} schema_version is missing or not a strict int: "
              f"{selection_doc.get('schema_version')!r}")
    if not gc.is_valid_sha256_hex(selection_doc.get("content_sha256")):
        _fail(f"{selection_path} content_sha256 is missing or not a 64-char lowercase hex string")
    if not isinstance(selection_doc.get("generation"), dict):
        _fail(f"{selection_path} generation is missing or not a JSON object")
    selection_content = selection_doc.get("content")
    if not isinstance(selection_content, dict):
        _fail(f"{selection_path} content is missing or not a JSON object")
    if not isinstance(selection_content.get("status"), str):
        _fail(f"{selection_path} content.status is missing or not a string")
    if not isinstance(selection_content.get("result"), dict):
        _fail(f"{selection_path} content.result is missing or not a JSON object")
    if not isinstance(selection_content.get("diagnostics"), dict):
        _fail(f"{selection_path} content.diagnostics is missing or not a JSON object")

    # ---- content_sha256 self-consistency: retained as defense in depth,
    # even though the binding checks above already bound this exact file
    # by byte identity. ------------------------------------------------------
    recomputed_selection_hash = gc.compute_content_sha256(selection_content)
    if selection_doc["content_sha256"] != recomputed_selection_hash:
        _fail(f"{selection_path} content_sha256 does not match its own recomputed hash -- "
              f"the file has been altered since it was written")
    if selection_content["status"] != gc.STATUS_CANDIDATE_SELECTED:
        _fail(f"calibration_v2_selection.json status must be {gc.STATUS_CANDIDATE_SELECTED!r}, "
              f"got {selection_content['status']!r}")

    # ---- mechanically recompute the selection; require exact agreement --
    recomputed_result = sel.select_threshold(scores, selection_contract)
    stored_result = selection_content["result"]
    recomputed_no_diag = {k: v for k, v in recomputed_result.items() if k != "diagnostics"}
    if recomputed_no_diag != stored_result:
        _fail("recomputed Gate v2 selection result does not exactly match the stored "
              "calibration_v2_selection.json result")
    if recomputed_result["diagnostics"] != selection_content["diagnostics"]:
        _fail("recomputed Gate v2 selection diagnostics do not exactly match the stored "
              "calibration_v2_selection.json diagnostics")

    candidate = stored_result.get("candidate", {})
    threshold_integer, threshold = candidate.get("threshold_integer"), candidate.get("threshold")
    # Compared against the ONE frozen source of truth for this threshold
    # (gate_v2_evaluation_contract.py's own constants, which are themselves
    # bound to the frozen evaluation contract) -- never a literal
    # re-hardcoded here, so there is nothing for this module and the
    # evaluation contract to silently drift apart on.
    if threshold_integer != ec.FROZEN_THRESHOLD_INTEGER or threshold != ec.FROZEN_THRESHOLD:
        _fail(f"selected threshold must be exactly {ec.FROZEN_THRESHOLD_INTEGER!r} / "
              f"{ec.FROZEN_THRESHOLD!r}, got {threshold_integer!r} / {threshold!r}")
    if (candidate.get("comparison") != "strict_less_than"
            or candidate.get("equal_threshold_action") != "normal_results"):
        _fail("selected decision shape is not strict_less_than with equality accepted")

    eval_out_path = args.repo / "data/unknown_test_v2/unknown_test_v2_eval.json"
    if not eval_out_path.exists():
        _fail(f"independent evaluation result does not exist yet: {eval_out_path} -- "
              f"Gate v2 has not been independently validated")
    evaluation = ec.load_and_verify_eval_file(eval_out_path, eval_contract, args.repo)

    validation = evaluation["content"]["validation"]
    if validation["status"] != ec.EVAL_STATUS_PASSED:
        _fail(f"independent evaluation status must be exactly {ec.EVAL_STATUS_PASSED!r}, got "
              f"{validation['status']!r} -- refusing to prepare a policy for a candidate that "
              f"did not pass independent validation")
    if validation["threshold_integer"] != ec.FROZEN_THRESHOLD_INTEGER or validation["threshold"] != ec.FROZEN_THRESHOLD:
        _fail(f"independent evaluation was not run against threshold "
              f"{ec.FROZEN_THRESHOLD_INTEGER!r} / {ec.FROZEN_THRESHOLD!r}")

    diagnostic_far = validation["metrics"]["diagnostic_ood_far"]["out_of_scope_ant"]["false_acceptance_rate"]
    marker_path = args.repo / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"

    # ---- candidate artifact hashes: bound in the eval contract AND
    # independently re-hashed from the ACTUAL --artifacts-dir --------------
    gc.verify_artifact_directory_matches_contract(args.repo, eval_contract, artifacts_dir)
    current_hashes = {
        "backbone.onnx": gc.sha256_file(artifacts_dir / "backbone.onnx"),
        "prototypes.npy": gc.sha256_file(artifacts_dir / "prototypes.npy"),
        "taxonomy.json": gc.sha256_file(artifacts_dir / "taxonomy.json"),
    }

    # ---- parity report: validated and bound, never rerun/rewritten -------
    parity_path = args.repo / FROZEN_PARITY_REPORT_PATH
    if not parity_path.exists():
        _fail(f"parity report does not exist: {parity_path}")
    parity_bytes = parity_path.read_bytes()
    parity_byte_hash = gc.sha256_bytes(parity_bytes)
    if parity_byte_hash != FROZEN_PARITY_REPORT_BYTE_SHA256:
        _fail(f"{parity_path} byte sha256 {parity_byte_hash} != frozen "
              f"{FROZEN_PARITY_REPORT_BYTE_SHA256}")
    try:
        parity = json.loads(parity_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail(f"{parity_path} is not valid JSON: {exc}")
    if not isinstance(parity, dict):
        _fail(f"{parity_path} is not a JSON object")
    dc = parity.get("deterministic_content", {})
    parity_candidate_hashes = dc.get("candidate", {}).get("final_artifact_hashes", {})
    for name in ("backbone.onnx", "prototypes.npy", "taxonomy.json"):
        if parity_candidate_hashes.get(name) != current_hashes[name]:
            _fail(f"parity report's candidate.final_artifact_hashes[{name!r}] does not match "
                  f"the current candidate artifact hash")
    model_parity = dc.get("model_parity", {})
    per_image = model_parity.get("per_image", [])
    if model_parity.get("n_images") != FROZEN_PARITY_N_IMAGES or len(per_image) != FROZEN_PARITY_N_IMAGES:
        _fail(f"parity report n_images must be exactly {FROZEN_PARITY_N_IMAGES}")
    top1_dis = sum(1 for r in per_image if not r.get("top1_agree"))
    top3_dis = sum(1 for r in per_image if not r.get("top3_set_agree"))
    if top1_dis != 0:
        _fail(f"parity report has {top1_dis} top-1 disagreement(s), expected 0")
    if top3_dis != 0:
        _fail(f"parity report has {top3_dis} top-3-set disagreement(s), expected 0")
    max_div = model_parity.get("max_cosine_abs_divergence", {}).get("max")
    if max_div != FROZEN_PARITY_MAX_COSINE_DIVERGENCE:
        _fail(f"parity report max_cosine_abs_divergence.max {max_div!r} != frozen "
              f"{FROZEN_PARITY_MAX_COSINE_DIVERGENCE!r}")
    # Recorded FAITHFULLY -- never concealed or reinterpreted, per instructions.
    workspace_git_dirty = parity.get("generation_metadata", {}).get("workspace_git_dirty")

    # ---- re-run the in-memory preprocessing parity check ------------------
    exec_check = ev.preprocessing_parity_check(HERE)

    # ---- construct and verify a CPU-exclusive ONNX session ----------------
    import onnxruntime as ort
    sess = ort.InferenceSession(str(artifacts_dir / "backbone.onnx"), providers=["CPUExecutionProvider"])
    if sess.get_providers() != ["CPUExecutionProvider"]:
        _fail(f"CPUExecutionProvider was not exclusively selected: {sess.get_providers()}")

    # ---- provenance: git HEAD + canonical source hashes --------------------
    # Read-only, never requires a clean tree (that gate is --write-only, in
    # cmd_write) -- --preflight and --check both need to be able to report/
    # reproduce this even against a dirty tree. The two policy_schema.py
    # copies must be byte-identical wherever this runs -- checked via raw
    # bytes (see _require_schema_copies_byte_identical), not the
    # canonical-LF hash used for provenance recording below.
    git_head = gc.get_git_head(args.repo)
    _require_schema_copies_byte_identical(args.repo)
    source_hashes = compute_source_provenance(args.repo)

    return {
        "ok": True,
        "selection_contract": selection_contract, "scores": scores, "selection_doc": selection_doc,
        "eval_contract": eval_contract, "evaluation": evaluation,
        "artifacts_dir": artifacts_dir, "current_hashes": current_hashes,
        "parity": parity, "parity_byte_hash": parity_byte_hash,
        "workspace_git_dirty": workspace_git_dirty, "diagnostic_far": diagnostic_far,
        "preprocessing_executable_check": exec_check, "onnxruntime_providers": sess.get_providers(),
        "scores_path": scores_path, "selection_path": selection_path,
        "eval_out_path": eval_out_path, "marker_path": marker_path,
        "repo": args.repo, "git_head": git_head, "source_hashes": source_hashes,
    }


# ------------------------------------------------------------- build content
def build_policy(pre: dict[str, Any]) -> tuple[dict, str]:
    """Pure function over an already-validated preflight result: no
    filesystem access, no network. Deterministic given `pre`: only
    `generation.generated_at` varies run-to-run, and that key is
    deliberately excluded from content_sha256 (compute_content_sha256 only
    ever hashes {policy_schema_version, content})."""
    decision = {
        "low_confidence_if": {"signal": schema.ALLOWED_DECISION_SIGNAL, "comparison": "strict_less_than",
                              "threshold": schema.FROZEN_THRESHOLD_V2},
        "normal_results_if": {"comparison": "greater_than_or_equal", "threshold": schema.FROZEN_THRESHOLD_V2},
        "equal_threshold_action": "normal_results",
        "non_finite_action": "request_error",
    }
    enc_errors = schema.decision_encodings_agree(decision)
    if enc_errors:
        _fail("generator built an internally inconsistent decision block: " + "; ".join(enc_errors))
    for probe, expected in schema.SCHEMA_VERSION_TO_PROBES[schema.SCHEMA_VERSION_V2]:
        if schema.should_abstain(probe, decision) != expected:
            _fail(f"decision boundary self-check failed for should_abstain({probe})")

    validation_evidence = {
        "gate_v2_selection_contract_content_sha256": pre["selection_contract"]["content_sha256"],
        "calibration_v2_scores_byte_sha256": gc.sha256_file(pre["scores_path"]),
        "calibration_v2_scores_content_sha256": pre["scores"]["content_sha256"],
        "calibration_v2_selection_byte_sha256": gc.sha256_file(pre["selection_path"]),
        "calibration_v2_selection_content_sha256": pre["selection_doc"]["content_sha256"],
        "gate_v2_evaluation_contract_content_sha256": pre["eval_contract"]["content_sha256"],
        "unknown_test_v2_evaluation_attempt_byte_sha256": gc.sha256_file(pre["marker_path"]),
        "unknown_test_v2_eval_byte_sha256": gc.sha256_file(pre["eval_out_path"]),
        "unknown_test_v2_eval_content_sha256": pre["evaluation"]["content_sha256"],
        "parity_report_byte_sha256": pre["parity_byte_hash"],
        "validation_status": pre["evaluation"]["content"]["validation"]["status"],
        "diagnostic_out_of_scope_ant_far": pre["diagnostic_far"],
    }
    missing = schema.REQUIRED_V2_VALIDATION_EVIDENCE_KEYS - set(validation_evidence)
    if missing:
        _fail(f"generator built an incomplete validation_evidence block, missing: {sorted(missing)}")

    content = {
        "rule": {
            "operator_verbatim": schema.ALLOWED_OPERATOR,
            "operator_normalized": dict(schema.ALLOWED_OPERATOR_NORMALIZED),
            "value": schema.FROZEN_THRESHOLD_V2, "signal": schema.ALLOWED_RULE_SIGNAL,
            "evaluated": "raw pre-geo max cosine similarity, strict <, before any rounding",
        },
        "decision": decision,
        "artifact_hashes": dict(pre["current_hashes"]),
        "provider_policy": {"providers": ["CPUExecutionProvider"], "exclusive": True},
        "preprocessing": {
            "contract": dict(ev.PREPROCESSING_CONTRACT),
            "executable_check": pre["preprocessing_executable_check"],
        },
        "validation_evidence": validation_evidence,
        "gate_framing": "independently validated selective confidence/abstention gate; "
                        "NOT an unknown-species detector",
        "parity_report": {
            "path": FROZEN_PARITY_REPORT_PATH, "byte_sha256": pre["parity_byte_hash"],
            "n_images": FROZEN_PARITY_N_IMAGES, "top1_disagreements": 0, "top3_set_disagreements": 0,
            "max_cosine_abs_divergence": FROZEN_PARITY_MAX_COSINE_DIVERGENCE,
            "generation_metadata_workspace_git_dirty": pre["workspace_git_dirty"],
            "note": "Parity evidence generation recorded workspace_git_dirty="
                    f"{pre['workspace_git_dirty']!r} -- carried through here verbatim, never "
                    "concealed or reinterpreted. Parity is validated and bound, never rerun or "
                    "rewritten by this generator.",
        },
        "not_validated_for": list(NOT_VALIDATED_FOR),
        "not_in_scope": "No per-species exceptions. No geo dependency.",
        "provenance": {
            "git_head": pre["git_head"],
            "source_hashes": dict(pre["source_hashes"]),
            "source_hash_algorithm": "sha256 of canonical-LF-normalized (CRLF->LF) file bytes",
            "note": "Diagnostic provenance only -- the API loader never reads or checks this block, "
                    "so an edit to any of these files after this policy was generated does not "
                    "retroactively deactivate an already-active gate. It DOES mean this policy's "
                    "recorded provenance is stale relative to the current tree and the policy should "
                    "be regenerated (via --write) before deployment. This applies equally to "
                    "api/inference.py, whose only RUNTIME binding is the separate "
                    "preprocessing.contract/executable_check fields above, not this hash.",
        },
    }

    content_sha256 = schema.compute_content_sha256(schema.SCHEMA_VERSION_V2, content)
    policy = {
        "policy_schema_version": schema.SCHEMA_VERSION_V2, "content": content,
        "content_sha256": content_sha256,
        "generation": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "generator_version": GENERATOR_VERSION,
        },
    }
    errors = schema.validate(policy)
    if errors:
        _fail("generated policy failed schema/semantic self-validation: " + "; ".join(errors))
    return policy, content_sha256


# ------------------------------------------------------------ write / check
def publish_policy_v2(dest_path: Path, text: str) -> None:
    """Atomically AND exclusively publishes the candidate policy -- mirrors
    eval_unknown_test_v2.publish_eval_output(): write a unique temp file in
    the SAME directory, fsync it, then os.link() it to the destination
    (which fails with FileExistsError if the destination already exists),
    and always remove the temp file afterward. An existing different (or
    even byte-identical) policy is never silently overwritten -- os.replace
    is never used here."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + f".tmp{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        try:
            os.link(str(tmp_path), str(dest_path))
        except FileExistsError:
            _fail(f"refusing to overwrite existing candidate policy: {dest_path}")
        except OSError as exc:
            import errno
            if exc.errno == errno.EEXIST:
                _fail(f"refusing to overwrite existing candidate policy: {dest_path}")
            raise
    finally:
        tmp_path.unlink(missing_ok=True)


def _prevalidate_via_isolated_api_loader(pre: dict[str, Any], text: str) -> None:
    """Pre-validates a candidate policy's TEXT through the real API loader
    in a TEMPORARY, ISOLATED artifact directory (never the real candidate
    directory) containing hard links to the three matching artifact files.
    Every mismatch must surface here, BEFORE any publication is attempted --
    a failure here writes nothing to the real candidate directory and
    leaves no temp files behind (the TemporaryDirectory context manager
    removes them even on exception)."""
    import inference_policy as api_loader
    import inference as api_inference
    with tempfile.TemporaryDirectory(prefix="policy_v2_preverify_") as tmp:
        tmp_dir = Path(tmp)
        for name in ("backbone.onnx", "prototypes.npy", "taxonomy.json"):
            os.link(str(pre["artifacts_dir"] / name), str(tmp_dir / name))
        (tmp_dir / "inference_policy.json").write_text(text, encoding="utf-8")
        state = api_loader.load_inference_policy(tmp_dir, pre["onnxruntime_providers"],
                                                  api_inference.PREPROCESSING_CONTRACT)
    if not state.active or state.reason != "active" or state.threshold != schema.FROZEN_THRESHOLD_V2:
        _fail(f"candidate policy failed API-loader pre-validation in an isolated temporary directory "
              f"(nothing was published to {pre['artifacts_dir']}): "
              f"active={state.active} reason={state.reason} threshold={state.threshold}")


def cmd_write(args) -> dict[str, Any]:
    """Reserved for a separately authorized phase. Writes
    --artifacts-dir/inference_policy.json (a candidate colocated with the
    candidate artifacts -- never the live training/artifacts root).

    Order of operations, each step a hard gate on the next: (1) require a
    clean tracked tree -- write-time only, --preflight/--check stay usable
    on a dirty tree; (2) run the complete preflight and build the
    deterministic policy content (embeds git HEAD + source provenance);
    (3) pre-validate the built policy through the REAL API loader in an
    isolated temp directory -- every mismatch fails HERE, before any
    publication; (4) only then, the atomic-exclusive final publish; (5) a
    final readback + schema + loader check against the REAL published file,
    as a last confirmation."""
    gc.require_clean_tracked_tree(args.repo)
    pre = run_preflight(args)
    policy, content_sha256 = build_policy(pre)
    text = json.dumps(policy, sort_keys=True, indent=2) + "\n"

    _prevalidate_via_isolated_api_loader(pre, text)

    dest = pre["artifacts_dir"] / "inference_policy.json"
    publish_policy_v2(dest, text)

    written = json.loads(dest.read_text(encoding="utf-8"))
    errors = schema.validate(written)
    if errors:
        _fail(f"just-written {dest} failed schema self-validation on readback: {errors}")
    if written["content_sha256"] != content_sha256:
        _fail(f"just-written {dest} content_sha256 does not match the in-memory value")

    import inference_policy as api_loader
    import inference as api_inference
    state = api_loader.load_inference_policy(pre["artifacts_dir"], pre["onnxruntime_providers"],
                                             api_inference.PREPROCESSING_CONTRACT)
    if not state.active or state.reason != "active" or state.threshold != schema.FROZEN_THRESHOLD_V2:
        _fail(f"just-written {dest} did not load ACTIVE through the API loader: "
              f"active={state.active} reason={state.reason} threshold={state.threshold}")
    return {"dest": dest, "content_sha256": content_sha256, "loader_state": state}


def cmd_check(args) -> dict[str, Any]:
    """Re-derives authority for a previously-written candidate policy
    rather than trusting the file's own claims about itself: reruns the
    COMPLETE read-only preflight, rebuilds the expected deterministic
    policy content from that fresh preflight result, and requires the
    stored schema version, content (gate_framing, parity facts,
    not_validated_for, validation_evidence, provenance -- every field), and
    content_sha256 to match the freshly rebuilt values EXACTLY. Only
    `generation` (timestamps, generator_version) is permitted to differ.
    Only after that structural/content match succeeds does it verify
    through the API loader. Writes nothing."""
    artifacts_dir = require_explicit_non_live_artifacts_dir(args.repo, args.artifacts_dir)
    dest = artifacts_dir / "inference_policy.json"
    if not dest.exists():
        _fail(f"no candidate policy to check: {dest}")
    try:
        stored = json.loads(dest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail(f"{dest} is not valid JSON: {exc}")
    if not isinstance(stored, dict):
        _fail(f"{dest} is not a JSON object")

    pre = run_preflight(args)
    rebuilt_policy, rebuilt_content_sha256 = build_policy(pre)

    # ---- top-level envelope: exact key set. Extra/missing top-level keys
    # must fail even though they would never change content_sha256 (which
    # only ever covers {policy_schema_version, content}). --------------------
    if set(stored) != set(rebuilt_policy):
        _fail(f"{dest} top-level key set {sorted(stored)} != freshly rebuilt "
              f"{sorted(rebuilt_policy)}")

    if stored.get("policy_schema_version") != rebuilt_policy["policy_schema_version"]:
        _fail(f"{dest} policy_schema_version {stored.get('policy_schema_version')!r} != freshly "
              f"rebuilt {rebuilt_policy['policy_schema_version']!r}")
    if stored.get("content") != rebuilt_policy["content"]:
        _fail(f"{dest} content does not exactly match the freshly rebuilt deterministic policy "
              f"content -- gate_framing, a parity fact, not_validated_for, validation_evidence, "
              f"provenance, or some other content field has changed or been altered")
    if stored.get("content_sha256") != rebuilt_content_sha256:
        _fail(f"{dest} content_sha256 {stored.get('content_sha256')!r} != freshly recomputed "
              f"{rebuilt_content_sha256!r}")

    # ---- generation: excluded from content_sha256, so checked separately.
    # Only generated_at's timestamp VALUE is permitted to differ -- the key
    # set and generator_version must match the freshly rebuilt policy
    # exactly, even though a key-set/version change alone never touches
    # content_sha256. ----------------------------------------------------
    stored_generation = stored.get("generation")
    rebuilt_generation = rebuilt_policy["generation"]
    if not isinstance(stored_generation, dict):
        _fail(f"{dest} generation is not a JSON object")
    if set(stored_generation) != set(rebuilt_generation):
        _fail(f"{dest} generation key set {sorted(stored_generation)} != freshly rebuilt "
              f"{sorted(rebuilt_generation)} -- extra or missing generation keys are never permitted")
    if stored_generation.get("generator_version") != rebuilt_generation["generator_version"]:
        _fail(f"{dest} generation.generator_version {stored_generation.get('generator_version')!r} != "
              f"freshly rebuilt {rebuilt_generation['generator_version']!r}")

    errors = schema.validate(stored)
    if errors:
        _fail(f"{dest} failed schema validation: {errors}")

    import inference_policy as api_loader
    import inference as api_inference
    state = api_loader.load_inference_policy(artifacts_dir, pre["onnxruntime_providers"],
                                             api_inference.PREPROCESSING_CONTRACT)
    if not state.active:
        _fail(f"{dest} did not load ACTIVE through the API loader: reason={state.reason}")
    return {"dest": dest, "content_sha256": stored["content_sha256"], "loader_state": state}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--artifacts-dir", type=Path, required=True,
                    help="Explicit candidate artifact directory. Never the live training/artifacts root.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        if args.preflight:
            result = run_preflight(args)
            skip = ("selection_contract", "scores", "selection_doc", "eval_contract", "evaluation", "parity")
            print(json.dumps({k: v for k, v in result.items() if k not in skip}, indent=2, default=str))
        elif args.write:
            result = cmd_write(args)
            print(f"wrote {result['dest']}")
            print(f"content_sha256: {result['content_sha256']}")
        else:
            result = cmd_check(args)
            print(f"{result['dest']} PASS")
            print(f"content_sha256: {result['content_sha256']}")
        return 0
    except (GeneratorV2Error, gc.ContractError, ec.EvaluationError) as e:
        print(f"[generate_inference_policy_v2] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
