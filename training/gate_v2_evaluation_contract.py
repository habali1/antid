#!/usr/bin/env python3
"""gate_v2_evaluation_contract.py — the single shared schema/arithmetic
implementation for the Gate v2 unknown_test_v2 evaluation contract (Phase
5D1).

This is the SINGLE-USE, INDEPENDENT-TEST counterpart to gate_v2_contract.py's
threshold-SELECTION contract. It never re-selects a threshold: it loads the
one already-frozen and already-reviewed candidate (threshold 0.61, from
data/calibration_v2/calibration_v2_selection.json) and defines the schema,
hash-bindings, and precommitted pass/fail arithmetic for applying that
threshold, EXACTLY ONCE, to the independent unknown_test_v2 set.

gate_v2_contract.py is reused here for every helper that isn't specific to
evaluation (hashing, strict-int/finite-number checks, git/tree helpers,
identity-order hashing, exact-fraction arithmetic) -- there is no reason to
reimplement any of that, and gate_v2_contract.py is itself one of this
contract's bound implementation sources.

Nothing in this module touches an image, a model, or the network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import gate_v2_contract as gc
import select_gate_v2_threshold as sel

EVAL_CONTRACT_SCHEMA_VERSION = 1
EVAL_CONTRACT_STATUS_FROZEN = "frozen_before_unknown_test_v2_evaluation"
EVAL_OUTPUT_SCHEMA_VERSION = 1
ATTEMPT_MARKER_SCHEMA_VERSION = 1

EVAL_STATUS_PASSED = "validation_passed"
EVAL_STATUS_FAILED = "validation_failed"

DATASET_NAME_EVAL = "unknown_test_v2"
KNOWN_CATEGORY = gc.KNOWN_CATEGORY
OOD_CATEGORIES = gc.OOD_CATEGORIES
COSINE_TOLERANCE = gc.COSINE_TOLERANCE

SHA256_HEX_RE = gc.SHA256_HEX_RE
ContractError = gc.ContractError


class EvaluationError(RuntimeError):
    """An evaluation-contract, hash-binding, or eval-artifact validation
    problem. Always fails closed."""


# ---------------------------------------------------------- frozen semantics
# Every value below is the ONE frozen definition of a piece of the approved
# unknown_test_v2 evaluation contract. freeze_gate_v2_evaluation_contract.py
# reads these rather than duplicating literals, and validate_evaluation_
# contract() checks the loaded contract against these by exact equality.
FROZEN_POLICY_NAME = "gate_v2_unknown_test_v2_evaluation_v1"

# The frozen, already-reviewed candidate from Phase 5C2 -- loaded, never
# re-derived, re-swept, or overridden. See docs/plans/gate-v2-threshold-
# selection.md for the full calibration_v2 result this comes from.
FROZEN_SELECTION_CONTRACT_CONTENT_SHA256 = (
    "991f7a0b8e83654e45575566eb0648ddcd69866e29c94b8a2030cc6f4bc19f77"
)
FROZEN_SCORES_BYTE_SHA256 = "4fec18a938ef22e06d6073016d1692512e8fb1e4e41ec645d9aeb33afbebe46b"
FROZEN_SCORES_CONTENT_SHA256 = "35c7bf36042470dde8fc922106c6528fcb6a06e3bac454c4d196880796e778f4"
FROZEN_SELECTION_BYTE_SHA256 = "00e56d2e64941086ee1d1c663890cb95d3daa4dcce1e729ef972879376f1489b"
FROZEN_SELECTION_CONTENT_SHA256 = "82bc754dbf46643862504916f4c273922e9ac3cefc45bfa28e5eb4a15dfdc5b5"

FROZEN_THRESHOLD_INTEGER = 61
FROZEN_THRESHOLD = 0.61
FROZEN_THRESHOLD_SOURCE = "data/calibration_v2/calibration_v2_selection.json"
FROZEN_THRESHOLD_STATUS_REQUIRED = gc.STATUS_CANDIDATE_SELECTED

# Candidate artifacts -- the SAME northeast_v1_b4_dev_v2 candidate already
# bound in the selection contract. Rebinding here (rather than only trusting
# the selection contract's own bindings) means this evaluation contract is
# independently verifiable even in isolation.
FROZEN_CANDIDATE_ARTIFACT_PATHS = {
    "candidate_run_manifest": "training/artifacts/northeast_v1_b4_dev_v2/run_manifest.json",
    "backbone_onnx": "training/artifacts/northeast_v1_b4_dev_v2/backbone.onnx",
    "prototypes_npy": "training/artifacts/northeast_v1_b4_dev_v2/prototypes.npy",
    "taxonomy_json": "training/artifacts/northeast_v1_b4_dev_v2/taxonomy.json",
}
FROZEN_CANDIDATE_ARTIFACT_HASHES = {
    "candidate_run_manifest": "6f8bfe3141a9870c37f6810da6b9a5459cf701242f9cbb3d32a8b01c2f0f265b",
    "backbone_onnx": "fc22d26ae5c73d20613dafcef02e75291c8779ec8a1296ebc9a72f0e7d7f826b",
    "prototypes_npy": "0e52a7f350996a48f4129e1924e9b0af79517e4c3273dd9ecc7e851121d43825",
    "taxonomy_json": "c8672287d59b5b9d3fb9aec5fd85a008ad30441f2ce3159713a3752b37a04767",
}

FROZEN_UNKNOWN_TEST_V2_BINDING_PATHS = {
    "unknown_test_v2_csv": "data/unknown_test_v2/unknown_test_v2.csv",
    "unknown_test_v2_json": "data/unknown_test_v2/unknown_test_v2.json",
}
FROZEN_UNKNOWN_TEST_V2_HASHES = {
    "unknown_test_v2_csv": "d706cb6dbda7a672594fc425ce5c8f8f0f6f868662c3d52c5f9c88083163b09f",
    "unknown_test_v2_json": "eea1cbb3118dfc05f42f477ef58d6ccbc5458df1e8a5496845a146d161a89e6d",
}

REQUIRED_BINDING_KEYS = set(FROZEN_CANDIDATE_ARTIFACT_PATHS) | set(FROZEN_UNKNOWN_TEST_V2_BINDING_PATHS)
FROZEN_BINDING_PATHS = {**FROZEN_CANDIDATE_ARTIFACT_PATHS, **FROZEN_UNKNOWN_TEST_V2_BINDING_PATHS}
FROZEN_BINDING_HASHES = {**FROZEN_CANDIDATE_ARTIFACT_HASHES, **FROZEN_UNKNOWN_TEST_V2_HASHES}

# Exact quotas -- identical numbers already frozen in the SELECTION contract
# (gc.FROZEN_DATASET_QUOTAS["unknown_test_v2"]), reused rather than
# retyped, so the two contracts can never silently disagree about them.
FROZEN_DATASET_QUOTA = dict(gc.FROZEN_DATASET_QUOTAS[DATASET_NAME_EVAL])

# Every implementation source this evaluator imports/consumes -- including
# itself and this contract module. score_calibration_v2.py and
# select_gate_v2_threshold.py are reused for pure image-decode/ONNX-session
# and grid-selection-recomputation helpers respectively; api/inference.py
# for preprocessing; gate_v2_contract.py for every shared primitive.
IMPLEMENTATION_SOURCE_KEYS = {
    "api_inference", "gate_v2_contract", "score_calibration_v2", "select_gate_v2_threshold",
    "gate_v2_evaluation_contract", "eval_unknown_test_v2",
}
IMPLEMENTATION_SOURCE_PATHS = {
    "api_inference": "api/inference.py",
    "gate_v2_contract": "training/gate_v2_contract.py",
    "score_calibration_v2": "training/score_calibration_v2.py",
    "select_gate_v2_threshold": "training/select_gate_v2_threshold.py",
    "gate_v2_evaluation_contract": "training/gate_v2_evaluation_contract.py",
    "eval_unknown_test_v2": "training/eval_unknown_test_v2.py",
}

FORBIDDEN_OUTPUT_PREFIXES = ("training/artifacts/", "api/", "mobile/")
FROZEN_APPROVED_OUTPUTS = {
    "unknown_test_v2_evaluation_attempt": "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json",
    "unknown_test_v2_eval": "data/unknown_test_v2/unknown_test_v2_eval.json",
}
REQUIRED_APPROVED_OUTPUT_KEYS = set(FROZEN_APPROVED_OUTPUTS)

FROZEN_RUNTIME = dict(gc.FROZEN_RUNTIME)
FROZEN_DECISION_SHAPE = dict(gc.FROZEN_DECISION_SHAPE)

# The three precommitted conditions -- see module docstring / freeze script
# for the full arithmetic. Conditions 1-2 are the SAME algorithm as the
# selection contract's feasibility/usefulness rules, applied at the ONE
# frozen threshold instead of swept over a grid. Condition 3 is a health
# check only, never a performance floor.
FROZEN_VALIDATION_CRITERIA = {
    "coverage_floor_numerator": 65, "coverage_floor_denominator": 100,
    "usefulness_floor_numerator": 5, "usefulness_floor_denominator": 100,
    "coverage_rule": "100 * accepted_known >= 65 * total_known (exact integer comparison), "
                     "at the single frozen threshold -- never swept or re-searched",
    "usefulness_rule": "accepted_accuracy must improve on baseline_accuracy by at least 5/100, "
                       "checked by exact integer cross multiplication; equality at exactly 5.0 "
                       "percentage points passes (>=, not >)",
    "health_check_rule": "incorrect_prediction_rejection_rate must be STRICTLY GREATER than "
                         "correct_prediction_rejection_rate (exact integer cross multiplication); "
                         "this confirms only that the gate is not operating backwards -- passing it "
                         "does not itself establish strong gate performance",
    "pass_requires_all_three": True,
    "diagnostic_only_categories": list(OOD_CATEGORIES),
    "ood_never_affects_validation_status": True,
}

FROZEN_SINGLE_USE_RULE = {
    "unknown_test_v2": "exactly_one_evaluation_ever",
    "retuning_after_evaluation": "forbidden",
    "authorization_required_before_evaluation": "this frozen evaluation contract plus the reviewed "
                                                 "Phase 5C2 calibration_v2 candidate threshold",
}
FROZEN_CLOSED_SOURCES = list(gc.FROZEN_CLOSED_SOURCES)
FROZEN_GATE_FRAMING = gc.FROZEN_GATE_FRAMING

CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256 = "unknown_test_v2_row_identity_order_sha256"

REQUIRED_TOP_KEYS = {"schema_version", "status", "content", "content_sha256"}
OPTIONAL_TOP_KEYS = {"generation"}
GENERATION_ALLOWED_KEYS = {"note", "generator"}

REQUIRED_CONTENT_KEYS = {
    "policy_name", "selection_contract", "scores_binding", "selection_binding", "threshold",
    "bindings", "implementation_sources", "approved_outputs", "dataset_quota", "runtime",
    "decision_shape", "validation_criteria", "single_use_rule", "closed_sources", "gate_framing",
    CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256,
}
OPTIONAL_CONTENT_KEYS = {"notes"}

REQUIRED_THRESHOLD_KEYS = {"threshold_integer", "threshold", "source", "comparison",
                          "equal_threshold_action", "required_status"}
REQUIRED_SCORES_BINDING_KEYS = {"byte_sha256", "content_sha256"}
REQUIRED_SELECTION_BINDING_KEYS = {"byte_sha256", "content_sha256"}


def compute_content_sha256(content: dict) -> str:
    return gc.compute_content_sha256(content)


def is_strict_int(value: Any) -> bool:
    return gc.is_strict_int(value)


def is_finite_number(value: Any) -> bool:
    return gc.is_finite_number(value)


def is_valid_sha256_hex(value: Any) -> bool:
    return gc.is_valid_sha256_hex(value)


def is_safe_relative_path(value: Any) -> bool:
    return gc.is_safe_relative_path(value)


def sha256_bytes(data: bytes) -> str:
    return gc.sha256_bytes(data)


def sha256_file(path: Path) -> str:
    return gc.sha256_file(path)


def canonical_lf_sha256_file(path: Path) -> str:
    return gc.canonical_lf_sha256_file(path)


def compute_identity_order_sha256(identity_triples: list[tuple[str, str, str]]) -> str:
    return gc.compute_identity_order_sha256(identity_triples)


def get_git_head(repo_root: Path) -> str:
    return gc.get_git_head(repo_root)


def require_clean_tracked_tree(repo_root: Path) -> None:
    gc.require_clean_tracked_tree(repo_root)


# --------------------------------------------------------- contract loading
def verify_implementation_sources(repo_root: Path, contract: dict) -> dict[str, str]:
    """Recomputes the canonical-LF hash of every bound implementation
    source for THIS evaluation contract (the eval-specific key set, not
    gate_v2_contract.py's IMPLEMENTATION_SOURCE_KEYS)."""
    bound = contract["content"].get("implementation_sources", {})
    actual: dict[str, str] = {}
    problems: list[str] = []
    for name in IMPLEMENTATION_SOURCE_KEYS:
        entry = bound.get(name)
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            problems.append(f"implementation_sources.{name} missing/malformed in contract")
            continue
        path = Path(repo_root) / entry["path"]
        if not path.exists():
            problems.append(f"implementation source missing: {path} (binding {name!r})")
            continue
        actual_hash = canonical_lf_sha256_file(path)
        actual[name] = actual_hash
        if actual_hash != entry["sha256"]:
            problems.append(
                f"implementation source {name!r} changed since the evaluation contract was frozen: "
                f"{path} is {actual_hash}, contract expects {entry['sha256']}"
            )
    if problems:
        raise EvaluationError(f"implementation-source verification failed: {problems}")
    return actual


def validate_evaluation_contract(contract: dict) -> list[str]:
    """Structural + exact-semantics validation of the frozen unknown_test_v2
    evaluation contract. Returns a list of problems (empty == valid). Never
    raises for malformed input."""
    problems: list[str] = []
    if not isinstance(contract, dict):
        return ["contract is not a JSON object"]

    missing_top = REQUIRED_TOP_KEYS - set(contract)
    if missing_top:
        problems.append(f"missing top-level key(s): {sorted(missing_top)}")
        return problems
    extra_top = set(contract) - REQUIRED_TOP_KEYS - OPTIONAL_TOP_KEYS
    if extra_top:
        problems.append(f"contract has unexpected top-level key(s): {sorted(extra_top)}")

    if "generation" in contract:
        generation = contract["generation"]
        if not isinstance(generation, dict):
            problems.append("generation is not a JSON object")
        else:
            extra_generation = set(generation) - GENERATION_ALLOWED_KEYS
            if extra_generation:
                problems.append(f"generation has unexpected key(s): {sorted(extra_generation)}")
            for key, value in generation.items():
                if key in GENERATION_ALLOWED_KEYS and not isinstance(value, str):
                    problems.append(f"generation.{key} must be a string, got {value!r}")

    if not is_strict_int(contract.get("schema_version")) or contract["schema_version"] != EVAL_CONTRACT_SCHEMA_VERSION:
        problems.append(f"schema_version must be strict int {EVAL_CONTRACT_SCHEMA_VERSION}, "
                        f"got {contract.get('schema_version')!r}")
    if contract.get("status") != EVAL_CONTRACT_STATUS_FROZEN:
        problems.append(f"status must be {EVAL_CONTRACT_STATUS_FROZEN!r}, got {contract.get('status')!r}")

    content = contract.get("content")
    if not isinstance(content, dict):
        problems.append("content is not a JSON object")
        return problems

    missing_content = REQUIRED_CONTENT_KEYS - set(content)
    if missing_content:
        problems.append(f"content missing key(s): {sorted(missing_content)}")
    extra_content = set(content) - REQUIRED_CONTENT_KEYS - OPTIONAL_CONTENT_KEYS
    if extra_content:
        problems.append(f"content has unexpected key(s): {sorted(extra_content)}")

    recomputed = compute_content_sha256(content)
    if contract.get("content_sha256") != recomputed:
        problems.append(f"content_sha256 {contract.get('content_sha256')!r} != recomputed {recomputed!r} "
                        f"-- content has been altered since it was frozen")

    if content.get("policy_name") != FROZEN_POLICY_NAME:
        problems.append(f"policy_name must be exactly {FROZEN_POLICY_NAME!r}, got {content.get('policy_name')!r}")

    if content.get("selection_contract") != {"content_sha256": FROZEN_SELECTION_CONTRACT_CONTENT_SHA256}:
        problems.append(f"selection_contract must be exactly "
                        f"{{'content_sha256': {FROZEN_SELECTION_CONTRACT_CONTENT_SHA256!r}}}, "
                        f"got {content.get('selection_contract')!r}")

    scores_binding = content.get("scores_binding")
    expected_scores_binding = {"byte_sha256": FROZEN_SCORES_BYTE_SHA256, "content_sha256": FROZEN_SCORES_CONTENT_SHA256}
    if scores_binding != expected_scores_binding:
        problems.append(f"scores_binding must be exactly {expected_scores_binding!r}, got {scores_binding!r}")

    selection_binding = content.get("selection_binding")
    expected_selection_binding = {"byte_sha256": FROZEN_SELECTION_BYTE_SHA256,
                                  "content_sha256": FROZEN_SELECTION_CONTENT_SHA256}
    if selection_binding != expected_selection_binding:
        problems.append(f"selection_binding must be exactly {expected_selection_binding!r}, "
                        f"got {selection_binding!r}")

    threshold = content.get("threshold")
    expected_threshold = {
        "threshold_integer": FROZEN_THRESHOLD_INTEGER, "threshold": FROZEN_THRESHOLD,
        "source": FROZEN_THRESHOLD_SOURCE, "comparison": FROZEN_DECISION_SHAPE["comparison"],
        "equal_threshold_action": FROZEN_DECISION_SHAPE["equal_threshold_action"],
        "required_status": FROZEN_THRESHOLD_STATUS_REQUIRED,
    }
    if threshold != expected_threshold:
        problems.append(f"threshold must be exactly {expected_threshold!r}, got {threshold!r}")

    if not is_valid_sha256_hex(content.get(CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256)):
        problems.append(f"{CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256} is not 64 lowercase hex "
                        f"chars: {content.get(CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256)!r}")

    # ---- bindings: presence, hash format, exact paths, exact hashes -------
    bindings = content.get("bindings", {})
    if isinstance(bindings, dict):
        missing_bindings = REQUIRED_BINDING_KEYS - set(bindings)
        if missing_bindings:
            problems.append(f"bindings missing key(s): {sorted(missing_bindings)}")
        extra_bindings = set(bindings) - REQUIRED_BINDING_KEYS
        if extra_bindings:
            problems.append(f"bindings has unexpected key(s): {sorted(extra_bindings)}")
        for name, entry in bindings.items():
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                problems.append(f"binding {name!r} must have exactly path and sha256")
                continue
            if not is_safe_relative_path(entry["path"]):
                problems.append(f"binding {name!r} path is not a safe repo-relative path: {entry['path']!r}")
            elif name in FROZEN_BINDING_PATHS and entry["path"] != FROZEN_BINDING_PATHS[name]:
                problems.append(f"binding {name!r} path must be exactly {FROZEN_BINDING_PATHS[name]!r}, "
                                f"got {entry['path']!r}")
            if not is_valid_sha256_hex(entry["sha256"]):
                problems.append(f"binding {name!r} sha256 is not 64 lowercase hex chars: {entry['sha256']!r}")
            elif name in FROZEN_BINDING_HASHES and entry["sha256"] != FROZEN_BINDING_HASHES[name]:
                problems.append(f"binding {name!r} sha256 must be exactly {FROZEN_BINDING_HASHES[name]!r}, "
                                f"got {entry['sha256']!r}")
    else:
        problems.append("bindings is not a JSON object")

    # ---- implementation sources ---------------------------------------------
    impl = content.get("implementation_sources", {})
    if isinstance(impl, dict):
        missing_impl = IMPLEMENTATION_SOURCE_KEYS - set(impl)
        if missing_impl:
            problems.append(f"implementation_sources missing key(s): {sorted(missing_impl)}")
        extra_impl = set(impl) - IMPLEMENTATION_SOURCE_KEYS
        if extra_impl:
            problems.append(f"implementation_sources has unexpected key(s): {sorted(extra_impl)}")
        for name, entry in impl.items():
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                problems.append(f"implementation_sources[{name!r}] must have exactly path and sha256")
                continue
            if not is_safe_relative_path(entry["path"]):
                problems.append(f"implementation_sources[{name!r}] path unsafe: {entry['path']!r}")
            if not is_valid_sha256_hex(entry["sha256"]):
                problems.append(f"implementation_sources[{name!r}] sha256 malformed: {entry['sha256']!r}")
            expected_path = IMPLEMENTATION_SOURCE_PATHS.get(name)
            if expected_path is not None and entry.get("path") != expected_path:
                problems.append(f"implementation_sources[{name!r}] path {entry.get('path')!r} "
                                f"!= expected {expected_path!r}")
    else:
        problems.append("implementation_sources is not a JSON object")

    # ---- approved outputs ----------------------------------------------------
    approved_outputs = content.get("approved_outputs", {})
    if isinstance(approved_outputs, dict):
        missing_outputs = REQUIRED_APPROVED_OUTPUT_KEYS - set(approved_outputs)
        if missing_outputs:
            problems.append(f"approved_outputs missing key(s): {sorted(missing_outputs)}")
        extra_outputs = set(approved_outputs) - REQUIRED_APPROVED_OUTPUT_KEYS
        if extra_outputs:
            problems.append(f"approved_outputs has unexpected key(s): {sorted(extra_outputs)}")
        for key, rel_path in approved_outputs.items():
            if not is_safe_relative_path(rel_path):
                problems.append(f"approved_outputs[{key!r}] is not a safe repo-relative path: {rel_path!r}")
                continue
            if any(rel_path.startswith(prefix) for prefix in FORBIDDEN_OUTPUT_PREFIXES):
                problems.append(f"approved_outputs[{key!r}] {rel_path!r} is under a forbidden prefix")
        if approved_outputs != FROZEN_APPROVED_OUTPUTS:
            problems.append(f"approved_outputs does not exactly match the frozen output paths: "
                            f"{approved_outputs!r} != {FROZEN_APPROVED_OUTPUTS!r}")
    else:
        problems.append("approved_outputs is not a JSON object")

    # ---- dataset quota, runtime, decision shape, criteria, rules: exact ----
    dataset_quota = content.get("dataset_quota", {})
    if isinstance(dataset_quota, dict):
        if dataset_quota != FROZEN_DATASET_QUOTA:
            problems.append(f"dataset_quota does not exactly match the frozen unknown_test_v2 quota: "
                            f"{dataset_quota!r} != {FROZEN_DATASET_QUOTA!r}")
    else:
        problems.append("dataset_quota is not a JSON object")

    runtime = content.get("runtime", {})
    if isinstance(runtime, dict):
        if runtime != FROZEN_RUNTIME:
            problems.append(f"runtime does not exactly match the frozen runtime policy: "
                            f"{runtime!r} != {FROZEN_RUNTIME!r}")
    else:
        problems.append("runtime is not a JSON object")

    decision_shape = content.get("decision_shape", {})
    if isinstance(decision_shape, dict):
        if decision_shape != FROZEN_DECISION_SHAPE:
            problems.append(f"decision_shape does not exactly match the frozen decision shape: "
                            f"{decision_shape!r} != {FROZEN_DECISION_SHAPE!r}")
    else:
        problems.append("decision_shape is not a JSON object")

    validation_criteria = content.get("validation_criteria", {})
    if isinstance(validation_criteria, dict):
        if validation_criteria != FROZEN_VALIDATION_CRITERIA:
            problems.append(f"validation_criteria does not exactly match the frozen criteria: "
                            f"{validation_criteria!r} != {FROZEN_VALIDATION_CRITERIA!r}")
    else:
        problems.append("validation_criteria is not a JSON object")

    single_use = content.get("single_use_rule", {})
    if isinstance(single_use, dict):
        if single_use != FROZEN_SINGLE_USE_RULE:
            problems.append(f"single_use_rule does not exactly match the frozen rule: "
                            f"{single_use!r} != {FROZEN_SINGLE_USE_RULE!r}")
    else:
        problems.append("single_use_rule is not a JSON object")

    if content.get("closed_sources") != FROZEN_CLOSED_SOURCES:
        problems.append(f"closed_sources must be exactly {FROZEN_CLOSED_SOURCES!r}, "
                        f"got {content.get('closed_sources')!r}")
    if content.get("gate_framing") != FROZEN_GATE_FRAMING:
        problems.append(f"gate_framing must be exactly {FROZEN_GATE_FRAMING!r}, "
                        f"got {content.get('gate_framing')!r}")

    return problems


def load_and_verify_evaluation_contract(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise EvaluationError(f"evaluation contract file does not exist: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_evaluation_contract(contract)
    if problems:
        raise EvaluationError(f"evaluation contract at {path} failed validation: {problems}")
    return contract


def verify_binding(repo_root: Path, contract: dict, binding_name: str) -> None:
    entry = contract["content"]["bindings"][binding_name]
    path = Path(repo_root) / entry["path"]
    if not path.exists():
        raise EvaluationError(f"bound file missing: {path} (binding {binding_name!r})")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise EvaluationError(
            f"binding {binding_name!r} hash mismatch: {path} is {actual}, contract expects {entry['sha256']}"
        )


def verify_all_bindings(repo_root: Path, contract: dict) -> None:
    for name in contract["content"]["bindings"]:
        verify_binding(repo_root, contract, name)


def verify_output_path_is_approved(repo_root: Path, contract: dict, output_key: str, requested_path: Path) -> Path:
    approved_outputs = contract["content"].get("approved_outputs", {})
    approved_rel = approved_outputs.get(output_key)
    if not approved_rel:
        raise EvaluationError(f"contract has no approved_outputs[{output_key!r}]")
    if any(approved_rel.startswith(prefix) for prefix in FORBIDDEN_OUTPUT_PREFIXES):
        raise EvaluationError(f"approved_outputs[{output_key!r}] {approved_rel!r} is under a forbidden prefix")
    approved_abs = (Path(repo_root) / approved_rel).resolve()
    requested_abs = Path(requested_path).resolve()
    if requested_abs != approved_abs:
        raise EvaluationError(
            f"{output_key} output path {requested_abs} does not resolve to the contract-approved "
            f"destination {approved_abs}"
        )
    return approved_abs


# ------------------------------------------------------------ attempt marker
REQUIRED_MARKER_TOP_KEYS = {"schema_version", "content", "content_sha256"}
REQUIRED_MARKER_CONTENT_KEYS = {
    "git_head", "evaluation_contract_content_sha256", "implementation_source_hashes",
    "threshold_integer", "threshold", "scores_binding", "selection_binding", "bindings",
    "created_at_utc",
}


def build_attempt_marker_content(git_head: str, contract: dict, implementation_hashes: dict[str, str],
                                 created_at_utc: str) -> dict:
    return {
        "git_head": git_head,
        "evaluation_contract_content_sha256": contract["content_sha256"],
        "implementation_source_hashes": dict(implementation_hashes),
        "threshold_integer": contract["content"]["threshold"]["threshold_integer"],
        "threshold": contract["content"]["threshold"]["threshold"],
        "scores_binding": dict(contract["content"]["scores_binding"]),
        "selection_binding": dict(contract["content"]["selection_binding"]),
        "bindings": {k: dict(v) for k, v in contract["content"]["bindings"].items()},
        "created_at_utc": created_at_utc,
    }


CREATED_AT_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate_attempt_marker(marker: dict, contract: dict) -> list[str]:
    """Structural + FULL binding validation of an attempt-marker file
    against the loaded evaluation contract -- every field the marker
    carries must exactly agree with the contract it was created from, not
    merely be well-formed. Never raises on malformed input."""
    problems: list[str] = []
    if not isinstance(marker, dict):
        return ["marker is not a JSON object"]
    missing_top = REQUIRED_MARKER_TOP_KEYS - set(marker)
    if missing_top:
        return [f"marker missing top-level key(s): {sorted(missing_top)}"]
    extra_top = set(marker) - REQUIRED_MARKER_TOP_KEYS
    if extra_top:
        problems.append(f"marker has unexpected top-level key(s): {sorted(extra_top)}")
    if not is_strict_int(marker.get("schema_version")) or marker["schema_version"] != ATTEMPT_MARKER_SCHEMA_VERSION:
        problems.append(f"marker.schema_version must be strict int {ATTEMPT_MARKER_SCHEMA_VERSION}, "
                        f"got {marker.get('schema_version')!r}")
    content = marker.get("content")
    if not isinstance(content, dict):
        return problems + ["marker.content is not a JSON object"]
    recomputed = compute_content_sha256(content)
    if marker.get("content_sha256") != recomputed:
        problems.append(f"marker.content_sha256 {marker.get('content_sha256')!r} != recomputed {recomputed!r}")
    missing_content = REQUIRED_MARKER_CONTENT_KEYS - set(content)
    if missing_content:
        problems.append(f"marker.content missing key(s): {sorted(missing_content)}")
    extra_content = set(content) - REQUIRED_MARKER_CONTENT_KEYS
    if extra_content:
        problems.append(f"marker.content has unexpected key(s): {sorted(extra_content)}")

    git_head = content.get("git_head")
    if not isinstance(git_head, str) or not re.fullmatch(r"[0-9a-f]{40}", git_head):
        problems.append(f"marker.content.git_head is not a valid 40-hex-char commit id: {git_head!r}")

    if content.get("evaluation_contract_content_sha256") != contract.get("content_sha256"):
        problems.append("marker.content.evaluation_contract_content_sha256 does not match the loaded contract")

    contract_content = contract.get("content", {})
    contract_content = contract_content if isinstance(contract_content, dict) else {}
    contract_impl = contract_content.get("implementation_sources", {})
    contract_impl = contract_impl if isinstance(contract_impl, dict) else {}
    expected_impl_hashes = {name: (entry.get("sha256") if isinstance(entry, dict) else None)
                            for name, entry in contract_impl.items()}
    if content.get("implementation_source_hashes") != expected_impl_hashes:
        problems.append("marker.content.implementation_source_hashes does not exactly match the "
                        "contract's implementation_sources (missing, extra, or altered entries)")

    if content.get("threshold_integer") != FROZEN_THRESHOLD_INTEGER:
        problems.append(f"marker.content.threshold_integer must be exactly {FROZEN_THRESHOLD_INTEGER!r}")
    if content.get("threshold") != FROZEN_THRESHOLD:
        problems.append(f"marker.content.threshold must be exactly {FROZEN_THRESHOLD!r}")

    if content.get("scores_binding") != contract_content.get("scores_binding"):
        problems.append("marker.content.scores_binding does not match the loaded contract's scores_binding")
    if content.get("selection_binding") != contract_content.get("selection_binding"):
        problems.append("marker.content.selection_binding does not match the loaded contract's selection_binding")
    if content.get("bindings") != contract_content.get("bindings"):
        problems.append("marker.content.bindings does not exactly match the loaded contract's bindings "
                        "(every candidate/dataset binding must agree)")

    created_at = content.get("created_at_utc")
    if not isinstance(created_at, str) or not CREATED_AT_UTC_RE.fullmatch(created_at):
        problems.append(f"marker.content.created_at_utc must be a strict UTC timestamp "
                        f"'YYYY-MM-DDTHH:MM:SSZ', got {created_at!r}")

    return problems


def _rejection_rate_strictly_greater(incorrect_rejected: int, total_incorrect: int,
                                     correct_rejected: int, total_correct: int) -> bool:
    """Exact cross-multiplication test of
        incorrect_rejected/total_incorrect > correct_rejected/total_correct
    Fails closed (returns False) if either total is zero -- an undefined
    comparison is never treated as passing."""
    if total_incorrect <= 0 or total_correct <= 0:
        return False
    return incorrect_rejected * total_correct > correct_rejected * total_incorrect


def compute_validation(records: list[dict]) -> dict[str, Any]:
    """Applies the ONE frozen threshold to the known_holdout rows and
    evaluates the three precommitted criteria via exact integer arithmetic.
    OOD rows are reported as diagnostics ONLY -- they never enter any
    criterion below.

    This is THE single shared computation: both the evaluator (which calls
    it once, at write time, over real records) and validate_eval_content
    (which calls it again, at read/verify time, over the SAME records taken
    from the file on disk) call this exact function and require exact
    equality between the two results -- a modified metric/criterion/status
    can no longer be smuggled past validation merely by recomputing
    content_sha256 over the altered `validation` block, since that block is
    never trusted on its own; it must reproduce from records."""
    known = [r for r in records if r["category"] == KNOWN_CATEGORY]
    n_known = len(known)
    threshold = FROZEN_THRESHOLD

    counts = sel._classify_known(known, threshold)
    baseline_correct_top1 = counts["total_correct"]
    baseline_correct_top3 = sum(1 for r in known if sel._top3_correct(r))

    coverage_ok = gc.coverage_feasible(counts["accepted"], n_known, 65, 100)
    usefulness_ok = gc.usefulness_ge_floor(counts["correct_accepted_top1"], counts["accepted"],
                                           baseline_correct_top1, n_known, 5, 100)
    health_check_ok = _rejection_rate_strictly_greater(
        counts["incorrect_rejected"], counts["total_incorrect"],
        counts["correct_rejected"], counts["total_correct"],
    )
    status = EVAL_STATUS_PASSED if (coverage_ok and usefulness_ok and health_check_ok) else EVAL_STATUS_FAILED

    rejection_metrics = sel._rejection_metrics(counts)
    per_species = sel._per_species_known_report(records, threshold)

    ood_by_cat = {cat: [r for r in records if r["category"] == cat] for cat in OOD_CATEGORIES}
    diagnostic_ood_far = {}
    for cat, rows in ood_by_cat.items():
        accepted_ood = sum(1 for r in rows if r["max_cosine"] >= threshold)
        diagnostic_ood_far[cat] = {"n": len(rows), "false_acceptance_rate": gc.rate_or_none(accepted_ood, len(rows))}
    diagnostic_auc = {
        cat: sel._auc_known_vs_ood([r["max_cosine"] for r in known],
                                   [r["max_cosine"] for r in records if r["category"] == cat])
        for cat in OOD_CATEGORIES
    }

    accepted_top1_accuracy = gc.rate_or_none(counts["correct_accepted_top1"], counts["accepted"])
    baseline_accuracy_top1 = gc.rate_or_none(baseline_correct_top1, n_known)
    improvement_pp = ((accepted_top1_accuracy - baseline_accuracy_top1) * 100.0
                      if accepted_top1_accuracy is not None and baseline_accuracy_top1 is not None else None)

    metrics = {
        "baseline": {
            "n": n_known, "correct_top1": baseline_correct_top1,
            "accuracy_top1": baseline_accuracy_top1,
            "correct_top3": baseline_correct_top3,
            "accuracy_top3": gc.rate_or_none(baseline_correct_top3, n_known),
        },
        "accepted": {
            "accepted": counts["accepted"], "rejected": counts["rejected"],
            "coverage": gc.rate_or_none(counts["accepted"], n_known),
            "accepted_top1_accuracy": accepted_top1_accuracy,
            "accepted_top3_accuracy": gc.rate_or_none(counts["correct_accepted_top3"], counts["accepted"]),
        },
        "improvement_over_baseline_pp": improvement_pp,
        "rejection_metrics": rejection_metrics,
        "per_species_known": per_species,
        "diagnostic_ood_far": diagnostic_ood_far,
        "diagnostic_auc_known_vs_ood": diagnostic_auc,
    }

    return {
        "status": status,
        "threshold_integer": FROZEN_THRESHOLD_INTEGER,
        "threshold": FROZEN_THRESHOLD,
        "criteria": {
            "coverage_ok": bool(coverage_ok), "usefulness_ok": bool(usefulness_ok),
            "health_check_ok": bool(health_check_ok),
        },
        "metrics": metrics,
    }


# ------------------------------------------------------------- eval output
REQUIRED_EVAL_TOP_KEYS = {"schema_version", "content", "content_sha256"}
OPTIONAL_EVAL_TOP_KEYS = {"generation"}
EVAL_GENERATION_ALLOWED_KEYS = {"evaluator_source_sha256", "python_version"}
REQUIRED_EVAL_CONTENT_KEYS = {
    "dataset", "row_order", "n_rows", "bindings", "runtime", "provenance", "records", "validation",
}
REQUIRED_EVAL_RUNTIME_KEYS = {"onnxruntime_version", "registered_providers", "pillow_version",
                              "preprocessing_contract"}
REQUIRED_EVAL_PROVENANCE_KEYS = {"git_head", "implementation_source_hashes"}
REQUIRED_EVAL_RECORD_KEYS = set(gc.REQUIRED_SCORE_RECORD_KEYS)
REQUIRED_EVAL_IDENTITY_FIELDS = gc.REQUIRED_IDENTITY_FIELDS
REQUIRED_EVAL_BINDING_KEYS = {
    "evaluation_contract_content_sha256", "unknown_test_v2_csv_sha256",
    CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256, "attempt_marker_sha256",
}


def validate_eval_records(records: Any, quota: dict) -> tuple[list[str], bool, list[tuple[str, str, str]]]:
    """Full per-record structural + semantic validation, mirroring
    gate_v2_contract.validate_score_content's record loop exactly (same
    record shape, same category set) but against unknown_test_v2's own
    quota. Never raises on malformed/unhashable input -- every value is
    type-checked before being used as a set/Counter key or compared."""
    problems: list[str] = []
    if not isinstance(records, list):
        return ["records is not a list"], False, []

    species_count = quota.get("species_count")
    known_per_species_quota = quota.get("known_holdout_per_species")
    seen: dict[str, set] = {f: set() for f in REQUIRED_EVAL_IDENTITY_FIELDS}
    dup_problems: list[str] = []
    from collections import Counter
    cat_counts: Counter = Counter()
    known_slug_counts: Counter = Counter()
    identity_triples: list[tuple[str, str, str]] = []
    identity_extraction_ok = True

    for i, r in enumerate(records):
        if not isinstance(r, dict):
            problems.append(f"record {i} is not a JSON object")
            continue
        missing_r = REQUIRED_EVAL_RECORD_KEYS - set(r)
        if missing_r:
            problems.append(f"record {i} missing key(s): {sorted(missing_r)}")
            continue
        extra_r = set(r) - REQUIRED_EVAL_RECORD_KEYS
        if extra_r:
            problems.append(f"record {i} has unexpected key(s): {sorted(extra_r)}")

        cat = r["category"]
        if not isinstance(cat, str):
            problems.append(f"record {i}: category must be a string, got {cat!r}")
            continue
        cat_counts[cat] += 1
        for field in ("slug", "species", "taxon_id", "top1_slug", *REQUIRED_EVAL_IDENTITY_FIELDS):
            if r.get(field) in (None, ""):
                problems.append(f"record {i}: blank required field {field!r}")

        for field in REQUIRED_EVAL_IDENTITY_FIELDS:
            value = r.get(field)
            if not isinstance(value, str):
                if value not in (None, ""):
                    problems.append(f"record {i}: {field} must be a string, got {value!r}")
                continue
            if value in seen[field]:
                dup_problems.append(f"duplicate {field} {value!r} at record {i}")
            seen[field].add(value)

        pid, uid, img_sha = r.get("photo_id"), r.get("observation_uuid"), r.get("image_sha256")
        if isinstance(pid, str) and pid and isinstance(uid, str) and uid and isinstance(img_sha, str) and img_sha:
            identity_triples.append((pid, uid, img_sha))
        else:
            identity_extraction_ok = False

        if cat == KNOWN_CATEGORY and isinstance(r.get("slug"), str) and r.get("slug"):
            known_slug_counts[r["slug"]] += 1

        top3_idx, top3_slugs, top3_sims = r.get("top3_indices"), r.get("top3_slugs"), r.get("top3_similarities")
        if not (isinstance(top3_idx, list) and len(top3_idx) == 3):
            problems.append(f"record {i}: top3_indices must have exactly 3 entries, got {top3_idx!r}")
            top3_idx = None
        elif not all(is_strict_int(x) for x in top3_idx):
            problems.append(f"record {i}: top3_indices must be 3 strict ints, got {top3_idx!r}")
            top3_idx = None
        elif len(set(top3_idx)) != 3:
            problems.append(f"record {i}: top3_indices must be 3 distinct values, got {top3_idx!r}")
        if not (isinstance(top3_slugs, list) and len(top3_slugs) == 3):
            problems.append(f"record {i}: top3_slugs must have exactly 3 entries, got {top3_slugs!r}")
            top3_slugs = None
        else:
            if any(not isinstance(s, str) or not s for s in top3_slugs):
                problems.append(f"record {i}: top3_slugs must be 3 nonblank strings, got {top3_slugs!r}")
            elif len(set(top3_slugs)) != 3:
                problems.append(f"record {i}: top3_slugs must be 3 distinct values, got {top3_slugs!r}")
        if not (isinstance(top3_sims, list) and len(top3_sims) == 3):
            problems.append(f"record {i}: top3_similarities must have exactly 3 entries, got {top3_sims!r}")
            top3_sims = None

        if top3_sims is not None:
            for sim in top3_sims:
                if not is_finite_number(sim):
                    problems.append(f"record {i}: non-finite or non-numeric similarity {sim!r}")
                elif abs(sim) > 1.0 + COSINE_TOLERANCE:
                    problems.append(f"record {i}: similarity {sim} outside [-1,1] tolerance {COSINE_TOLERANCE}")
            if all(is_finite_number(s) for s in top3_sims) and \
                    not (top3_sims[0] >= top3_sims[1] >= top3_sims[2]):
                problems.append(f"record {i}: top3_similarities must be non-increasing, got {top3_sims!r}")

        top1_slug = r.get("top1_slug")
        if top3_slugs is not None and isinstance(top1_slug, str) and top1_slug and top1_slug != top3_slugs[0]:
            problems.append(f"record {i}: top1_slug {top1_slug!r} != first of top3_slugs {top3_slugs[0]!r}")

        max_cosine = r.get("max_cosine")
        if not is_finite_number(max_cosine):
            problems.append(f"record {i}: max_cosine not finite numeric: {max_cosine!r}")
        elif top3_sims is not None and max_cosine != top3_sims[0]:
            problems.append(f"record {i}: max_cosine {max_cosine!r} != first top-3 similarity {top3_sims[0]!r}")

        top1_index = r.get("top1_index")
        if not is_strict_int(top1_index):
            problems.append(f"record {i}: top1_index must be a strict int, got {top1_index!r}")
        elif top3_idx is not None and top1_index != top3_idx[0]:
            problems.append(f"record {i}: top1_index {top1_index} != first of top3_indices {top3_idx[0]}")

        if top3_idx is not None and is_strict_int(species_count):
            for idx in top3_idx:
                if not is_strict_int(idx) or not (0 <= idx < species_count):
                    problems.append(f"record {i}: top index {idx!r} out of range 0..{species_count - 1}")

        true_idx, top1_correct = r.get("true_class_index"), r.get("top1_correct")
        if cat == KNOWN_CATEGORY:
            if not is_strict_int(true_idx) or (is_strict_int(species_count) and not (0 <= true_idx < species_count)):
                problems.append(f"record {i}: known_holdout true_class_index invalid: {true_idx!r}")
            if not isinstance(top1_correct, bool):
                problems.append(f"record {i}: known_holdout top1_correct must be a strict bool, got {top1_correct!r}")
            elif is_strict_int(true_idx) and is_strict_int(top1_index):
                expected_correct = (top1_index == true_idx)
                if top1_correct != expected_correct:
                    problems.append(f"record {i}: top1_correct {top1_correct} disagrees with "
                                    f"top1_index==true_class_index ({expected_correct})")
        elif cat in OOD_CATEGORIES:
            if true_idx is not None:
                problems.append(f"record {i}: OOD row true_class_index must be null, got {true_idx!r}")
            if top1_correct is not None:
                problems.append(f"record {i}: OOD row top1_correct must be null, got {top1_correct!r}")
        else:
            problems.append(f"record {i}: unrecognized category {cat!r}")

    if dup_problems:
        problems.extend(dup_problems[:20])
        if len(dup_problems) > 20:
            problems.append(f"... and {len(dup_problems) - 20} more duplicate-identity problem(s)")

    if is_strict_int(quota.get(KNOWN_CATEGORY)) and cat_counts.get(KNOWN_CATEGORY, 0) != quota[KNOWN_CATEGORY]:
        problems.append(f"known_holdout record count {cat_counts.get(KNOWN_CATEGORY, 0)} != "
                        f"contract {quota[KNOWN_CATEGORY]}")
    for cat in OOD_CATEGORIES:
        if is_strict_int(quota.get(cat)) and cat_counts.get(cat, 0) != quota[cat]:
            problems.append(f"{cat} record count {cat_counts.get(cat, 0)} != contract {quota[cat]}")

    if is_strict_int(species_count) and is_strict_int(known_per_species_quota):
        if len(known_slug_counts) != species_count:
            problems.append(f"known_holdout has {len(known_slug_counts)} distinct slug(s), "
                            f"contract expects exactly {species_count}")
        bad_species = {s: n for s, n in known_slug_counts.items() if n != known_per_species_quota}
        if bad_species:
            problems.append(f"known_holdout per-species row count must be exactly "
                            f"{known_per_species_quota} each, mismatched: {bad_species}")

    return problems, identity_extraction_ok, identity_triples


def validate_eval_content(evaluation: dict, contract: dict) -> list[str]:
    """Full structural + semantic validation of an unknown_test_v2_eval.json
    artifact against the frozen evaluation contract. Returns a list of
    problems (empty == valid). Never raises KeyError/TypeError."""
    problems: list[str] = []
    if not isinstance(evaluation, dict):
        return ["evaluation is not a JSON object"]

    missing_top = REQUIRED_EVAL_TOP_KEYS - set(evaluation)
    if missing_top:
        return [f"evaluation missing top-level key(s): {sorted(missing_top)}"]
    extra_top = set(evaluation) - REQUIRED_EVAL_TOP_KEYS - OPTIONAL_EVAL_TOP_KEYS
    if extra_top:
        problems.append(f"evaluation has unexpected top-level key(s): {sorted(extra_top)}")

    if "generation" in evaluation:
        generation = evaluation["generation"]
        if not isinstance(generation, dict):
            problems.append("evaluation.generation is not a JSON object")
        else:
            extra_generation = set(generation) - EVAL_GENERATION_ALLOWED_KEYS
            if extra_generation:
                problems.append(f"evaluation.generation has unexpected key(s): {sorted(extra_generation)}")
            for key, value in generation.items():
                if key in EVAL_GENERATION_ALLOWED_KEYS and not isinstance(value, str):
                    problems.append(f"evaluation.generation.{key} must be a string, got {value!r}")

    if not is_strict_int(evaluation["schema_version"]) or evaluation["schema_version"] != EVAL_OUTPUT_SCHEMA_VERSION:
        problems.append(f"evaluation.schema_version must be strict int {EVAL_OUTPUT_SCHEMA_VERSION}, "
                        f"got {evaluation['schema_version']!r}")

    content = evaluation["content"]
    if not isinstance(content, dict):
        return problems + ["evaluation.content is not a JSON object"]

    missing_content = REQUIRED_EVAL_CONTENT_KEYS - set(content)
    if missing_content:
        problems.append(f"evaluation.content missing key(s): {sorted(missing_content)}")
    extra_content = set(content) - REQUIRED_EVAL_CONTENT_KEYS
    if extra_content:
        problems.append(f"evaluation.content has unexpected key(s): {sorted(extra_content)}")

    recomputed = compute_content_sha256(content)
    if evaluation.get("content_sha256") != recomputed:
        problems.append(f"evaluation.content_sha256 {evaluation.get('content_sha256')!r} != recomputed "
                        f"{recomputed!r} -- the evaluation file has been altered since it was written")

    if content.get("dataset") != DATASET_NAME_EVAL:
        problems.append(f"evaluation.content.dataset must be {DATASET_NAME_EVAL!r}, got {content.get('dataset')!r}")
    if not content.get("row_order"):
        problems.append("evaluation.content.row_order must be a non-empty description of the row ordering")

    bindings = content.get("bindings")
    if not isinstance(bindings, dict):
        problems.append("evaluation.content.bindings is not a JSON object")
        bindings = {}
    else:
        missing_b = REQUIRED_EVAL_BINDING_KEYS - set(bindings)
        if missing_b:
            problems.append(f"evaluation.content.bindings missing key(s): {sorted(missing_b)}")
        extra_b = set(bindings) - REQUIRED_EVAL_BINDING_KEYS
        if extra_b:
            problems.append(f"evaluation.content.bindings has unexpected key(s): {sorted(extra_b)}")
    if bindings.get("evaluation_contract_content_sha256") != contract.get("content_sha256"):
        problems.append("evaluation.content.bindings.evaluation_contract_content_sha256 does not match "
                        "the loaded evaluation contract")
    expected_csv_hash = FROZEN_UNKNOWN_TEST_V2_HASHES["unknown_test_v2_csv"]
    if bindings.get("unknown_test_v2_csv_sha256") != expected_csv_hash:
        problems.append(f"evaluation.content.bindings.unknown_test_v2_csv_sha256 "
                        f"{bindings.get('unknown_test_v2_csv_sha256')!r} != frozen {expected_csv_hash!r}")
    if not is_valid_sha256_hex(bindings.get("attempt_marker_sha256")):
        problems.append("evaluation.content.bindings.attempt_marker_sha256 is not 64 lowercase hex chars")

    quota = FROZEN_DATASET_QUOTA
    records = content.get("records")
    record_problems, identity_extraction_ok, identity_triples = validate_eval_records(records, quota)
    problems.extend(record_problems)
    if isinstance(records, list):
        if not is_strict_int(content.get("n_rows")) or content["n_rows"] != len(records):
            problems.append(f"evaluation.content.n_rows {content.get('n_rows')!r} != len(records) {len(records)}")
        expected_total = quota.get("total")
        if is_strict_int(expected_total) and len(records) != expected_total:
            problems.append(f"{len(records)} evaluation records, contract expects {expected_total}")

        contract_identity_hash = bindings.get(CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256)
        frozen_identity_hash = contract.get("content", {}).get(CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256)
        if contract_identity_hash != frozen_identity_hash:
            problems.append("evaluation.content.bindings row identity-order hash does not match the "
                            "frozen evaluation contract")
        if not identity_extraction_ok or len(identity_triples) != len(records):
            problems.append("could not compute row identity-order hash: one or more records has a "
                            "missing/blank photo_id, observation_uuid, or image_sha256")
        else:
            try:
                recomputed_identity_hash = compute_identity_order_sha256(identity_triples)
            except ContractError as exc:
                problems.append(f"could not recompute row identity-order hash: {exc}")
            else:
                if recomputed_identity_hash != frozen_identity_hash:
                    problems.append(
                        "row identity-order hash does not match the frozen evaluation contract -- "
                        "records have been reordered, are missing, or have been substituted"
                    )

    # ---- runtime / provenance ------------------------------------------------
    runtime = content.get("runtime")
    if not isinstance(runtime, dict):
        problems.append("evaluation.content.runtime is not a JSON object")
    else:
        missing_rt = REQUIRED_EVAL_RUNTIME_KEYS - set(runtime)
        if missing_rt:
            problems.append(f"evaluation.content.runtime missing key(s): {sorted(missing_rt)}")
        extra_rt = set(runtime) - REQUIRED_EVAL_RUNTIME_KEYS
        if extra_rt:
            problems.append(f"evaluation.content.runtime has unexpected key(s): {sorted(extra_rt)}")
        if runtime.get("registered_providers") != ["CPUExecutionProvider"]:
            problems.append(f"evaluation.content.runtime.registered_providers must be exactly "
                            f"['CPUExecutionProvider'], got {runtime.get('registered_providers')!r}")
        preprocessing_contract = runtime.get("preprocessing_contract")
        if not isinstance(preprocessing_contract, dict) or not preprocessing_contract:
            problems.append("evaluation.content.runtime.preprocessing_contract must be a non-empty JSON object")
        if not runtime.get("onnxruntime_version"):
            problems.append("evaluation.content.runtime.onnxruntime_version must be a non-empty string")
        if not runtime.get("pillow_version"):
            problems.append("evaluation.content.runtime.pillow_version must be a non-empty string")

    provenance = content.get("provenance")
    contract_impl = contract.get("content", {}).get("implementation_sources", {})
    contract_impl = contract_impl if isinstance(contract_impl, dict) else {}
    if not isinstance(provenance, dict):
        problems.append("evaluation.content.provenance is not a JSON object")
    else:
        missing_pr = REQUIRED_EVAL_PROVENANCE_KEYS - set(provenance)
        if missing_pr:
            problems.append(f"evaluation.content.provenance missing key(s): {sorted(missing_pr)}")
        extra_pr = set(provenance) - REQUIRED_EVAL_PROVENANCE_KEYS
        if extra_pr:
            problems.append(f"evaluation.content.provenance has unexpected key(s): {sorted(extra_pr)}")
        git_head = provenance.get("git_head")
        if not isinstance(git_head, str) or not re.fullmatch(r"[0-9a-f]{40}", git_head):
            problems.append(f"evaluation.content.provenance.git_head is not a valid 40-hex-char "
                            f"commit id: {git_head!r}")
        impl_hashes = provenance.get("implementation_source_hashes")
        if not isinstance(impl_hashes, dict):
            problems.append("evaluation.content.provenance.implementation_source_hashes is not a JSON object")
        else:
            for name, entry in contract_impl.items():
                expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
                actual_hash = impl_hashes.get(name)
                if actual_hash != expected_hash:
                    problems.append(f"evaluation.content.provenance.implementation_source_hashes[{name!r}] "
                                    f"{actual_hash!r} != contract-bound {expected_hash!r}")
            extra_impl_hashes = set(impl_hashes) - set(contract_impl)
            if extra_impl_hashes:
                problems.append(f"evaluation.content.provenance.implementation_source_hashes has "
                                f"unexpected key(s): {sorted(extra_impl_hashes)}")

    # ---- validation result: DERIVED, not merely internally consistent -----
    # The stored `validation` block (status, criteria, every metric, every
    # per-species/diagnostic entry) is never trusted on its own -- it is
    # fully RECOMPUTED here from `records` via the same compute_validation()
    # the evaluator itself calls, and required to match by exact equality.
    # A file that alters any metric/criterion/status and rehashes
    # content_sha256 fresh is still rejected, because content_sha256 was
    # never the thing being checked here.
    stored_validation = content.get("validation")
    if not isinstance(stored_validation, dict):
        problems.append("evaluation.content.validation is not a JSON object")
    elif not isinstance(records, list) or record_problems or not identity_extraction_ok:
        problems.append("evaluation.content.validation cannot be verified because records failed "
                        "structural or identity validation above")
    else:
        try:
            recomputed_validation = compute_validation(records)
        except Exception as exc:  # noqa: BLE001 -- system boundary: never propagate, always report
            problems.append(f"could not recompute evaluation.content.validation from records: {exc}")
        else:
            extra_v = set(stored_validation) - set(recomputed_validation)
            missing_v = set(recomputed_validation) - set(stored_validation)
            if extra_v:
                problems.append(f"evaluation.content.validation has unexpected key(s): {sorted(extra_v)}")
            if missing_v:
                problems.append(f"evaluation.content.validation missing key(s): {sorted(missing_v)}")
            for key, expected_value in recomputed_validation.items():
                if key in stored_validation and stored_validation[key] != expected_value:
                    problems.append(
                        f"evaluation.content.validation.{key} does not match the value recomputed "
                        f"from records -- expected {expected_value!r}, got {stored_validation[key]!r}"
                    )

    return problems


def load_and_verify_eval_file(path: Path, contract: dict, repo_root: Path) -> dict:
    """Loads and fully validates an unknown_test_v2_eval.json artifact,
    THEN locates the contract-approved attempt-marker path, loads and fully
    validates that marker too, and requires its actual byte sha256 to equal
    the value the evaluation output itself recorded. A merely well-formed
    64-hex-character attempt_marker_sha256 is never sufficient on its own."""
    path = Path(path)
    if not path.exists():
        raise EvaluationError(f"evaluation output file does not exist: {path}")
    try:
        evaluation = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path} is not valid JSON: {exc}") from exc
    problems = validate_eval_content(evaluation, contract)
    if problems:
        raise EvaluationError(f"evaluation file {path} failed validation: {problems}")

    marker_rel = contract["content"]["approved_outputs"]["unknown_test_v2_evaluation_attempt"]
    marker_path = Path(repo_root) / marker_rel
    if not marker_path.exists():
        raise EvaluationError(f"attempt marker missing at contract-approved path: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{marker_path} is not valid JSON: {exc}") from exc
    marker_problems = validate_attempt_marker(marker, contract)
    if marker_problems:
        raise EvaluationError(f"attempt marker at {marker_path} failed validation: {marker_problems}")
    actual_marker_sha256 = sha256_file(marker_path)
    expected_marker_sha256 = evaluation.get("content", {}).get("bindings", {}).get("attempt_marker_sha256")
    if actual_marker_sha256 != expected_marker_sha256:
        raise EvaluationError(f"attempt marker byte sha256 {actual_marker_sha256} != evaluation-recorded "
                              f"attempt_marker_sha256 {expected_marker_sha256!r}")
    return evaluation
