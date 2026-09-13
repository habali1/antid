"""Pure schema, canonical hashing, and decision semantics for
inference_policy.json.

This module is duplicated byte-for-byte in ``training/`` and ``api/``. Each
stage imports its local copy; neither stage imports code across the deployment
boundary. ``inference_policy.json`` is the interface between them, and a test
asserts that the two schema copies remain byte-identical.

validate() must never raise: a malformed or hostile policy file must not
crash the loader that calls it. Every access is guarded and the whole
function is wrapped in a final try/except as a defense-in-depth net.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA_VERSION = 1  # unchanged: the V1 generator/loader default. Never repurposed for V2.
FROZEN_THRESHOLD = 0.6  # unchanged: schema v1's frozen threshold. Never repurposed for V2.

# ---- schema v2 (Gate v2, independently validated) -------------------------
# Added ALONGSIDE v1, never replacing it: SCHEMA_VERSION/FROZEN_THRESHOLD
# above keep their original names and values so the existing V1 generator
# (training/inference_policy_generator.py) and the V1 loader path need zero
# changes to keep emitting/accepting exactly schema v1 at exactly 0.60.
SCHEMA_VERSION_V2 = 2
FROZEN_THRESHOLD_V2 = 0.61
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION, SCHEMA_VERSION_V2}
# Each supported version selects its own permitted threshold value AND its
# own should_abstain() self-check probes (offset from that version's exact
# threshold) -- a policy can never mix a v1 schema_version with the v2
# threshold or vice versa; validate()/loader both check this exactly.
SCHEMA_VERSION_TO_THRESHOLD = {SCHEMA_VERSION: FROZEN_THRESHOLD, SCHEMA_VERSION_V2: FROZEN_THRESHOLD_V2}
SCHEMA_VERSION_TO_PROBES = {
    SCHEMA_VERSION: ((0.5999, True), (0.6000, False), (0.6001, False)),
    SCHEMA_VERSION_V2: ((0.6099, True), (0.6100, False), (0.6101, False)),
}
# Schema v2 additionally requires this block: independently-validated Gate
# v2 provenance (exact byte/content hashes of the calibration selection,
# the independent unknown_test_v2 evaluation, and the parity report bound
# to the same candidate artifacts), plus the two informational facts that
# must be carried verbatim. validate() requires this EXACT key set (missing
# or extra keys both fail) and type/value-checks every entry -- see
# _validate_v2_validation_evidence(). A v2 policy with this block missing,
# altered, or structurally incomplete is rejected even after content_sha256
# is freshly recomputed over the altered content, because content_sha256
# only proves the content wasn't tampered with AFTER being frozen; it says
# nothing about whether the frozen content was ever valid in the first
# place -- that is what this block's own field-level checks establish.
REQUIRED_V2_VALIDATION_EVIDENCE_KEYS = {
    "gate_v2_selection_contract_content_sha256",
    "calibration_v2_scores_byte_sha256", "calibration_v2_scores_content_sha256",
    "calibration_v2_selection_byte_sha256", "calibration_v2_selection_content_sha256",
    "gate_v2_evaluation_contract_content_sha256",
    "unknown_test_v2_evaluation_attempt_byte_sha256",
    "unknown_test_v2_eval_byte_sha256", "unknown_test_v2_eval_content_sha256",
    "parity_report_byte_sha256",
    "validation_status", "diagnostic_out_of_scope_ant_far",
}
REQUIRED_V2_VALIDATION_EVIDENCE_HASH_KEYS = REQUIRED_V2_VALIDATION_EVIDENCE_KEYS - {
    "validation_status", "diagnostic_out_of_scope_ant_far",
}
EXPECTED_V2_VALIDATION_STATUS = "validation_passed"

# Schema v2's envelope, checked EXACTLY (missing and extra keys both
# rejected) -- v1's envelope is intentionally left unconstrained beyond its
# existing checks, so this never changes v1 behavior. `V2_GENERATOR_VERSION`
# is the ONE frozen generator-version string; generate_inference_policy_v2.py
# imports it rather than defining its own copy, so there is nothing for the
# generator and this schema to silently drift apart on.
V2_GENERATOR_VERSION = "1.0.0"
REQUIRED_V2_TOP_KEYS = {"policy_schema_version", "content", "content_sha256", "generation"}
REQUIRED_V2_GENERATION_KEYS = {"generated_at", "generator_version"}
V2_GENERATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# The EXACT frozen values schema-v2's validation_evidence block must carry --
# not merely well-formed hashes/status/FAR, but THESE specific ones, bound
# to the ONE candidate (northeast_v1_b4_dev_v2) this schema-v2 policy exists
# for. A policy carrying a different, individually well-formed 64-hex-char
# hash in any of these fields (e.g. swapped in from some other evidence
# artifact) is rejected here even though format/type checks alone would
# have accepted it. Derived by re-running generate_inference_policy_v2.py's
# --preflight + build_policy() against the real, frozen repo evidence
# (read-only; never a write) -- see docs/plans/gate-v2-threshold-
# selection.md Phase 5E1 for provenance.
EXPECTED_V2_VALIDATION_EVIDENCE = {
    "gate_v2_selection_contract_content_sha256": "991f7a0b8e83654e45575566eb0648ddcd69866e29c94b8a2030cc6f4bc19f77",
    "calibration_v2_scores_byte_sha256": "4fec18a938ef22e06d6073016d1692512e8fb1e4e41ec645d9aeb33afbebe46b",
    "calibration_v2_scores_content_sha256": "35c7bf36042470dde8fc922106c6528fcb6a06e3bac454c4d196880796e778f4",
    "calibration_v2_selection_byte_sha256": "00e56d2e64941086ee1d1c663890cb95d3daa4dcce1e729ef972879376f1489b",
    "calibration_v2_selection_content_sha256": "82bc754dbf46643862504916f4c273922e9ac3cefc45bfa28e5eb4a15dfdc5b5",
    "gate_v2_evaluation_contract_content_sha256": "49bb0c4ee5749513b62e7bdfa7f50a7619fe16e2afc75a47d60bda25b4bdb10c",
    "unknown_test_v2_evaluation_attempt_byte_sha256": "fd790252ffbe4b20a6f52c26fb902e0cd7fbfedbd88b5613da048992b1a062bf",
    "unknown_test_v2_eval_byte_sha256": "b695e44ecd902b3763b6f202695e0495308a9f10d43400c47a46253c6029447e",
    "unknown_test_v2_eval_content_sha256": "e7c1d565c7ecd0999e00c65cc238296cf78525fc8efd00da8af91b3a0dbf7616",
    "parity_report_byte_sha256": "bec08235a48e9583f2b40556c9c8d860c41904729bcded9d6f91c37716c8297e",
    "validation_status": "validation_passed",
    "diagnostic_out_of_scope_ant_far": 0.51,
}
assert set(EXPECTED_V2_VALIDATION_EVIDENCE) == REQUIRED_V2_VALIDATION_EVIDENCE_KEYS

ALLOWED_OPERATOR = "max_sim < value"
ALLOWED_OPERATOR_NORMALIZED = {"comparison": "strict_less_than"}
ALLOWED_RULE_SIGNAL = "raw max cosine similarity before geo re-ranking"
ALLOWED_DECISION_SIGNAL = "raw_pre_geo_max_cosine"
REQUIRED_ARTIFACT_HASH_KEYS = {"backbone.onnx", "prototypes.npy", "taxonomy.json"}
REQUIRED_PREPROCESSING_CONTRACT_FIELDS = {
    "rgb_conversion", "resize", "interpolation", "scale_divisor",
    "normalize_mean", "normalize_std", "dtype", "channel_layout",
}
_HEX = frozenset("0123456789abcdef")


def is_sha256_hex(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s) <= _HEX


def canonical_bytes(schema_version, content) -> bytes:
    """The exact bytes the content_sha256 assertion covers: only
    {policy_schema_version, content}, never the whole file. Raises
    (TypeError/ValueError) on non-JSON-serializable or non-finite content --
    callers that must not raise should catch around this."""
    canonical = {"policy_schema_version": schema_version, "content": content}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False).encode("utf-8")


def compute_content_sha256(schema_version, content) -> str:
    return hashlib.sha256(canonical_bytes(schema_version, content)).hexdigest()


def should_abstain(max_sim: float, decision: dict) -> bool:
    """The single behavioral source of truth: reads ONLY
    decision['low_confidence_if']. normal_results_if and
    equal_threshold_action are redundant encodings of the same boundary,
    asserted consistent by decision_encodings_agree() -- never consulted
    here, so this function can't itself be the source of an inversion."""
    lo = decision["low_confidence_if"]
    if lo["comparison"] != "strict_less_than":
        raise ValueError(f"unsupported comparison: {lo['comparison']!r}")
    if not math.isfinite(max_sim):
        raise ValueError("non-finite similarity is a request error, not a decision input")
    return max_sim < lo["threshold"]


def decision_encodings_agree(decision: dict) -> list[str]:
    """The decision boundary is encoded three times (low_confidence_if,
    normal_results_if, equal_threshold_action) so a reviewer can catch an
    inversion by inspection. That redundancy is itself an inversion risk
    if the three ever disagree -- this asserts they don't."""
    errors = []
    lo = decision.get("low_confidence_if") if isinstance(decision, dict) else None
    hi = decision.get("normal_results_if") if isinstance(decision, dict) else None
    lo, hi = lo if isinstance(lo, dict) else {}, hi if isinstance(hi, dict) else {}
    t_lo, t_hi = lo.get("threshold"), hi.get("threshold")
    if t_lo != t_hi:
        errors.append(f"low_confidence_if.threshold {t_lo!r} != normal_results_if.threshold {t_hi!r}")
    if not isinstance(t_lo, (int, float)) or isinstance(t_lo, bool) or not math.isfinite(t_lo):
        errors.append(f"low_confidence_if.threshold is not a finite number: {t_lo!r}")
    if lo.get("comparison") != "strict_less_than":
        errors.append(f"low_confidence_if.comparison must be 'strict_less_than', got {lo.get('comparison')!r}")
    if lo.get("signal") != ALLOWED_DECISION_SIGNAL:
        errors.append(f"low_confidence_if.signal must be {ALLOWED_DECISION_SIGNAL!r}, "
                      f"got {lo.get('signal')!r}")
    if hi.get("comparison") != "greater_than_or_equal":
        errors.append(f"normal_results_if.comparison must be 'greater_than_or_equal', got {hi.get('comparison')!r}")
    if not isinstance(decision, dict) or decision.get("equal_threshold_action") != "normal_results":
        errors.append("equal_threshold_action must be 'normal_results'")
    if not isinstance(decision, dict) or decision.get("non_finite_action") != "request_error":
        errors.append("non_finite_action must be 'request_error'")
    return errors


def _obj(x) -> dict:
    return x if isinstance(x, dict) else {}


def _validate_v2_validation_evidence(content: dict) -> list[str]:
    """Schema-v2-only: content.validation_evidence must carry the EXACT key
    set (never more, never fewer) and every hash field must be a well-formed
    sha256 hex string; validation_status must be exactly 'validation_passed'
    and diagnostic_out_of_scope_ant_far a finite number. This module never
    touches a file, so it cannot re-verify these hashes against the actual
    evidence artifacts on disk -- that is the generator's job at
    --preflight/--write time. What this DOES guarantee is that a v2 policy
    can never be accepted with this evidence missing, incomplete, or of the
    wrong shape/type, regardless of how content_sha256 was recomputed."""
    errors: list[str] = []
    ve = content.get("validation_evidence")
    if not isinstance(ve, dict):
        errors.append("content.validation_evidence is not a JSON object")
        return errors
    missing = REQUIRED_V2_VALIDATION_EVIDENCE_KEYS - set(ve)
    if missing:
        errors.append(f"content.validation_evidence missing key(s): {sorted(missing)}")
    extra = set(ve) - REQUIRED_V2_VALIDATION_EVIDENCE_KEYS
    if extra:
        errors.append(f"content.validation_evidence has unexpected key(s): {sorted(extra)}")
    for key in REQUIRED_V2_VALIDATION_EVIDENCE_HASH_KEYS & set(ve):
        if not is_sha256_hex(ve.get(key)):
            errors.append(f"content.validation_evidence.{key} is not a 64-char lowercase hex string")
    if "validation_status" in ve and ve["validation_status"] != EXPECTED_V2_VALIDATION_STATUS:
        errors.append(f"content.validation_evidence.validation_status must be exactly "
                      f"{EXPECTED_V2_VALIDATION_STATUS!r}, got {ve['validation_status']!r}")
    if "diagnostic_out_of_scope_ant_far" in ve:
        far = ve["diagnostic_out_of_scope_ant_far"]
        if not isinstance(far, (int, float)) or isinstance(far, bool) or not math.isfinite(far):
            errors.append(f"content.validation_evidence.diagnostic_out_of_scope_ant_far must be a "
                          f"finite number, got {far!r}")

    # Exact-value freeze: every field present must equal the ONE frozen
    # value bound to this candidate's evidence, not merely be well-formed.
    # A rehashed policy that swaps in a different (but individually valid)
    # hash/status/FAR is rejected here regardless of content_sha256.
    for key, expected in EXPECTED_V2_VALIDATION_EVIDENCE.items():
        if key in ve and ve[key] != expected:
            errors.append(f"content.validation_evidence.{key} must be exactly {expected!r}, got {ve[key]!r}")
    return errors


def validate(policy) -> list[str]:
    """Structural + semantic validation of the fields a loader actually
    reads: schema version, decision block, artifact_hashes,
    provider_policy, content_sha256, preprocessing contract. Does not
    exhaustively validate phase_c/provenance (evidence bookkeeping the
    loader doesn't need). Returns a list of error strings; empty = valid.
    Never raises."""
    errors: list[str] = []
    try:
        if not isinstance(policy, dict):
            return ["policy is not a JSON object"]

        # Strict type check FIRST: `True == 1` and `1.0 == 1` in Python, so a
        # bare `sv not in SUPPORTED_SCHEMA_VERSIONS` comparison would
        # silently accept policy_schema_version=True or =1.0 --
        # `type(sv) is int` rejects both, and a bool/float/str/None/an
        # unsupported integer (e.g. 3) are all rejected the same way, before
        # any version-specific check ever runs.
        sv = policy.get("policy_schema_version")
        sv_valid = type(sv) is int and sv in SUPPORTED_SCHEMA_VERSIONS
        if not sv_valid:
            errors.append(f"policy_schema_version must be exactly one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}; "
                          f"got {sv!r} (type {type(sv).__name__})")

        content = _obj(policy.get("content"))
        if not isinstance(policy.get("content"), dict):
            errors.append("content is not a JSON object")

        # Schema v2 additionally requires the validation_evidence block.
        # Gated on sv_valid and sv == SCHEMA_VERSION_V2 specifically: an
        # invalid version is already reported above, and a v1 policy must
        # never be asked for a block it doesn't have.
        if sv_valid and sv == SCHEMA_VERSION_V2:
            errors.extend(_validate_v2_validation_evidence(content))
            # ---- schema-v2 envelope: exact top-level/generation key sets,
            # a strict generated_at timestamp form, and the ONE frozen
            # generator_version -- v1's envelope is untouched by this block.
            top_keys = set(policy.keys())
            if top_keys != REQUIRED_V2_TOP_KEYS:
                missing_top = REQUIRED_V2_TOP_KEYS - top_keys
                extra_top = top_keys - REQUIRED_V2_TOP_KEYS
                if missing_top:
                    errors.append(f"policy missing top-level key(s): {sorted(missing_top)}")
                if extra_top:
                    errors.append(f"policy has unexpected top-level key(s): {sorted(extra_top)}")

        gen = policy.get("generation")
        if not isinstance(gen, dict):
            errors.append("generation is not a JSON object")
        else:
            if not isinstance(gen.get("generated_at"), str):
                errors.append("generation.generated_at missing or not a string")
            if not isinstance(gen.get("generator_version"), str):
                errors.append("generation.generator_version missing or not a string")
            if "hostname" in gen:
                errors.append("generation must not record hostname")

            if sv_valid and sv == SCHEMA_VERSION_V2:
                gen_keys = set(gen.keys())
                if gen_keys != REQUIRED_V2_GENERATION_KEYS:
                    missing_gen = REQUIRED_V2_GENERATION_KEYS - gen_keys
                    extra_gen = gen_keys - REQUIRED_V2_GENERATION_KEYS
                    if missing_gen:
                        errors.append(f"generation missing key(s): {sorted(missing_gen)}")
                    if extra_gen:
                        errors.append(f"generation has unexpected key(s): {sorted(extra_gen)}")
                generated_at = gen.get("generated_at")
                if isinstance(generated_at, str) and not V2_GENERATED_AT_RE.match(generated_at):
                    errors.append(f"generation.generated_at must match the strict UTC form "
                                  f"'YYYY-MM-DDTHH:MM:SSZ', got {generated_at!r}")
                if gen.get("generator_version") != V2_GENERATOR_VERSION:
                    errors.append(f"generation.generator_version must be exactly "
                                  f"{V2_GENERATOR_VERSION!r}, got {gen.get('generator_version')!r}")

        # content_sha256 is only meaningful once the version itself is valid --
        # an invalid version is a validation error on its own (appended above)
        # regardless of what content_sha256 contains.
        if sv_valid:
            recorded_hash = policy.get("content_sha256")
            if not is_sha256_hex(recorded_hash):
                errors.append(f"content_sha256 is not a 64-char lowercase hex string: {recorded_hash!r}")
            else:
                try:
                    expected = compute_content_sha256(sv, content)
                except (TypeError, ValueError) as e:
                    errors.append(f"content is not canonicalizable: {e}")
                else:
                    if recorded_hash != expected:
                        errors.append(f"content_sha256 mismatch: recorded={recorded_hash} recomputed={expected}")

        decision = content.get("decision")
        if not isinstance(decision, dict):
            errors.append("content.decision is not a JSON object")
        else:
            enc_errors = decision_encodings_agree(decision)
            errors.extend(f"decision: {e}" for e in enc_errors)
            if not enc_errors and sv_valid:
                for probe, expected in SCHEMA_VERSION_TO_PROBES[sv]:
                    try:
                        got = should_abstain(probe, decision)
                    except (KeyError, ValueError, TypeError) as e:
                        errors.append(f"decision.should_abstain({probe}) raised: {e}")
                    else:
                        if got != expected:
                            errors.append(f"decision.should_abstain({probe}) = {got}, expected {expected}")

        rule = content.get("rule")
        if not isinstance(rule, dict):
            errors.append("content.rule is not a JSON object")
        else:
            if rule.get("operator_verbatim") != ALLOWED_OPERATOR:
                errors.append(f"rule.operator_verbatim must be {ALLOWED_OPERATOR!r}, "
                              f"got {rule.get('operator_verbatim')!r}")
            if rule.get("operator_normalized") != ALLOWED_OPERATOR_NORMALIZED:
                errors.append(f"rule.operator_normalized must be {ALLOWED_OPERATOR_NORMALIZED!r}, "
                              f"got {rule.get('operator_normalized')!r}")
            if rule.get("signal") != ALLOWED_RULE_SIGNAL:
                errors.append(f"rule.signal must be {ALLOWED_RULE_SIGNAL!r}, "
                              f"got {rule.get('signal')!r}")

            rule_value = rule.get("value")
            rule_value_is_number = (isinstance(rule_value, (int, float))
                                    and not isinstance(rule_value, bool)
                                    and math.isfinite(rule_value))
            if not rule_value_is_number:
                errors.append(f"rule.value is not a finite non-bool number: {rule_value!r}")
            elif sv_valid and rule_value != SCHEMA_VERSION_TO_THRESHOLD[sv]:
                errors.append(f"rule.value must be the frozen threshold for schema v{sv} "
                              f"({SCHEMA_VERSION_TO_THRESHOLD[sv]!r}); got {rule_value!r}")
            if rule_value_is_number and isinstance(decision, dict):
                lo = _obj(decision.get("low_confidence_if"))
                hi = _obj(decision.get("normal_results_if"))
                if rule_value != lo.get("threshold"):
                    errors.append(f"rule.value {rule_value!r} != "
                                  f"decision.low_confidence_if.threshold {lo.get('threshold')!r}")
                if rule_value != hi.get("threshold"):
                    errors.append(f"rule.value {rule_value!r} != "
                                  f"decision.normal_results_if.threshold {hi.get('threshold')!r}")

        hashes = _obj(content.get("artifact_hashes"))
        if not isinstance(content.get("artifact_hashes"), dict):
            errors.append("content.artifact_hashes is not a JSON object")
        else:
            if set(hashes.keys()) != REQUIRED_ARTIFACT_HASH_KEYS:
                errors.append(f"artifact_hashes keys must be exactly {sorted(REQUIRED_ARTIFACT_HASH_KEYS)}, "
                              f"got {sorted(hashes.keys())}")
            for k, v in hashes.items():
                if not is_sha256_hex(v):
                    errors.append(f"artifact_hashes.{k} is not a 64-char lowercase hex string")

        if content.get("provider_policy") != {"providers": ["CPUExecutionProvider"], "exclusive": True}:
            errors.append("provider_policy must be exactly "
                          "{'providers': ['CPUExecutionProvider'], 'exclusive': True}")

        pre = _obj(content.get("preprocessing"))
        contract = _obj(pre.get("contract"))
        if not isinstance(pre.get("contract"), dict):
            errors.append("preprocessing.contract is not a JSON object")
        else:
            missing = REQUIRED_PREPROCESSING_CONTRACT_FIELDS - contract.keys()
            if missing:
                errors.append(f"preprocessing.contract missing fields: {sorted(missing)}")
            if contract.get("normalize_mean") != [0.485, 0.456, 0.406]:
                errors.append(f"preprocessing.contract.normalize_mean must be [0.485,0.456,0.406], "
                              f"got {contract.get('normalize_mean')!r}")
            if contract.get("normalize_std") != [0.229, 0.224, 0.225]:
                errors.append(f"preprocessing.contract.normalize_std must be [0.229,0.224,0.225], "
                              f"got {contract.get('normalize_std')!r}")
    except Exception as e:  # noqa: BLE001 -- validate() must never raise
        errors.append(f"internal validation error: {type(e).__name__}: {e}")
    return errors
