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

SCHEMA_VERSION = 1
FROZEN_THRESHOLD = 0.6
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
        # bare `sv != SCHEMA_VERSION` comparison would silently accept
        # policy_schema_version=True or =1.0 -- `type(sv) is int` rejects
        # both before the value comparison ever runs.
        sv = policy.get("policy_schema_version")
        sv_valid = type(sv) is int and sv == SCHEMA_VERSION
        if not sv_valid:
            errors.append(f"policy_schema_version must be the exact int {SCHEMA_VERSION}; "
                          f"got {sv!r} (type {type(sv).__name__})")

        content = _obj(policy.get("content"))
        if not isinstance(policy.get("content"), dict):
            errors.append("content is not a JSON object")

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
            if not enc_errors:
                for probe, expected in ((0.5999, True), (0.6000, False), (0.6001, False)):
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
            if (not isinstance(rule_value, (int, float)) or isinstance(rule_value, bool)
                    or not math.isfinite(rule_value)):
                errors.append(f"rule.value is not a finite non-bool number: {rule_value!r}")
            elif rule_value != FROZEN_THRESHOLD:
                errors.append(f"rule.value must be the frozen threshold {FROZEN_THRESHOLD!r}; "
                              f"got {rule_value!r}")
            elif isinstance(decision, dict):
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
