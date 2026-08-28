"""Optional, fail-safe loader for the serving-side abstention policy.

The three model artifacts remain the mandatory serving contract. A missing or
invalid ``inference_policy.json`` disables only the confidence gate; it never
turns an unknown policy state into ``low_confidence=False`` and never imports
training-side code at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path

import policy_schema as schema


POLICY_FILENAME = "inference_policy.json"
CPU_ONLY_PROVIDERS = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class PolicyState:
    """Loaded policy state exposed to inference and ``/health``.

    ``threshold`` is present only when ``active`` is true. ``classify`` always
    rejects non-finite inputs, even when the gate is inactive; a numerical
    model failure must never be converted into an apparently normal result.
    """

    active: bool
    reason: str
    threshold: float | None = None

    def classify(self, raw_max_similarity: float) -> bool | None:
        if isinstance(raw_max_similarity, bool):
            raise ValueError("similarity must be a finite number, not bool")
        try:
            score = float(raw_max_similarity)
        except (TypeError, ValueError) as exc:
            raise ValueError("similarity must be a finite number") from exc
        if not math.isfinite(score):
            raise ValueError("non-finite similarity is a request error")
        if not self.active:
            return None
        if self.threshold is None:  # impossible through load_inference_policy
            raise RuntimeError("active policy has no threshold")
        # Deliberately compare the widened, unrounded raw score using strict <.
        return score < self.threshold


def _inactive(reason: str) -> PolicyState:
    return PolicyState(active=False, reason=reason, threshold=None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash_status(policy: dict) -> str | None:
    """Return a stable failure reason, or None when the content hash matches."""
    version = policy.get("policy_schema_version")
    if type(version) is not int or version != schema.SCHEMA_VERSION:
        return "invalid_schema"
    content = policy.get("content")
    if not isinstance(content, dict):
        return "invalid_schema"
    recorded = policy.get("content_sha256")
    if not schema.is_sha256_hex(recorded):
        return "content_hash_mismatch"
    try:
        expected = schema.compute_content_sha256(version, content)
    except (TypeError, ValueError):
        return "invalid_schema"
    if not hmac.compare_digest(recorded, expected):
        return "content_hash_mismatch"
    return None


def _rule_is_supported(content: dict) -> bool:
    rule = content.get("rule")
    decision = content.get("decision")
    if not isinstance(rule, dict) or not isinstance(decision, dict):
        return False
    low = decision.get("low_confidence_if")
    high = decision.get("normal_results_if")
    if not isinstance(low, dict) or not isinstance(high, dict):
        return False
    value = rule.get("value")
    low_threshold = low.get("threshold")
    high_threshold = high.get("threshold")
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value)):
        return False
    if value != schema.FROZEN_THRESHOLD:
        return False
    if value != low_threshold or value != high_threshold:
        return False
    return (
        rule.get("operator_verbatim") == schema.ALLOWED_OPERATOR
        and rule.get("operator_normalized") == schema.ALLOWED_OPERATOR_NORMALIZED
        and rule.get("signal") == schema.ALLOWED_RULE_SIGNAL
        and low.get("signal") == schema.ALLOWED_DECISION_SIGNAL
        and low.get("comparison") == "strict_less_than"
        and high.get("comparison") == "greater_than_or_equal"
        and decision.get("equal_threshold_action") == "normal_results"
        and decision.get("non_finite_action") == "request_error"
    )


def load_inference_policy(
    artifacts_dir: Path,
    actual_providers: list[str],
    expected_preprocessing_contract: dict,
) -> PolicyState:
    """Load and bind the optional policy to the live serving environment.

    Every failure returns an inactive state with a stable reason. Only the
    three required training-to-serving artifacts are hash-bound; the optional
    geo index is intentionally outside this confidence-gate policy.
    """
    artifacts_dir = Path(artifacts_dir)
    policy_path = artifacts_dir / POLICY_FILENAME
    try:
        raw = policy_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _inactive("policy_missing")
    except UnicodeDecodeError:
        return _inactive("invalid_json")
    except OSError:
        return _inactive("io_error")

    try:
        policy = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _inactive("invalid_json")
    if not isinstance(policy, dict):
        return _inactive("invalid_schema")

    hash_reason = _content_hash_status(policy)
    if hash_reason is not None:
        return _inactive(hash_reason)

    content = policy["content"]
    if not _rule_is_supported(content):
        return _inactive("unsupported_rule")

    if content.get("provider_policy") != {
        "providers": CPU_ONLY_PROVIDERS,
        "exclusive": True,
    }:
        return _inactive("provider_mismatch")
    try:
        providers = list(actual_providers)
    except TypeError:
        return _inactive("provider_mismatch")
    if providers != CPU_ONLY_PROVIDERS:
        return _inactive("provider_mismatch")

    preprocessing = content.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return _inactive("preprocessing_mismatch")
    if preprocessing.get("contract") != expected_preprocessing_contract:
        return _inactive("preprocessing_mismatch")

    if schema.validate(policy):
        return _inactive("invalid_schema")

    recorded_hashes = content["artifact_hashes"]
    for name in sorted(schema.REQUIRED_ARTIFACT_HASH_KEYS):
        path = artifacts_dir / name
        try:
            actual_hash = _sha256(path)
        except FileNotFoundError:
            return _inactive("artifact_hash_mismatch")
        except OSError:
            return _inactive("io_error")
        if not hmac.compare_digest(recorded_hashes[name], actual_hash):
            return _inactive("artifact_hash_mismatch")

    threshold = float(content["decision"]["low_confidence_if"]["threshold"])
    return PolicyState(active=True, reason="active", threshold=threshold)
