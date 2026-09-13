#!/usr/bin/env python3
"""gate_v2_contract.py — the single shared schema/arithmetic implementation
for the Gate v2 threshold-selection contract.

Both the scorer (score_calibration_v2.py) and the selector
(select_gate_v2_threshold.py) import this module rather than each
reimplementing the contract's schema, hash-binding, or decision arithmetic.
This is deliberate: a writer and a reader with independent assumptions about
"what counts as the grid" or "what counts as feasible" is exactly the kind
of drift this project has been correcting elsewhere (see
data_pipeline/scrape_gate_v2.py's history). There is exactly one
implementation of each rule here.

Nothing in this module touches an image, a model, or the network.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_VERSION = 1
CONTRACT_STATUS_FROZEN = "frozen_before_calibration_scoring"
SCORE_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1

STATUS_CANDIDATE_SELECTED = "candidate_selected"
STATUS_NO_USEFUL_GATE = "no_useful_gate_found"

KNOWN_CATEGORY = "known_holdout"
OOD_CATEGORIES = ("out_of_scope_ant", "non_ant_insect", "unrelated")
ALL_CATEGORIES = (KNOWN_CATEGORY,) + OOD_CATEGORIES
DATASET_NAME = "calibration_v2"

# The small numerical tolerance a raw cosine similarity is allowed to exceed
# [-1, 1] by before it is treated as a hard failure rather than a legitimate
# floating-point artifact of ONNX/BLAS reduction order. Defined once, here,
# before any real scoring happens -- never invented ad hoc at scoring time.
COSINE_TOLERANCE = 1e-4

# ---- the grid: 201 integer-hundredths points, -1.00 .. +1.00 -------------
GRID_START = -100
GRID_STOP = 100
GRID_STEP = 1
GRID_DIVISOR = 100
GRID_N_POINTS = (GRID_STOP - GRID_START) // GRID_STEP + 1  # 201

SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# The four implementation sources whose canonical-LF content hash is bound
# into the frozen contract -- verified before the scorer or selector does
# any real work, so a code change after the contract is frozen is caught
# structurally rather than relying on discipline alone.
IMPLEMENTATION_SOURCE_KEYS = {
    "api_inference", "gate_v2_contract", "score_calibration_v2", "select_gate_v2_threshold",
}
IMPLEMENTATION_SOURCE_PATHS = {
    "api_inference": "api/inference.py",
    "gate_v2_contract": "training/gate_v2_contract.py",
    "score_calibration_v2": "training/score_calibration_v2.py",
    "select_gate_v2_threshold": "training/select_gate_v2_threshold.py",
}

# Output paths the contract approves in advance -- a CLI --out must resolve
# to exactly one of these, never anywhere the caller likes. Any output under
# one of these prefixes is refused outright, defense-in-depth even though
# the approved paths below never target them.
FORBIDDEN_OUTPUT_PREFIXES = ("training/artifacts/", "api/", "mobile/")
REQUIRED_APPROVED_OUTPUT_KEYS = {"calibration_v2_scores", "calibration_v2_selection"}

# ---------------------------------------------------------- frozen semantics
# Every value below is the ONE frozen definition of that piece of the
# approved Gate v2 contract. freeze_gate_v2_contract.py's build_content()
# reads these rather than duplicating the literals, and validate_contract()
# below checks the loaded contract against these by exact equality -- not
# merely "present and internally consistent". A contract that is internally
# self-consistent but semantically different from the approved policy (e.g.
# a smaller calibration_v2 quota, a different approved output path, altered
# wording of the no-useful-gate action) must fail validation even though its
# content_sha256 was freshly recomputed over the altered content.
FROZEN_POLICY_NAME = "gate_v2_threshold_selection_v1"
FROZEN_IS_V1_RECONSTRUCTION = False

FROZEN_DATASET_QUOTAS = {
    "calibration_v2": {
        "total": 1250, "species_count": 65,
        "known_holdout": 650, "known_holdout_per_species": 10,
        "out_of_scope_ant": 300, "non_ant_insect": 150, "unrelated": 150,
    },
    "unknown_test_v2": {
        "total": 790, "species_count": 65,
        "known_holdout": 390, "known_holdout_per_species": 6,
        "out_of_scope_ant": 200, "non_ant_insect": 100, "unrelated": 100,
    },
}
FROZEN_UNKNOWN_TEST_V2_PURPOSE = (
    "single-use independent test; bound here for future identity "
    "verification only -- no unknown_test_v2 image may be opened until "
    "after this contract is frozen and a candidate threshold is selected "
    "and reviewed."
)
# Extra (beyond REQUIRED_DATASET_QUOTA_FIELDS) keys each dataset's quota
# entry must carry, and the exact frozen value each must equal -- REQUIRED,
# not merely tolerated if present: a contract with the field silently
# stripped (rehashed) must fail just as loudly as one with it altered.
DATASET_QUOTA_REQUIRED_EXTRA_FIELDS = {"unknown_test_v2": {"purpose": FROZEN_UNKNOWN_TEST_V2_PURPOSE}}

FROZEN_APPROVED_OUTPUTS = {
    "calibration_v2_scores": "data/calibration_v2/calibration_v2_scores.json",
    "calibration_v2_selection": "data/calibration_v2/calibration_v2_selection.json",
}

FROZEN_BINDING_PATHS = {
    "calibration_v2_csv": "data/calibration_v2/calibration_v2.csv",
    "calibration_v2_json": "data/calibration_v2/calibration_v2.json",
    "unknown_test_v2_csv": "data/unknown_test_v2/unknown_test_v2.csv",
    "unknown_test_v2_json": "data/unknown_test_v2/unknown_test_v2.json",
    "candidate_run_manifest": "training/artifacts/northeast_v1_b4_dev_v2/run_manifest.json",
    "parity_report": "training/reports/northeast_v1_b4_dev_v2_parity.json",
    "backbone_onnx": "training/artifacts/northeast_v1_b4_dev_v2/backbone.onnx",
    "prototypes_npy": "training/artifacts/northeast_v1_b4_dev_v2/prototypes.npy",
    "taxonomy_json": "training/artifacts/northeast_v1_b4_dev_v2/taxonomy.json",
}
BINDING_ALLOWED_EXTRA_KEYS = {"applicability"}

FROZEN_RUNTIME = {
    "provider_policy": {"providers": ["CPUExecutionProvider"], "exclusive": True},
    "signal": "raw_pre_geo_max_cosine",
    "rounding": "none_before_storage",
    "geo_reranking": "never_applied",
    "inference_policy_json": "never_consulted_by_scorer_or_selector",
}

FROZEN_DECISION_SHAPE = {
    "comparison": "strict_less_than",
    "equal_threshold_action": "normal_results",
    "scope": "single_global_threshold",
    "per_species_or_per_class": False,
}

FROZEN_GRID = {
    "start": GRID_START, "stop": GRID_STOP, "step": GRID_STEP,
    "divisor": GRID_DIVISOR, "n_points": GRID_N_POINTS,
    "description": "threshold = integer / 100.0 for integer in -100..100 inclusive; "
                   "generated from range(), never by repeated float addition.",
}

FROZEN_SELECTION_POLICY = {
    "coverage_floor_numerator": 65, "coverage_floor_denominator": 100,
    "usefulness_floor_numerator": 5, "usefulness_floor_denominator": 100,
    "primary_objective": "maximize_accepted_known_top1_accuracy",
    "feasibility_rule": "100 * accepted_known >= 65 * total_known (exact integer comparison)",
    "usefulness_rule": "accepted_accuracy must improve on baseline_accuracy by at least "
                       "5/100, checked by exact integer cross multiplication; equality "
                       "at exactly 5.0 percentage points passes (>=, not >)",
    "tie_break_order": ["higher_accepted_coverage", "lower_threshold"],
    "diagnostic_only_categories": list(OOD_CATEGORIES),
    "ood_never_affects_selection": True,
    "no_useful_gate_found_status": STATUS_NO_USEFUL_GATE,
    "no_useful_gate_found_action": "emit no candidate threshold or policy; "
                                   "unknown_test_v2 evaluation remains prohibited",
}

FROZEN_SINGLE_USE_RULE = {
    "unknown_test_v2": "exactly_one_evaluation_after_freeze_and_review",
    "retuning_after_unknown_test_v2": "forbidden",
    "authorization_required_before_evaluation": "a reviewed, frozen candidate threshold "
                                                 "from this contract's selection step",
}

FROZEN_CLOSED_SOURCES = ["northeast_final_test_v1"]
FROZEN_LOW_QUALITY_KNOWN = "absent_from_gate_v2_categories_and_not_used"
FROZEN_GATE_FRAMING = "selective_confidence_gate_not_unknown_species_detector"


class ContractError(RuntimeError):
    """A contract-schema, hash-binding, or score/selection-artifact
    validation problem. Always fails closed."""


# --------------------------------------------------------------- grid/arith
def generate_grid_integers() -> list[int]:
    """The 201 candidate thresholds as integer hundredths, in ascending
    order. Built from range() over integers -- never by repeated float
    addition, so there is no float accumulation error to reason about."""
    return list(range(GRID_START, GRID_STOP + GRID_STEP, GRID_STEP))


def grid_integer_to_float(value: int) -> float:
    """Display/storage-only conversion. Never used for a decision -- every
    decision in this module takes the integer directly."""
    if not is_strict_int(value):
        raise ContractError(f"grid integer must be a strict int, got {value!r}")
    return value / float(GRID_DIVISOR)


def is_strict_int(value: Any) -> bool:
    """True only for an actual int, never for bool (bool is an int subclass
    in Python, but a contract field typed as an integer must reject True/False
    the same way it would reject any other wrong type)."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    """True only for a real int/float (never bool) that is finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def coverage_feasible(accepted: int, total: int, floor_numerator: int, floor_denominator: int) -> bool:
    """Exact integer test of accepted/total >= floor_numerator/floor_denominator,
    i.e. floor_denominator*accepted >= floor_numerator*total. No floats."""
    for name, v in (("accepted", accepted), ("total", total),
                    ("floor_numerator", floor_numerator), ("floor_denominator", floor_denominator)):
        if not is_strict_int(v):
            raise ContractError(f"coverage_feasible: {name} must be a strict int, got {v!r}")
    if total <= 0 or floor_denominator <= 0:
        raise ContractError("coverage_feasible: total and floor_denominator must be positive")
    return floor_denominator * accepted >= floor_numerator * total


def accuracy_ge(correct_a: int, total_a: int, correct_b: int, total_b: int) -> bool:
    """Exact fraction comparison correct_a/total_a >= correct_b/total_b via
    integer cross multiplication. Never compares rounded display floats."""
    for name, v in (("correct_a", correct_a), ("total_a", total_a),
                    ("correct_b", correct_b), ("total_b", total_b)):
        if not is_strict_int(v):
            raise ContractError(f"accuracy_ge: {name} must be a strict int, got {v!r}")
    if total_a <= 0 or total_b <= 0:
        raise ContractError("accuracy_ge: both totals must be positive")
    return correct_a * total_b >= correct_b * total_a


def usefulness_ge_floor(correct_accepted: int, accepted_total: int,
                        correct_baseline: int, baseline_total: int,
                        floor_numerator: int, floor_denominator: int) -> bool:
    """Exact integer test of:
        correct_accepted/accepted_total - correct_baseline/baseline_total
        >= floor_numerator/floor_denominator
    Derived by clearing all three denominators (all positive), so this is
    a single integer comparison with no float arithmetic and no rounding.
    Equality at exactly the floor passes (>=, not >) -- the gate's accepted
    accuracy must IMPROVE on baseline by at least the floor; equality is
    explicitly allowed, never required to strictly exceed it."""
    for name, v in (("correct_accepted", correct_accepted), ("accepted_total", accepted_total),
                    ("correct_baseline", correct_baseline), ("baseline_total", baseline_total),
                    ("floor_numerator", floor_numerator), ("floor_denominator", floor_denominator)):
        if not is_strict_int(v):
            raise ContractError(f"usefulness_ge_floor: {name} must be a strict int, got {v!r}")
    if accepted_total <= 0 or baseline_total <= 0 or floor_denominator <= 0:
        raise ContractError("usefulness_ge_floor: accepted_total, baseline_total, "
                            "floor_denominator must be positive")
    lhs = floor_denominator * (correct_accepted * baseline_total - correct_baseline * accepted_total)
    rhs = floor_numerator * accepted_total * baseline_total
    return lhs >= rhs


def rate_or_none(numerator: int, denominator: int) -> float | None:
    """numerator/denominator as a float, or None when the denominator is
    zero -- a rate over zero observations is undefined, never reported as 0."""
    if not is_strict_int(numerator) or not is_strict_int(denominator):
        raise ContractError(f"rate_or_none: numerator/denominator must be strict ints, got "
                            f"{numerator!r}/{denominator!r}")
    if denominator == 0:
        return None
    return numerator / denominator


def ratio_or_none(a: float | None, b: float | None) -> float | None:
    """a/b as a float, or None when either input is None or b is exactly 0."""
    if a is None or b is None or b == 0:
        return None
    return a / b


# ------------------------------------------------------------------- hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_lf_bytes(data: bytes) -> bytes:
    """Normalizes CRLF to LF so a source file's identity hash is stable
    across checkouts with different line-ending settings."""
    return data.replace(b"\r\n", b"\n")


def canonical_lf_sha256_file(path: Path) -> str:
    return sha256_bytes(canonical_lf_bytes(Path(path).read_bytes()))


def compute_content_sha256(content: dict) -> str:
    """Deterministic hash over the contract's `content` block only.
    Generation metadata (timestamps, tool versions) must never be part of
    `content` -- they live in a sibling `generation` key excluded from this
    hash, so re-running the freeze step at a different time never changes
    the frozen identity of the contract."""
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def is_valid_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_HEX_RE.match(value))


def compute_identity_order_sha256(identity_triples: list[tuple[str, str, str]]) -> str:
    """The one canonical serialization binding an ORDERED sequence of
    (photo_id, observation_uuid, sha256) identity triples to a single hash.
    Order matters -- this is deliberately NOT a set/sorted hash, so it
    catches a reordering of otherwise-identical rows, not only a changed or
    substituted identity. Used to bind score records to the exact row order
    of the frozen calibration_v2.csv without requiring the selector (or this
    function itself) to ever open an image.

    Each triple must be exactly (photo_id, observation_uuid, sha256) as
    non-empty strings -- callers pass the CSV's own string fields (photo_id,
    observation_uuid, sha256) at contract-generation time, and the scorer's
    own record fields (photo_id, observation_uuid, image_sha256) at score
    time, so the two are directly comparable."""
    for t in identity_triples:
        if not (isinstance(t, tuple) and len(t) == 3 and all(isinstance(x, str) and x for x in t)):
            raise ContractError(f"identity triple must be exactly 3 non-empty strings, got {t!r}")
    canonical = json.dumps([list(t) for t in identity_triples], ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def is_safe_relative_path(path_str: Any) -> bool:
    """A bound path must be a repo-relative, forward-slash, normalized path
    that cannot escape the repository root: no absolute path, no drive
    letter, no backslashes, no '.' or '..' segment, no empty segment."""
    if not isinstance(path_str, str) or not path_str:
        return False
    if "\\" in path_str:
        return False
    if path_str.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:", path_str):
        return False
    segments = path_str.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        return False
    return True


# ------------------------------------------------------ git / implementation
def get_git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.PIPE
        ).decode("utf-8").strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ContractError(f"could not determine git HEAD in {repo_root}: {exc}") from exc


def require_clean_tracked_tree(repo_root: Path) -> None:
    """Requires the TRACKED working tree to be clean (no staged or unstaged
    modification to a tracked file). Untracked files (this preparation
    pass's own new, not-yet-committed files) are explicitly allowed --
    only drift in already-tracked content is a problem here."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root, stderr=subprocess.PIPE,
        ).decode("utf-8")
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ContractError(f"could not check git status in {repo_root}: {exc}") from exc
    if out.strip():
        raise ContractError(f"tracked working tree is not clean:\n{out}")


def verify_implementation_sources(repo_root: Path, contract: dict) -> dict[str, str]:
    """Recomputes the canonical-LF hash of every bound implementation
    source and requires it to match the frozen contract. Returns the
    actual hashes (for embedding in score/selection provenance)."""
    bound = contract["content"].get("implementation_sources", {})
    actual: dict[str, str] = {}
    problems = []
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
                f"implementation source {name!r} changed since the contract was frozen: "
                f"{path} is {actual_hash}, contract expects {entry['sha256']}"
            )
    if problems:
        raise ContractError(f"implementation-source verification failed: {problems}")
    return actual


# --------------------------------------------------------------- validation
REQUIRED_TOP_KEYS = {"schema_version", "status", "content", "content_sha256"}
# "generation" is the ONE top-level key allowed outside REQUIRED_TOP_KEYS --
# deliberately excluded from content_sha256 (see compute_content_sha256's
# docstring) so re-running the freeze step at a different time never changes
# the frozen identity. Its own key set is still bounded and type-checked
# below so it can never be used to smuggle an unvalidated semantic field
# past content_sha256's reach.
OPTIONAL_TOP_KEYS = {"generation"}
GENERATION_ALLOWED_KEYS = {"note", "generator"}
CONTENT_KEY_IDENTITY_ORDER_SHA256 = "calibration_v2_row_identity_order_sha256"
REQUIRED_CONTENT_KEYS = {
    "policy_name", "is_v1_reconstruction", "bindings", "implementation_sources", "approved_outputs",
    "dataset_quotas", "runtime", "decision_shape", "grid", "selection", "single_use_rule",
    "closed_sources", "low_quality_known", "gate_framing", CONTENT_KEY_IDENTITY_ORDER_SHA256,
}
# Free-text/documentation keys allowed in `content` beyond the required
# semantic set above -- present for human readability, never validated for
# an exact value, and never allowed to silently carry a semantic field that
# should have been added to REQUIRED_CONTENT_KEYS and validated exactly.
OPTIONAL_CONTENT_KEYS = {"notes"}
REQUIRED_BINDING_KEYS = {
    "calibration_v2_csv", "calibration_v2_json", "unknown_test_v2_csv", "unknown_test_v2_json",
    "candidate_run_manifest", "parity_report", "backbone_onnx", "prototypes_npy", "taxonomy_json",
}
REQUIRED_GRID_INT_FIELDS = ("start", "stop", "step", "divisor", "n_points")
REQUIRED_SELECTION_INT_FIELDS = (
    "coverage_floor_numerator", "coverage_floor_denominator",
    "usefulness_floor_numerator", "usefulness_floor_denominator",
)
REQUIRED_DATASET_QUOTA_FIELDS = (
    "total", "species_count", "known_holdout", "known_holdout_per_species",
    "out_of_scope_ant", "non_ant_insect", "unrelated",
)


def _check_dataset_quota_exact(ds_name: str, dq: dict, problems: list[str]) -> None:
    """Exact equality against FROZEN_DATASET_QUOTAS[ds_name] -- not merely
    internally-consistent arithmetic. A contract mutated to a smaller but
    self-consistent quota (e.g. total=4 with matching category counts) must
    still fail here."""
    frozen = FROZEN_DATASET_QUOTAS[ds_name]
    required_extra = DATASET_QUOTA_REQUIRED_EXTRA_FIELDS.get(ds_name, {})
    extra = set(dq) - set(REQUIRED_DATASET_QUOTA_FIELDS) - set(required_extra)
    if extra:
        problems.append(f"dataset_quotas.{ds_name} has unexpected key(s): {sorted(extra)}")
    for field in REQUIRED_DATASET_QUOTA_FIELDS:
        if dq.get(field) != frozen[field]:
            problems.append(f"dataset_quotas.{ds_name}.{field} must be exactly {frozen[field]!r}, "
                            f"got {dq.get(field)!r}")
    for field, expected_value in required_extra.items():
        if field not in dq:
            problems.append(f"dataset_quotas.{ds_name} missing required key {field!r}")
        elif dq[field] != expected_value:
            problems.append(f"dataset_quotas.{ds_name}.{field} must be exactly {expected_value!r}, "
                            f"got {dq[field]!r}")


def validate_contract(contract: dict) -> list[str]:
    """Structural + type + hash-consistency + exact-semantics validation.
    Returns a list of problems (empty list == valid). Never raises for a
    malformed contract -- the caller decides whether to treat problems as
    fatal. Never divides by a value that hasn't first been checked positive,
    so a malformed grid/selection field is always a validation problem,
    never a ZeroDivisionError."""
    problems: list[str] = []
    if not isinstance(contract, dict):
        return ["contract is not a JSON object"]

    missing_top = REQUIRED_TOP_KEYS - set(contract)
    if missing_top:
        problems.append(f"missing top-level key(s): {sorted(missing_top)}")
        return problems  # nothing else is safe to inspect
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

    if not is_strict_int(contract["schema_version"]) or contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        problems.append(f"schema_version must be strict int {CONTRACT_SCHEMA_VERSION}, "
                        f"got {contract['schema_version']!r}")
    if contract["status"] != CONTRACT_STATUS_FROZEN:
        problems.append(f"status must be {CONTRACT_STATUS_FROZEN!r}, got {contract['status']!r}")

    content = contract["content"]
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
    if contract["content_sha256"] != recomputed:
        problems.append(f"content_sha256 {contract['content_sha256']!r} != recomputed {recomputed!r} "
                        f"-- content has been altered since it was frozen")

    # ---- top-level scalar/list semantic fields: exact equality ------------
    if content.get("policy_name") != FROZEN_POLICY_NAME:
        problems.append(f"policy_name must be exactly {FROZEN_POLICY_NAME!r}, got {content.get('policy_name')!r}")
    if content.get("is_v1_reconstruction") != FROZEN_IS_V1_RECONSTRUCTION:
        problems.append(f"is_v1_reconstruction must be exactly {FROZEN_IS_V1_RECONSTRUCTION!r}, "
                        f"got {content.get('is_v1_reconstruction')!r}")
    if content.get("closed_sources") != FROZEN_CLOSED_SOURCES:
        problems.append(f"closed_sources must be exactly {FROZEN_CLOSED_SOURCES!r}, "
                        f"got {content.get('closed_sources')!r}")
    if content.get("low_quality_known") != FROZEN_LOW_QUALITY_KNOWN:
        problems.append(f"low_quality_known must be exactly {FROZEN_LOW_QUALITY_KNOWN!r}, "
                        f"got {content.get('low_quality_known')!r}")
    if content.get("gate_framing") != FROZEN_GATE_FRAMING:
        problems.append(f"gate_framing must be exactly {FROZEN_GATE_FRAMING!r}, "
                        f"got {content.get('gate_framing')!r}")
    if not is_valid_sha256_hex(content.get(CONTENT_KEY_IDENTITY_ORDER_SHA256)):
        problems.append(f"{CONTENT_KEY_IDENTITY_ORDER_SHA256} is not 64 lowercase hex chars: "
                        f"{content.get(CONTENT_KEY_IDENTITY_ORDER_SHA256)!r}")

    # ---- bindings: presence, hash format, path safety, exact paths --------
    bindings = content.get("bindings", {})
    if isinstance(bindings, dict):
        missing_bindings = REQUIRED_BINDING_KEYS - set(bindings)
        if missing_bindings:
            problems.append(f"bindings missing key(s): {sorted(missing_bindings)}")
        extra_bindings = set(bindings) - REQUIRED_BINDING_KEYS
        if extra_bindings:
            problems.append(f"bindings has unexpected key(s): {sorted(extra_bindings)}")
        for name, entry in bindings.items():
            if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
                problems.append(f"binding {name!r} must have path and sha256")
                continue
            extra_entry_keys = set(entry) - {"path", "sha256"} - BINDING_ALLOWED_EXTRA_KEYS
            if extra_entry_keys:
                problems.append(f"binding {name!r} has unexpected key(s): {sorted(extra_entry_keys)}")
            if not is_safe_relative_path(entry["path"]):
                problems.append(f"binding {name!r} path is not a safe repo-relative path: {entry['path']!r}")
            elif name in FROZEN_BINDING_PATHS and entry["path"] != FROZEN_BINDING_PATHS[name]:
                problems.append(f"binding {name!r} path must be exactly {FROZEN_BINDING_PATHS[name]!r}, "
                                f"got {entry['path']!r}")
            if not is_valid_sha256_hex(entry["sha256"]):
                problems.append(f"binding {name!r} sha256 is not 64 lowercase hex chars: {entry['sha256']!r}")
    else:
        problems.append("bindings is not a JSON object")

    # ---- implementation sources: presence, hash format, path safety -------
    impl = content.get("implementation_sources", {})
    if isinstance(impl, dict):
        missing_impl = IMPLEMENTATION_SOURCE_KEYS - set(impl)
        if missing_impl:
            problems.append(f"implementation_sources missing key(s): {sorted(missing_impl)}")
        extra_impl = set(impl) - IMPLEMENTATION_SOURCE_KEYS
        if extra_impl:
            problems.append(f"implementation_sources has unexpected key(s): {sorted(extra_impl)}")
        for name, entry in impl.items():
            if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
                problems.append(f"implementation_sources[{name!r}] must have path and sha256")
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

    # ---- grid: exact frozen shape, no unguarded division -------------------
    grid = content.get("grid", {})
    if isinstance(grid, dict):
        for field in REQUIRED_GRID_INT_FIELDS:
            if field not in grid or not is_strict_int(grid[field]):
                problems.append(f"grid.{field} must be a strict int, got {grid.get(field)!r}")
        ints_ok = all(is_strict_int(grid.get(f)) for f in ("start", "stop", "step"))
        if ints_ok and grid["step"] <= 0:
            problems.append(f"grid.step must be positive, got {grid['step']!r}")
        elif ints_ok and grid["step"] > 0:
            expected_n = (grid["stop"] - grid["start"]) // grid["step"] + 1
            if grid.get("n_points") != expected_n:
                problems.append(f"grid.n_points {grid.get('n_points')} != computed {expected_n}")
        if is_strict_int(grid.get("divisor")) and grid["divisor"] <= 0:
            problems.append(f"grid.divisor must be positive, got {grid['divisor']!r}")
        if (grid.get("start"), grid.get("stop"), grid.get("step"), grid.get("divisor")) != \
                (GRID_START, GRID_STOP, GRID_STEP, GRID_DIVISOR):
            problems.append(f"grid does not match the frozen -100..100 step 1 divisor 100 shape: {grid}")
        if grid != FROZEN_GRID:
            problems.append(f"grid does not exactly match the frozen grid definition: {grid!r} != {FROZEN_GRID!r}")
    else:
        problems.append("grid is not a JSON object")

    # ---- selection: exact frozen policy shape ------------------------------
    selection = content.get("selection", {})
    if isinstance(selection, dict):
        if selection != FROZEN_SELECTION_POLICY:
            problems.append(f"selection does not exactly match the frozen selection policy: "
                            f"{selection!r} != {FROZEN_SELECTION_POLICY!r}")
    else:
        problems.append("selection is not a JSON object")

    # ---- dataset quotas: every field, exact frozen values ------------------
    dataset_quotas = content.get("dataset_quotas", {})
    if isinstance(dataset_quotas, dict):
        extra_ds = set(dataset_quotas) - set(FROZEN_DATASET_QUOTAS)
        if extra_ds:
            problems.append(f"dataset_quotas has unexpected dataset key(s): {sorted(extra_ds)}")
        for ds_name in FROZEN_DATASET_QUOTAS:
            dq = dataset_quotas.get(ds_name)
            if not isinstance(dq, dict):
                problems.append(f"dataset_quotas.{ds_name} is not a JSON object")
                continue
            _check_dataset_quota_exact(ds_name, dq, problems)
    else:
        problems.append("dataset_quotas is not a JSON object")

    # ---- runtime: exact frozen shape ----------------------------------------
    runtime = content.get("runtime", {})
    if isinstance(runtime, dict):
        if runtime != FROZEN_RUNTIME:
            problems.append(f"runtime does not exactly match the frozen runtime policy: "
                            f"{runtime!r} != {FROZEN_RUNTIME!r}")
    else:
        problems.append("runtime is not a JSON object")

    # ---- decision shape: exact frozen shape ---------------------------------
    decision_shape = content.get("decision_shape", {})
    if isinstance(decision_shape, dict):
        if decision_shape != FROZEN_DECISION_SHAPE:
            problems.append(f"decision_shape does not exactly match the frozen decision shape: "
                            f"{decision_shape!r} != {FROZEN_DECISION_SHAPE!r}")
    else:
        problems.append("decision_shape is not a JSON object")

    # ---- single-use / no-retuning -------------------------------------------
    single_use = content.get("single_use_rule", {})
    if isinstance(single_use, dict):
        if single_use != FROZEN_SINGLE_USE_RULE:
            problems.append(f"single_use_rule does not exactly match the frozen rule: "
                            f"{single_use!r} != {FROZEN_SINGLE_USE_RULE!r}")
    else:
        problems.append("single_use_rule is not a JSON object")

    # ---- approved output destinations --------------------------------------
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

    return problems


def load_and_verify_contract(path: Path) -> dict:
    """Loads the frozen contract file and validates it structurally. Raises
    ContractError on any problem. Never touches an image or a model."""
    path = Path(path)
    if not path.exists():
        raise ContractError(f"contract file does not exist: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_contract(contract)
    if problems:
        raise ContractError(f"contract at {path} failed validation: {problems}")
    return contract


def verify_binding(repo_root: Path, contract: dict, binding_name: str) -> None:
    """Verifies one bindings.<binding_name> entry's sha256 against the
    actual file on disk. Raises ContractError on mismatch or a missing file."""
    entry = contract["content"]["bindings"][binding_name]
    path = Path(repo_root) / entry["path"]
    if not path.exists():
        raise ContractError(f"bound file missing: {path} (binding {binding_name!r})")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ContractError(
            f"binding {binding_name!r} hash mismatch: {path} is {actual}, contract expects {entry['sha256']}"
        )


def verify_all_bindings(repo_root: Path, contract: dict) -> None:
    for name in contract["content"]["bindings"]:
        verify_binding(repo_root, contract, name)


def verify_artifact_directory_matches_contract(repo_root: Path, contract: dict, artifacts_dir: Path) -> None:
    """Verifies that backbone.onnx / prototypes.npy / taxonomy.json /
    run_manifest.json under an ARBITRARY --artifacts-dir hash-match the
    contract's bound candidate artifacts. Identity is by byte hash, not by
    the directory string -- a copy of the exact same files at a different
    path passes; a different file set at the expected path fails."""
    b = contract["content"]["bindings"]
    checks = {
        "backbone.onnx": b["backbone_onnx"]["sha256"],
        "prototypes.npy": b["prototypes_npy"]["sha256"],
        "taxonomy.json": b["taxonomy_json"]["sha256"],
        "run_manifest.json": b["candidate_run_manifest"]["sha256"],
    }
    artifacts_dir = Path(artifacts_dir)
    problems = []
    for filename, expected_hash in checks.items():
        path = artifacts_dir / filename
        if not path.exists():
            problems.append(f"{path} does not exist")
            continue
        actual = sha256_file(path)
        if actual != expected_hash:
            problems.append(f"{path} sha256 {actual} != contract-bound {expected_hash}")
    if problems:
        raise ContractError(f"--artifacts-dir {artifacts_dir} does not match the contract-bound "
                            f"candidate artifact set: {problems}")


def verify_output_path_is_approved(repo_root: Path, contract: dict, output_key: str, requested_path: Path) -> Path:
    """Requires `requested_path` to resolve to exactly the contract's
    approved_outputs[output_key] destination. Returns the resolved,
    approved absolute path for the caller to write to. Raises ContractError
    otherwise -- including if the approved destination itself were ever
    under a forbidden prefix (defense in depth)."""
    approved_outputs = contract["content"].get("approved_outputs", {})
    approved_rel = approved_outputs.get(output_key)
    if not approved_rel:
        raise ContractError(f"contract has no approved_outputs[{output_key!r}]")
    if any(approved_rel.startswith(prefix) for prefix in FORBIDDEN_OUTPUT_PREFIXES):
        raise ContractError(f"approved_outputs[{output_key!r}] {approved_rel!r} is under a forbidden prefix")
    approved_abs = (Path(repo_root) / approved_rel).resolve()
    requested_abs = Path(requested_path).resolve()
    if requested_abs != approved_abs:
        raise ContractError(
            f"{output_key} output path {requested_abs} does not resolve to the contract-approved "
            f"destination {approved_abs}"
        )
    return approved_abs


# ------------------------------------------------------- score/selection I/O
REQUIRED_SCORE_TOP_KEYS = {"schema_version", "content", "content_sha256"}
REQUIRED_SCORE_CONTENT_KEYS = {"dataset", "row_order", "n_rows", "bindings", "runtime", "provenance", "records"}
REQUIRED_SCORE_RECORD_KEYS = {
    "category", "slug", "species", "taxon_id", "photo_id", "observation_uuid", "image_sha256",
    "true_class_index", "top1_index", "top1_slug", "top3_indices", "top3_slugs", "top3_similarities",
    "max_cosine", "top1_correct",
}
REQUIRED_IDENTITY_FIELDS = ("photo_id", "observation_uuid", "image_sha256")


def validate_score_content(scores: dict, contract: dict) -> list[str]:
    """Full structural + semantic validation of a calibration_v2 score
    artifact against the frozen contract. Returns a list of problems
    (empty == valid). Never raises KeyError/TypeError on malformed input --
    every access is guarded."""
    problems: list[str] = []
    if not isinstance(scores, dict):
        return ["scores is not a JSON object"]

    missing_top = REQUIRED_SCORE_TOP_KEYS - set(scores)
    if missing_top:
        return [f"scores missing top-level key(s): {sorted(missing_top)}"]

    if not is_strict_int(scores["schema_version"]) or scores["schema_version"] != SCORE_SCHEMA_VERSION:
        problems.append(f"scores.schema_version must be strict int {SCORE_SCHEMA_VERSION}, "
                        f"got {scores['schema_version']!r}")

    content = scores["content"]
    if not isinstance(content, dict):
        return problems + ["scores.content is not a JSON object"]

    missing_content = REQUIRED_SCORE_CONTENT_KEYS - set(content)
    if missing_content:
        problems.append(f"scores.content missing key(s): {sorted(missing_content)}")

    recomputed = compute_content_sha256(content)
    if scores.get("content_sha256") != recomputed:
        problems.append(f"scores.content_sha256 {scores.get('content_sha256')!r} != recomputed "
                        f"{recomputed!r} -- the score file has been altered since it was written")

    if content.get("dataset") != DATASET_NAME:
        problems.append(f"scores.content.dataset must be {DATASET_NAME!r}, got {content.get('dataset')!r}")
    if not content.get("row_order"):
        problems.append("scores.content.row_order must be a non-empty description of the row ordering")

    bindings = content.get("bindings", {})
    if not isinstance(bindings, dict):
        problems.append("scores.content.bindings is not a JSON object")
        bindings = {}
    contract_content = contract.get("content", {})
    contract_content = contract_content if isinstance(contract_content, dict) else {}
    contract_bindings = contract_content.get("bindings", {})
    contract_bindings = contract_bindings if isinstance(contract_bindings, dict) else {}
    if bindings.get("contract_content_sha256") != contract.get("content_sha256"):
        problems.append(f"scores bound to contract_content_sha256 {bindings.get('contract_content_sha256')!r}, "
                        f"but the loaded contract is {contract.get('content_sha256')!r}")
    calibration_v2_csv_binding = contract_bindings.get("calibration_v2_csv", {})
    calibration_v2_csv_binding = calibration_v2_csv_binding if isinstance(calibration_v2_csv_binding, dict) else {}
    expected_csv_hash = calibration_v2_csv_binding.get("sha256")
    if bindings.get("calibration_v2_csv_sha256") != expected_csv_hash:
        problems.append(f"scores bound to calibration_v2_csv_sha256 {bindings.get('calibration_v2_csv_sha256')!r}, "
                        f"but the contract binds {expected_csv_hash!r}")

    quotas = contract_content.get("dataset_quotas", {})
    quotas = quotas.get(DATASET_NAME, {}) if isinstance(quotas, dict) else {}
    quotas = quotas if isinstance(quotas, dict) else {}
    records = content.get("records")
    if not isinstance(records, list):
        return problems + ["scores.content.records is not a list"]

    if not is_strict_int(content.get("n_rows")) or content["n_rows"] != len(records):
        problems.append(f"scores.content.n_rows {content.get('n_rows')!r} != len(records) {len(records)}")
    expected_total = quotas.get("total")
    if is_strict_int(expected_total) and len(records) != expected_total:
        problems.append(f"{len(records)} score records, contract expects {expected_total}")

    # ---- runtime: object shape, CPU-exclusive, preprocessing provenance ----
    runtime = content.get("runtime")
    if not isinstance(runtime, dict):
        problems.append("scores.content.runtime is not a JSON object")
    else:
        if runtime.get("registered_providers") != ["CPUExecutionProvider"]:
            problems.append(f"scores.content.runtime.registered_providers must be exactly "
                            f"['CPUExecutionProvider'], got {runtime.get('registered_providers')!r}")
        preprocessing_contract = runtime.get("preprocessing_contract")
        if not isinstance(preprocessing_contract, dict) or not preprocessing_contract:
            problems.append("scores.content.runtime.preprocessing_contract must be a non-empty JSON object")
        if not runtime.get("onnxruntime_version"):
            problems.append("scores.content.runtime.onnxruntime_version must be a non-empty string")
        if not runtime.get("pillow_version"):
            problems.append("scores.content.runtime.pillow_version must be a non-empty string")

    # ---- provenance: git_head + implementation-source hashes vs. contract --
    provenance = content.get("provenance")
    contract_impl = contract_content.get("implementation_sources", {})
    contract_impl = contract_impl if isinstance(contract_impl, dict) else {}
    if not isinstance(provenance, dict):
        problems.append("scores.content.provenance is not a JSON object")
    else:
        git_head = provenance.get("git_head")
        if not isinstance(git_head, str) or not re.fullmatch(r"[0-9a-f]{40}", git_head):
            problems.append(f"scores.content.provenance.git_head is not a valid 40-hex-char commit id: {git_head!r}")
        impl_hashes = provenance.get("implementation_source_hashes")
        if not isinstance(impl_hashes, dict):
            problems.append("scores.content.provenance.implementation_source_hashes is not a JSON object")
        else:
            if isinstance(contract_impl, dict):
                for name, entry in contract_impl.items():
                    expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
                    actual_hash = impl_hashes.get(name)
                    if actual_hash != expected_hash:
                        problems.append(f"scores.content.provenance.implementation_source_hashes[{name!r}] "
                                        f"{actual_hash!r} != contract-bound {expected_hash!r}")
                extra_impl_hashes = set(impl_hashes) - set(contract_impl)
                if extra_impl_hashes:
                    problems.append(f"scores.content.provenance.implementation_source_hashes has "
                                    f"unexpected key(s): {sorted(extra_impl_hashes)}")

    species_count = quotas.get("species_count")
    known_per_species_quota = quotas.get("known_holdout_per_species")
    seen: dict[str, set] = {f: set() for f in REQUIRED_IDENTITY_FIELDS}
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
        missing_r = REQUIRED_SCORE_RECORD_KEYS - set(r)
        if missing_r:
            problems.append(f"record {i} missing key(s): {sorted(missing_r)}")
            continue

        cat = r["category"]
        if not isinstance(cat, str):
            problems.append(f"record {i}: category must be a string, got {cat!r}")
            continue  # every check below assumes cat is a hashable, comparable string
        cat_counts[cat] += 1
        for field in ("slug", "species", "taxon_id", "top1_slug", *REQUIRED_IDENTITY_FIELDS):
            if r.get(field) in (None, ""):
                problems.append(f"record {i}: blank required field {field!r}")

        for field in REQUIRED_IDENTITY_FIELDS:
            value = r.get(field)
            if not isinstance(value, str):
                # Already reported as a blank/missing field above when value
                # is None/""; a wrong TYPE (list/dict/bool/number) gets its
                # own message here and is never added to the dedup set,
                # since only a string is hashable-and-comparable the way an
                # identity field is supposed to be.
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
            # An element that isn't even an int (a list/dict/bool/None) is
            # never hashable-safe to feed into set() below -- reported here,
            # once, instead of letting set() raise TypeError on it.
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

    if is_strict_int(quotas.get(KNOWN_CATEGORY)) and cat_counts.get(KNOWN_CATEGORY, 0) != quotas[KNOWN_CATEGORY]:
        problems.append(f"known_holdout record count {cat_counts.get(KNOWN_CATEGORY, 0)} != "
                        f"contract {quotas[KNOWN_CATEGORY]}")
    for cat in OOD_CATEGORIES:
        if is_strict_int(quotas.get(cat)) and cat_counts.get(cat, 0) != quotas[cat]:
            problems.append(f"{cat} record count {cat_counts.get(cat, 0)} != contract {quotas[cat]}")

    # ---- known_holdout distribution: exactly N distinct slugs, M each -----
    # A skewed-but-globally-correct-total distribution (e.g. one slug with
    # 20 rows and another with 0, while the sum still matches quotas.total)
    # is NOT caught by the aggregate count check above -- checked per-slug
    # here instead.
    if is_strict_int(species_count) and is_strict_int(known_per_species_quota):
        if len(known_slug_counts) != species_count:
            problems.append(f"known_holdout has {len(known_slug_counts)} distinct slug(s), "
                            f"contract expects exactly {species_count}")
        bad_species = {s: n for s, n in known_slug_counts.items() if n != known_per_species_quota}
        if bad_species:
            problems.append(f"known_holdout per-species row count must be exactly "
                            f"{known_per_species_quota} each, mismatched: {bad_species}")

    # ---- row identity-order binding: catches reorder/substitution even ----
    # after content_sha256 has been freshly recomputed over the altered
    # records, because the identity-order hash is recomputed here from the
    # records themselves and compared to the value FROZEN in the contract,
    # never to any value merely carried inside the score file itself.
    contract_identity_hash = contract.get("content", {}).get(CONTENT_KEY_IDENTITY_ORDER_SHA256)
    if not identity_extraction_ok or len(identity_triples) != len(records):
        problems.append("could not compute row identity-order hash: one or more records has a "
                        "missing/blank photo_id, observation_uuid, or image_sha256")
    else:
        try:
            recomputed_identity_hash = compute_identity_order_sha256(identity_triples)
        except ContractError as exc:
            problems.append(f"could not recompute row identity-order hash: {exc}")
        else:
            if recomputed_identity_hash != contract_identity_hash:
                problems.append(
                    "row identity-order hash does not match the frozen contract -- records have "
                    "been reordered, are missing, or have been substituted relative to "
                    "calibration_v2.csv's row order"
                )

    return problems


def load_and_verify_score_file(path: Path, contract: dict) -> dict:
    """Loads a calibration_v2 score file and fully validates it against the
    frozen contract. Raises ContractError (never KeyError/TypeError) on any
    problem."""
    path = Path(path)
    if not path.exists():
        raise ContractError(f"score file does not exist: {path}")
    try:
        scores = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path} is not valid JSON: {exc}") from exc
    problems = validate_score_content(scores, contract)
    if problems:
        raise ContractError(f"score file {path} failed validation: {problems}")
    return scores
