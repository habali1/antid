"""run_manifest_schema.py — the ONE shared definition of run_manifest.json's
schema, imported by train.py (writer) and evaluate.py (reader) so the two
can never silently drift apart.

The schema is staged, matching the actual lifecycle of a real run:

  "initialized"     fields present the instant run_manifest.json is first
                    created (or, on resume, loaded) -- before any data has
                    been verified.
  "data_verified"   + manifest / taxonomy_source / val_split -- written once,
                    after data/taxonomy/split verification, strictly before
                    model initialization.
  "epoch_committed" + last_completed_epoch / best -- written after every
                    completed epoch (including the pause-for-smoke epoch).
  "completed"       + final_artifact_hashes -- written only at successful
                    finalization.
  "any"             validates whatever is present without requiring fields
                    beyond "initialized" -- used by a reader (evaluate.py)
                    that must tolerate a run at any point in its lifecycle,
                    though evaluate.py itself additionally requires
                    "data_verified" fields since it cannot resolve a data
                    source without them.

Every validation failure raises RunManifestValidationError with a specific,
field-named message -- never a raw KeyError/TypeError from whichever caller
happened to touch a missing or malformed field first.

Schema versions:
  1  the historical schema, from before per-epoch validation cadence existed.
     Implicitly meant "validate every epoch" (cadence 1); carries no
     `validation_cadence` field at all. Still fully READABLE (e.g. by
     evaluate.py, for a completed run) -- SUPPORTED_SCHEMA_VERSIONS keeps it
     valid to read -- but train.py refuses to RESUME a v1 run: this harness
     version's per-epoch commit shape (nullable validation_ran/metrics on
     skipped epochs) is not what a v1 run's checkpoints were written under.
  2  adds the required top-level `validation_cadence` (strict positive int).
     This is the version new runs write (RUN_MANIFEST_SCHEMA_VERSION); only
     a v2 run may be resumed.
"""
from __future__ import annotations

RUN_MANIFEST_SCHEMA_VERSION = 2

# Versions this code can still READ (e.g. for evaluate.py against a
# completed run). Only RUN_MANIFEST_SCHEMA_VERSION itself may be RESUMED --
# see train.py's bootstrap_run_manifest.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

STATUSES = frozenset({"initialized", "running", "paused_for_smoke", "failed", "completed"})

# "paused_for_smoke" is deliberately resumable: it is written only after a
# fully committed epoch (see checkpoint.commit_epoch), so resuming from it is
# exactly as safe as resuming from "running". "completed" is deliberately
# NOT resumable -- see train.py's check_resumable_status.
RESUMABLE_STATUSES = ("initialized", "running", "paused_for_smoke", "failed")

FINAL_ARTIFACT_NAMES = (
    "model.pth", "prototypes.npy", "taxonomy.json", "geo_index.json",
    "eval.json", "backbone.onnx", "val_split.json",
)

_STAGES = ("initialized", "data_verified", "epoch_committed", "completed", "any")


class RunManifestValidationError(RuntimeError):
    """run_manifest.json did not conform to the schema. Always fail closed."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RunManifestValidationError(msg)


def _is_strict_int(v) -> bool:
    """True ints only -- bool is a subclass of int in Python and must never
    silently pass as a count/epoch/version number."""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_nonneg_int(v) -> bool:
    return _is_strict_int(v) and v >= 0


def _is_sha256(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _is_nonempty_str(v) -> bool:
    return isinstance(v, str) and len(v) > 0


def validate_manifest_source(obj, field: str = "manifest") -> None:
    _require(isinstance(obj, dict), f"{field} must be an object")
    _require(_is_nonempty_str(obj.get("path")), f"{field}.path must be a nonempty string")
    _require(_is_sha256(obj.get("sha256")), f"{field}.sha256 must be a 64-char lowercase hex string")
    _require(_is_nonneg_int(obj.get("rows")), f"{field}.rows must be a non-negative integer")


def validate_taxonomy_source(obj, field: str = "taxonomy_source") -> None:
    _require(isinstance(obj, dict), f"{field} must be an object")
    _require(_is_nonempty_str(obj.get("path")), f"{field}.path must be a nonempty string")
    _require(_is_sha256(obj.get("sha256")), f"{field}.sha256 must be a 64-char lowercase hex string")
    _require(_is_nonneg_int(obj.get("num_classes")), f"{field}.num_classes must be a non-negative integer")


def validate_val_split(obj, field: str = "val_split") -> None:
    _require(isinstance(obj, dict), f"{field} must be an object")
    _require(_is_nonempty_str(obj.get("path")), f"{field}.path must be a nonempty string")
    _require(_is_sha256(obj.get("sha256")), f"{field}.sha256 must be a 64-char lowercase hex string")
    for k in ("n_train", "n_val", "n_total"):
        _require(_is_nonneg_int(obj.get(k)), f"{field}.{k} must be a non-negative integer")
    _require(obj["n_total"] == obj["n_train"] + obj["n_val"],
             f"{field}.n_total must equal n_train + n_val")


def validate_best_ref(obj, field: str = "best") -> None:
    if obj is None:
        return
    _require(isinstance(obj, dict), f"{field} must be an object or null")
    _require(_is_nonneg_int(obj.get("epoch")), f"{field}.epoch must be a non-negative integer")
    metrics = obj.get("metrics")
    _require(isinstance(metrics, dict), f"{field}.metrics must be an object")
    for k in ("top1", "top3"):
        v = metrics.get(k)
        _require(isinstance(v, (int, float)) and not isinstance(v, bool),
                 f"{field}.metrics.{k} must be numeric")
    _require(_is_nonempty_str(obj.get("filename")), f"{field}.filename must be a nonempty string")
    _require(_is_sha256(obj.get("sha256")), f"{field}.sha256 must be a 64-char lowercase hex string")


def validate_final_artifact_hashes(obj, field: str = "final_artifact_hashes") -> None:
    _require(isinstance(obj, dict), f"{field} must be an object")
    missing = [n for n in FINAL_ARTIFACT_NAMES if n not in obj]
    _require(not missing, f"{field} is missing required entries: {missing}")
    for name in FINAL_ARTIFACT_NAMES:
        _require(_is_sha256(obj[name]), f"{field}[{name!r}] must be a 64-char lowercase hex string")


def validate_run_manifest(rm: dict, *, stage: str = "any") -> None:
    """Validate a run_manifest dict for a given lifecycle `stage` (see the
    module docstring). Raises RunManifestValidationError with a specific,
    field-named message on the first problem found; returns None on success.
    """
    _require(stage in _STAGES, f"internal error: unknown validation stage {stage!r}")
    _require(isinstance(rm, dict), "run_manifest must be a JSON object")

    _require(_is_strict_int(rm.get("run_manifest_schema_version")),
             "run_manifest_schema_version must be an integer")
    schema_version = rm["run_manifest_schema_version"]
    _require(schema_version in SUPPORTED_SCHEMA_VERSIONS,
             f"unsupported run_manifest_schema_version {schema_version!r} "
             f"(this code understands {SUPPORTED_SCHEMA_VERSIONS!r})")

    status = rm.get("status")
    _require(status in STATUSES, f"status must be one of {sorted(STATUSES)}, got {status!r}")

    _require(rm.get("run_kind") in ("full", "smoke"),
             f"run_kind must be 'full' or 'smoke', got {rm.get('run_kind')!r}")

    # Schema v1 predates per-epoch validation cadence and implicitly meant
    # "validate every epoch" (cadence 1) -- it carries no validation_cadence
    # field at all, and that absence is not an error for a v1 manifest.
    # Schema v2+ must carry an explicit, strictly positive integer cadence.
    if schema_version >= 2:
        vc = rm.get("validation_cadence")
        _require(_is_strict_int(vc) and vc > 0,
                 f"validation_cadence must be a strict positive integer for "
                 f"run_manifest_schema_version >= 2, got {vc!r}")
    _require(rm.get("git_head") is None or isinstance(rm["git_head"], str),
             "git_head must be a string or null")
    _require(isinstance(rm.get("git_dirty"), bool) or rm.get("git_dirty") is None,
             "git_dirty must be a boolean or null")
    _require(isinstance(rm.get("invocations"), list) and len(rm["invocations"]) >= 1,
             "invocations must be a non-empty list")
    _require(_is_nonempty_str(rm.get("started_at_utc")), "started_at_utc must be a nonempty string")
    _require(_is_nonempty_str(rm.get("updated_at_utc")), "updated_at_utc must be a nonempty string")
    _require("finished_at_utc" in rm, "finished_at_utc key must be present (string or null)")
    _require(rm["finished_at_utc"] is None or isinstance(rm["finished_at_utc"], str),
             "finished_at_utc must be a string or null")

    needs_data = stage in ("data_verified", "epoch_committed", "completed")
    if needs_data or "val_split" in rm:
        _require("val_split" in rm, "val_split is required at this stage")
        validate_val_split(rm["val_split"])
    if needs_data or "manifest" in rm:
        _require("manifest" in rm, "manifest key is required at this stage (may be null "
                                   "for a non-explicit data source)")
        if rm["manifest"] is not None:
            validate_manifest_source(rm["manifest"])
    if needs_data or "taxonomy_source" in rm:
        _require("taxonomy_source" in rm, "taxonomy_source key is required at this stage "
                                          "(may be null for a non-explicit data source)")
        if rm["taxonomy_source"] is not None:
            validate_taxonomy_source(rm["taxonomy_source"])

    needs_epoch = stage in ("epoch_committed", "completed")
    if needs_epoch or "last_completed_epoch" in rm:
        _require(_is_nonneg_int(rm.get("last_completed_epoch")),
                 "last_completed_epoch must be a non-negative integer")
    if needs_epoch or "best" in rm:
        _require("best" in rm, "best key is required at this stage")
        validate_best_ref(rm["best"])

    is_completed_status = status == "completed"
    if stage == "completed" or (stage == "any" and is_completed_status):
        _require(rm.get("final_artifact_hashes") is not None,
                 "final_artifact_hashes is required when status is 'completed'")
        validate_final_artifact_hashes(rm["final_artifact_hashes"])
    elif rm.get("final_artifact_hashes") is not None:
        validate_final_artifact_hashes(rm["final_artifact_hashes"])
