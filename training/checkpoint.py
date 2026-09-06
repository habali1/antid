"""checkpoint.py — crash-consistent, atomic checkpoint I/O and
resume-compatibility verification for the Northeast B4 development harness
(train.py).

Design (checkpoint_last.pth is the canonical epoch-commit marker):

  checkpoint_best_epoch_NNN.pth   Immutable, versioned. Written once, when
                                  epoch NNN becomes the new best. Never
                                  overwritten again (a later best gets its
                                  own NNN). Carries the model state, epoch,
                                  metrics, resolved config, and provenance.
  checkpoint_last.pth             The resumable, canonical commit marker for
                                  the whole run. Written after EVERY epoch,
                                  atomically. Carries: completed_epoch, a
                                  reference to the current best file
                                  (epoch/metrics/filename/sha256), the
                                  CANONICAL history (every completed epoch's
                                  row, embedded directly -- not just in
                                  history.jsonl), the current model +
                                  optimizer + RNG/generator state, and
                                  provenance/resolved_config.
  history.jsonl                   A pure DERIVED/cache file, always fully
                                  rewritten from checkpoint_last's canonical
                                  history AFTER checkpoint_last itself has
                                  committed -- never the other way around,
                                  and never appended to independently. If a
                                  crash happens between the two writes,
                                  resume detects the mismatch and repairs
                                  history.jsonl deterministically from
                                  checkpoint_last.
  checkpoint_best.pth             Materialized ONLY at successful run
                                  finalization, as a plain copy of whichever
                                  checkpoint_best_epoch_NNN.pth
                                  checkpoint_last currently references.

This ordering means: a crash can leave an orphan checkpoint_best_epoch_NNN.pth
(if it crashed after that write but before checkpoint_last committed), or a
stale history.jsonl (if it crashed after checkpoint_last committed but before
the history.jsonl rewrite) -- but checkpoint_last.pth, once it exists for a
given epoch, is always internally self-consistent and always the sole source
of truth for resume. run_manifest.json is written last of all and is never
used to infer committed state.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# Every key here must still match between the running process's freshly
# computed provenance and a checkpoint's saved provenance before a resume is
# allowed to proceed. Order is fixed so a mismatch report is stable/readable.
# run_kind/limit_batches/wandb_enabled were added because they are
# behavior-changing options that must not silently differ across a resume
# (a smoke run resuming as a full run, or vice versa, must be refused).
PROVENANCE_KEYS = (
    "manifest_sha256",
    "taxonomy_sha256",
    "val_split_sha256",
    "resolved_config_sha256",
    "backbone",
    "num_classes",
    "git_commit",
    "numerical_policy",
    "run_kind",
    "limit_batches",
    "wandb_enabled",
)


class CheckpointIntegrityError(RuntimeError):
    """A required checkpoint invariant did not hold. Always fail closed."""


def _tmp_path_for(path: Path) -> Path:
    """A temp path that preserves the real suffix (unlike
    path.with_suffix(suffix + '.tmpNNN')) so suffix-sensitive writers (e.g.
    numpy's np.save, which auto-appends '.npy' to any name not already
    ending in it) behave correctly against the temp path too."""
    return path.parent / f"{path.stem}.tmp{os.getpid()}{path.suffix}"


def torch_load_trusted(path: Path, map_location=None):
    """torch.load with weights_only=False.

    PyTorch >=2.6 defaults torch.load to weights_only=True, which refuses to
    unpickle the rich, non-tensor metadata every checkpoint here carries
    (provenance dicts, resolved_config, RNG state tuples containing numpy
    arrays, etc.) -- our own checkpoint files, always written by
    atomic_torch_save in this same codebase, are trusted, so
    weights_only=False is safe here specifically. Never use this to load an
    artifact whose origin isn't this training harness."""
    return torch.load(path, map_location=map_location, weights_only=False)


def atomic_torch_save(obj, path: Path) -> None:
    """torch.save through a temp file + os.replace so a checkpoint is never
    observed half-written -- a crash mid-save leaves the previous checkpoint
    (or nothing, on the very first write) intact, never a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def best_checkpoint_filename(epoch: int) -> str:
    return f"checkpoint_best_epoch_{epoch:03d}.pth"


def serialize_history_jsonl(rows: list[dict]) -> bytes:
    import json
    return ("".join(json.dumps(r) + "\n" for r in rows)).encode("utf-8")


def provenance_mismatches(current: dict, saved: dict) -> list[str]:
    """Human-readable list of provenance keys that differ between the
    current run's provenance and a checkpoint's saved provenance.

    Empty list means resuming from `saved` is safe. Comparing each key
    individually (rather than a single combined hash) keeps every specific
    mismatch reportable: a manifest substitution, a config change, a
    numerics change, a different git commit, a smoke/full-run mismatch, or a
    different pinned split must each be named, not just "provenance changed".
    """
    problems = []
    for key in PROVENANCE_KEYS:
        cur, sav = current.get(key), saved.get(key)
        if cur != sav:
            problems.append(f"{key}: current={cur!r} != checkpoint={sav!r}")
    return problems


def commit_epoch(art: Path, *, epoch: int, is_best: bool, model_state, optimizer_state,
                 provenance: dict, resolved_config: dict, rng_state: dict,
                 train_generator_state, metrics: dict, canonical_history: list[dict],
                 history_row: dict, previous_best) -> dict:
    """Perform one crash-consistent epoch commit. Returns the new `best`
    reference dict ({"epoch", "metrics", "filename", "sha256"} or the
    unchanged `previous_best` if this epoch was not a new best).

    Order (each step only begins once the previous one has durably
    committed):
      1. If is_best: atomically write the new versioned best file, hash it.
      2. Atomically write checkpoint_last.pth -- model, optimizer, RNG/
         generator state, completed_epoch, the (possibly updated) best
         reference, and the full canonical history including this epoch's
         row.
      3. Only now: atomically rewrite history.jsonl from that same
         canonical history list.
      4. Only now: if a *different* best file was superseded, remove it
         (best-effort; a failure to remove is logged, never fatal and never
         mistaken for a resume-blocking problem -- an orphan is inert).
    """
    new_history = canonical_history + [history_row]

    best_ref = previous_best
    if is_best:
        filename = best_checkpoint_filename(epoch)
        atomic_torch_save({
            "model": model_state, "epoch": epoch, "metrics": metrics,
            "resolved_config": resolved_config, "provenance": provenance,
        }, art / filename)
        import hashlib
        sha256 = hashlib.sha256((art / filename).read_bytes()).hexdigest()
        best_ref = {"epoch": epoch, "metrics": metrics, "filename": filename, "sha256": sha256}

    last_payload = {
        "completed_epoch": epoch, "best": best_ref, "history": new_history,
        "model": model_state, "optimizer": optimizer_state,
        "rng_state": rng_state, "train_generator_state": train_generator_state,
        "provenance": provenance, "resolved_config": resolved_config,
    }
    atomic_torch_save(last_payload, art / "checkpoint_last.pth")

    atomic_write_bytes(art / "history.jsonl", serialize_history_jsonl(new_history))

    if is_best and previous_best is not None and previous_best["filename"] != best_ref["filename"]:
        old_path = art / previous_best["filename"]
        try:
            if old_path.exists():
                old_path.unlink()
        except OSError as e:  # noqa: BLE001
            print(f"[checkpoint] WARNING: could not remove superseded best file "
                 f"{old_path}: {e} (harmless orphan; will be reported, not used, on resume)")

    return best_ref


def verify_referenced_best(art: Path, checkpoint_last: dict) -> dict:
    """Verify the best file checkpoint_last["best"] references actually
    exists, hashes to what checkpoint_last recorded, and its own internal
    epoch/metrics/provenance agree with checkpoint_last. Raises
    CheckpointIntegrityError on any mismatch. Returns the loaded best
    payload (map_location="cpu") on success."""
    best_ref = checkpoint_last.get("best")
    if best_ref is None:
        raise CheckpointIntegrityError("checkpoint_last.pth has no 'best' reference at all.")
    best_path = art / best_ref["filename"]
    if not best_path.exists():
        raise CheckpointIntegrityError(
            f"checkpoint_last.pth references {best_ref['filename']}, but that file does "
            f"not exist in {art} -- refusing to resume or finalize from a missing best."
        )
    import hashlib
    actual_sha256 = hashlib.sha256(best_path.read_bytes()).hexdigest()
    if actual_sha256 != best_ref["sha256"]:
        raise CheckpointIntegrityError(
            f"{best_path} sha256 {actual_sha256} does not match checkpoint_last.pth's "
            f"recorded {best_ref['sha256']} -- refusing to trust a tampered/corrupted best file."
        )
    best_payload = torch_load_trusted(best_path, map_location="cpu")
    if best_payload.get("epoch") != best_ref["epoch"]:
        raise CheckpointIntegrityError(
            f"{best_path}'s internal epoch {best_payload.get('epoch')} does not match "
            f"checkpoint_last.pth's recorded best epoch {best_ref['epoch']}."
        )
    if best_payload.get("metrics") != best_ref["metrics"]:
        raise CheckpointIntegrityError(
            f"{best_path}'s internal metrics {best_payload.get('metrics')} do not match "
            f"checkpoint_last.pth's recorded best metrics {best_ref['metrics']}."
        )
    if best_payload.get("provenance") != checkpoint_last.get("provenance"):
        raise CheckpointIntegrityError(
            f"{best_path}'s internal provenance does not match checkpoint_last.pth's "
            f"provenance -- refusing to trust a best file from a different run lineage."
        )
    return best_payload


def repair_history_if_needed(history_path: Path, canonical_history: list[dict]) -> bool:
    """Compares history.jsonl's actual bytes to what checkpoint_last's
    canonical history would serialize to; if they differ (missing, ahead,
    truncated, duplicated -- any deviation at all), deterministically
    rewrites history.jsonl from the canonical list. Returns True iff a
    repair was actually performed (for the caller to log)."""
    canonical_bytes = serialize_history_jsonl(canonical_history)
    if history_path.exists() and history_path.read_bytes() == canonical_bytes:
        return False
    atomic_write_bytes(history_path, canonical_bytes)
    return True


def list_orphan_best_files(art: Path, referenced_filename: str | None) -> list[Path]:
    """checkpoint_best_epoch_*.pth files NOT referenced by checkpoint_last.
    Never deleted automatically -- only reported, so a crash-created orphan
    can be inspected before removal."""
    all_best_files = sorted(art.glob("checkpoint_best_epoch_*.pth"))
    return [p for p in all_best_files if p.name != referenced_filename]
