"""data_provenance.py — shared, fail-closed data/provenance helpers used by
train.py, evaluate.py, and eval_benchmark.py.

Split out from train.py so evaluate.py (which train.py already imports
topk_accuracy from) can share this logic without a circular import.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from data import EXT_PROBE_ORDER, _manifest_from_csv


class DataIntegrityError(RuntimeError):
    """A required data invariant did not hold. Always fail closed."""


# Postconditions of the frozen 65-species Northeast catalog. Asserted only
# for the explicit-manifest source path, never for the legacy DB/bare-walk
# paths (which may load a different, smaller dataset, e.g. a CPU smoke run)
# -- and never used to construct the data, only to gate it after the fact.
# See data_pipeline/build_northeast_training_catalog.py for the generator
# that independently enforces the same counts.
EXPECTED_SAMPLE_COUNT = 13581
EXPECTED_CLASS_COUNT = 65
EXPECTED_TRAIN_COUNT = 10985
EXPECTED_VAL_COUNT = 2596

# The 15 new Northeast species slugs. Intentionally duplicated here (also
# duplicated in training/test_geo_split.py and
# data_pipeline/build_northeast_training_catalog.py's NEW_SPECIES_SLUGS) --
# each of these three call sites is a small, frozen, self-contained fact
# rather than a cross-package import, consistent with the "loosely-coupled
# stages" convention in AGENTS.md.
NORTHEAST_NEW_SPECIES_SLUGS = frozenset({
    "aphaenogaster-rudis", "camponotus-americanus", "camponotus-nearcticus",
    "camponotus-novaeboracensis", "camponotus-subbarbatus", "formica-exsectoides",
    "lasius-americanus", "lasius-aphidicola", "lasius-claviger", "lasius-emarginatus",
    "lasius-interjectus", "lasius-neoniger", "nylanderia-flavipes",
    "ponera-pennsylvanica", "temnothorax-curvispinosus",
})


def resolved_config_sha256(cfg: dict) -> str:
    """Deterministic hash of a resolved training config, EXCLUDING
    artifacts_dir (the output location doesn't affect training numerics,
    and comparing it as part of resume/eval-compat could false-mismatch on
    two equivalent, e.g. relative vs. absolute, paths to the same dir).

    Shared by train.py (provenance binding) and evaluate.py/eval_benchmark.py
    (verifying a user-supplied --config against a checkpoint's own recorded
    config) so the two can never silently compute this differently.
    """
    without_artifacts_dir = {k: v for k, v in cfg.items() if k != "artifacts_dir"}
    return hashlib.sha256(
        json.dumps(without_artifacts_dir, indent=2, sort_keys=True).encode()
    ).hexdigest()


def taxonomy_matches_committed(taxonomy: dict[int, dict], committed: dict) -> bool:
    """True iff `taxonomy` (int-keyed, as returned by any data.py loader) is
    exactly equal -- full object, not just genus/slug -- to `committed` (the
    str-keyed JSON object loaded from a committed taxonomy file)."""
    return {str(k): v for k, v in taxonomy.items()} == committed


def load_explicit_manifest_source(manifest_csv: Path, local_data_dir: Path,
                                  taxonomy_json: Path, expected_manifest_sha256: str,
                                  expected_taxonomy_sha256: str, *,
                                  database_url: str | None):
    """Fail-closed, explicit-source data loading for a controlled run.

    Bypasses data.load_manifest()'s DATABASE_URL > MANIFEST_CSV > bare-walk
    precedence entirely: `database_url` being truthy is itself a hard
    failure here -- never silently unset, never silently overridden -- and
    there is no fallback to a bare directory walk. Calls
    data._manifest_from_csv directly: the real, production CSV-loading code,
    not a reimplementation.

    Returns (samples, taxonomy, manifest_sha256, taxonomy_sha256,
    committed_taxonomy). Raises DataIntegrityError (never returns partial
    results) on any check failure.
    """
    if database_url:
        raise DataIntegrityError(
            "DATABASE_URL is set in the environment. This explicit-manifest run must "
            "not silently prefer, fall back to, or bypass it -- unset DATABASE_URL "
            "yourself before running, or do not pass an explicit manifest source at all."
        )
    if not manifest_csv.exists():
        raise DataIntegrityError(f"manifest does not exist: {manifest_csv}")
    if not local_data_dir.exists():
        raise DataIntegrityError(f"local data dir does not exist: {local_data_dir}")
    if not taxonomy_json.exists():
        raise DataIntegrityError(f"taxonomy source does not exist: {taxonomy_json}")

    manifest_bytes = manifest_csv.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise DataIntegrityError(
            f"manifest sha256 {manifest_sha256} does not match expected "
            f"{expected_manifest_sha256} -- refusing to proceed against an unexpected manifest."
        )
    taxonomy_bytes = taxonomy_json.read_bytes()
    taxonomy_sha256 = hashlib.sha256(taxonomy_bytes).hexdigest()
    if taxonomy_sha256 != expected_taxonomy_sha256:
        raise DataIntegrityError(
            f"taxonomy sha256 {taxonomy_sha256} does not match expected "
            f"{expected_taxonomy_sha256} -- refusing to proceed against an unexpected taxonomy."
        )
    committed_taxonomy = json.loads(taxonomy_bytes)

    samples, taxonomy = _manifest_from_csv(manifest_csv, local_data_dir)

    if not taxonomy_matches_committed(taxonomy, committed_taxonomy):
        raise DataIntegrityError(
            f"Taxonomy derived from {manifest_csv.name} does not exactly match the "
            f"committed taxonomy object at {taxonomy_json.name} -- refusing to proceed."
        )
    return samples, taxonomy, manifest_sha256, taxonomy_sha256, committed_taxonomy


def assert_dataset_shape(samples, taxonomy) -> tuple[int, int]:
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise DataIntegrityError(f"expected {EXPECTED_SAMPLE_COUNT} samples, got {len(samples)}")
    if len(taxonomy) != EXPECTED_CLASS_COUNT:
        raise DataIntegrityError(f"expected {EXPECTED_CLASS_COUNT} classes, got {len(taxonomy)}")
    n_train = sum(1 for s in samples if s.__dict__.get("split") == "train")
    n_val = sum(1 for s in samples if s.__dict__.get("split") == "val")
    if n_train != EXPECTED_TRAIN_COUNT or n_val != EXPECTED_VAL_COUNT:
        raise DataIntegrityError(
            f"expected {EXPECTED_TRAIN_COUNT} train / {EXPECTED_VAL_COUNT} val samples, "
            f"got {n_train} train / {n_val} val"
        )
    return n_train, n_val


def check_per_class_counts(per_class_counts: dict[str, dict[str, int]]) -> list[str]:
    """Postcondition check (report-only -- callers decide whether to fail
    closed on the returned problem list): every Northeast new-species slug
    must be exactly 200 train / 40 val; every other (legacy) slug must have
    158-160 train and a nonempty (>0) val count -- the documented range,
    never rebalanced here."""
    problems: list[str] = []
    seen_new = set()
    for slug, counts in per_class_counts.items():
        train_n, val_n = counts.get("train", 0), counts.get("val", 0)
        if slug in NORTHEAST_NEW_SPECIES_SLUGS:
            seen_new.add(slug)
            if train_n != 200 or val_n != 40:
                problems.append(f"{slug}: expected 200 train / 40 val, got "
                                f"{train_n} train / {val_n} val")
        else:
            if not (158 <= train_n <= 160):
                problems.append(f"{slug}: expected 158-160 legacy train images, got {train_n}")
            if val_n <= 0:
                problems.append(f"{slug}: expected a nonempty val set, got {val_n}")
    missing_new = NORTHEAST_NEW_SPECIES_SLUGS - seen_new
    if missing_new:
        problems.append(f"missing Northeast species entirely: {sorted(missing_new)}")
    return problems


def verify_image_bytes(manifest_csv: Path, local_data_dir: Path) -> dict:
    """Independently re-reads manifest_csv and, for every logical
    slug/photo_id row, resolves {local_data_dir}/{slug}/{photo_id}.{ext}
    through the same deterministic extension probe order data.py's loader
    uses, and verifies the resolved file's actual sha256 against the row's
    recorded sha256.

    Fails closed (raises DataIntegrityError) on ANY: duplicate logical key,
    missing file, resolution ambiguity (more than one probed extension
    exists for the same key), missing recorded sha256, or a sha256 mismatch.
    Runs once, before model initialization -- never per epoch.

    Returns {"files_verified": int, "total_bytes": int}.
    """
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    seen_keys: dict[str, dict] = {}
    problems: list[str] = []
    for r in rows:
        key = f"{r['slug']}/{r['photo_id']}"
        if key in seen_keys:
            problems.append(f"duplicate logical key {key!r} in {manifest_csv.name}")
            continue
        seen_keys[key] = r

    total_bytes = 0
    verified = 0
    for key, r in seen_keys.items():
        slug, photo_id = r["slug"], r["photo_id"]
        matches = [ext for ext in EXT_PROBE_ORDER
                  if (local_data_dir / slug / f"{photo_id}{ext}").exists()]
        if not matches:
            problems.append(f"{key}: no resolved file under {local_data_dir}")
            continue
        if len(matches) > 1:
            problems.append(f"{key}: ambiguous -- {len(matches)} extensions exist {matches}")
            continue
        path = local_data_dir / slug / f"{photo_id}{matches[0]}"
        expected = (r.get("sha256") or "").strip()
        if not expected:
            problems.append(f"{key}: manifest row has no recorded sha256")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"{key}: sha256 mismatch (file={actual[:12]}..., "
                            f"manifest={expected[:12]}...)")
            continue
        verified += 1
        total_bytes += path.stat().st_size

    if problems:
        shown = "; ".join(problems[:10])
        more = f" ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise DataIntegrityError(
            f"{len(problems)} image-byte verification problem(s), e.g.: {shown}{more}"
        )
    return {"files_verified": verified, "total_bytes": total_bytes}
