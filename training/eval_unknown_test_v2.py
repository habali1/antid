#!/usr/bin/env python3
"""eval_unknown_test_v2.py — the single-use, independent evaluator that
mechanically applies the ALREADY-SELECTED and reviewed Phase 5C2 Gate v2
candidate threshold (0.61, from calibration_v2_selection.json) to the
independent unknown_test_v2 set, exactly once.

This module NEVER selects, sweeps, or searches a threshold -- it loads the
one frozen candidate from gate_v2_evaluation_contract.json (itself bound to
the exact byte/content hashes of the frozen selection artifacts) and applies
it. There is no --threshold, --operator, --out, or search/sweep mode by
construction.

Two modes:
  --preflight   metadata-only. Verifies the evaluation contract, every
                implementation-source hash, a clean git tree, the frozen
                selection contract, the frozen calibration_v2 scores and
                selection artifacts (including mechanically RECOMPUTING the
                selection and requiring exact equality with the stored
                result), every candidate artifact hash, and unknown_test_v2
                CSV/JSON metadata (hashes, quotas, per-species distribution,
                identity uniqueness, row-order identity binding). Never
                opens an image, never creates an ONNX inference session,
                never imports PIL, never writes any file -- including the
                attempt marker.
  --evaluate    the real, one-shot run. Repeats every --preflight check,
                THEN atomically creates the immutable attempt marker (O_EXCL
                exclusive creation) immediately before opening the first
                unknown_test_v2 image. Once the marker exists, the dataset
                is considered consumed: a crash or failure after that point
                leaves the marker in place and writes no partial output --
                there is no automatic retry.

Preprocessing/model math and image-decode/ONNX-session helpers are reused,
not reimplemented, from score_calibration_v2.py (itself one of this
evaluator's bound implementation sources); grid-independent pure arithmetic
and per-species/AUC reporting are reused from select_gate_v2_threshold.py
(also bound) to recompute the calibration_v2 selection mechanically before
ever touching unknown_test_v2.
"""
from __future__ import annotations

import argparse
import datetime
import errno
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "api"))
sys.path.insert(0, str(REPO / "data_pipeline"))
sys.path.insert(0, str(HERE))

import gate_v2_contract as gc  # noqa: E402
import gate_v2_evaluation_contract as ec  # noqa: E402
import score_calibration_v2 as sc  # noqa: E402
import select_gate_v2_threshold as sel  # noqa: E402

# Every input path this evaluator reads is derived from --repo ONLY, never
# separately overridable -- a byte-identical copy of, say, unknown_test_v2.csv
# supplied from another directory must never cause image resolution relative
# to that alternate directory. --repo itself stays overridable (needed for
# synthetic tests to point the whole tree at a tmpdir), but every other path
# always resolves to the canonical repo-relative location beneath it.
def contract_path_for(repo: Path) -> Path:
    return repo / "training/gate_v2_evaluation_contract.json"


def selection_contract_path_for(repo: Path) -> Path:
    return repo / "training/gate_v2_selection_contract.json"


def scores_path_for(repo: Path) -> Path:
    return repo / "data/calibration_v2/calibration_v2_scores.json"


def selection_path_for(repo: Path) -> Path:
    return repo / "data/calibration_v2/calibration_v2_selection.json"


def unknown_test_csv_path_for(repo: Path) -> Path:
    return repo / "data/unknown_test_v2/unknown_test_v2.csv"


def unknown_test_json_path_for(repo: Path) -> Path:
    return repo / "data/unknown_test_v2/unknown_test_v2.json"


class EvaluationRunError(RuntimeError):
    """A source-contract or evaluation-run problem. Always fails closed,
    before or during evaluation -- never produces a partial/silently
    degraded output."""


def verify_unknown_test_v2_quotas(rows: list[dict[str, str]]) -> None:
    """Exact row-count and per-category/per-species quota checks against
    the frozen unknown_test_v2 quota -- analogous to
    score_calibration_v2.verify_manifest_row_counts_and_quotas, but against
    ec.FROZEN_DATASET_QUOTA (that function hardcodes 'calibration_v2')."""
    quota = ec.FROZEN_DATASET_QUOTA
    if len(rows) != quota["total"]:
        raise EvaluationRunError(f"unknown_test_v2.csv has {len(rows)} rows, contract expects {quota['total']}")

    from collections import Counter
    counts = Counter(r["category"] for r in rows)
    for cat in gc.OOD_CATEGORIES:
        if counts.get(cat, 0) != quota[cat]:
            raise EvaluationRunError(f"{cat}: {counts.get(cat, 0)} rows, contract expects {quota[cat]}")
    known = [r for r in rows if r["category"] == gc.KNOWN_CATEGORY]
    if len(known) != quota["known_holdout"]:
        raise EvaluationRunError(f"known_holdout: {len(known)} rows, contract expects {quota['known_holdout']}")
    per_species = Counter(r["slug"] for r in known)
    bad = {s: n for s, n in per_species.items() if n != quota["known_holdout_per_species"]}
    if bad:
        raise EvaluationRunError(f"known_holdout per-species count mismatch: {bad}")
    if len(per_species) != quota["species_count"]:
        raise EvaluationRunError(f"known_holdout species count {len(per_species)} != {quota['species_count']}")


def run_preflight(args) -> dict[str, Any]:
    """Metadata-only. Never opens an image, never creates an ONNX session,
    never imports PIL, never writes any file. Every input path below is
    derived from args.repo ONLY -- see contract_path_for() and friends."""
    contract_path = contract_path_for(args.repo)
    selection_contract_path = selection_contract_path_for(args.repo)
    scores_path = scores_path_for(args.repo)
    selection_path = selection_path_for(args.repo)
    unknown_test_csv = unknown_test_csv_path_for(args.repo)
    unknown_test_json = unknown_test_json_path_for(args.repo)

    contract = ec.load_and_verify_evaluation_contract(contract_path)
    ec.require_clean_tracked_tree(args.repo)
    git_head = ec.get_git_head(args.repo)
    implementation_hashes = ec.verify_implementation_sources(args.repo, contract)

    selection_contract = gc.load_and_verify_contract(selection_contract_path)
    if selection_contract["content_sha256"] != ec.FROZEN_SELECTION_CONTRACT_CONTENT_SHA256:
        raise EvaluationRunError(
            f"selection contract content_sha256 {selection_contract['content_sha256']!r} != frozen "
            f"{ec.FROZEN_SELECTION_CONTRACT_CONTENT_SHA256!r}"
        )

    scores_byte_hash = gc.sha256_file(scores_path)
    if scores_byte_hash != ec.FROZEN_SCORES_BYTE_SHA256:
        raise EvaluationRunError(f"{scores_path} byte sha256 {scores_byte_hash} != frozen "
                                 f"{ec.FROZEN_SCORES_BYTE_SHA256}")
    scores = gc.load_and_verify_score_file(scores_path, selection_contract)
    if scores["content_sha256"] != ec.FROZEN_SCORES_CONTENT_SHA256:
        raise EvaluationRunError(f"{scores_path} content_sha256 {scores['content_sha256']} != frozen "
                                 f"{ec.FROZEN_SCORES_CONTENT_SHA256}")

    selection_byte_hash = gc.sha256_file(selection_path)
    if selection_byte_hash != ec.FROZEN_SELECTION_BYTE_SHA256:
        raise EvaluationRunError(f"{selection_path} byte sha256 {selection_byte_hash} != frozen "
                                 f"{ec.FROZEN_SELECTION_BYTE_SHA256}")
    selection_doc = json.loads(selection_path.read_text(encoding="utf-8"))
    recomputed_selection_content_hash = gc.compute_content_sha256(selection_doc.get("content", {}))
    if selection_doc.get("content_sha256") != recomputed_selection_content_hash:
        raise EvaluationRunError(f"{selection_path} content_sha256 does not match its own recomputed hash "
                                 f"-- the file has been altered since it was written")
    if selection_doc["content_sha256"] != ec.FROZEN_SELECTION_CONTENT_SHA256:
        raise EvaluationRunError(f"{selection_path} content_sha256 {selection_doc['content_sha256']} != "
                                 f"frozen {ec.FROZEN_SELECTION_CONTENT_SHA256}")
    if selection_doc["content"]["status"] != ec.FROZEN_THRESHOLD_STATUS_REQUIRED:
        raise EvaluationRunError(f"selection status must be {ec.FROZEN_THRESHOLD_STATUS_REQUIRED!r}, "
                                 f"got {selection_doc['content']['status']!r}")

    recomputed_result = sel.select_threshold(scores, selection_contract)
    stored_result = selection_doc["content"]["result"]
    recomputed_no_diag = {k: v for k, v in recomputed_result.items() if k != "diagnostics"}
    if recomputed_no_diag != stored_result:
        raise EvaluationRunError("recomputed selection result does not exactly match the stored "
                                 "calibration_v2_selection.json result -- refusing to evaluate against "
                                 "a threshold that cannot be mechanically reproduced")
    if recomputed_result["diagnostics"] != selection_doc["content"]["diagnostics"]:
        raise EvaluationRunError("recomputed selection diagnostics do not exactly match the stored "
                                 "calibration_v2_selection.json diagnostics")

    candidate = stored_result.get("candidate", {})
    if candidate.get("threshold_integer") != ec.FROZEN_THRESHOLD_INTEGER:
        raise EvaluationRunError(f"selected threshold_integer {candidate.get('threshold_integer')!r} != "
                                 f"frozen {ec.FROZEN_THRESHOLD_INTEGER!r}")
    if candidate.get("threshold") != ec.FROZEN_THRESHOLD:
        raise EvaluationRunError(f"selected threshold {candidate.get('threshold')!r} != frozen "
                                 f"{ec.FROZEN_THRESHOLD!r}")
    if candidate.get("comparison") != ec.FROZEN_DECISION_SHAPE["comparison"]:
        raise EvaluationRunError("selected comparison does not match the frozen strict_less_than rule")
    if candidate.get("equal_threshold_action") != ec.FROZEN_DECISION_SHAPE["equal_threshold_action"]:
        raise EvaluationRunError("selected equal_threshold_action does not match the frozen rule "
                                 "(equality must be accepted, not rejected)")

    ec.verify_all_bindings(args.repo, contract)

    actual_csv_hash = gc.sha256_file(unknown_test_csv)
    if actual_csv_hash != ec.FROZEN_UNKNOWN_TEST_V2_HASHES["unknown_test_v2_csv"]:
        raise EvaluationRunError(f"{unknown_test_csv} sha256 {actual_csv_hash} != frozen "
                                 f"{ec.FROZEN_UNKNOWN_TEST_V2_HASHES['unknown_test_v2_csv']}")
    actual_json_hash = gc.sha256_file(unknown_test_json)
    if actual_json_hash != ec.FROZEN_UNKNOWN_TEST_V2_HASHES["unknown_test_v2_json"]:
        raise EvaluationRunError(f"{unknown_test_json} sha256 {actual_json_hash} != frozen "
                                 f"{ec.FROZEN_UNKNOWN_TEST_V2_HASHES['unknown_test_v2_json']}")

    rows = sc.read_csv_rows(unknown_test_csv)
    verify_unknown_test_v2_quotas(rows)
    sc.verify_manifest_provenance_fields(rows)
    sc.verify_manifest_identity_uniqueness(rows)
    slug_to_idx = sc.verify_taxonomy_and_prototypes(args.repo, selection_contract)
    sc.verify_known_holdout_slugs_map(args.repo, selection_contract, rows, slug_to_idx)

    identity_order_sha256 = gc.compute_identity_order_sha256(
        [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in rows]
    )
    if identity_order_sha256 != contract["content"][ec.CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256]:
        raise EvaluationRunError(
            f"recomputed unknown_test_v2 row identity-order hash {identity_order_sha256} does not match "
            f"the contract-bound {contract['content'][ec.CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256]} "
            f"-- unknown_test_v2.csv row order has changed since the evaluation contract was frozen"
        )

    marker_path = (args.repo / contract["content"]["approved_outputs"]
                  ["unknown_test_v2_evaluation_attempt"])
    eval_out_path = (args.repo / contract["content"]["approved_outputs"]["unknown_test_v2_eval"])
    if marker_path.exists():
        raise EvaluationRunError(f"refusing to run: attempt marker already exists: {marker_path}")
    if eval_out_path.exists():
        raise EvaluationRunError(f"refusing to run: evaluation output already exists: {eval_out_path}")

    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise EvaluationRunError(f"CPUExecutionProvider not available in this runtime: {available}")

    return {
        "ok": True, "evaluation_contract_content_sha256": contract["content_sha256"],
        "selection_contract_content_sha256": selection_contract["content_sha256"],
        "scores_content_sha256": scores["content_sha256"],
        "selection_content_sha256": selection_doc["content_sha256"],
        "threshold_integer": ec.FROZEN_THRESHOLD_INTEGER, "threshold": ec.FROZEN_THRESHOLD,
        "n_rows": len(rows), "n_known_species": len(slug_to_idx),
        "unknown_test_v2_csv_sha256": actual_csv_hash,
        "unknown_test_v2_row_identity_order_sha256": identity_order_sha256,
        "onnxruntime_available_providers": available,
        "marker_path": str(marker_path), "eval_out_path": str(eval_out_path),
        "contract": contract, "selection_contract": selection_contract, "rows": rows,
        "slug_to_idx": slug_to_idx, "git_head": git_head, "implementation_hashes": implementation_hashes,
        "unknown_test_csv": unknown_test_csv,
    }


def _create_attempt_marker(marker_path: Path, contract: dict, git_head: str,
                           implementation_hashes: dict[str, str]) -> tuple[dict, str]:
    """Atomically creates the immutable attempt marker via EXCLUSIVE
    creation (O_CREAT|O_EXCL) -- a second concurrent/duplicate invocation
    fails here, not on a racy exists()-then-write. Never overwrites or
    updates an existing marker.

    Validates the in-memory marker BEFORE writing anything, then re-reads
    and re-validates the persisted bytes off disk (after flush/fsync) and
    only then computes the byte sha256 that gets embedded in the eventual
    evaluation output -- so the hash always refers to a marker that has
    itself been proven well-formed and contract-bound, not merely to
    whatever bytes happened to land on disk. Returns
    (marker_dict, marker_file_sha256)."""
    created_at_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = ec.build_attempt_marker_content(git_head, contract, implementation_hashes, created_at_utc)
    marker = {
        "schema_version": ec.ATTEMPT_MARKER_SCHEMA_VERSION,
        "content": content,
        "content_sha256": ec.compute_content_sha256(content),
    }
    problems = ec.validate_attempt_marker(marker, contract)
    if problems:
        raise EvaluationRunError(f"refusing to create attempt marker: it failed self-validation "
                                 f"before being written: {problems}")

    text = json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(marker_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise EvaluationRunError(f"refusing to run: attempt marker already exists: {marker_path}") from None
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise EvaluationRunError(f"refusing to run: attempt marker already exists: {marker_path}") from None
        raise
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())

    persisted_bytes = marker_path.read_bytes()
    try:
        persisted_marker = json.loads(persisted_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationRunError(f"attempt marker at {marker_path} is not valid JSON immediately "
                                 f"after being written: {exc}") from exc
    persisted_problems = ec.validate_attempt_marker(persisted_marker, contract)
    if persisted_problems:
        raise EvaluationRunError(f"attempt marker at {marker_path} failed validation immediately "
                                 f"after being written: {persisted_problems}")
    marker_sha256 = gc.sha256_bytes(persisted_bytes)
    return persisted_marker, marker_sha256


def publish_eval_output(eval_out_path: Path, text: str) -> None:
    """Atomically AND exclusively publishes the final evaluation output.

    os.replace() was the original approach, but os.replace() deliberately
    OVERWRITES an existing destination -- wrong for a single-use artifact
    that must refuse overwrite at the actual publication boundary, not only
    because an earlier existence check happened to run first. Instead:
    write a unique temp file in the SAME directory as the destination (so
    the hard link below is guaranteed to be same-filesystem), fsync it,
    then os.link() it to the destination. os.link() itself fails with
    FileExistsError if the destination already exists, so publication is
    exclusive by construction, not merely by a prior check -- there is no
    window in which two racing writers could both believe the destination
    is free. The temporary file is always removed afterward, on every
    success or failure path, and a pre-existing destination is left
    byte-for-byte untouched on failure."""
    eval_out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = eval_out_path.with_suffix(eval_out_path.suffix + f".tmp{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        try:
            os.link(str(tmp_path), str(eval_out_path))
        except FileExistsError:
            raise EvaluationRunError(
                f"refusing to overwrite existing evaluation output: {eval_out_path}"
            ) from None
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise EvaluationRunError(
                    f"refusing to overwrite existing evaluation output: {eval_out_path}"
                ) from None
            raise
    finally:
        tmp_path.unlink(missing_ok=True)


def run_evaluate(args) -> dict[str, Any]:
    """The real, one-shot evaluation run.

    Order (see module docstring): full metadata preflight; verify the
    explicit --artifacts-dir; import every required runtime module; load
    and validate taxonomy/prototypes and construct the ONNX session
    (load_candidate_session_and_prototypes both loads them and itself
    requires session.get_providers() == ["CPUExecutionProvider"]);
    atomically create and fsync the immutable attempt marker; ONLY THEN
    begin resolving/reading unknown_test_v2 images. No fallible
    model/session/import initialization is left pending after the marker
    if it could have been completed without opening an unknown_test_v2
    image -- a session-construction or taxonomy/prototype-load failure
    happens entirely before the marker exists, so it never consumes the
    single-use budget."""
    pre = run_preflight(args)
    contract = pre["contract"]
    rows, slug_to_idx = pre["rows"], pre["slug_to_idx"]
    git_head, implementation_hashes = pre["git_head"], pre["implementation_hashes"]
    marker_path = Path(pre["marker_path"])
    eval_out_path = Path(pre["eval_out_path"])
    unknown_test_csv = pre["unknown_test_csv"]

    gc.verify_artifact_directory_matches_contract(args.repo, contract, args.artifacts_dir)

    # ---- every fallible, image-free initialization step happens BEFORE
    # the attempt marker is created -- including numpy, which the original
    # code imported lazily inside the per-row loop (i.e. AFTER the marker
    # existed). Every required runtime import now happens here. -----------
    from PIL import Image
    import inference as api_inference
    import onnxruntime as ort
    import PIL
    import numpy as np

    session, input_name, prototypes, taxonomy = sc.load_candidate_session_and_prototypes(args.artifacts_dir)
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise EvaluationRunError(f"ONNX session providers are not CPU-exclusive: {session.get_providers()}")

    # ---- one-shot mechanism: create the marker, THEN begin reading images -
    marker, marker_sha256 = _create_attempt_marker(marker_path, contract, git_head, implementation_hashes)

    # From this point on, unknown_test_v2 is considered CONSUMED: any
    # failure below leaves the marker in place and writes no partial
    # output. There is no automatic retry.
    unknown_dir = unknown_test_csv.parent
    records = []
    for row in rows:
        slug, photo_id = row["slug"], row["photo_id"]
        path = sc.resolve_one_image(unknown_dir, slug, photo_id)
        data = path.read_bytes()
        actual_hash = gc.sha256_bytes(data)
        if actual_hash != row["sha256"]:
            raise EvaluationRunError(f"{slug}/{photo_id}: sha256 mismatch against manifest")
        if len(data) != int(row["byte_size"]):
            raise EvaluationRunError(f"{slug}/{photo_id}: byte_size mismatch against manifest")
        sc.decode_and_validate_image(data, int(row["width"]), int(row["height"]))

        img = Image.open(io.BytesIO(data))
        x = api_inference.AntIdentifier.preprocess(None, img)
        emb = session.run(None, {input_name: x})[0][0]
        emb_norm = float(np.linalg.norm(emb))
        if not np.isfinite(emb_norm) or emb_norm <= 1e-8:
            raise EvaluationRunError(f"{slug}/{photo_id}: invalid embedding (norm={emb_norm})")
        emb = emb / emb_norm
        sims = prototypes @ emb
        if sims.size == 0 or not bool(np.isfinite(sims).all()):
            raise EvaluationRunError(f"{slug}/{photo_id}: non-finite similarity score -- never clamped")
        if bool((np.abs(sims) > 1.0 + gc.COSINE_TOLERANCE).any()):
            bad = sims[np.abs(sims) > 1.0 + gc.COSINE_TOLERANCE]
            raise EvaluationRunError(f"{slug}/{photo_id}: raw cosine {bad.tolist()} outside "
                                    f"[-1,1] tolerance {gc.COSINE_TOLERANCE}")

        order = np.argsort(-sims, kind="stable")
        top3 = [int(i) for i in order[:3]]
        true_idx = slug_to_idx[slug] if row["category"] == gc.KNOWN_CATEGORY else None
        top1_idx = top3[0]
        records.append({
            "category": row["category"], "slug": slug, "species": row["species"],
            "taxon_id": row["taxon_id"], "photo_id": photo_id,
            "observation_uuid": row["observation_uuid"], "image_sha256": actual_hash,
            "true_class_index": true_idx,
            "top1_index": top1_idx, "top1_slug": taxonomy[top1_idx]["slug"],
            "top3_indices": top3,
            "top3_slugs": [taxonomy[i]["slug"] for i in top3],
            "top3_similarities": [float(sims[i]) for i in top3],
            "max_cosine": float(sims[top1_idx]),
            "top1_correct": (top1_idx == true_idx) if true_idx is not None else None,
        })

    assert [r["photo_id"] for r in records] == [row["photo_id"] for row in rows]

    identity_order_sha256 = gc.compute_identity_order_sha256(
        [(r["photo_id"], r["observation_uuid"], r["image_sha256"]) for r in records]
    )
    if identity_order_sha256 != contract["content"][ec.CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256]:
        raise EvaluationRunError("recomputed row identity-order hash does not match the contract-bound "
                                "value -- unknown_test_v2.csv row order has changed mid-run")

    validation = ec.compute_validation(records)

    content = {
        "dataset": ec.DATASET_NAME_EVAL,
        "row_order": "unknown_test_v2.csv row order, preserved exactly (verified by construction)",
        "n_rows": len(records),
        "bindings": {
            "evaluation_contract_content_sha256": contract["content_sha256"],
            "unknown_test_v2_csv_sha256": pre["unknown_test_v2_csv_sha256"],
            ec.CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256: identity_order_sha256,
            "attempt_marker_sha256": marker_sha256,
        },
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "registered_providers": session.get_providers(),
            "pillow_version": PIL.__version__,
            "preprocessing_contract": api_inference.PREPROCESSING_CONTRACT,
        },
        "provenance": {"git_head": git_head, "implementation_source_hashes": implementation_hashes},
        "records": records,
        "validation": validation,
    }
    content_sha256 = gc.compute_content_sha256(content)
    output = {
        "schema_version": ec.EVAL_OUTPUT_SCHEMA_VERSION, "content": content, "content_sha256": content_sha256,
        "generation": {"evaluator_source_sha256": implementation_hashes["eval_unknown_test_v2"],
                      "python_version": sys.version.split()[0]},
    }

    problems = ec.validate_eval_content(output, contract)
    if problems:
        raise EvaluationRunError(f"just-written evaluation content failed self-validation: {problems}")

    text = json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    publish_eval_output(eval_out_path, text)
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--artifacts-dir", type=Path, default=None,
                    help="Candidate artifact directory. REQUIRED for --evaluate; never defaults "
                        "to the live serving training/artifacts/. Its CONTENTS (byte hash), not "
                        "its path string, are verified against the contract-bound candidate.")
    # Deliberately NO --contract / --selection-contract / --scores / --selection /
    # --unknown-test-csv / --unknown-test-json flags: every input path is derived
    # from --repo alone (see contract_path_for() and friends) so a byte-identical
    # copy supplied from another directory can never redirect image resolution.
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    if args.evaluate and args.artifacts_dir is None:
        ap.error("--evaluate requires --artifacts-dir (no default; must never be the live serving "
                "directory implicitly)")

    try:
        if args.preflight:
            result = run_preflight(args)
            skip_keys = ("contract", "selection_contract", "rows", "slug_to_idx",
                        "implementation_hashes", "unknown_test_csv")
            print(json.dumps({k: v for k, v in result.items() if k not in skip_keys}, indent=2, default=str))
        else:
            result = run_evaluate(args)
            print("wrote unknown_test_v2_eval.json")
            print(f"status: {result['content']['validation']['status']}")
            print(f"content_sha256: {result['content_sha256']}")
        return 0
    except (EvaluationRunError, sc.ScoringError, ec.EvaluationError, gc.ContractError) as e:
        print(f"[eval_unknown_test_v2] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
