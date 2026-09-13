#!/usr/bin/env python3
"""score_calibration_v2.py — ONNX-CPU scorer for calibration_v2 (Gate v2
threshold selection). Reads the frozen contract (gate_v2_contract.py /
gate_v2_selection_contract.json), verifies every bound hash and manifest
invariant BEFORE any image is opened, then (real-score mode only) scores
every calibration_v2 image with the candidate's ONNX backbone under
CPUExecutionProvider and writes one record per row.

Two modes:
  --preflight   metadata-only. Verifies contract, bindings, manifest row
                counts/quotas/provenance/identity-uniqueness, taxonomy/
                prototype shape, and the committed parity report's per-node
                CPU profiling evidence. Never opens an image, never creates
                an ONNX inference session, never writes an output file. Does
                NOT require a clean git tree -- it is meant to be runnable
                at any time during preparation.
  --score       the real run (NOT executed by Phase 5C1 -- code preparation
                only). Additionally verifies the four implementation-source
                canonical hashes against the frozen contract and requires a
                clean TRACKED git working tree before doing any work, then
                scores every calibration_v2 row and writes the output
                artifact atomically, refusing to overwrite an existing one.

This script can ONLY be pointed at calibration_v2 -- there is no unknown_test
CLI path here at all, by construction (see select_gate_v2_threshold.py for
the separate, image-free selector). --artifacts-dir has no default and its
CONTENTS (not its path string) are verified byte-for-byte against the
contract-bound candidate artifacts before any image access or ORT session
construction.

Preprocessing/model math: this module never constructs api.inference's
AntIdentifier (whose __init__ reads the optional inference_policy.json and
geo_index.json). It calls AntIdentifier.preprocess as an unbound function
(that method never references `self`) and builds its own explicitly
CPU-only ONNX Runtime session plus prototype/taxonomy loading, using the
identical math api/inference.py uses -- so scoring reuses the real
production preprocessing implementation without ever reading a policy or
geo file, and without invoking AntIdentifier.__init__ at all.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "api"))                 # api/inference.py's AntIdentifier.preprocess, reused verbatim
sys.path.insert(0, str(REPO / "data_pipeline"))        # scrape_gate_v2's manifest schema, reused verbatim
sys.path.insert(0, str(HERE))

import gate_v2_contract as gc  # noqa: E402

DATASET_NAME = gc.DATASET_NAME
DEFAULT_CONTRACT_PATH = HERE / "gate_v2_selection_contract.json"
DEFAULT_CALIBRATION_CSV = REPO / "data/calibration_v2/calibration_v2.csv"
DEFAULT_CALIBRATION_JSON = REPO / "data/calibration_v2/calibration_v2.json"
DEFAULT_OUT_PATH = REPO / "data/calibration_v2/calibration_v2_scores.json"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
MIN_DIMENSION_PX = 200


class ScoringError(RuntimeError):
    """A source-contract or scoring problem. Always fails closed, before or
    during scoring -- never produces a partial/silently-degraded output."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------ verification
def verify_manifest_row_counts_and_quotas(rows: list[dict[str, str]], contract: dict) -> None:
    quotas = contract["content"]["dataset_quotas"][DATASET_NAME]
    if len(rows) != quotas["total"]:
        raise ScoringError(f"calibration_v2.csv has {len(rows)} rows, contract expects {quotas['total']}")

    from collections import Counter
    counts = Counter(r["category"] for r in rows)
    for cat in gc.OOD_CATEGORIES:
        if counts.get(cat, 0) != quotas[cat]:
            raise ScoringError(f"{cat}: {counts.get(cat, 0)} rows, contract expects {quotas[cat]}")
    known = [r for r in rows if r["category"] == gc.KNOWN_CATEGORY]
    if len(known) != quotas["known_holdout"]:
        raise ScoringError(f"known_holdout: {len(known)} rows, contract expects {quotas['known_holdout']}")
    per_species = Counter(r["slug"] for r in known)
    bad = {s: n for s, n in per_species.items() if n != quotas["known_holdout_per_species"]}
    if bad:
        raise ScoringError(f"known_holdout per-species count mismatch: {bad}")
    if len(per_species) != quotas["species_count"]:
        raise ScoringError(f"known_holdout species count {len(per_species)} != {quotas['species_count']}")


def verify_manifest_provenance_fields(rows: list[dict[str, str]]) -> None:
    import scrape_gate_v2 as sg  # data_pipeline -- the single source of truth for the manifest schema

    for i, r in enumerate(rows):
        for field in sg.MANIFEST_REQUIRED_FIELDS:
            if r.get(field) in (None, ""):
                raise ScoringError(f"calibration_v2.csv row {i} ({r.get('photo_id')}): blank required field {field!r}")


def verify_manifest_identity_uniqueness(rows: list[dict[str, str]]) -> None:
    """Every row's photo_id, observation_uuid, and sha256 must be unique
    within calibration_v2 -- checked BEFORE any image access. Never
    deduplicates; names the exact field and value on the first collision
    found (and reports up to 20)."""
    problems: list[str] = []
    for field in ("photo_id", "observation_uuid", "sha256"):
        seen: dict[str, int] = {}
        for i, r in enumerate(rows):
            value = r.get(field)
            if value in seen:
                problems.append(f"duplicate {field} {value!r}: rows {seen[value]} and {i}")
            else:
                seen[value] = i
    if problems:
        raise ScoringError(f"calibration_v2.csv contains duplicate identity value(s): {problems[:20]}")


def verify_taxonomy_and_prototypes(repo: Path, contract: dict) -> dict[str, int]:
    """Returns slug -> class index. Raises if shape/order/coverage disagree."""
    b = contract["content"]["bindings"]
    taxonomy = json.loads((repo / b["taxonomy_json"]["path"]).read_text(encoding="utf-8"))
    prototypes = np.load(repo / b["prototypes_npy"]["path"])
    expected_n = contract["content"]["dataset_quotas"][DATASET_NAME]["species_count"]

    if len(taxonomy) != expected_n:
        raise ScoringError(f"taxonomy.json has {len(taxonomy)} entries, expected {expected_n}")
    if prototypes.shape[0] != expected_n:
        raise ScoringError(f"prototypes.npy has {prototypes.shape[0]} rows, expected {expected_n}")
    keys = sorted(int(k) for k in taxonomy)
    if keys != list(range(expected_n)):
        raise ScoringError(f"taxonomy.json keys are not contiguous 0..{expected_n - 1}: {keys}")
    slugs = [taxonomy[str(i)]["slug"] for i in range(expected_n)]
    if slugs != sorted(slugs):
        raise ScoringError("taxonomy.json slugs are not sorted -- class index ordering contract violated")
    return {taxonomy[str(i)]["slug"]: i for i in range(expected_n)}


def verify_known_holdout_slugs_map(repo: Path, contract: dict, rows: list[dict[str, str]],
                                   slug_to_idx: dict[str, int]) -> None:
    """Every known_holdout row's slug must be one of the 65 taxonomy
    classes, AND its taxon_id must agree with that class's own taxon_id --
    a slug match alone is not proof of identity (see the eciton-burchellii
    canonicalization work in data_pipeline for exactly why)."""
    taxonomy = json.loads((repo / contract["content"]["bindings"]["taxonomy_json"]["path"]).read_text(encoding="utf-8"))
    known = [r for r in rows if r["category"] == gc.KNOWN_CATEGORY]
    unmapped = sorted({r["slug"] for r in known} - set(slug_to_idx))
    if unmapped:
        raise ScoringError(f"known_holdout slug(s) not present in the 65-class taxonomy: {unmapped}")
    for r in known:
        idx = slug_to_idx[r["slug"]]
        expected_taxon_id = int(taxonomy[str(idx)]["taxon_id"])
        if int(r["taxon_id"]) != expected_taxon_id:
            raise ScoringError(
                f"{r['slug']}/{r['photo_id']}: manifest taxon_id {r['taxon_id']} != "
                f"taxonomy taxon_id {expected_taxon_id} for class index {idx}"
            )


def verify_parity_provider_evidence(repo: Path, contract: dict) -> dict[str, Any]:
    """Binds to the ALREADY-COMMITTED parity report's per-node CPU
    profiling evidence. This is deliberately not re-profiled per scoring
    run -- the model/runtime pairing is unchanged and already hash-bound;
    re-profiling every image would be redundant. `session.get_providers()`
    at scoring time is a secondary corroborating check, never the primary
    evidence -- get_providers() reports what was REQUESTED, not what every
    graph node actually executed on."""
    path = repo / contract["content"]["bindings"]["parity_report"]["path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    evidence = report.get("deterministic_content", {}).get("ort_provider_evidence", {})
    if evidence.get("any_node_executed_outside_cpu") is not False:
        raise ScoringError(f"parity report's ort_provider_evidence does not confirm CPU-only execution: {evidence}")
    if evidence.get("registered_providers") != ["CPUExecutionProvider"]:
        raise ScoringError(f"parity report's registered_providers is not CPU-exclusive: {evidence}")
    return evidence


def run_preflight(args) -> dict[str, Any]:
    """Metadata-only. Never opens an image. Never creates an ONNX inference
    session (not even an unused one) -- only a static provider-availability
    query, which requires no model to be loaded. Does not require a clean
    git tree -- see module docstring."""
    contract = gc.load_and_verify_contract(args.contract)
    gc.verify_all_bindings(args.repo, contract)

    calib_csv_path = args.calibration_csv
    actual_csv_hash = sha256_file(calib_csv_path)
    expected_csv_hash = contract["content"]["bindings"]["calibration_v2_csv"]["sha256"]
    if actual_csv_hash != expected_csv_hash:
        raise ScoringError(f"{calib_csv_path} sha256 {actual_csv_hash} != contract-bound {expected_csv_hash}")

    rows = read_csv_rows(calib_csv_path)
    verify_manifest_row_counts_and_quotas(rows, contract)
    verify_manifest_provenance_fields(rows)
    verify_manifest_identity_uniqueness(rows)
    slug_to_idx = verify_taxonomy_and_prototypes(args.repo, contract)
    verify_known_holdout_slugs_map(args.repo, contract, rows, slug_to_idx)
    parity_evidence = verify_parity_provider_evidence(args.repo, contract)

    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise ScoringError(f"CPUExecutionProvider not available in this runtime: {available}")

    return {
        "ok": True, "contract_content_sha256": contract["content_sha256"],
        "calibration_v2_csv_sha256": actual_csv_hash, "n_rows": len(rows),
        "n_known_species": len(slug_to_idx),
        "parity_provider_evidence": parity_evidence,
        "onnxruntime_available_providers": available,
        "contract": contract, "rows": rows, "slug_to_idx": slug_to_idx,
    }


# ---------------------------------------------------------------- decoding
def decode_and_validate_image(data: bytes, expected_width: int, expected_height: int) -> None:
    """Verifies the image decodes cleanly (verify() + a full independent
    reopen+load(), so a truncated file that passes verify() but fails on
    full decode is still caught), and that its decoded dimensions match
    both the frozen manifest exactly and the >=200x200 floor. Raises
    ScoringError on any problem -- never silently accepts a truncated or
    undersized image."""
    from PIL import Image, UnidentifiedImageError
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
        width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ScoringError(f"image failed to decode: {exc}") from exc
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        raise ScoringError(f"decoded image is {width}x{height}, under the {MIN_DIMENSION_PX}px floor")
    if width != expected_width or height != expected_height:
        raise ScoringError(f"decoded image is {width}x{height}, manifest expects {expected_width}x{expected_height}")


def resolve_one_image(calibration_dir: Path, slug: str, photo_id: str) -> Path:
    matches = sorted((calibration_dir / slug).glob(f"{photo_id}.*"))
    matches = [p for p in matches if p.suffix.lower() in IMG_EXTS]
    if len(matches) != 1:
        raise ScoringError(f"{slug}/{photo_id}: expected exactly one allowed-extension image, found {len(matches)}")
    return matches[0]


# ---------------------------------------------------------------- scoring
def load_candidate_session_and_prototypes(artifacts_dir: Path):
    """Builds the explicitly CPU-only ORT session and hash-verified,
    normalized prototypes/taxonomy directly -- the SAME math
    api/inference.py's AntIdentifier.__init__ uses -- without ever calling
    AntIdentifier.__init__ itself, so inference_policy.json and
    geo_index.json are never read, opened, or able to affect a score."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(artifacts_dir / "backbone.onnx"), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ScoringError(f"ONNX session providers are not CPU-exclusive: {session.get_providers()}")
    input_name = session.get_inputs()[0].name

    protos = np.load(artifacts_dir / "prototypes.npy").astype(np.float32)
    norms = np.linalg.norm(protos, axis=1, keepdims=True)
    prototypes = protos / np.clip(norms, 1e-8, None)

    raw = json.loads((artifacts_dir / "taxonomy.json").read_text(encoding="utf-8"))
    taxonomy = {int(k): v for k, v in raw.items()}
    if len(taxonomy) != prototypes.shape[0]:
        raise ScoringError(f"taxonomy has {len(taxonomy)} classes but prototypes has {prototypes.shape[0]} rows")

    return session, input_name, prototypes, taxonomy


def run_score(args) -> dict[str, Any]:
    """The real scoring run. NOT invoked by Phase 5C1 against real data --
    exists as prepared, tested code for a later, separately-authorized
    execution turn."""
    contract = gc.load_and_verify_contract(args.contract)
    approved_out = gc.verify_output_path_is_approved(args.repo, contract, "calibration_v2_scores", args.out)
    if approved_out.exists():
        raise ScoringError(f"refusing to overwrite existing score file: {approved_out}")

    # Code-identity gates -- real execution only (see module docstring for
    # why --preflight does not require these).
    gc.require_clean_tracked_tree(args.repo)
    implementation_hashes = gc.verify_implementation_sources(args.repo, contract)
    git_head = gc.get_git_head(args.repo)

    pre = run_preflight(args)
    rows, slug_to_idx = pre["rows"], pre["slug_to_idx"]
    gc.verify_artifact_directory_matches_contract(args.repo, contract, args.artifacts_dir)

    from PIL import Image
    import inference as api_inference  # api/inference.py -- only its PREPROCESSING_CONTRACT and the
                                        # unbound AntIdentifier.preprocess method are used; __init__ is
                                        # never called, so inference_policy.json/geo_index.json are
                                        # never read.

    session, input_name, prototypes, taxonomy = load_candidate_session_and_prototypes(args.artifacts_dir)

    calibration_dir = args.calibration_csv.parent
    manifest_by_key: dict[str, dict] = {(r["slug"], r["photo_id"]): r for r in rows}
    records = []
    out_path = approved_out
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    try:
        for row in rows:  # manifest (CSV) row order preserved -- the one deterministic ordering
            slug, photo_id = row["slug"], row["photo_id"]
            path = resolve_one_image(calibration_dir, slug, photo_id)
            data = path.read_bytes()
            actual_hash = sha256_bytes(data)
            if actual_hash != row["sha256"]:
                raise ScoringError(f"{slug}/{photo_id}: sha256 mismatch against manifest")
            if len(data) != int(row["byte_size"]):
                raise ScoringError(f"{slug}/{photo_id}: byte_size mismatch against manifest")
            decode_and_validate_image(data, int(row["width"]), int(row["height"]))

            img = Image.open(io.BytesIO(data))
            x = api_inference.AntIdentifier.preprocess(None, img)  # unbound call; the method never uses self
            emb = session.run(None, {input_name: x})[0][0]
            emb_norm = float(np.linalg.norm(emb))
            if not np.isfinite(emb_norm) or emb_norm <= 1e-8:
                raise ScoringError(f"{slug}/{photo_id}: invalid embedding (norm={emb_norm})")
            emb = emb / emb_norm
            sims = prototypes @ emb  # raw cosine, (65,), float32
            if sims.size == 0 or not bool(np.isfinite(sims).all()):
                raise ScoringError(f"{slug}/{photo_id}: non-finite similarity score -- never clamped, always fatal")
            if bool((np.abs(sims) > 1.0 + gc.COSINE_TOLERANCE).any()):
                bad = sims[np.abs(sims) > 1.0 + gc.COSINE_TOLERANCE]
                raise ScoringError(f"{slug}/{photo_id}: raw cosine {bad.tolist()} outside "
                                  f"[-1,1] tolerance {gc.COSINE_TOLERANCE} -- reporting raw value, never clamping")

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
                "top3_similarities": [float(sims[i]) for i in top3],  # raw float32, no rounding/clipping
                "max_cosine": float(sims[top1_idx]),
                "top1_correct": (top1_idx == true_idx) if true_idx is not None else None,
            })

        assert [r["photo_id"] for r in records] == [row["photo_id"] for row in rows]  # order preserved, always

        identity_order_sha256 = gc.compute_identity_order_sha256(
            [(r["photo_id"], r["observation_uuid"], r["image_sha256"]) for r in records]
        )
        if identity_order_sha256 != contract["content"][gc.CONTENT_KEY_IDENTITY_ORDER_SHA256]:
            raise ScoringError(
                f"recomputed row identity-order hash {identity_order_sha256} does not match the "
                f"contract-bound {contract['content'][gc.CONTENT_KEY_IDENTITY_ORDER_SHA256]} -- "
                f"calibration_v2.csv row order has changed since the contract was frozen"
            )

        import onnxruntime as ort
        import PIL

        content = {
            "dataset": DATASET_NAME,
            "row_order": "calibration_v2.csv row order, preserved exactly (verified by construction)",
            "n_rows": len(records),
            "bindings": {
                "contract_content_sha256": contract["content_sha256"],
                "calibration_v2_csv_sha256": pre["calibration_v2_csv_sha256"],
                gc.CONTENT_KEY_IDENTITY_ORDER_SHA256: identity_order_sha256,
            },
            "runtime": {
                "onnxruntime_version": ort.__version__,
                "registered_providers": session.get_providers(),
                "pillow_version": PIL.__version__,
                "preprocessing_contract": api_inference.PREPROCESSING_CONTRACT,
            },
            "provenance": {
                "git_head": git_head,
                "implementation_source_hashes": implementation_hashes,
            },
            "records": records,
        }
        content_sha256 = gc.compute_content_sha256(content)
        output = {
            "schema_version": gc.SCORE_SCHEMA_VERSION, "content": content, "content_sha256": content_sha256,
            "generation": {
                "scorer_source_sha256": implementation_hashes["score_calibration_v2"],
                "python_version": sys.version.split()[0],
            },
        }

        problems = gc.validate_score_content(output, contract)
        if problems:
            raise ScoringError(f"just-written score content failed self-validation: {problems}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--artifacts-dir", type=Path, default=None,
                    help="Candidate artifact directory. REQUIRED for --score; never defaults "
                        "to the live serving training/artifacts/. Its CONTENTS (byte hash), not "
                        "its path string, are verified against the contract-bound candidate.")
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    ap.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION_CSV)
    ap.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.score and args.artifacts_dir is None:
        ap.error("--score requires --artifacts-dir (no default; must never be the live serving directory implicitly)")

    try:
        if args.preflight:
            result = run_preflight(args)
            print(json.dumps({k: v for k, v in result.items() if k not in ("contract", "rows", "slug_to_idx")},
                             indent=2, default=str))
        else:
            result = run_score(args)
            print(f"wrote {args.out}")
            print(f"content_sha256: {result['content_sha256']}")
        return 0
    except (ScoringError, gc.ContractError) as e:
        print(f"[score_calibration_v2] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
