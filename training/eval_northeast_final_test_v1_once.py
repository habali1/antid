#!/usr/bin/env python3
"""Precommitted, single-use 65-class B4 final-test evaluator.

--preflight reads metadata and artifacts, never a dataset image. --evaluate
creates an exclusive attempt marker after session initialization and before
the first image read. An interrupted attempt is consumed, never retried.
Neither mode consults final-test results to select a model or threshold.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "api")]

import gate_v2_contract as gc
import gate_v2_evaluation_contract as ec

RULE_PATH = "training/final_test_v1_reporting_rule.json"
CSV_PATH = "data/northeast_final_test_v1/northeast_final_test_v1.csv"
JSON_PATH = "data/northeast_final_test_v1/northeast_final_test_v1.json"
V3_PATH = "training/perceptual_independence_v3_decision.json"
ARTIFACTS_PATH = "training/artifacts/northeast_v1_b4_dev_v2"
POLICY_PATH = ARTIFACTS_PATH + "/inference_policy.json"
CONTRACT_PATH = "training/gate_v2_evaluation_contract.json"
MARKER_PATH = "data/northeast_final_test_v1/northeast_final_test_v1_evaluation_attempt.json"
RESULT_PATH = "data/northeast_final_test_v1/northeast_final_test_v1_eval.json"
EXPECTED_CSV_SHA256 = "0c230bb51f8f2d074a796fce4b1d35a50595a0843d8955291a9829fc4f56f992"
EXPECTED_JSON_SHA256 = "cd652c5a49d9afd86ab81fcb6f8a2b8015466ff609e0af7788e399f2d2afbd5a"
EXPECTED_V3_SHA256 = "e6040cb06d0fabd99f61bd68645585154682e3ff6c5397b95565654e46f84b4f"
EXPECTED_POLICY_SHA256 = "9feeae83013ecf72266084421fc74bbfe21757de528ad7ed258c01dcffc9422d"
EXPECTED_DECISION_STATUS = "perceptual_independence_incomplete_by_decision"
EXPECTED_MANUAL = {"unusable_final_test_images_included": 3,
                   "poor_quality_usable_final_test_images_included": 41}


class FinalTestError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalTestError(message)


def require_committed_clean(repo: Path) -> str:
    def git(*argv: str) -> str:
        r = subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    require(not git("status", "--porcelain", "--untracked-files=no"), "tracked tree is dirty")
    head = git("rev-parse", "HEAD")
    for rel in (RULE_PATH, V3_PATH, "training/eval_northeast_final_test_v1_once.py"):
        require(git("ls-files", "--error-unmatch", rel) == rel, f"uncommitted source: {rel}")
        blob = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{rel}"],
                              capture_output=True, check=True).stdout
        require(hashlib.sha256(blob).hexdigest() == sha256(repo / rel), f"source differs from HEAD: {rel}")
    return head


def load_preflight(repo: Path) -> dict:
    repo = repo.resolve()
    head = require_committed_clean(repo)
    rule = json.loads((repo / RULE_PATH).read_text(encoding="utf-8"))
    expected = {
        "dataset_csv": CSV_PATH, "dataset_json": JSON_PATH,
        "candidate_artifacts": ARTIFACTS_PATH, "candidate_policy": POLICY_PATH,
        "v3_decision": V3_PATH, "v2_evaluation_contract": CONTRACT_PATH,
        "attempt_marker": MARKER_PATH, "result": RESULT_PATH,
        "total_images": 450, "species_count": 15, "images_per_species": 30,
        "threshold": .61, "comparison": "raw_pre_geo_max_cosine_strict_less_than_0.61_equality_accepted",
        "provider": "CPUExecutionProvider_only", "geo": "disabled",
        "independence_label": EXPECTED_DECISION_STATUS,
        "v2_perceptual_contract_result": "not_passed_not_finalized",
        "unusable_final_test_images_included": 3,
        "poor_quality_usable_final_test_images_included": 41,
        "metric_denominator": "all_450_frozen_images_no_exclusions",
        "single_use": "marker_before_first_image_no_retry_no_threshold_retuning",
        "report_status": "descriptive_result_not_perceptual_independence_pass",
    }
    for key, value in expected.items():
        require(rule.get(key) == value, f"report rule {key} differs from frozen implementation")
    require(set(rule) == set(expected) | {"schema_version", "name", "reporting"}, "report rule keys changed")
    require(rule["schema_version"] == 1 and rule["name"] == "northeast_final_test_v1_single_use_v3",
            "report rule identity changed")
    require(rule["reporting"] == [
        "micro_top1_and_top3_counts_and_rates_on_all_450",
        "macro_per_species_top1_and_top3_on_15_species",
        "gate_accepted_rejected_coverage_and_accepted_top1_top3",
        "correct_vs_incorrect_rejection_counts_and_rates",
        "per_species_counts_and_gate_metrics",
        "manual_quality_counts_and_explicit_incomplete_independence_label",
    ], "reporting fields changed")
    decision = json.loads((repo / V3_PATH).read_text(encoding="utf-8"))
    require(sha256(repo / V3_PATH) == EXPECTED_V3_SHA256, "v3 decision byte hash mismatch")
    require(decision["status"] == EXPECTED_DECISION_STATUS and
            decision["decision"]["v2_contract_result"] == "not_passed_not_finalized" and
            decision["evidence"]["remaining_in_scope"]["count"] == 3210, "v3 decision mismatch")
    for entry in (decision["evidence"][k] for k in (
            "v2_queue", "v2_contract", "partial_ledger", "perceptual_candidate_report",
            "perceptual_hash_report", "metadata_leakage_report")):
        require(sha256(repo / entry["path"]) == entry["byte_sha256"],
                f"v3 evidence changed: {entry['path']}")

    csv_path = repo / CSV_PATH
    require(sha256(csv_path) == EXPECTED_CSV_SHA256, "final-test CSV hash mismatch")
    sidecar = json.loads((repo / JSON_PATH).read_text(encoding="utf-8"))
    require(sha256(repo / JSON_PATH) == EXPECTED_JSON_SHA256, "final-test sidecar byte hash mismatch")
    require(sidecar["counts"]["per_split"] == {"final_test": 450} and
            set(sidecar["counts"]["per_species"].values()) == {30}, "final-test sidecar mismatch")
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    require(len(rows) == 450 and all(r["split"] == "final_test" for r in rows), "wrong final population")
    counts = collections.Counter(r["slug"] for r in rows)
    require(len(counts) == 15 and set(counts.values()) == {30}, "wrong per-species final population")
    keys = [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in rows]
    require(len(set(keys)) == 450, "duplicate final row identity")
    for row in rows:
        require(len(row["sha256"]) == 64 and int(row["byte_size"]) > 0 and
                int(row["width"]) > 0 and int(row["height"]) > 0, "invalid image metadata")

    contract = ec.load_and_verify_evaluation_contract(repo / CONTRACT_PATH)
    ec.verify_all_bindings(repo, contract)
    artifacts = repo / ARTIFACTS_PATH
    gc.verify_artifact_directory_matches_contract(repo, contract, artifacts)
    raw_taxonomy = json.loads((artifacts / "taxonomy.json").read_text(encoding="utf-8"))
    taxonomy = {int(k): v for k, v in raw_taxonomy.items()}
    require(set(taxonomy) == set(range(65)), "candidate taxonomy is not 65 contiguous classes")
    slug_to_idx = {v["slug"]: k for k, v in taxonomy.items()}
    require(len(slug_to_idx) == 65 and all(row["slug"] in slug_to_idx and
            taxonomy[slug_to_idx[row["slug"]]]["taxon_id"] == int(row["taxon_id"]) for row in rows),
            "final-test taxonomy mismatch")
    require(sha256(repo / POLICY_PATH) == EXPECTED_POLICY_SHA256, "candidate policy byte hash mismatch")
    import onnxruntime as ort
    import inference as api_inference
    from inference_policy import load_inference_policy
    require("CPUExecutionProvider" in ort.get_available_providers(), "CPU provider unavailable")
    state = load_inference_policy(artifacts, ["CPUExecutionProvider"], api_inference.PREPROCESSING_CONTRACT)
    require(state.active and state.threshold == .61 and state.reason == "active", "candidate gate inactive/mismatched")
    require(state.classify(.61) is False and state.classify(.609999) is True, "gate boundary mismatch")
    marker, result = repo / MARKER_PATH, repo / RESULT_PATH
    require(not marker.exists() and not result.exists(), "one-shot marker or result already exists")
    return {"head": head, "rule_sha256": sha256(repo / RULE_PATH), "v3_sha256": sha256(repo / V3_PATH),
            "csv_sha256": sha256(csv_path), "json_sha256": sha256(repo / JSON_PATH),
            "policy_sha256": sha256(repo / POLICY_PATH), "row_identity_order_sha256": gc.compute_identity_order_sha256(keys),
            "rows": rows, "artifacts": artifacts, "marker": marker, "result": result,
            "gate": state, "contract": contract, "repo": repo, "taxonomy": taxonomy,
            "slug_to_idx": slug_to_idx}


def compute_metrics(records: list[dict]) -> dict:
    require(len(records) == 450, "metrics require all 450 records")
    species = collections.defaultdict(list)
    for record in records:
        species[record["true_slug"]].append(record)
    require(len(species) == 15 and all(len(v) == 30 for v in species.values()), "metrics require 15x30")
    n1 = sum(r["top1_correct"] for r in records)
    n3 = sum(r["top3_correct"] for r in records)
    accepted = [r for r in records if not r["low_confidence"]]
    rejected = [r for r in records if r["low_confidence"]]
    correct_rejected = sum(r["top1_correct"] for r in rejected)
    incorrect_rejected = len(rejected) - correct_rejected
    by_species = {}
    for slug, rs in sorted(species.items()):
        a = [r for r in rs if not r["low_confidence"]]
        by_species[slug] = {"total": 30, "top1_correct": sum(r["top1_correct"] for r in rs),
                            "top3_correct": sum(r["top3_correct"] for r in rs),
                            "accepted": len(a), "rejected": 30 - len(a),
                            "coverage": len(a) / 30,
                            "accepted_top1_correct": sum(r["top1_correct"] for r in a),
                            "accepted_top3_correct": sum(r["top3_correct"] for r in a),
                            "accepted_top1": sum(r["top1_correct"] for r in a) / len(a) if a else None,
                            "accepted_top3": sum(r["top3_correct"] for r in a) / len(a) if a else None}
    return {"total": 450, "top1_correct": n1, "top1": n1 / 450,
            "top3_correct": n3, "top3": n3 / 450,
            "macro_top1": sum(v["top1_correct"] / 30 for v in by_species.values()) / 15,
            "macro_top3": sum(v["top3_correct"] / 30 for v in by_species.values()) / 15,
            "accepted": len(accepted), "rejected": len(rejected), "coverage": len(accepted) / 450,
            "accepted_top1_correct": sum(r["top1_correct"] for r in accepted),
            "accepted_top3_correct": sum(r["top3_correct"] for r in accepted),
            "accepted_top1": sum(r["top1_correct"] for r in accepted) / len(accepted) if accepted else None,
            "accepted_top3": sum(r["top3_correct"] for r in accepted) / len(accepted) if accepted else None,
            "correct_rejected": correct_rejected, "incorrect_rejected": incorrect_rejected,
            "correct_rejection_rate": correct_rejected / n1 if n1 else None,
            "incorrect_rejection_rate": incorrect_rejected / (450 - n1) if n1 < 450 else None,
            "per_species": by_species}


def validate_result(pre: dict, result: dict) -> None:
    require(set(result) == {"schema_version", "content", "content_sha256"} and
            result["schema_version"] == 1, "invalid result envelope")
    content = result["content"]
    require(result["content_sha256"] == canonical_hash(content), "result content hash mismatch")
    require(set(content) == {"dataset", "status", "perceptual_independence", "v2_perceptual_contract_result",
                             "manual_quality", "threshold", "provider", "geo_applied", "bindings",
                             "attempt_marker_sha256", "records", "metrics"}, "unexpected result fields")
    require(content["dataset"] == "northeast_final_test_v1" and
            content["status"] == "descriptive_result_not_perceptual_independence_pass" and
            content["perceptual_independence"] == EXPECTED_DECISION_STATUS and
            content["v2_perceptual_contract_result"] == "not_passed_not_finalized" and
            content["manual_quality"] == EXPECTED_MANUAL and content["threshold"] == .61 and
            content["provider"] == ["CPUExecutionProvider"] and content["geo_applied"] is False,
            "result framing/model rule mismatch")
    required_bindings = ("head", "rule_sha256", "v3_sha256", "csv_sha256", "json_sha256",
                         "policy_sha256", "row_identity_order_sha256")
    require(content["bindings"] == {k: pre[k] for k in required_bindings}, "result binding mismatch")
    require(content["attempt_marker_sha256"] == sha256(pre["marker"]), "attempt marker hash mismatch")
    records = content["records"]
    require(len(records) == len(pre["rows"]) == 450, "result population size mismatch")
    require(set(content["metrics"]) == set(compute_metrics(records)) and
            content["metrics"] == compute_metrics(records), "result metrics do not recompute")
    for row, record in zip(pre["rows"], records):
        require(set(record) == {"photo_id", "observation_uuid", "image_sha256", "true_slug",
                                "true_class_index", "top3_indices", "top3_slugs", "top3_similarities",
                                "max_cosine", "low_confidence", "top1_correct", "top3_correct"},
                "record schema mismatch")
        require((record["photo_id"], record["observation_uuid"], record["image_sha256"], record["true_slug"]) ==
                (row["photo_id"], row["observation_uuid"], row["sha256"], row["slug"]),
                "result row identity/order mismatch")
        require(record["true_class_index"] == pre["slug_to_idx"][row["slug"]],
                "true index differs from frozen taxonomy")
        indices, slugs, sims = (record[k] for k in ("top3_indices", "top3_slugs", "top3_similarities"))
        require(len(indices) == len(slugs) == len(sims) == 3 and
                all(type(i) is int and 0 <= i < 65 for i in indices) and len(set(indices)) == 3 and
                len(set(slugs)) == 3 and all(type(s) is str and s for s in slugs) and
                all(type(s) is float and math.isfinite(s) and abs(s) <= 1 + gc.COSINE_TOLERANCE for s in sims) and
                sims[0] >= sims[1] >= sims[2] and record["max_cosine"] == sims[0],
                "invalid top3 scores/ranking")
        require(slugs == [pre["taxonomy"][i]["slug"] for i in indices], "top3 taxonomy mismatch")
        require(type(record["low_confidence"]) is bool and
                record["low_confidence"] == (sims[0] < .61) and
                type(record["top1_correct"]) is bool and
                record["top1_correct"] == (indices[0] == record["true_class_index"]) and
                type(record["top3_correct"]) is bool and
                record["top3_correct"] == (record["true_class_index"] in indices),
                "record decision/correctness mismatch")
    require(gc.compute_identity_order_sha256([(r["photo_id"], r["observation_uuid"], r["image_sha256"])
                                                    for r in records]) == pre["row_identity_order_sha256"],
            "result identity-order binding mismatch")


def publish_exclusive(dest: Path, payload: bytes) -> None:
    tmp = dest.with_name(dest.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, dest)  # structural exclusivity; never replace an existing result
    finally:
        tmp.unlink(missing_ok=True)


def run_once(repo: Path) -> dict:
    pre = load_preflight(repo)
    from PIL import Image
    import numpy as np
    import onnxruntime as ort
    import inference as api_inference
    import score_calibration_v2 as sc
    session, input_name, prototypes, taxonomy = sc.load_candidate_session_and_prototypes(pre["artifacts"])
    require(session.get_providers() == ["CPUExecutionProvider"] and len(taxonomy) == 65, "not the 65-class CPU model")
    slug_to_idx = pre["slug_to_idx"]
    require(all(r["slug"] in slug_to_idx and taxonomy[slug_to_idx[r["slug"]]]["taxon_id"] == int(r["taxon_id"])
                for r in pre["rows"]), "final-test labels do not match candidate taxonomy")
    marker_doc = {"schema_version": 1, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "git_head": pre["head"], "rule_sha256": pre["rule_sha256"],
                  "v3_sha256": pre["v3_sha256"], "csv_sha256": pre["csv_sha256"],
                  "row_identity_order_sha256": pre["row_identity_order_sha256"]}
    payload = (json.dumps(marker_doc, sort_keys=True, indent=2) + "\n").encode()
    pre["marker"].parent.mkdir(parents=True, exist_ok=True)
    with pre["marker"].open("xb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    require(pre["marker"].read_bytes() == payload, "marker persistence mismatch")

    records = []
    for row in pre["rows"]:
        rel = Path(row["clean_relative_path"])
        require(not rel.is_absolute() and ".." not in rel.parts and rel.parts[0] == "clean" and
                rel.parts[1] == row["slug"] and rel.stem == row["photo_id"], "unsafe/unbound image path")
        path = pre["repo"] / "data/northeast_final_test_v1" / rel
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == row["sha256"] and len(data) == int(row["byte_size"]),
                "final-test image hash/size mismatch")
        probe = Image.open(io.BytesIO(data)); probe.verify()
        img = Image.open(io.BytesIO(data)); img.load()
        require(img.size == (int(row["width"]), int(row["height"])), "image dimensions changed")
        x = api_inference.AntIdentifier.preprocess(None, img)
        emb = session.run(None, {input_name: x})[0][0]
        norm = float(np.linalg.norm(emb))
        require(math.isfinite(norm) and norm > 1e-8, "invalid embedding")
        sims = prototypes @ (emb / norm)
        require(bool(np.isfinite(sims).all()) and bool((np.abs(sims) <= 1 + gc.COSINE_TOLERANCE).all()),
                "non-finite/out-of-range cosine")
        order = np.argsort(-sims, kind="stable")[:3]
        indices = [int(i) for i in order]
        score = float(sims[indices[0]])
        records.append({"photo_id": row["photo_id"], "observation_uuid": row["observation_uuid"],
                        "image_sha256": row["sha256"], "true_slug": row["slug"],
                        "true_class_index": slug_to_idx[row["slug"]],
                        "top3_indices": indices, "top3_slugs": [taxonomy[i]["slug"] for i in indices],
                        "top3_similarities": [float(sims[i]) for i in indices],
                        "max_cosine": score, "low_confidence": pre["gate"].classify(score),
                        "top1_correct": indices[0] == slug_to_idx[row["slug"]],
                        "top3_correct": slug_to_idx[row["slug"]] in indices})
    require(gc.compute_identity_order_sha256([(r["photo_id"], r["observation_uuid"], r["image_sha256"])
                                                    for r in records]) == pre["row_identity_order_sha256"],
            "row identity order changed during evaluation")
    content = {"dataset": "northeast_final_test_v1", "status": "descriptive_result_not_perceptual_independence_pass",
               "perceptual_independence": EXPECTED_DECISION_STATUS,
               "v2_perceptual_contract_result": "not_passed_not_finalized", "manual_quality": EXPECTED_MANUAL,
               "threshold": .61, "provider": session.get_providers(), "geo_applied": False,
               "bindings": {k: pre[k] for k in ("head", "rule_sha256", "v3_sha256", "csv_sha256",
                                                  "json_sha256", "policy_sha256", "row_identity_order_sha256")},
               "attempt_marker_sha256": sha256(pre["marker"]), "records": records,
               "metrics": compute_metrics(records)}
    result = {"schema_version": 1, "content": content, "content_sha256": canonical_hash(content)}
    validate_result(pre, result)
    publish_exclusive(pre["result"], (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    require(json.loads(pre["result"].read_text(encoding="utf-8")) == result, "persisted result differs")
    validate_result(pre, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            pre = load_preflight(args.repo)
            print(json.dumps({"ok": True, "n_rows": len(pre["rows"]), "head": pre["head"],
                              "independence": EXPECTED_DECISION_STATUS, "threshold": .61}, indent=2))
        else:
            result = run_once(args.repo)
            print(json.dumps({"content_sha256": result["content_sha256"],
                              "metrics": result["content"]["metrics"],
                              "independence": EXPECTED_DECISION_STATUS}, indent=2))
        return 0
    except (FinalTestError, gc.ContractError, ec.EvaluationError, OSError, ValueError, KeyError,
            subprocess.CalledProcessError) as exc:
        print(f"[eval_northeast_final_test_v1_once] FAILURE: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
