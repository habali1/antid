#!/usr/bin/env python3
"""select_gate_v2_threshold.py — applies the frozen Gate v2 selection
contract to an already-produced, hash-bound calibration_v2 score artifact.

This module NEVER imports PIL or onnxruntime, and never opens a file under
data/calibration_v2/<slug>/ or data/unknown_test_v2/. It consumes exactly
two inputs: the frozen contract (gate_v2_selection_contract.json) and a
calibration_v2 score file (score_calibration_v2.py's output) -- both plain
JSON, fully schema-validated via gate_v2_contract.validate_score_content
before any metric is computed.

Usage (not run in Phase 5C1 -- no real score file exists yet):
    python select_gate_v2_threshold.py --scores data/calibration_v2/calibration_v2_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import gate_v2_contract as gc  # noqa: E402

DEFAULT_CONTRACT_PATH = HERE / "gate_v2_selection_contract.json"
DEFAULT_SCORES_PATH = REPO / "data/calibration_v2/calibration_v2_scores.json"
DEFAULT_OUT_PATH = REPO / "data/calibration_v2/calibration_v2_selection.json"
STATUS_CANDIDATE_SELECTED = gc.STATUS_CANDIDATE_SELECTED
STATUS_NO_USEFUL_GATE = gc.STATUS_NO_USEFUL_GATE


class SelectionError(RuntimeError):
    """A source-contract, input-identity, or artifact-validation problem.
    Always fails closed."""


def load_scores(path: Path, contract: dict) -> dict[str, Any]:
    """Loads and FULLY validates the score artifact via the one shared
    schema validator before returning it -- no metric is ever computed
    against an unvalidated or partially-trusted score file."""
    try:
        return gc.load_and_verify_score_file(path, contract)
    except gc.ContractError as exc:
        raise SelectionError(str(exc)) from exc


# ------------------------------------------------------------- rank/AUC
def _auc_known_vs_ood(known_scores: list[float], ood_scores: list[float]) -> float | None:
    """Mann-Whitney-U-based AUC of separating `known_scores` (positive)
    from `ood_scores` (negative) by raw max-cosine. Ties are handled by
    average-rank assignment (the standard definition). Returns None when
    AUC is not properly defined (either side empty)."""
    n_pos, n_neg = len(known_scores), len(ood_scores)
    if n_pos == 0 or n_neg == 0:
        return None
    combined = sorted((s, 0) for s in known_scores) + sorted((s, 1) for s in ood_scores)
    combined.sort(key=lambda t: t[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for (r, (_, label)) in zip(ranks, combined) if label == 0)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def _top3_correct(r: dict) -> bool:
    return r["true_class_index"] in r["top3_indices"]


# ------------------------------------------------------------- per-grid
def _classify_known(records: list[dict], threshold: float) -> dict[str, int]:
    """Exact integer counts only -- the frozen decision shape: strict '<'
    rejects, equality is accepted. Returns raw counts; rates are derived
    from these elsewhere, never the other way around."""
    counts = {
        "accepted": 0, "correct_accepted_top1": 0, "correct_accepted_top3": 0,
        "rejected": 0, "correct_rejected": 0, "incorrect_rejected": 0,
        "total_correct": 0, "total_incorrect": 0,
    }
    for r in records:
        is_correct = bool(r["top1_correct"])
        if is_correct:
            counts["total_correct"] += 1
        else:
            counts["total_incorrect"] += 1
        if r["max_cosine"] < threshold:
            counts["rejected"] += 1
            if is_correct:
                counts["correct_rejected"] += 1
            else:
                counts["incorrect_rejected"] += 1
        else:
            counts["accepted"] += 1
            if is_correct:
                counts["correct_accepted_top1"] += 1
            if _top3_correct(r):
                counts["correct_accepted_top3"] += 1
    return counts


def _rejection_metrics(counts: dict[str, int]) -> dict[str, Any]:
    """The CORRECT rejection-rate definitions (point 3 of the correction):
    each rate is rejected-of-that-outcome divided by the TOTAL count of
    that outcome across the whole known set (correct or incorrect
    predictions), never divided by total-rejected. The ratio of the two
    rates is reported separately from the ratio of the two raw counts --
    they are NOT the same number in general, which is exactly the bug this
    replaces."""
    correct_rate = gc.rate_or_none(counts["correct_rejected"], counts["total_correct"])
    incorrect_rate = gc.rate_or_none(counts["incorrect_rejected"], counts["total_incorrect"])
    return {
        "correct_rejected": counts["correct_rejected"], "total_correct_predictions": counts["total_correct"],
        "correct_prediction_rejection_rate": correct_rate,
        "incorrect_rejected": counts["incorrect_rejected"], "total_incorrect_predictions": counts["total_incorrect"],
        "incorrect_prediction_rejection_rate": incorrect_rate,
        "incorrect_to_correct_rejection_ratio": gc.ratio_or_none(incorrect_rate, correct_rate),
        # retained for transparency ONLY -- this is the raw-count ratio that
        # the old (incorrect) implementation effectively used; it is NOT
        # the same quantity as incorrect_to_correct_rejection_ratio above
        # whenever total_correct_predictions != total_incorrect_predictions.
        "diagnostic_raw_count_ratio_incorrect_to_correct_rejected": gc.ratio_or_none(
            float(counts["incorrect_rejected"]) if counts["correct_rejected"] else None,
            float(counts["correct_rejected"]) if counts["correct_rejected"] else None,
        ) if counts["correct_rejected"] else None,
    }


def build_grid_report(scores: dict, contract: dict) -> list[dict[str, Any]]:
    records = scores["content"]["records"]
    known = [r for r in records if r["category"] == gc.KNOWN_CATEGORY]
    ood_by_cat = {cat: [r for r in records if r["category"] == cat] for cat in gc.OOD_CATEGORIES}
    n_known = len(known)

    grid_ints = gc.generate_grid_integers()
    report = []
    for k in grid_ints:
        threshold = gc.grid_integer_to_float(k)
        counts = _classify_known(known, threshold)
        accepted, rejected = counts["accepted"], counts["rejected"]
        feasible = gc.coverage_feasible(accepted, n_known,
                                        contract["content"]["selection"]["coverage_floor_numerator"],
                                        contract["content"]["selection"]["coverage_floor_denominator"])
        entry = {
            "threshold_integer": k, "threshold": threshold, "feasible": feasible,
            "known": {
                "n": n_known, "accepted": accepted, "rejected": rejected,
                "coverage": gc.rate_or_none(accepted, n_known),
                "rejection_rate": gc.rate_or_none(rejected, n_known),
                "accepted_top1_accuracy": gc.rate_or_none(counts["correct_accepted_top1"], accepted),
                "accepted_top3_accuracy": gc.rate_or_none(counts["correct_accepted_top3"], accepted),
                **_rejection_metrics(counts),
            },
            "diagnostic_ood_far": {},
            "_exact": {  # exact integer counts used for the decision itself -- never derived from the floats above
                "accepted": accepted, "correct_accepted": counts["correct_accepted_top1"], "n_known": n_known,
            },
        }
        for cat, rows in ood_by_cat.items():
            accepted_ood = sum(1 for r in rows if r["max_cosine"] >= threshold)
            entry["diagnostic_ood_far"][cat] = {
                "n": len(rows), "false_acceptance_rate": gc.rate_or_none(accepted_ood, len(rows)),
            }
        report.append(entry)
    return report


def _per_species_known_report(records: list[dict], threshold: float) -> list[dict[str, Any]]:
    from collections import defaultdict
    by_species: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["category"] == gc.KNOWN_CATEGORY:
            by_species[r["slug"]].append(r)

    report = []
    for slug in sorted(by_species):
        rows = by_species[slug]
        n = len(rows)
        accepted_rows = [r for r in rows if r["max_cosine"] >= threshold]
        rejected_rows = [r for r in rows if r["max_cosine"] < threshold]
        baseline_top1 = sum(1 for r in rows if bool(r["top1_correct"]))
        baseline_top3 = sum(1 for r in rows if _top3_correct(r))
        accepted_top1 = sum(1 for r in accepted_rows if bool(r["top1_correct"]))
        accepted_top3 = sum(1 for r in accepted_rows if _top3_correct(r))
        report.append({
            "slug": slug, "n": n, "accepted": len(accepted_rows), "rejected": len(rejected_rows),
            "coverage": gc.rate_or_none(len(accepted_rows), n),
            "baseline_top1_accuracy": gc.rate_or_none(baseline_top1, n),
            "baseline_top3_accuracy": gc.rate_or_none(baseline_top3, n),
            "accepted_top1_accuracy": gc.rate_or_none(accepted_top1, len(accepted_rows)),
            "accepted_top3_accuracy": gc.rate_or_none(accepted_top3, len(accepted_rows)),
        })
    return report


def select_threshold(scores: dict, contract: dict) -> dict[str, Any]:
    """Pure computation over already-validated `scores`/`contract` dicts.
    Never touches a file, an image, or the network."""
    records = scores["content"]["records"]
    known = [r for r in records if r["category"] == gc.KNOWN_CATEGORY]
    n_known = len(known)
    if n_known == 0:
        raise SelectionError("no known_holdout records in the score file")

    baseline_correct_top1 = sum(1 for r in known if bool(r["top1_correct"]))
    baseline_correct_top3 = sum(1 for r in known if _top3_correct(r))
    baseline_accuracy_top1 = gc.rate_or_none(baseline_correct_top1, n_known)
    baseline_accuracy_top3 = gc.rate_or_none(baseline_correct_top3, n_known)

    grid_report = build_grid_report(scores, contract)
    sel = contract["content"]["selection"]
    cov_num, cov_den = sel["coverage_floor_numerator"], sel["coverage_floor_denominator"]
    use_num, use_den = sel["usefulness_floor_numerator"], sel["usefulness_floor_denominator"]

    feasible = [e for e in grid_report if e["feasible"]]
    diagnostics = {
        "baseline": {
            "n": n_known, "correct_top1": baseline_correct_top1, "accuracy_top1": baseline_accuracy_top1,
            "correct_top3": baseline_correct_top3, "accuracy_top3": baseline_accuracy_top3,
        },
        "auc_known_vs_ood": {
            cat: _auc_known_vs_ood([r["max_cosine"] for r in known],
                                   [r["max_cosine"] for r in records if r["category"] == cat])
            for cat in gc.OOD_CATEGORIES
        },
        "grid_report": grid_report,
    }

    if not feasible:
        return {"status": STATUS_NO_USEFUL_GATE, "reason": "no_feasible_threshold",
               "coverage_floor": f"{cov_num}/{cov_den}", "diagnostics": diagnostics}

    best = None  # (correct_accepted, accepted, threshold_integer)
    for e in feasible:
        ca, at, k = e["_exact"]["correct_accepted"], e["_exact"]["accepted"], e["threshold_integer"]
        if best is None:
            best = (ca, at, k)
            continue
        bca, bat, bk = best
        if gc.accuracy_ge(ca, at, bca, bat) and not gc.accuracy_ge(bca, bat, ca, at):
            best = (ca, at, k)  # strictly higher accuracy
        elif gc.accuracy_ge(ca, at, bca, bat) and gc.accuracy_ge(bca, bat, ca, at):
            # exactly equal accuracy -- tie-break 1: higher coverage, tie-break 2: lower threshold
            if at > bat or (at == bat and k < bk):
                best = (ca, at, k)

    correct_accepted, accepted_total, winning_k = best
    winning_threshold = gc.grid_integer_to_float(winning_k)
    useful = gc.usefulness_ge_floor(correct_accepted, accepted_total, baseline_correct_top1, n_known, use_num, use_den)

    winning_entry = next(e for e in feasible if e["threshold_integer"] == winning_k)
    per_species = _per_species_known_report(records, winning_threshold)

    if not useful:
        return {"status": STATUS_NO_USEFUL_GATE, "reason": "usefulness_guard_failed",
               "usefulness_floor_pp": use_num * 100.0 / use_den,
               "best_feasible_threshold_integer": winning_k,
               "best_feasible_accepted_top1_accuracy": gc.rate_or_none(correct_accepted, accepted_total),
               "baseline_accuracy_top1": baseline_accuracy_top1,
               "diagnostics": diagnostics}

    return {
        "status": STATUS_CANDIDATE_SELECTED,
        "candidate": {
            "threshold_integer": winning_k, "threshold": winning_threshold,
            "comparison": "strict_less_than", "equal_threshold_action": "normal_results",
            "accepted": accepted_total, "correct_accepted_top1": correct_accepted,
            "accepted_top1_accuracy": gc.rate_or_none(correct_accepted, accepted_total),
            "accepted_top3_accuracy": winning_entry["known"]["accepted_top3_accuracy"],
            "coverage": gc.rate_or_none(accepted_total, n_known),
            "improvement_over_baseline_pp": (gc.rate_or_none(correct_accepted, accepted_total)
                                            - baseline_accuracy_top1) * 100.0,
            "rejection_metrics": {k: v for k, v in winning_entry["known"].items()
                                  if k in ("correct_rejected", "total_correct_predictions",
                                         "correct_prediction_rejection_rate", "incorrect_rejected",
                                         "total_incorrect_predictions", "incorrect_prediction_rejection_rate",
                                         "incorrect_to_correct_rejection_ratio")},
            "per_species_known": per_species,
            "validation_status": "unvalidated_pending_independent_unknown_test_v2_evaluation",
        },
        "diagnostics": diagnostics,
    }


def write_selection_output(args, contract: dict, scores: dict, result: dict[str, Any]) -> Path:
    """Atomically writes calibration_v2_selection.json for EITHER outcome
    (candidate_selected or no_useful_gate_found) -- there is no later code
    path required to make selection executable; this is it. Refuses to
    overwrite an existing selection file. Verifies implementation-source
    hashes and a clean tracked git tree first, same discipline as the
    scorer's real run, so the code that computed `result` is provably the
    code bound in the frozen contract."""
    approved_out = gc.verify_output_path_is_approved(args.repo, contract, "calibration_v2_selection", args.out)
    if approved_out.exists():
        raise SelectionError(f"refusing to overwrite existing selection file: {approved_out}")

    gc.require_clean_tracked_tree(args.repo)
    implementation_hashes = gc.verify_implementation_sources(args.repo, contract)
    git_head = gc.get_git_head(args.repo)

    content = {
        "dataset": gc.DATASET_NAME,
        "status": result["status"],
        "bindings": {
            "contract_content_sha256": contract["content_sha256"],
            "scores_content_sha256": scores["content_sha256"],
        },
        "provenance": {"git_head": git_head, "implementation_source_hashes": implementation_hashes},
        "result": {k: v for k, v in result.items() if k != "diagnostics"},
        "diagnostics": result["diagnostics"],
    }
    content_sha256 = gc.compute_content_sha256(content)
    output = {
        "schema_version": gc.SELECTION_SCHEMA_VERSION, "content": content, "content_sha256": content_sha256,
        "generation": {"selector_source_sha256": implementation_hashes["select_gate_v2_threshold"]},
    }

    approved_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = approved_out.with_suffix(approved_out.suffix + f".tmp{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, approved_out)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return approved_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the result without writing calibration_v2_selection.json")
    args = ap.parse_args()

    try:
        contract = gc.load_and_verify_contract(args.contract)
        scores = load_scores(args.scores, contract)
        result = select_threshold(scores, contract)
        if not args.dry_run:
            out_path = write_selection_output(args, contract, scores, result)
        else:
            out_path = None
    except (SelectionError, gc.ContractError) as e:
        print(f"[select_gate_v2_threshold] FAILURE: {e}")
        return 1

    print(f"status: {result['status']}")
    if result["status"] == STATUS_CANDIDATE_SELECTED:
        print(json.dumps({k: v for k, v in result["candidate"].items() if k != "per_species_known"}, indent=2))
    else:
        print(f"reason: {result.get('reason')}")
    if out_path is not None:
        print(f"wrote {out_path}")
    return 0 if result["status"] == STATUS_CANDIDATE_SELECTED else 2


if __name__ == "__main__":
    sys.exit(main())
