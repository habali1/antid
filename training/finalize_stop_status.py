#!/usr/bin/env python3
"""finalize_stop_status.py — the post-adjudication stop-decision gate.

Produces training/perceptual_duplicate_post_adjudication_stop_status.json,
a SEPARATE, contract-bound artifact from scan_perceptual_duplicates.py's
scan-time stop_status_report. That report is generated BEFORE any human
pair adjudication and can never claim a confirmed same_source_image stop
(scan_perceptual_duplicates.validate_reports_against_population/
mrc.validate_stop_status_report both reject one that tries). THIS artifact
is generated only once every candidate pair across all 36 domains has an
adjudication record in the pair-adjudication ledger; it folds in confirmed
same-source-image findings by domain, PRESERVES every scan-time
metadata-leakage stop verbatim, and activates the mandatory stop for every
frozen critical (mrc.STOP_BEFORE_INFERENCE_DOMAINS) domain that needs one.
It is the actual required gate before northeast_final_test_v1 model
inference -- no remediation policy is invented here, the already-frozen
decisions (STOP_BEFORE_INFERENCE_DOMAINS, the two stop reasons, the
remediation_rule text) are the only ones this artifact acts on.

CLI modes:
  --preflight  loads/verifies the contract, all five scan reports (schema/
               binding AND full re-derivation against the frozen
               manifests), and the full adjudication ledger; reports
               adjudication progress (total/adjudicated/remaining
               candidates) WITHOUT requiring completion. No image access.
  --finalize   (NOT run in Phase 5F1) refuses to complete until every
               candidate pair has an adjudication record; derives the
               post-adjudication stop decision and publishes it
               exclusively/atomically. NEVER modifies the five scan
               reports, the adjudication ledger, or any dataset.
  --check      validates the ALREADY-PUBLISHED finalization artifact:
               schema, every cross-report/ledger binding, and that its
               content is EXACTLY re-derivable from the current scan
               reports and adjudication ledger -- WITHOUT rerunning or
               replacing it.

Never imports onnxruntime or any inference/model module. --finalize is NOT
run in Phase 5F1 (there is no completed, fully-adjudicated ledger yet).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import scan_perceptual_duplicates as spd  # noqa: E402
import adjudicate_pairs as adj  # noqa: E402

FINALIZATION_ARTIFACT_NAME = "post_adjudication_stop_status_report"


class FinalizationError(RuntimeError):
    """A verification or completeness problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise FinalizationError(msg)


def _report_path(repo: Path, name: str) -> Path:
    return Path(repo) / mrc.APPROVED_OUTPUT_PATHS[name]


def verify_everything(repo: Path) -> dict:
    """The one required first step for every mode: the full shared
    scan-state verification (spd.load_and_verify_full_scan_state --
    contract, all five scan reports, schema/binding, AND genuine
    re-derivation against the real frozen manifests), THEN the
    pair-adjudication ledger's full semantic verification via
    adjudicate_pairs.load_verified_ledger(..., require_canonical_tail=
    True) -- THE SAME check --preflight/--next/--record apply (never a
    second, divergent copy of "is this ledger trustworthy"), plus the
    additional canonical-newline-terminated-bytes requirement finalization
    alone needs. Raises FinalizationError before anything else runs.
    Confirmed same-source stops are only ever derived from a ledger that
    has passed every one of these layers."""
    state = spd.load_and_verify_full_scan_state(repo)
    verified, reports = state["verified"], state["reports"]

    cp_path = Path(repo) / mrc.APPROVED_OUTPUT_PATHS["candidate_pairs_report"]
    adj_state = {"reports": reports, "candidate_pairs_report_bytes": cp_path.read_bytes()}
    ledger_path = adj.ledger_path_for(repo)
    ledger_bytes = ledger_path.read_bytes() if ledger_path.exists() else b""
    try:
        ledger_records = adj.load_verified_ledger(repo, adj_state, require_canonical_tail=True)
    except adj.AdjudicationError as exc:
        _fail(f"adjudication ledger failed verification, refusing to finalize: {exc}")

    return {"verified": verified, "reports": reports, "ledger_records": ledger_records, "ledger_bytes": ledger_bytes}


def _all_candidates_by_domain(candidate_pairs_report: dict) -> dict[str, list[dict]]:
    return candidate_pairs_report["content"]["domains"]


def derive_post_adjudication_domains(reports: dict, ledger_records: list[dict]) -> dict[str, dict]:
    """Pure decision function: folds CONFIRMED same_source_image
    adjudications (by domain) together with the scan-time metadata-leakage
    stop reasons (preserved verbatim, never dropped) into one
    per-domain post-adjudication decision. A candidate pair adjudicated as
    anything other than same_source_image, or not yet adjudicated, never
    triggers a stop on its own -- mirrors
    scan_perceptual_duplicates.evaluate_stop_conditions's semantics, now
    with real (rather than always-empty) confirmed-pair data."""
    confirmed_pair_ids_by_domain: dict[str, set] = {}
    for rec in ledger_records:
        if rec.get("label") == "same_source_image":
            confirmed_pair_ids_by_domain.setdefault(rec.get("domain"), set()).add(rec.get("pair_id"))

    candidates_by_domain = _all_candidates_by_domain(reports["candidate_pairs_report"])
    ss_domains = reports["stop_status_report"]["content"]["domains"]

    domains: dict[str, dict] = {}
    for domain in mrc.COMPARISON_DOMAINS:
        name = domain["name"]
        reasons = []
        if name in mrc.STOP_BEFORE_INFERENCE_DOMAINS:
            confirmed_here = confirmed_pair_ids_by_domain.get(name, set())
            candidate_ids_here = {p["pair_id"] for p in candidates_by_domain.get(name, [])}
            if confirmed_here & candidate_ids_here:
                reasons.append(mrc.STOP_REASON_SAME_SOURCE_IMAGE)
            if mrc.STOP_REASON_MATCHING_OBSERVATION_UUID in ss_domains.get(name, {}).get("reasons", []):
                reasons.append(mrc.STOP_REASON_MATCHING_OBSERVATION_UUID)
        domains[name] = {"stop_before_inference": bool(reasons), "reasons": reasons}
    return domains


def build_post_adjudication_content(state: dict) -> dict:
    """Pure function over already-verified state -- no filesystem access,
    no image access. Deterministic: the same state always produces the
    same content, which is what makes --check's exact re-derivation
    comparison meaningful."""
    verified, reports, ledger_records = state["verified"], state["reports"], state["ledger_records"]
    contract = verified["contract"]
    candidates_by_domain = _all_candidates_by_domain(reports["candidate_pairs_report"])
    all_candidate_ids = {p["pair_id"] for pairs in candidates_by_domain.values() for p in pairs}
    total_candidates = len(all_candidate_ids)
    adjudicated_pair_ids = {r.get("pair_id") for r in ledger_records}
    adjudicated_count = len(all_candidate_ids & adjudicated_pair_ids)

    domains = derive_post_adjudication_domains(reports, ledger_records)
    overall_stop = any(v["stop_before_inference"] for v in domains.values())

    return {
        "contract_content_sha256": contract["content_sha256"],
        "stop_status_report_content_sha256": reports["stop_status_report"]["content_sha256"],
        "candidate_pairs_report_content_sha256": reports["candidate_pairs_report"]["content_sha256"],
        "adjudication_ledger_byte_sha256": mrc.sha256_bytes(state["ledger_bytes"]),
        "domains": domains,
        "overall_stop_before_inference": overall_stop,
        "total_candidates": total_candidates,
        "adjudicated_count": adjudicated_count,
        "complete": adjudicated_count == total_candidates,
    }


# --------------------------------------------------------------- commands --
def cmd_preflight(args) -> dict:
    state = verify_everything(args.repo)
    content = build_post_adjudication_content(state)
    return {
        "ok": True, "contract_content_sha256": content["contract_content_sha256"],
        "total_candidates": content["total_candidates"], "adjudicated_count": content["adjudicated_count"],
        "remaining_count": content["total_candidates"] - content["adjudicated_count"],
        "complete": content["complete"],
    }


def cmd_finalize(args) -> dict:
    state = verify_everything(args.repo)
    content = build_post_adjudication_content(state)
    if not content["complete"]:
        _fail(f"refusing to finalize: {content['adjudicated_count']}/{content['total_candidates']} "
              f"candidates adjudicated -- every candidate must have an adjudication record first "
              f"(this is the required gate before northeast_final_test_v1 inference)")

    content_sha256 = mrc.compute_content_sha256(content)
    report = {"schema_version": mrc.SCHEMA_VERSION, "content": content, "content_sha256": content_sha256}
    dest = _report_path(args.repo, FINALIZATION_ARTIFACT_NAME)
    if dest.exists():
        _fail(f"refusing to overwrite existing finalization artifact: {dest}")

    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(str(tmp), str(dest))
    finally:
        tmp.unlink(missing_ok=True)

    return {"ok": True, "path": str(dest), "content_sha256": content_sha256,
            "overall_stop_before_inference": content["overall_stop_before_inference"]}


def cmd_check(args) -> dict:
    state = verify_everything(args.repo)
    dest = _report_path(args.repo, FINALIZATION_ARTIFACT_NAME)
    if not dest.exists():
        _fail(f"finalization artifact does not exist yet: {dest} -- run --finalize first "
              f"(a separately authorized phase)")
    try:
        report = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{dest} is not valid JSON: {exc}")

    problems = mrc.validate_post_adjudication_report(
        report, state["verified"]["contract"], state["reports"]["stop_status_report"],
        state["reports"]["candidate_pairs_report"], mrc.sha256_bytes(state["ledger_bytes"]))
    if problems:
        _fail(f"finalization artifact failed schema/binding validation: {problems}")

    expected_content = build_post_adjudication_content(state)
    if report["content"] != expected_content:
        _fail("finalization artifact does not exactly match the current re-derivation from the scan "
              "reports and adjudication ledger")

    return {"ok": True, "path": str(dest), "content_sha256": report["content_sha256"],
            "overall_stop_before_inference": report["content"]["overall_stop_before_inference"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        if args.preflight:
            result = cmd_preflight(args)
        elif args.finalize:
            result = cmd_finalize(args)
        else:
            result = cmd_check(args)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (FinalizationError, mrc.ContractError, spd.ScanError, adj.AdjudicationError) as e:
        print(f"[finalize_stop_status] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
