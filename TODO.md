# TODO

## Active roadmap: Northeast expansion

- Scope approved: keep the existing 50 species and add up to 15 missing
  Northeast species. One catalog, not a Northeast-only replacement.
- [Plan, provisional shortlist, and next bounded task](docs/plans/northeast-expansion-v1.md)
  and [public aggregate-count snapshot](docs/plans/northeast-counts-2026-09-05.json).
- Train-only geo export, stale-sidecar replacement, and pinned train+val
  membership are implemented and covered by focused synthetic tests.
- Next: candidate metadata/quality and post-exclusion availability review.
  No training or photo collection has started.
- Keep personal local history and optional accounts on the roadmap. Public
  maps/social sharing remain deferred; no paid infrastructure without approval.

## Policy maintenance: verified closeout and boundaries

- Training/API `policy_schema.py` copies are byte-identical; the existing
  `test_training_and_api_schema_copies_are_byte_identical` test passed on
  2026-09-05. Preserve this test when changing either copy.
- The recorded `api/inference.py` **source hash is diagnostic provenance only**.
  A source edit alone does not disable the gate. Runtime checks enforce the
  three mandatory artifact hashes, preprocessing contract, CPU provider policy,
  and schema/content integrity. Inspect `/health` reason fields when inactive.
- After serving edits, verify behavior and regenerate provenance as appropriate.
  Artifact/preprocessing changes need compatible evidence; regeneration alone
  is not validation. The optional geo sidecar is intentionally not hash-bound.
- Catalog expansion invalidates the current gate's evidence scope even without
  backbone retraining. New prototypes/taxonomy need a freshly validated policy;
  do not reuse old 0.60 evidence by merely replacing hashes. Preserve all old
  benchmark/calibration/unknown-test files and parity reports.

## Parity scripts: deferred audit items (must be fixed before any future parity run)

`training/parity_check.py`, `training/parity_diagnostic.py`, and
`training/parity_flag_ablation.py`, along with the contemporaneous `v1`
reports associated with them (`training/artifacts/parity_report.json`,
`parity_diagnostic.json`, `parity_flag_ablation.json`), are **frozen** — see
"Parity evidence and limitations" in `training/artifacts/README.md` for
their hashes and known result-level limitations. These are separate,
script-level issues found during review but not fixed, so as not to
invalidate the frozen v1 evidence by changing the contemporaneous script
bytes preserved alongside those reports:

- `parity_check.py` hardcodes the abstention threshold (0.60) instead of
  reading `data/calibration_v1/calibration_v1.json`'s
  `frozen_candidate_abstention_threshold.machine_readable_rule.value`.
- `parity_diagnostic.py` independently hardcodes the same 0.60 threshold
  instead of reading the same authoritative
  `data/calibration_v1/calibration_v1.json`
  `frozen_candidate_abstention_threshold.machine_readable_rule`, preferably
  through a shared evaluation-side loader. The current `parity_check.py`
  hardcode is itself a deferred defect and must not become another source
  of truth.
- `parity_check.py` hardcodes `N_SAMPLES` (200) instead of deriving it from
  the stratified sample it actually builds.
- `parity_check.py` does not fail closed on duplicate `(slug, photo_id)` rows
  in its sample.
- `parity_check.py` does not fail closed on duplicate `file_sha256` values
  in its sample (the same image selected twice under different identifiers).
- `parity_flag_ablation.py` trusts its stored `sample_list_hash_sha256`
  rather than recomputing it from the ordered row list at run time, so a
  hand-edited sample list with a stale-but-matching stored hash would pass
  silently.
- `parity_flag_ablation.py` records `n_rows` but does not enforce it in the
  fail-closed integrity gate (`integrity_gate` checks `n_samples_valid`, not
  `n_rows`, against expectations).

**All of the above must be fixed before any future parity run.** A future
run must write versioned `v2` reports (e.g. `parity_report_v2.json`) rather
than overwrite the frozen `v1` reports listed above.
