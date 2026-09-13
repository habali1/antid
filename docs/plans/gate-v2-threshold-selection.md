# Gate v2 threshold-selection contract

Status: **frozen_before_calibration_scoring** (`training/gate_v2_selection_contract.json`,
`content_sha256: 991f7a0b8e83654e45575566eb0648ddcd69866e29c94b8a2030cc6f4bc19f77`).

This document is the human-readable companion to the machine-readable
contract. The JSON file is authoritative; this document explains and
justifies it. If the two ever disagree, the JSON file wins and this
document is wrong and must be fixed.

## This is a new contract, not a reconstruction of v1

`data/calibration_v1/calibration_v1.json` records that 0.60 was selected and
what its measured behavior was, but **not a reproducible selection
algorithm**. `training/eval_unknown_test.py`'s own docstring says so
explicitly: the original 0.60 selection and the original one-pass
`unknown_test_v1` evaluation both predate `machine_readable_rule` and used
0.60 as a hardcoded constant in that script; the machine-readable rule block
was added *after the fact*, purely so the already-chosen value could be
reproduced, not as a record of how it was chosen. `calibration_v1.json`
records **one measured operating point** at 0.60 (accuracy, FRR, FAR at that
single value) — not a swept grid, not a stated objective function, not a
recorded tie-break rule.

This contract does not attempt to recover that undocumented process. It
defines a **new, fully deterministic algorithm** for choosing a v2
threshold, informed by v1's category split and calibrate-once/test-once
discipline, but not bound to arrive at any particular number.

## Roles: calibration_v2 selects, unknown_test_v2 grades once

- `data/calibration_v2/` (1,250 rows: 650 known_holdout @ 10/species × 65
  species, 300 out_of_scope_ant, 150 non_ant_insect, 150 unrelated) is the
  **only** data used to select a candidate threshold.
- `data/unknown_test_v2/` (790 rows: 390 known_holdout @ 6/species × 65
  species, 200 out_of_scope_ant, 100 non_ant_insect, 100 unrelated) is a
  **single-use independent test**. It is evaluated **exactly once**, after
  a candidate threshold has already been selected, reviewed, and frozen
  from calibration_v2 alone. No unknown_test_v2 image may be opened before
  that point.
- **No retuning after unknown_test_v2.** If the frozen candidate performs
  poorly there, that is a finding to report, not a cue to pick a new
  threshold from the same data — this is the identical discipline
  `calibration_v1.json`'s own `threshold_selection_policy.data_separation`
  already established for v1, carried forward deliberately.
- `data/northeast_final_test_v1/` remains closed to inference throughout
  Phase 5C. It is not part of this contract's data flow at all.

## Purpose: a selective confidence gate, not an unknown-species detector

Consistent with the existing `inference_policy.json` framing (and v1's own
documented finding that 0.60 still accepted ~45.5% of out-of-scope ant
photographs), this gate is understood to **improve confidence in accepted
predictions**, not to reliably detect out-of-catalog species. The
out-of-scope categories (`out_of_scope_ant`, `non_ant_insect`, `unrelated`)
are measured and reported for every candidate threshold, but **never
influence which threshold is selected** — they are diagnostic only.

`low_quality_known` does not exist as a category in Gate v2 at all (Gate v2
has no such quota), so there is nothing to exclude — it is simply absent,
matching the instruction that it must not be used.

## The algorithm (see the JSON contract for the exact arithmetic)

1. **Grid**: 201 candidate thresholds, `t = k / 100.0` for integer `k` in
   `-100..100` inclusive, generated from integer `range()` — never by
   repeated float addition, so there is no accumulation error in the grid
   itself.
2. **Feasibility**: a threshold is feasible only if accepted known_holdout
   coverage is at least 65% (equivalently, known rejection at most 35%),
   checked by the exact integer test `100 * accepted >= 65 * total`.
3. **Primary objective**: among feasible thresholds, maximize accepted-known
   top-1 accuracy, compared as exact fractions via integer cross
   multiplication — never by comparing rounded display floats.
4. **Tie-break**: (a) higher accepted coverage, then (b) lower threshold.
5. **Usefulness guard**: the selected candidate's accepted-known accuracy
   must improve on the no-abstention baseline accuracy by at least 5.0
   percentage points, checked exactly (integer cross multiplication);
   equality at exactly 5.0 points passes.
6. **If no threshold is feasible, or the feasible optimum fails the
   usefulness guard**: the selector emits `status: "no_useful_gate_found"`,
   emits **no** candidate threshold or policy, and unknown_test_v2
   evaluation remains prohibited until a further decision is made (out of
   scope for this contract — that would require a different data or
   design decision, which is exactly the kind of retuning-after-the-fact
   this process is built to avoid doing silently).
7. **Decision shape carried into any future policy**: raw, unrounded,
   pre-geo maximum cosine; strict `<` comparison; equality is accepted
   (`normal_results`, matching v1's existing `equal_threshold_action`); one
   global threshold; no per-species or per-class behavior.

## What Phase 5C1 (this document's original scope) does not do

- No calibration_v2 image is scored in this phase (5C1 is code preparation
  only).
- No unknown_test_v2 image is ever opened before a threshold is selected,
  reviewed, and frozen.
- No live `inference_policy.json` or serving artifact is modified.
- No commit or push happens in this phase.

## Phase 5C2 calibration result -- candidate selected; independent unknown_test_v2 validation pending

Phase 5C2 ran the prepared scorer and selector, exactly once each, against
the frozen contract above (`content_sha256:
991f7a0b8e83654e45575566eb0648ddcd69866e29c94b8a2030cc6f4bc19f77`) at commit
`2fc9261f6c8ca704a2fd220b3b6dc5ff4415f482`. This section records that result
as calibration evidence. **It is not a serving policy** -- no
`inference_policy.json` was generated -- **and it has not been validated
against `unknown_test_v2`**, which remains unopened by inference and whose
single permitted evaluation has not been used.

Artifacts (both re-validated through the shared strict validator and the
selector's own mechanical recomputation before this record was written):

- `data/calibration_v2/calibration_v2_scores.json` --
  byte sha256 `4fec18a938ef22e06d6073016d1692512e8fb1e4e41ec645d9aeb33afbebe46b`,
  `content_sha256 35c7bf36042470dde8fc922106c6528fcb6a06e3bac454c4d196880796e778f4`.
- `data/calibration_v2/calibration_v2_selection.json` --
  byte sha256 `00e56d2e64941086ee1d1c663890cb95d3daa4dcce1e729ef972879376f1489b`,
  `content_sha256 82bc754dbf46643862504916f4c273922e9ac3cefc45bfa28e5eb4a15dfdc5b5`.

Result: **`status: candidate_selected`**, threshold **0.61** (grid integer
61), under the frozen `strict_less_than` decision shape -- a raw max-cosine
of exactly 0.61 is accepted, never rejected (equality accepted).

| Metric | Value |
|---|---|
| Baseline (no-abstention) top-1 / top-3 | 59.85% / 77.23% |
| Accepted-known coverage | 66.46% (432/650) |
| Accepted-known top-1 / top-3 accuracy | 78.70% / 91.44% |
| Improvement over baseline (top-1) | +18.86 pp |
| Correct-prediction rejection rate | 12.60% (49/389) |
| Incorrect-prediction rejection rate | 64.75% (169/261) |
| Incorrect:correct rejection ratio | 5.14x |
| Diagnostic OOD false-acceptance rate at 0.61 | out_of_scope_ant 50.33%, non_ant_insect 17.33%, unrelated 7.33% |

**The diagnostic OOD false-acceptance and AUC numbers above did not affect
threshold selection in any way.** Per this contract's frozen
`ood_never_affects_selection: true` policy, the selector's grid search reads
only known_holdout accuracy/coverage; OOD rows are scored and reported purely
for post-hoc diagnostic visibility. The permissive out_of_scope_ant
false-acceptance rate (50.33%) at the selected threshold is itself evidence
for, not against, the project's existing framing: **this is a selective
confidence gate, not an unknown-species detector** (consistent with v1's
own ~45.5% out-of-scope pass rate cited in the top-level project docs).

**unknown_test_v2 remains completely unopened by inference.** Only
`data/calibration_v2/` images were read during scoring. The next gate is a
review of this result, followed by a separately authorized turn for the
single, independent `unknown_test_v2` evaluation this contract's
`single_use_rule` reserves -- not run, implemented, or scheduled here.
