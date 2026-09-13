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

## Phase 5D1 -- the unknown_test_v2 evaluator (prepared, not run)

The single-use evaluator that will mechanically apply the 0.61 threshold
above to `unknown_test_v2` is code-prepared and frozen, but has NOT been
run: `training/gate_v2_evaluation_contract.py` /
`training/gate_v2_evaluation_contract.json`
(`content_sha256: 49bb0c4ee5749513b62e7bdfa7f50a7619fe16e2afc75a47d60bda25b4bdb10c`)
bind the selection contract above, both calibration_v2 artifact hashes, the
frozen threshold, all four candidate artifact hashes, and the unknown_test_v2
CSV/JSON hashes and quotas. `training/eval_unknown_test_v2.py` loads the
threshold from `calibration_v2_selection.json` only -- there is no
`--threshold`/`--operator`/`--out` override, sweep mode, or any input-path
override flag (every path derives from `--repo` alone) -- and applies it
exactly once: every fallible, image-free step (artifact-directory
verification, ONNX session construction, taxonomy/prototype loading) runs
BEFORE the atomic, exclusive-creation attempt marker is created immediately
before the first unknown_test_v2 image is opened (including the numpy
import, moved out of the per-row loop in a final correction pass so it can
no longer land after the marker), so a session/load failure never consumes
the single-use budget. Final publication of the result uses `os.link()`,
not `os.replace()`, so overwrite is refused at the actual write boundary,
not only by an earlier existence check. The evaluation output's
`validation` block is fully recomputed from records at load/verify time
(not merely checked for internal consistency), and the attempt marker
itself is verified for exact contract agreement, both at creation and
again whenever an evaluation output is loaded. See `TODO.md`'s "Gate v2:
Phase 5D1" section for full detail.

## Phase 5D2/5D3 -- unknown_test_v2 consumed exactly once; Gate v2 independently validated

`unknown_test_v2` has been evaluated -- **exactly once**, as the frozen
`single_use_rule` requires -- and must never be evaluated again. Result:
**`validation_status: validation_passed`**, all three precommitted criteria
passed, under the frozen decision shape (raw unrounded pre-geo max cosine
**< 0.61**, equality at exactly 0.61 accepted). The threshold was **not
adjusted** in response to this result.

- Baseline (n=390): top-1 61.03% (238/390), top-3 78.72% (307/390).
- After the gate: accepted 256/390 = 65.64% coverage; accepted top-1
  79.6875%, top-3 91.40625%; **+18.66 percentage points** top-1 improvement.
- Correct-prediction rejection rate 14.29% vs. incorrect-prediction
  rejection rate 65.79% (ratio 4.61x) -- health check passes.
- Diagnostic-only OOD false-acceptance rate at 0.61: `out_of_scope_ant`
  51%, `non_ant_insect` 11%, `unrelated` 19% -- these numbers played **no
  role** in `validation_status` and the permissive out_of_scope_ant rate
  reconfirms this remains a **selective confidence gate, not an
  unknown-species detector**.
- Per-species rows (65 species, n=6 each) are descriptive at this sample
  size, not stable population-level per-species estimates.

Full detail, exact hashes/sizes for the frozen `unknown_test_v2_evaluation_attempt.json` and `unknown_test_v2_eval.json` evidence artifacts, and the
complete per-species table are recorded in `TODO.md`'s "Gate v2: Phase
5D2/5D3" section -- this document only summarizes the outcome.
`calibration_v2_selection.json` remains immutable (`status:
candidate_selected`); this independent validation is represented solely by
the separate evaluation artifact. **Gate v2 is independently validated but
not yet deployed** -- no v2 `inference_policy.json` has been generated and
no serving artifact has changed.

## Phase 5E1 -- policy schema v2 and a separate V2 generator (prepared, not run)

Independent validation above stays frozen and unmodified. This phase adds
only the machinery a future promotion will use: `policy_schema.py` (kept
byte-identical between `training/` and `api/`) now supports schema v2
alongside the unmodified v1 default -- `SCHEMA_VERSION`/`FROZEN_THRESHOLD`
(1 / 0.60) are untouched; `SCHEMA_VERSION_V2`/`FROZEN_THRESHOLD_V2` (2 /
0.61) were added alongside, never replacing them. A schema-v2 policy
additionally requires a `validation_evidence` block, and
`policy_schema.EXPECTED_V2_VALIDATION_EVIDENCE` freezes the EXACT byte/
content hashes of the calibration selection, the independent
`unknown_test_v2` evaluation, and the parity report, plus the exact
`validation_status`/diagnostic FAR values -- not merely their format.
`policy_schema.validate()` checks every field for exact equality against
that frozen mapping, wired in for schema v2 only, so a rehashed 0.61
policy that swaps in a different, individually well-formed hash (or a
different in-range FAR) in any evidence field is rejected, not only one
with the block missing or malformed. The API loader
(`api/inference_policy.py`) is now version-aware: it selects the permitted
threshold from the policy's own `policy_schema_version` before comparing,
so a v1 policy carrying 0.61 or a v2 policy carrying 0.60 both fail closed
as `unsupported_rule` -- neither is silently accepted.

A new, separate generator, `training/generate_inference_policy_v2.py`,
reads the frozen Gate v2 evidence chain through the already-committed
shared validators (never duplicating their logic), mechanically
recomputes the calibration selection and requires exact agreement with the
frozen selection artifact, requires the independent evaluation's
`validation_status == "validation_passed"`, binds the current candidate's
three artifact hashes and the parity report's byte hash, and constructs a
real ONNX Runtime session on `CPUExecutionProvider` exclusively. It loads
the frozen evaluation contract EARLY and, immediately after reading
`calibration_v2_selection.json`, compares its actual byte hash against
`evaluation_contract.content.selection_binding.byte_sha256` BEFORE
touching status/result/diagnostics -- a byte-tampered selection file is
caught by identity alone, before any of its claimed content is trusted
enough to index into. Every required selection field (`schema_version`,
`content_sha256`, `generation`, `content.status`, `content.result`,
`content.diagnostics`) is then checked via `.get()` -- never direct
indexing that could raise `KeyError` -- so a missing or malformed field
produces a controlled `GeneratorV2Error`, never a traceback; the
self-consistency `content_sha256` recompute and status/result/diagnostics
recomputation checks are retained afterward as defense in depth. The same
scores/selection byte+content cross-check (against `scores_binding`) and
controlled-JSON-parse handling apply throughout the preflight path.

The **required synchronization gate** between `training/policy_schema.py`
and `api/policy_schema.py` compares raw `read_bytes()` -- not a hash, and
deliberately not canonical-LF-normalized -- so a divergence that is only
CRLF-vs-LF is still caught; canonical-LF hashes are retained separately
for the `content.provenance` recording of six implementation files (this
generator, both `policy_schema.py` copies, `api/inference_policy.py`,
`api/inference.py`, `training/data.py`), which stays diagnostic-only (the
API loader never reads it). The original V1 generator
(`training/inference_policy_generator.py`) is untouched and still reads
only the unrenamed `SCHEMA_VERSION`/`FROZEN_THRESHOLD` names, so it cannot
accidentally emit schema v2.

Schema v2's envelope is now checked exactly (v1's is unconstrained by
this, unchanged): the top-level keys must be exactly
`policy_schema_version, content, content_sha256, generation`; `generation`
must be exactly `generated_at, generator_version`; `generated_at` must
match the strict `YYYY-MM-DDTHH:MM:SSZ` form; and `generator_version` must
equal `policy_schema.V2_GENERATOR_VERSION`, the ONE frozen version string
the generator imports rather than redefining. `--write` now: (1) requires
a clean tracked tree (write-time only -- `--preflight`/`--check` remain
usable on a dirty tree, consistent with faithfully recording the parity
report's own `workspace_git_dirty` flag); (2) records the generating git
HEAD and source provenance as above; (3) pre-validates the built candidate
policy through the REAL API loader in a temporary, isolated artifact
directory (hard-linked artifact files) BEFORE any publication -- every
mismatch fails there, leaving the real candidate directory untouched and
no temp files behind; only then does the atomic-exclusive final publish
happen, followed by a last readback + schema + loader check against the
real published file. `--check` now reruns the COMPLETE read-only
preflight, rebuilds the expected deterministic policy content from it, and
requires the stored top-level key set, schema version, content, and
`content_sha256` to match EXACTLY, and separately requires the stored
`generation` key set and `generator_version` to match exactly too --
**only `generated_at`'s timestamp value is permitted to differ** -- before
verifying through the API loader. An extra top-level or `generation` key
is rejected even when it leaves `content_sha256` untouched (that hash only
ever covers `{policy_schema_version, content}`), and a stored candidate
with an altered `gate_framing`, a parity fact, `not_validated_for`, or any
single evidence hash is rejected even if its `content_sha256` was freshly
recomputed over the altered content.

**Only `--preflight` has been run in this phase** -- against
`training/artifacts/northeast_v1_b4_dev_v2`, producing `"ok": true` with
every evidence hash binding correctly, git HEAD and all six provenance
source hashes recorded, both `policy_schema.py` copies confirmed
byte-identical (raw bytes), and zero files written anywhere. No
`inference_policy.json` has been generated, no artifact has been copied or
promoted, and the live 50-species v1 gate (threshold 0.60) remains the
only active gate. `--write`/`--check` were exercised only against
synthetic fixtures in this phase's test suite, never against real
evidence. Promotion (an actual `--write` run, review of its output, and
atomic publication to the live serving directory) is a separate,
not-yet-started phase.
