# Perceptual-pair adjudication v2 protocol

Status: **generated and frozen; adjudication not started**. The source-
preparation commit is `7e847379bfadc378f1fa1b57c6ebd3f7390e81c1`.
`training/pair_adjudication_v2_queue.json` and
`training/pair_adjudication_v2_contract.json` were then generated exactly
once and verified twice each. The machine-readable contract is authoritative;
this document is its human-readable companion and must be reconciled to it if
the two ever disagree.

Frozen evidence:

- Queue byte sha256:
  `28f9d3f984d14d88905cee5357e39802d384d87586fb10cab4e078e9057ca70d`
  (2,500,250 bytes, 3,972 rows); identity-order sha256:
  `7e906cc2a66017940a8aef586e0eb9be7743c36f1dddec4ec22d46341562d14f`.
- Contract byte sha256:
  `27d9f66949a0ec2b9cb22a8993646a3c8a301955d7e3b664d1281ec44c230fb3`
  (9,579 bytes); content sha256:
  `852275c4a6d3107ece2fef9a2f97b3834d0ca21078c8cfb164ef015c21b415f0`.
- Outside-scope population: 12,567 candidates, identity-order sha256
  `d43fb8839beb641df258385d1b83c291e68d68bf1732dc5bc701db93470d92db`.
- Both runtime preflights pass with zero adjudications: 3,972 remaining,
  every workstream pending, and no stop condition.

## What this is, and what it is not

Phase 5F3 produced a frozen, real scan of all 9,259 resolvable images across
36 domain-pairs, yielding 16,539 perceptual-duplicate *candidate* pairs
(`training/perceptual_duplicate_candidate_pairs.json`). The original Phase
5F1 finalizer requires every one of those 16,539 candidates to be manually
adjudicated before any stop decision can be drawn -- an undifferentiated
workload that does not distinguish "this pair threatens final_test/gate
independence" from "this pair is scientifically interesting but harmless."

This v2 protocol is a **separate, additive layer that revises scope only**.
It does not alter the Phase 5F1 scan, the frozen pHash≤10/dHash≤8 candidate
rule, the frozen 0.61 Gate v2 threshold, the four adjudication labels' own
meanings, or any candidate record. It defines *which* candidates must be
adjudicated before *which* conclusion may be drawn, and reports the result
per-channel rather than as one collapsed pass/fail.

## Three independent workstreams

| Workstream | Role | Candidates | Domains |
|---|---|---|---|
| `final_test_independence` | blocking | 1,681 | the 7 domains pairing `final_test` against every other evidence set |
| `final_test_internal_repetition` | diagnostic | 53 | `within_final_test` only |
| `gate_evidence_independence` | blocking | 2,238 | the 5 domains binding `calibration_v2`/`expansion_development`/`expansion_train` against each other and `unknown_test_v2` |

Scoped total: 3,972. Outside-scope: 12,567. Full frozen scan: 16,539.
`3,972 + 12,567 = 16,539` — verified mechanically, never assumed, both at
module-load time (`assert` statements in `pair_adjudication_v2_contract.py`)
and at every queue-build/contract-freeze/runtime-load against the real
committed candidate report.

A confirmed `same_source_image` in a **blocking** channel sets that
channel's status to `failed` and its own stop flag
(`stop_before_final_test_inference` / `stop_before_gate_promotion`). A
confirmed `same_source_image` inside `final_test_internal_repetition` is
reported as an **effective-sample-size limitation** of
`northeast_final_test_v1` — never development-data leakage, and never a
stop condition by itself. An `uncertain` label makes its blocking channel's
status `inconclusive` and blocking — it is never counted as cleared or
passed; resolving it requires a separately reviewed correction protocol
(mirroring the Phase 5F2 single-target correction mechanism), never a
silent re-adjudication.

Every one of the 12,567 outside-scope candidates, and every row inside a
stopped blocking channel that was never reached, is reported as
`not_adjudicated` / `not_adjudicated_due_to_early_stop` — **never**
`not_duplicate`. Silence about a candidate is not evidence of anything. The
excluded population itself is hash-bound: `derive_outside_scope_pair_ids` +
`compute_outside_scope_identity_order_sha256` produce a deterministic,
order-independent identity over the 12,567 excluded pair_ids, verified fresh
against the real candidate report at every runtime load
(`adjudicate_pairs_v2.load_verified_v2_state`) -- a jointly-tampered
queue+contract pair that relabels a scoped candidate "outside scope" (while
keeping every hash internally self-consistent) is rejected because the
rederived population no longer matches the frozen binding.

## Early stop, per channel, independently

`final_test_independence` and `gate_evidence_independence` each stop the
moment their OWN first `same_source_image` or `uncertain` decision lands —
independently of each other. `final_test_internal_repetition` never stops
early; being diagnostic and small (53 rows), it is always reviewed in full.
`finalize_stop_status_v2.py`'s `derive_workstream_status` implements this
per-workstream, and `ready_to_finalize` requires each blocking channel to be
either exhausted cleanly or stopped by a recorded result, and the diagnostic
channel to be fully reviewed — never a merged single flag.

## Deterministic queue construction — the literal frozen priority order

The order is literal, not an interpretive resolution: **(a)** all 18
both-pHash-and-dHash-rule candidates first, globally, ahead of every
standard candidate; **(b)** `calibration_v2_vs_unknown_test_v2` (238);
**(c)** `within_final_test` (53); **(d)** the remaining six
`final_test_independence` domains; **(e)** the remaining four
`gate_evidence_independence` domains.

The 18 both-rule candidates happen to fall in exactly 5 of the 13 domains
(`expansion_train_vs_final_test`, `final_test_vs_unknown_test_v2`,
`calibration_v2_vs_expansion_train`, `expansion_development_vs_unknown_test_v2`,
`expansion_train_vs_unknown_test_v2` — verified mechanically, never assumed).
`derive_v2_queue_rows` (`training/pair_adjudication_v2_contract.py`)
implements the literal order as two passes over `DOMAIN_VISIT_ORDER`: **Phase
1** pulls each of those 5 domains' both-rule rows into their own small,
single-domain priority block (queue_index 0–17, globally first); **Phase 2**
then queues every remaining candidate exactly as if the 18 phase-1 rows had
never existed. This is how "all 18 both-rule pairs first" and "one domain
per session" are both satisfied without contradiction: a domain with
both-rule rows gets TWO session blocks (its small priority block, then its
later bulk block(s)) rather than one block mixing both tiers.
`assign_session_blocks` enforces this by never merging a domain's tier-1
(priority) rows into the same block as its tier-2 (standard) rows, with
block numbering continuing sequentially per domain across the two phases.

Real, mechanically-verified result of this construction against the
committed candidate report: 3,972 rows, 18 both-rule pairs at queue_index
0–17 (globally first, verified — not merely counted), zero pairs with exact
`phash_distance == 0 AND dhash_distance == 0` in scope,
`identity_order_sha256 = 7e906cc2a66017940a8aef586e0eb9be7743c36f1dddec4ec22d46341562d14f`
(this value will be reconfirmed, not assumed, when the queue is actually
generated in the later phase).

## Sessions

Each session/block holds rows from exactly one domain AND exactly one
priority tier, capped at 300 rows, with a recommended 60–90 minute human
review limit. A domain larger than 300 splits into numbered blocks in queue
order. The real frozen queue produces **25 session blocks** — never
described as "3–4 sessions," and 5 more than the earlier (superseded)
within-domain-first design produced, because the 5 both-rule-bearing
domains now each contribute one extra small priority block.

| # | Session | Rows |
|---|---|---|
| 1 | `expansion_train_vs_final_test__block0` (priority) | 6 |
| 2 | `final_test_vs_unknown_test_v2__block0` (priority) | 1 |
| 3 | `calibration_v2_vs_expansion_train__block0` (priority) | 4 |
| 4 | `expansion_development_vs_unknown_test_v2__block0` (priority) | 1 |
| 5 | `expansion_train_vs_unknown_test_v2__block0` (priority) | 6 |
| 6 | `calibration_v2_vs_unknown_test_v2__block0` | 238 |
| 7 | `within_final_test__block0` | 53 |
| 8 | `benchmark_v1_vs_final_test__block0` | 300 |
| 9 | `benchmark_v1_vs_final_test__block1` | 32 |
| 10 | `calibration_v1_vs_final_test__block0` | 253 |
| 11 | `calibration_v2_vs_final_test__block0` | 188 |
| 12 | `expansion_development_vs_final_test__block0` | 104 |
| 13 | `expansion_train_vs_final_test__block1` | 300 |
| 14 | `expansion_train_vs_final_test__block2` | 263 |
| 15 | `final_test_vs_unknown_test_v1__block0` | 133 |
| 16 | `final_test_vs_unknown_test_v2__block1` | 101 |
| 17 | `calibration_v2_vs_expansion_development__block0` | 182 |
| 18 | `calibration_v2_vs_expansion_train__block1` | 300 |
| 19 | `calibration_v2_vs_expansion_train__block2` | 300 |
| 20 | `calibration_v2_vs_expansion_train__block3` | 300 |
| 21 | `calibration_v2_vs_expansion_train__block4` | 137 |
| 22 | `expansion_development_vs_unknown_test_v2__block1` | 114 |
| 23 | `expansion_train_vs_unknown_test_v2__block1` | 300 |
| 24 | `expansion_train_vs_unknown_test_v2__block2` | 300 |
| 25 | `expansion_train_vs_unknown_test_v2__block3` | 56 |

## Session identity is mechanically bound, never freeform

A record's `session_id` must equal the current eligible queue row's own
`suggested_session` exactly — `adjudicate_pairs_v2.record_adjudication`
enforces this before touching the ledger (not only the review UI), and
`validate_record_bindings` re-checks it for every stored ledger record too,
so a plausible-but-wrong session label is rejected rather than silently
accepted or auto-corrected. `pair_adjudication_v2_review.ReviewSession`
refuses to even launch/construct for a `session_id` that does not match the
current frozen block. The UI checks that boundary again after every recorded
decision: if an early stop or an exhausted block makes the globally next
eligible row belong to another session, it reports that next session for
handoff but does not resolve or open that row's image.

## The new per-write lock

Concurrent-terminal writes to the same ledger were a real risk in a prior
phase (Phase 5F2's TOCTOU publication race). `adjudicate_pairs_v2.py`'s
`--record` now acquires a structurally exclusive per-write lock
(`training/pair_adjudication_v2.lock`, atomic `O_CREAT|O_EXCL` creation)
before touching the ledger. A pre-existing lock is **never** silently
removed or overridden — it fails closed and is reported for review. Inside
the lock: the ledger is reverified, the record is appended and fsynced,
then reloaded and fully reverified, and only the lock this exact
invocation created is ever released (checked by token, not by path alone).
`--record` additionally requires a clean tracked tree before it will touch
the ledger at all.

## Source provenance and the two-commit lifecycle

Every one of the six approved implementation sources
(`APPROVED_IMPLEMENTATION_SOURCE_PATHS` in `pair_adjudication_v2_contract.py`)
must be bound in the contract to a **real commit that actually contains that
exact path with that exact content** — never the working tree, never a
plausible-looking but wrong commit. This requires two separate commits, in
order:

1. **Source-preparation commit** —
   commits the six approved source files (and their tests/docs) for real,
   with the tracked tree otherwise clean.
2. **Evidence-generation phase** —
   only once (1) exists does `build_pair_adjudication_v2_queue.py --write`
   and then `freeze_pair_adjudication_v2_contract.py --write` run.
   `--write` refuses outright unless the tracked tree is clean AND every
   approved source is tracked (`git show HEAD:<path>` succeeds) at HEAD —
   generation against an uncommitted or partially-committed source set is
   refused, not merely warned about.

`freeze_pair_adjudication_v2_contract.py --write` records that single HEAD
commit as `generator_git_commit` for every source. `--check` never rebuilds
the recorded provenance against a fresh current HEAD (an unrelated later
commit must never invalidate already-generated evidence — the immutable-
generation-commit pattern from `generate_inference_policy_v2.py`); instead
it reuses the published contract's own `implementation_sources` dict
verbatim for the byte-for-byte reconstruction comparison, and separately
re-verifies (`pair_adjudication_v2_contract.verify_implementation_source_provenance`)
that: the recorded commit is a strict 40-hex hash resolving to a real commit
that is an ancestor of (or equal to) current HEAD; the source's canonical-LF
content at that commit hashes to the recorded value; and the CURRENT
working-tree copy of that source ALSO hashes to the recorded value. Unlike
Gate v2's diagnostic-only provenance, this last check is **authoritative**:
any edit to a bound source since generation — committed or not — fails
`--check`, not merely marks it stale.

At runtime, `adjudicate_pairs_v2.load_verified_v2_state` performs this same
provenance verification on the loaded contract before trusting anything else
in it, and `record_adjudication` re-checks a clean tracked tree immediately
before any real write. Every stored ledger record binds
`generator_git_commit`/`generator_source_sha256`/`generator_source_path`
to the CONTRACT's own recorded `adjudicate_pairs_v2` entry — never an
independently computed current-HEAD/current-working-tree value — so a
forged-but-well-formed commit or hash in a ledger record is rejected even
after its hash chain is recomputed to match.

## Queue rederivation, not just self-consistency

`load_verified_v2_state` does not stop at confirming the published queue's
bytes hash to what the contract recorded (which a jointly-tampered
queue+contract pair could satisfy trivially by recomputing both). It
rederives the canonical queue fresh from the verified committed
`candidate_pairs_report` (`derive_v2_queue_rows`) and requires the loaded
queue to match it row-for-row, and does the same for the outside-scope
population. Both the queue and the contract remain the sole authorities for
*scope*; the actual candidate data they scope is always re-proven against
the frozen Phase 5F1/5F3 evidence, never trusted merely because two files
agree with each other.

The freeze step enforces the same rule before a contract can exist:
`freeze_pair_adjudication_v2_contract.py` independently rederives the
canonical queue from the verified candidate report and requires the proposed
queue artifact to match both the exact parsed object and exact canonical
bytes. It also validates the published contract's top-level `generation`
envelope (exact keys and frozen values) before `--check` reconstruction;
`generation` is outside `content_sha256`, but it is not outside validation.

## Files

- `training/pair_adjudication_v2_contract.py` — schema/constants/pure
  derivation functions plus git-backed provenance verification (shared by
  every other v2 module).
- `training/build_pair_adjudication_v2_queue.py` — the queue builder
  (`--preflight`/`--write`/`--check`); the generated queue is frozen by the
  hashes at the top of this document.
- `training/freeze_pair_adjudication_v2_contract.py` — the contract
  builder (`--check`/`--write`), enforcing the two-commit lifecycle above;
  the generated contract is frozen by the hashes at the top of this document.
- `training/adjudicate_pairs_v2.py` — the ledger/lock/eligibility CLI (only
  the current next eligible pair may ever be recorded; session_id is
  mechanically bound; generator provenance is bound to the contract, not
  computed live).
- `training/pair_adjudication_v2_review.py` — the local side-by-side
  reviewer (all PIL/Tkinter imports deferred to `launch_review_ui()`;
  refuses to launch/record for the wrong session).
- `training/finalize_stop_status_v2.py` — the per-channel independent
  status/stop-flag finalizer, including the outside-scope identity binding.
- `training/test_pair_adjudication_v2_contract.py`,
  `training/test_build_pair_adjudication_v2_queue.py`,
  `training/test_freeze_pair_adjudication_v2_contract.py`,
  `training/test_adjudicate_pairs_v2.py`,
  `training/test_finalize_stop_status_v2.py`,
  `training/test_pair_adjudication_v2_review.py` — offline/synthetic tests
  (98 tests total, including real-git-repo provenance fixtures and explicit
  regressions for jointly rehashed noncanonical queues, generation-envelope
  mutation, and UI session-boundary isolation).

## Current runtime state

`training/pair_adjudication_v2_ledger.jsonl` now contains the first 309
visually adjudicated rows. The ledger verified cleanly at the reviewer
handoff; `training/pair_adjudication_v2.lock` and
`training/perceptual_duplicate_post_adjudication_stop_status_v2.json` remain
absent. Finalization has not run.

## Reviewer handoff after the first 309 decisions

The first 309 queue rows (queue indexes 0 through 308) were visually
adjudicated by reviewer `huso1`. Beginning with queue index 309, the user
delegated the remaining visual comparison workload to Codex. Subsequent
records therefore use reviewer id `codex-visual-v1`; they must never be
written as, or attributed to, `huso1`.

This is an operational reviewer handoff, not a scope, threshold, label, or
finalization-contract change: `reviewer_id` is already a required non-empty
provenance field and is deliberately not frozen to one identity. The same
four frozen labels and the same per-channel early-stop behavior remain in
force. Codex must visually inspect both images in every pair; pHash/dHash
distances or dataset labels may prioritize/contextualize a pair but must not
alone determine its adjudication. Any pair that cannot be resolved visually
is recorded as `uncertain`, never guessed. Final reporting must stratify
adjudication counts by `reviewer_id` and disclose this mixed-reviewer method.
