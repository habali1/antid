# AntID: Northeast coverage expansion v1

Status: scope and personal/non-commercial license pool approved; Milestone 1 and
metadata-only Milestone 2A implemented and verified; species shortlist remains
provisional; no photos downloaded and no training started. Prepared 2026-09-05.

## Decision and boundaries

- Keep every existing supported species (50) and add **at most 15** missing
  Northeast species. One expanded catalog/model, not a Northeast-only replacement.
- First region: the nine-state U.S. Census Bureau Northeast (Connecticut, Maine,
  Massachusetts, New Hampshire, Rhode Island, Vermont, New Jersey, New York, and
  Pennsylvania). Broome County is the local Binghamton cross-check. This is a
  bounded first step toward North American coverage, not all of North America.
- Personal/local use first. No paid APIs, rented GPUs, hosted databases, public
  deployment, or new subscriptions. Estimate local disk/compute needs before
  downloads/training; stop and discuss any cost or capacity problem.
- Preserve the current working 50-species model and its policy. New experiments
  use separate data and artifact directories; never overwrite `training/artifacts/`.
- The approved cap is not approval of every provisional taxon. Review actual
  label/photo quality and post-exclusion availability before freezing membership.

## Current milestones: what is actually complete

1. Frozen `benchmark_v1`: 1,591 images; current B4 baseline 60.8%/79.3%
   top-1/top-3 raw, 64.2%/82.2% with geo. The historical 66.6%/81.9% split is
   unrecoverable; the contaminated ~93% result is not performance evidence.
2. Calibration and independent Phase C evaluation: the frozen 0.60 rule improved
   accepted **known-species** accuracy from 69.8% to 85.5% on `unknown_test_v1`,
   while accepting 45.5% of out-of-scope ant photographs. It is an abstention aid,
   not a reliable unknown-species detector or an 85.5%-accurate general ant identifier.
3. Numerical transfer evidence and hash-bound policy are recorded. Keep the
   frozen reports and their limitations; the 200-image diagnostic is not a
   population guarantee. No new parity investigation is needed just to plan coverage.
4. Closest-match wording, raw scores, scope warnings, and the optional gate are
   implemented in the API/mobile code. Missing/invalid policy means
   `gate_active: false`, `low_confidence: null`. This does not establish a public deployment.
5. Training/API schema files have the same SHA-256 and the existing byte-identity
   unit test passed on 2026-09-05. Shared hash:
   `4c1ca49fda7faf23472c49a8beb239b519b27f369065ccbd87c3d19b5bae150b`.

Maintenance correction: `api/inference.py`'s recorded source hash is diagnostic
provenance, not a live runtime binding. An arbitrary source edit does not itself
disable the gate. The loader enforces the three model-artifact hashes, the
preprocessing contract, schema/content integrity, and CPU-only provider policy.
After serving changes, verify behavior and regenerate provenance when appropriate;
changing artifacts/preprocessing requires compatible validation evidence, not just
updating hashes. `geo_index.json` is intentionally not bound by the pre-geo gate.

## Count-based shortlist

Evidence: [saved aggregate snapshot](northeast-counts-2026-09-05.json), with
retrieval completed 2026-09-05 15:47 UTC from the public
[iNaturalist API](https://api.inaturalist.org/v1/docs/).
Queries used Formicidae (47336), species rank, research grade, photos, and wild
observations, across all available dates. All ten responses were complete.
The snapshot preserves each query URL, state IDs, full returned species counts,
the current taxonomy hash, and this deterministic ranking rule:

> Exclude taxa already supported; sort by summed observations in the nine states,
> descending, breaking ties by taxon ID. Take the first 15 as provisional candidates.

Broome is already part of NY and is **not** added to the regional total.

| Priority | Provisional addition | iNat taxon ID | NY | PA | NJ | Northeast total | Broome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Nylanderia flavipes | 372712 | 1,574 | 466 | 160 | 2,494 | 0 |
| 2 | Lasius neoniger | 69311 | 525 | 265 | 151 | 1,399 | 2 |
| 3 | Lasius claviger | 222712 | 580 | 257 | 170 | 1,345 | 6 |
| 4 | Camponotus novaeboracensis | 143252 | 328 | 69 | 4 | 1,266 | 5 |
| 5 | Lasius emarginatus | 341143 | 1,002 | 0 | 40 | 1,042 | 0 |
| 6 | Camponotus americanus | 215970 | 197 | 132 | 90 | 1,027 | 0 |
| 7 | Camponotus subbarbatus | 143505 | 450 | 389 | 130 | 987 | 0 |
| 8 | Camponotus nearcticus | 146942 | 415 | 189 | 76 | 879 | 2 |
| 9 | Lasius americanus | 966095 | 293 | 184 | 71 | 711 | 5 |
| 10 | Lasius aphidicola | 1032788 | 228 | 143 | 54 | 667 | 4 |
| 11 | Aphaenogaster rudis | 213893 | 274 | 166 | 109 | 581 | 0 |
| 12 | Formica exsectoides | 133654 | 153 | 120 | 25 | 580 | 17 |
| 13 | Ponera pennsylvanica | 203419 | 235 | 108 | 59 | 495 | 2 |
| 14 | Temnothorax curvispinosus | 232366 | 181 | 105 | 75 | 484 | 0 |
| 15 | Lasius interjectus | 222713 | 178 | 122 | 81 | 474 | 0 |

Reserves, in count order: Aphaenogaster tennesseensis (454), Aphaenogaster fulva
(437), Temnothorax longispinosus (437), Aphaenogaster picea (434), and
Brachymyrmex depilis (406). Substitutions require a recorded
data-quality/availability reason and
must occur before the dataset and final test are frozen, never based on test scores.

Across 50,762 queried regional observations, taxa already supported account for
29,484 (58.1%); adding all 15 candidates would represent 43,915 (86.5%). In the
small Broome sample (121 observations), representation would move from 55 to 98.
These are **catalog representation counts, not identification accuracy, natural
abundance, unique ants, or encounter probabilities**. Research-grade observations
favor identifiable/photographable taxa; many difficult ants never reach species-level
research grade. Rare records can be erroneous; zero counts do not prove absence.
The top 15 are consequently a starting list, not a taxonomic assessment of Binghamton.
iNaturalist taxonomy and aggregate counts can change; these numbers are the dated,
all-time snapshot described above.

## Milestones still to do, in order

### 1. Close the bounded retraining prerequisites — completed 2026-09-05

Implemented in the working tree after the planning commit: train-only geo
export, unconditional stale-sidecar replacement, sorted train+val split pins,
fail-closed pinned-record validation, honest file-provenance labeling, and an
inactive empty-index evaluation path. Verification: 27 focused synthetic tests,
33 policy-generator tests, 11 policy-loader tests, 13 inference tests, 3 API-main
tests, and Python compilation all passed. No model or frozen evaluation ran.

The geo export format is already slug-keyed and compatible with the API. Do not
reopen the old format bug. The remaining issues found in the current code are:

- `training/train.py` builds its geo index from `samples`, not `train_s`, so
  validation coordinates enter the training-emitted index. Export from the
  actual training split only.
- An empty build leaves an existing geo file untouched. Write an explicitly
  schema-valid empty sidecar for the new run so stale cells cannot survive. The
  API should intentionally report it inactive with reason `no_usable_cells`.
- `training/evaluate.py --geo-source train` trusts manifest split labels even
  when validation comes from a pinned split. For new runs, pin sorted `train`
  and `val` membership in `val_split.json` (an additive, backward-compatible
  extension), then require complete, unique, disjoint resolution before calling
  an index train-only. A legacy val-only file's complement does not prove the
  historical training membership; label that limitation or fail closed rather
  than describing it as established fact.
- Document geo provenance accurately: old shipped indexes may include validation
  data or have unknown provenance; future train-only exports must not be
  automatically labeled train+val. A small `source_split: "train"` field is safe
  because the API ignores extra top-level fields; avoid changing the shared
  `load_geo_index` return contract just for labeling.
- Update the stale `build_geo_index` docstring: malformed legacy indexes no longer
  report geo as functionally loaded after the API health hardening.

Use synthetic samples and temporary artifact directories to test the emitted
schema, train/val separation, negative-coordinate flooring, empty/stale behavior,
and API consumer compatibility. Do not train, evaluate frozen datasets, change
the current live geo index, or regenerate the current policy for this work.

### 2. Audit candidate data, then freeze a versioned dataset

Metadata-only readiness audit completed 2026-09-05. See
[`northeast-readiness-v1.md`](northeast-readiness-v1.md) and the ignored
row-level snapshot under `data/northeast_readiness_v1/`. Across 14,431 unique
observations, only 13 matched an existing training/frozen-set photo or
observation. Exact active species and genus ancestry were verified. Under the
predeclared 230-observation quota, 1/15 candidates is numerically ready using
CC0/CC BY/CC BY-SA alone. The approved 2026-09-05 personal/non-commercial pool
also admits CC BY-NC and CC BY-NC-SA, under which all 15 are numerically ready.
No image-quality review, byte-hash check, perceptual-duplicate check, split
assignment, or species freeze has occurred.

The previously agreed **150-image admission floor is not currently binding**:
each of the 15 candidates clears the larger 230-observation quota under the
approved pool. Keep the 150 floor as a fail-closed eligibility rule if a current
candidate is later dropped or a reserve is proposed as a replacement; evaluate
the replacement before it enters the catalog, never after looking at model or
test scores.

Audit methodology retained for reproducibility: collect **metadata only** for
the provisional IDs; resolve exact active taxon IDs and lineage without a fuzzy
name fallback; record current genus IDs and names rather than inferring taxonomy
from string prefixes.

For each candidate, report research-grade observation counts remaining after
all old-data exclusions, distinct observers, geographic spread, photo-license
availability, and the likely train/development/final-test allocation. Aggregate
counts above cannot answer those questions. Before downloading, set a bounded
image/disk budget and record attribution/license handling. Follow
[iNaturalist's API practices](https://www.inaturalist.org/pages/api+recommended+practices),
including paced public metadata requests. No image scrape is authorized by this plan alone.

Then propose a small labeled-photo review to establish whether diagnostic
features are visible. Research grade is evidence, not guaranteed correct species
ground truth. Especially inspect lookalikes within Lasius, Camponotus, and
Aphaenogaster instead of admitting them solely because counts are high.

Once membership and collection budget are approved:

- Use a versioned manifest, not the existing scraper's completion-order split.
  Include observation ID/UUID, photo ID, taxon and genus IDs, source URL,
  `photo_license`, `photo_attribution`, observed/created timestamps, optional
  coordinates, file hash, and split membership. Preserve private/obscured
  location handling.
  Fail closed if license or attribution is blank for a selected expansion row;
  never attempt to reconstruct either field after downloading.
- Prefer one photo per observation. Group any additional photos from the same
  observation in one split; check hashes across splits and against existing
  training, benchmark, calibration, unknown-test, and duplicate-exclusion records.
  Hash matching misses recompressed duplicates; include a near-duplicate review.
- Reuse eligible existing training photos for the original 50 classes, without
  importing old held-out benchmark/calibration/test photos into training.
- Pin a species-stratified, observation-grouped train/development split with
  fixed seed and immutable membership before fitting. Audit observer overlap and
  geographic imbalance; document any limitation rather than promising independence.
- Build a separate frozen `northeast_test_v1` from independent eligible
  observations. Choose quotas and eligibility before seeing model scores; report
  shortfalls and uncertainty rather than backfill based on performance.
- Retain all original 50 classes. Add fewer than 15 if independent usable data is
  insufficient. Record the final catalog count; do not silently drop old classes.

### 3. Establish the expanded B4 baseline, then compare EfficientNetV2-S

- Start the controlled comparison from the intended pretrained initialization,
  not the old fine-tuned checkpoint if its training images now occur in a newly
  assigned development split. The old lost split must not contaminate a new baseline.
- Use the same frozen species catalog, training/development images, prototype
  construction, and selection metric. Compare raw cosine first; report geo separately.
  Keep prototypes train-only and taxonomy/prototype row order synchronized.
- Inspect each chosen pretrained model's actual preprocessing configuration;
  record the observed training and serving transforms. Model-appropriate
  normalization is allowed but must be explicit. Do not change the shipped B4
  preprocessing or infer defaults just from a model-name prefix.
- Fix and record evaluation numerics before new experiments (including disabling
  reduced-precision TF32 evaluation where applicable). Verify the ONNX-CPU path;
  TF32-off alone does not prove numerical equivalence to serving.
- First produce one expanded B4 baseline. Then run a bounded EfficientNetV2-S
  candidate using available local hardware. Record latency, memory, training time,
  top-1/top-3 and per-class results, especially the weak/local classes. No paid GPU.
- Before training, predeclare matched seeds and the same run/tuning budget. Use at
  least two matched seeds per architecture for a comparative claim. If local
  capacity permits only one run each, label the result exploratory and do not
  attribute a difference causally to the backbone or switch architecture on it alone.
- Select using development data only. An architecture swap is not inherently a
  tiny-ant detector or an unknown-species detector. Cropping/localization is a
  separate possible experiment, not a promised property of EfficientNetV2.
- Existing 20-to-50-class results are not a controlled demonstration that the
  next expansion will succeed. Freeze the new comparison instead of extrapolating.

### 4. Evaluate the final candidate and validate its own gate

- Never extend or rewrite `benchmark_v1`. Evaluate the finalized expanded model
  on it once as an **old-50-input regression test with the full expanded candidate
  catalog**. Clearly label the changed output space. Any optional old-50-only
  ranking result is a separate restricted-catalog diagnostic, not overall accuracy.
- Use `northeast_test_v1` for new regional coverage; report it separately. Keep
  the historical 60.8%/79.3% result and artifact hashes intact. Do not tune on either test.
- A new backbone, new prototypes, or new taxonomy invalidates the current gate's
  hash binding and evidence scope. Even prototype-only expansion changes the
  maximum over classes. Do not copy 0.60 into a new policy or rehash old evidence
  to make the loader accept it.
- Create new development calibration and an untouched independent unknown test
  for the final expanded catalog. Some former out-of-scope species will now be
  in-scope: recompute eligibility without altering historical files. Keep new
  out-of-scope test species disjoint from new calibration OOS species.
- Freeze the new candidate rule on development data, then evaluate it without
  retuning on the final unknown test. Preserve known-photo rejection, correct/
  incorrect rejection, accepted known accuracy, per-category OOS acceptance,
  class coverage and uncertainty. If it is not useful, ship closest matches
  without an active gate; inactive still means false/null, not reassurance.

### 5. Integrate the approved catalog and genus presentation

- Derive all supported-species counts from the API/catalog. The three hardcoded
  50-species messages in HomeScreen/ResultsScreen must not survive a catalog change.
- Keep raw match scores, optional geo flags, closest-match wording, and any active
  low-confidence warning. No percentages presented as probabilities.
- First genus feature: an explicitly labeled presentation of candidate genus,
  e.g. "Possible genus: Camponotus - the displayed matches belong to this genus."
  Only make the agreement statement when authoritative genus IDs actually agree.
  This is candidate grouping, **not a confirmed genus identification**. A narrow
  catalog or geo ranking can create agreement even for an unsupported ant.
- Do not add a second confidence threshold, per-class exception, summed-similarity
  probability, or automatic genus label to saved records. A genuinely validated
  genus fallback would be a separate experiment, not UI wording alone.
- Test the complete flow with and without location/policy, correct species count,
  schema/hash mismatch, and new taxon-to-prototype order. Deploy only after approval.

## Personal records and later plans: retained, not lost

After the identification/coverage milestone, add a private local observation
history: user's sample photo, original result list/raw scores, model/policy
version, date, optional spotted location, notes, and any later correction kept
separately from the original model result. Records should be exportable/deletable.
Use local storage first; account creation remains optional and is unnecessary
for single-user use. Cloud sync/authentication is not a prerequisite.

Still deferred: public spotted-location maps, photo publication/moderation,
friend/community additions, and social identification activity. Nothing becomes
public automatically. Assess privacy, consent, moderation, location precision,
hosting, and any recurring cost before enabling those features.

## Next bounded terminal task

Design a small, balanced labeled-photo review for diagnostic visibility, label
plausibility, license/attribution completeness, and lookalike risk under the
approved personal/non-commercial pool. Do not bulk-download images yet. Stop
again before the bulk image download, final species substitution, or dataset
freeze.
