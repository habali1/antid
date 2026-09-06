# AntID: Northeast coverage expansion v1

Status: scope and personal/non-commercial license pool approved; Milestone 1
implemented and verified; Milestone 2A metadata audit complete; the 15-species
train/development/final-test photo download is complete and frozen
(`northeast_expansion_v1`, `northeast_final_test_v1`); the resulting 65-species
training catalog is merged into a versioned manifest and taxonomy (see
`TODO.md`); no training has started. Prepared 2026-09-05.

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
6. The 15-species train/development/final-test photo download is complete and
   frozen: 3,600 train/development images (240/species: 200 train + 40
   development, `northeast_expansion_v1`) and a disjoint 450 final-test images
   (30/species, `northeast_final_test_v1`), zero sha256 overlap with each
   other or with `benchmark_v1`/`calibration_v1`/`unknown_test_v1`. A versioned
   65-species catalog now exists outside the live serving artifacts:
   `data/northeast_expansion_v1/northeast_taxonomy_v1.json` and
   `manifest_all_northeast_v1.csv` (13,581 usable rows — 8 legacy rows across
   **6 species** — Linepithema humile, Tetramorium immigrans, Paratrechina
   longicornis, Crematogaster scutellaris, Dolichoderus thoracicus, and
   Dorymyrmex bureni — were excluded as already rejected by `clean.py`). Every
   row carries a `provenance_status` (`northeast_v1_complete` or
   `legacy_partial`); legacy rows carry a real, freshly computed `sha256` for
   all 9,981 rows and a real `observation_uuid` for 9,912 of them (the
   remaining 69, plus all legacy license/attribution/source-URL fields, are
   left blank, never a placeholder string), verified 64-hex-char and
   file-byte-matching with zero duplicates or frozen-set overlap. **Known
   limitation, accepted rather than corrected:** the 15 new species get 200
   train images each while the original 50 mostly get ~160 (158–160 after the
   8 exclusions) — a roughly 25% per-class train-count gap that a first
   expanded-catalog run will simply inherit. `training/artifacts/taxonomy.json`
   and `data/manifest_all.csv` remain the original, unmodified 50-species
   files; no training has run against the 65-species catalog. **Still
   pending, before any training:** a manual labeled-photo quality review and a
   perceptual near-duplicate review (sha256 dedup does not catch
   recompressed/resized duplicates) — neither has been done.
7. Both versioned catalog files are reproducible, not hand-maintained:
   `data_pipeline/build_northeast_training_catalog.py` deterministically
   rebuilds them from three small **committed** metadata snapshots under
   `data/northeast_expansion_v1/catalog_inputs_v1/` (`base_manifest_50_v1.csv`,
   `base_taxonomy_50_v1.json`, `legacy_photo_observation_map_v1.json` --
   byte-identical, hash-verified copies of the original `data/manifest_all.csv`,
   the pinned `training/artifacts/v1_50species/taxonomy.json`, and the legacy
   photo->observation map) plus `data/clean/` and `northeast_train_dev_v1.csv`.
   The script never writes to `data/manifest_all.csv` or `training/artifacts/`.
   **Precisely stated:** only the metadata derivation is portable this way --
   a fresh clone gets the three small committed inputs, but a full check-mode
   run still needs the real `data/clean/` image tree (the original 50 species
   has no committed restore path at all; the 15 Northeast species are
   restorable via `data_pipeline/scrape_northeast_expansion.py --restore` then
   copied into `data/clean/`, but that copy step isn't scripted yet either).
   Do not call this "clean-clone reproducible" -- the image tree remains a
   separate, unproven, local prerequisite. No-argument run = check mode
   (rebuilds into a temp dir, diffs byte-for-byte against the frozen outputs,
   writes nothing); `--write` regenerates them in place. Every
   `northeast_v1_complete` row is required, fail-closed, to have non-blank
   `photo_license`/`photo_attribution`/`observation_uuid`/`photo_id`/
   `source_url`/`sha256`. Covered by
   `data_pipeline/test_build_northeast_training_catalog.py` (determinism,
   fail-closed fault injection including per-field blank-provenance checks)
   and `training/test_taxonomy_loaders.py` (the latter proves exact
   full-object taxonomy equality -- not just genus -- between what
   `training/data.py` derives from the manifest and the versioned taxonomy
   snapshot; this also caught and fixed a real gap where the manifest's
   missing `common_name` column would have silently dropped all 50 curated
   common names on retrain).
   **Retraining the 65-species catalog must use exactly**
   `MANIFEST_CSV=data/northeast_expansion_v1/manifest_all_northeast_v1.csv`,
   `LOCAL_DATA_DIR=data/clean`, and **`DATABASE_URL` unset** (`data.py` tries
   `DATABASE_URL` first and would otherwise silently ignore `MANIFEST_CSV`).
   Do not use the bare-directory loader for this catalog: it loses the pinned
   train/val split, taxon IDs, coordinates, and all other manifest metadata.
   The current 50-species serving artifacts are backed up and hash-verified at
   `training/artifacts/v1_50species/` (gitignored, local-only) so the live app
   is restorable if the 65-species retrain is worse.
8. **Geo coverage gap: resolved, Option A approved and verified.** Originally
   found: **9,974 / 9,981** legacy rows had a usable lat+lon; **0 / 3,600**
   Northeast rows did, because `northeast_train_dev_v1.csv` never carried
   coordinates. A train-only `geo_index.json` built as-is would have contained
   grid cells for the old 50 species only, systematically favoring them via
   `GEO_BOOST`. **User approved Option A** (audited public/obscured-coordinate
   sidecar) over Option B (disable geo re-ranking). For this fixed,
   already-frozen 3,600-row dataset, Option A means **complete** coverage --
   an earlier draft's 70% acceptance threshold was removed as an invented,
   unnecessary bar for a dataset whose size is already fixed and known.
   - Two-stage, reviewed build, not a single script: `fetch_northeast_coordinates.py`
     produced a raw, **unreviewed** capture (preserved byte-for-byte at
     `northeast_coordinates_capture_v1.json`, sha256
     `8275b487ff40d4095dfc9adc6e403299500b7185632104e2ad2f1fe2415a0677`);
     `finalize_northeast_coordinates.py` then builds the actual frozen sidecar
     **offline, from that capture alone, with no network access**, enforcing
     exactly 3,600/3,600 coverage (fail-closed, never a percentage), no
     duplicate/missing/unexpected observation UUID, every coordinate
     numeric/finite/in-range, every source row's `geoprivacy` refused if
     `private`, and taxonomic self-consistency per row. Atomic write, refuses
     to overwrite a frozen sidecar unless byte-identical. Frozen sidecar:
     `data/northeast_expansion_v1/northeast_coordinates_v1.json`, sha256
     `17680f64ab81573969e3994f202a01ab9dad89f7aa8467d56a857f88e0cd98aa`.
     `northeast_train_dev_v1.csv` itself was never touched (hash-verified
     unchanged: `9840997f...4215f7`).
   - **Known, documented limitation:** the raw capture recorded only
     latitude/longitude per observation, not the iNaturalist API's own
     per-observation `geoprivacy`/`obscured` fields, so the "private from
     captured API metadata" cross-check could not be independently repeated
     without a new fetch; the finalize step relies on the frozen source
     manifest's own `geoprivacy` column instead (verified: none are private).
   - **Correction to the obscured-coordinate reasoning:** an obscured location
     is *not* guaranteed to remain in the same 1-degree geo-index cell as the
     true one -- obscuring can cross a cell boundary. AntID's 3x3-neighbor
     cell check can mitigate a one-cell displacement in the Northeast, but
     that is a partial mitigation, not proof of zero precision loss.
   - `data_pipeline/test_finalize_northeast_coordinates.py` (21 tests, fully
     offline/mocked) covers deterministic finalization and every fault
     scenario (hash mismatch, missing/unexpected/duplicate UUID, private
     observation, NaN/infinite/out-of-range coordinates, incomplete coverage,
     taxonomy mismatch, overwrite refusal, atomic-failure safety,
     byte-identical verification mode).
   - **Integrated into the generator:** `build_northeast_training_catalog.py`
     takes the sidecar as a versioned, hash-bound input and joins coordinates
     by `observation_uuid`, requiring all 3,600 Northeast rows to resolve one.
     `manifest_all_northeast_v1.csv` was regenerated (hash changed, as
     expected, since lat/lon are no longer blank); `northeast_taxonomy_v1.json`
     stayed byte-identical, as expected. `training/test_geo_split.py` gained
     an integration test proving `training/data.py` resolves all 65 classes
     and a train-only `build_geo_index` produces usable cells for every one of
     the 15 new species, with validation rows structurally excluded from the
     train split passed to it.

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
The dataset has since been downloaded, byte-hashed (zero frozen-set
collisions), split-assigned (train/development/final-test), and frozen — see
milestone 6 above. No image-quality review or perceptual-duplicate check has
occurred.

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

## Phase 4A: training harness hardened for the first B4-65 development run

Before any training: the harness itself was hardened across two review
passes (the second specifically finding and fixing two blocking integration
bugs plus stale documentation), and every check was verified via
`--preflight-only`, unit tests, and a real orchestration integration test
only -- no model has been initialized, no weights downloaded, and no
training or evaluation has run. **This section describes the harness as
implemented today; see `TODO.md`'s "Phase 4A" entry for full detail**,
including the corrections list.

- **Frozen selection rule for this and the later EfficientNetV2-S
  comparison:** after every epoch, recompute augmentation-free L2-normalized
  prototypes from the pinned train split and evaluate raw (no-geo) cosine
  top-1/top-3 on the pinned val split. Select by (1) highest top-1, (2) tie
  -> highest top-3, (3) tie -> earliest epoch. Never the classifier-head
  result, never geo re-ranking, for selection. Run every configured epoch;
  no early stopping, so the comparison has a stable control.
- **Full-FP32 numerical policy**, pinned in `training/numerics.py` and
  applied/logged identically by `train.py`, `evaluate.py`, and
  `eval_benchmark.py`: TF32 disabled for both cudnn convolutions and
  matmul, `float32_matmul_precision("highest")`, `cudnn.benchmark=False`,
  `cudnn.deterministic=True`, no AMP/autocast/GradScaler. This only removes
  an ambient reduced-precision/algorithm-selection variable -- it is not
  evidence of ONNX-CPU serving parity, which remains a separate, later
  required step.
- **Corrected resume RNG ordering:** model (`pretrained=False`) -> optimizer
  -> `load_state_dict` -> DataLoaders -> RNG/generator state restored LAST,
  immediately before the resumed epoch loop. An earlier draft restored RNG
  state before model construction, which would have silently desynced a
  resumed run's random stream from an uninterrupted run's -- caught and
  proven fixed by a real interrupted-vs-uninterrupted integration test
  (`training/test_train_orchestration.py`), not just a unit test on the
  capture/restore functions in isolation.
- **Crash-consistent, versioned checkpointing:** `checkpoint_last.pth` is
  the sole canonical, resumable commit marker (model + optimizer +
  RNG/generator state + a reference to the current best checkpoint + the
  full canonical per-epoch history), written atomically after every epoch.
  Whenever the selection rule improves, an immutable, versioned
  `checkpoint_best_epoch_NNN.pth` (model + metrics + config + provenance
  only -- no optimizer/RNG state) is written *before* `checkpoint_last`
  references it, and a superseded version is removed only *after* that
  reference safely commits. `history.jsonl` is a pure cache, always
  rederivable from `checkpoint_last`'s canonical history, repaired
  deterministically on resume if stale or missing. The unversioned
  `checkpoint_best.pth` is materialized only at successful finalization,
  which additionally asserts the restored best model's recomputed top1/top3
  exactly match the metrics recorded at training time.
  `--resume` fails closed unless manifest/taxonomy/pinned-split hashes,
  resolved config, backbone/class count, git commit, the numerical policy,
  `run_kind`, `limit_batches`, and `wandb_enabled` all still match.
  `run_manifest.json` status must be `initialized`/`running`/
  `paused_for_smoke`/`failed` to resume -- `paused_for_smoke` is
  deliberately resumable (it is written only after a fully committed
  epoch), which an earlier draft of this harness got backwards; `completed`
  always refuses. A fresh run refuses a nonempty `--artifacts-dir` without
  `--resume`; this experiment always uses
  `training/artifacts/northeast_v1_b4_dev`, never the live
  `training/artifacts/` default.
- **One shared run_manifest.json schema** (`training/run_manifest_schema.py`),
  imported by both `train.py` (writer) and `evaluate.py` (reader), staged to
  match the real lifecycle (`initialized` -> `data_verified` -> `epoch_committed`
  -> `completed`) so the two can never silently drift and a malformed
  manifest produces a specific, field-named error rather than a raw
  `KeyError`. An earlier draft of `train.py` never persisted the
  `manifest`/`taxonomy_source`/`val_split` fields `evaluate.py`'s
  provenance-aware path actually reads -- a real bug that would have failed
  a completed run's standalone evaluation outright, found by review and
  fixed here.
- **Standalone evaluation bound to its own run:** for a provenance-aware
  checkpoint, `evaluate.py` resolves its data source from `run_manifest.json`
  (schema-validated, hash-and-image-byte re-verified) rather than
  `config.yaml`/`DATABASE_URL`/environment, cross-checks `run_manifest.json`'s
  recorded hashes against the checkpoint's own embedded provenance, and --
  when the run completed -- cross-checks the actual on-disk artifacts
  against `run_manifest.final_artifact_hashes`. A clearly labeled legacy
  fallback is preserved only for checkpoints with no embedded provenance
  (e.g. the live 50-species one). `eval_benchmark.py` reports the embedded
  `resolved_config_sha256` for a provenance-aware checkpoint rather than an
  unused `config.yaml`'s hash as if it were authoritative.
- **Known 158-160 vs. 200 per-class train-count imbalance is recorded, not
  corrected**, in this first control run -- no class weighting, resampling,
  or subsampling was added. The later EfficientNetV2-S run must reuse the
  same seed, data, split, epoch budget, optimizer, augmentations, selection
  rule, and numerical policy for the comparison to mean anything.
- **Development model selection may use only the pinned 2,596-image val
  split.** `benchmark_v1`, `calibration_v1`, `unknown_test_v1`,
  `northeast_final_test_v1`, and the live 50-species artifacts (plus their
  local `v1_50species/` backup) remain untouched during development/tuning.
- **The old 0.60 confidence-gate evidence cannot be reused for this
  catalog.** No `inference_policy.json` is created or updated by this
  harness; a new backbone/prototypes/taxonomy requires freshly validated
  policy evidence, not a hash swap onto the old evaluation.
- **`--pause-after-epoch N`** (one-based, requires `--limit-batches`, excluded
  from provenance) exists to smoke-test this real resume path end to end
  before committing to the full 30-epoch run: pause only after a fully
  committed epoch, resume with `--limit-batches` still present but
  `--pause-after-epoch` optionally omitted.
- See `TODO.md`'s "Phase 4A" entry for full detail, including the
  `dataset_selection_seed`/`seed` distinction and why
  `dataset_selection_seed` is a `train.py` constant rather than a
  `config.yaml` key (editing `config.yaml` for this purpose broke a frozen
  calibration-evidence hash check during this phase and was reverted).

## Next bounded terminal task

The bulk train/development/final-test download and dataset freeze are
complete (see milestone 6). Design a small, balanced labeled-photo review for
diagnostic visibility, label plausibility, license/attribution completeness,
and lookalike risk under the approved personal/non-commercial pool, using the
now-frozen data. Stop again before training or evaluating against the
65-species catalog.
