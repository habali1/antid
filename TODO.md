# TODO

## Active roadmap: Northeast expansion

- Scope approved: keep the existing 50 species and add up to 15 missing
  Northeast species. One catalog, not a Northeast-only replacement.
- [Plan, provisional shortlist, and next bounded task](docs/plans/northeast-expansion-v1.md)
  and [public aggregate-count snapshot](docs/plans/northeast-counts-2026-09-05.json).
- Train-only geo export, stale-sidecar replacement, and pinned train+val
  membership are implemented and covered by focused synthetic tests.
- **3,600 train/development photographs for the 15 new species have been
  downloaded and frozen** (`data/northeast_expansion_v1/northeast_train_dev_v1.csv`,
  240/species: 200 train + 40 development), plus a disjoint, untouched
  `northeast_final_test_v1` (30/species). Cleaned copies are staged at
  `data/clean/{slug}/{photo_id}.jpg` alongside the original 50 species, ready
  to resolve through `MANIFEST_CSV`/`LOCAL_DATA_DIR` for a retrain. This is
  metadata-plus-download progress, not model training — no training or
  evaluation has run against this data.
- The 65-species training catalog is versioned outside the live serving
  artifacts: `data/northeast_expansion_v1/northeast_taxonomy_v1.json` (65
  entries, contiguous slug-sorted indices 0–64, with `common_name` preserved
  from the original 50 and a `genus` field on every entry) and
  `data/northeast_expansion_v1/manifest_all_northeast_v1.csv` (13,581 usable
  rows: 9,981 legacy + 3,600 Northeast). Every row carries a `provenance_status`
  of either `northeast_v1_complete` or `legacy_partial` (the original 9,981
  rows). **Every `northeast_v1_complete` row is required, fail-closed, to have
  non-blank `photo_license`, `photo_attribution`, `observation_uuid`,
  `photo_id`, `source_url`, and `sha256`** — the generator refuses to write
  any output if even one is blank. Legacy rows carry a **real, freshly
  computed `sha256`** for every row (not a placeholder) and a real
  `observation_uuid` recovered from the legacy photo→observation map for
  9,912 of them; the remaining 69 legacy rows, and all legacy
  `photo_license`/`photo_attribution`/`source_url` values, are left **blank**
  (never a sentinel string, and never required — that's the documented
  difference from `northeast_v1_complete`) — those fields cannot be recovered
  without guessing. Every `sha256` in the file is verified 64 lowercase hex
  characters matching the resolved file's actual bytes, with zero duplicates
  and zero overlap against `benchmark_v1`/`calibration_v1`/`unknown_test_v1`/
  `northeast_final_test_v1`. `training/artifacts/taxonomy.json` and
  `data/manifest_all.csv` remain the original, untouched 50-species
  serving/training files — **do not treat either of those as the 65-species
  source of truth**; retraining should read the versioned Northeast files
  above.
- Both versioned files are **reproducible from authoritative inputs**, not
  hand-maintained: `data_pipeline/build_northeast_training_catalog.py`
  deterministically rebuilds them from three small **committed** metadata
  snapshots under `data/northeast_expansion_v1/catalog_inputs_v1/`
  (`base_manifest_50_v1.csv`, `base_taxonomy_50_v1.json`,
  `legacy_photo_observation_map_v1.json` — byte-identical copies of the
  original `data/manifest_all.csv`, the pinned
  `training/artifacts/v1_50species/taxonomy.json`, and the legacy
  photo→observation map, verified by hash) plus `data/clean/` and
  `northeast_train_dev_v1.csv`. The script never writes to
  `data/manifest_all.csv` or `training/artifacts/`. **Precisely stated:** only
  the metadata derivation is portable this way — a fresh clone gets the three
  small committed inputs, but a full check-mode run still needs the actual
  `data/clean/` image tree (both the original 50 species, which has no
  committed restore path at all, and the 15 Northeast species, restorable via
  `data_pipeline/scrape_northeast_expansion.py --restore` then copied into
  `data/clean/`, but that copy step itself isn't scripted yet). Do not call
  this "clean-clone reproducible" — the image tree is a separate, unproven,
  local prerequisite. Run the generator with no arguments to rebuild into a
  temp dir and diff byte-for-byte against the frozen outputs (writes nothing);
  `--write` regenerates them in place. Covered by
  `data_pipeline/test_build_northeast_training_catalog.py` (determinism,
  fail-closed fault injection including per-field blank-provenance checks,
  real-data postconditions) and `training/test_taxonomy_loaders.py`
  (genus/common_name survive every `training/data.py` loader path, including
  exact full-object equality against the versioned taxonomy — not just genus).
- **Retraining the 65-species catalog must use exactly:**
  `MANIFEST_CSV=data/northeast_expansion_v1/manifest_all_northeast_v1.csv` and
  `LOCAL_DATA_DIR=data/clean`, with **`DATABASE_URL` unset** — `data.py`'s
  `load_manifest` tries `DATABASE_URL` first and would otherwise silently
  ignore `MANIFEST_CSV` and train on whatever the DB holds instead. Do not use
  the bare-directory loader (`LOCAL_DATA_DIR` alone, no `MANIFEST_CSV`) for
  this catalog: it loses the pinned train/val split, taxon IDs, coordinates,
  and all other manifest metadata, silently reconstructing a different
  (unpinned, taxon-id-less) split instead.
- The current 50-species serving artifacts (`taxonomy.json`, `prototypes.npy`,
  `backbone.onnx`, `model.pth`, `geo_index.json`, `inference_policy.json`) are
  backed up, hash-verified, at `training/artifacts/v1_50species/` (gitignored,
  local-only) so the live app can be restored if the 65-species retrain is
  worse.
- **Geo coverage gap: resolved, Option A approved and verified.** Originally
  found: **9,974 / 9,981** legacy rows had a usable lat+lon but **0 / 3,600**
  Northeast rows did, because `northeast_train_dev_v1.csv` never carried
  coordinates — a train-only geo index built as-is would have contained cells
  for the old 50 species only, and `GEO_BOOST` would have systematically
  favored them. **User approved Option A** (build and freeze an audited
  public/obscured-coordinate sidecar) over Option B (disable geo re-ranking).
  For this fixed, already-frozen 3,600-row dataset, Option A means **complete**
  coverage, not a percentage threshold — an earlier draft's 70% acceptance bar
  was removed.
  - `data_pipeline/fetch_northeast_coordinates.py` fetched public/obscured
    locations for exactly the 3,600 observation UUIDs already in
    `northeast_train_dev_v1.csv` (no new observations, no pool expansion) and
    produced a raw, **unreviewed** capture. That capture is preserved
    byte-for-byte at `data/northeast_expansion_v1/northeast_coordinates_capture_v1.json`
    (sha256 `8275b487ff40d4095dfc9adc6e403299500b7185632104e2ad2f1fe2415a0677`)
    and is never treated as the production sidecar.
  - `data_pipeline/finalize_northeast_coordinates.py` builds the actual frozen
    sidecar **offline, from the capture alone — no network access** —
    validating: exactly 3,600/3,600 coverage (fail-closed, not a threshold);
    no duplicate or unexpected/missing observation UUID (source vs. capture,
    including duplicate-JSON-key detection); every source row's `geoprivacy`
    is refused if `private` (none are — but the raw capture never preserved
    the API's *own* per-observation geoprivacy/obscured fields, only lat/lon,
    so that specific cross-check is a **known, documented limitation**, not
    silently assumed clean); every coordinate numeric, finite, and in
    `[-90,90]`/`[-180,180]`; every source row taxonomically self-consistent
    (species' first token matches its genus; taxon\_id/genus\_id positive
    integers). Writes atomically (temp file + replace) and refuses to
    overwrite an existing frozen sidecar unless byte-identical. Frozen sidecar:
    `data/northeast_expansion_v1/northeast_coordinates_v1.json`, sha256
    `17680f64ab81573969e3994f202a01ab9dad89f7aa8467d56a857f88e0cd98aa`.
    **Corrected claim:** an obscured location is *not* guaranteed to stay in
    the same 1° geo-index cell as the true one — obscuring can cross a cell
    boundary. AntID's 3×3-neighbor cell check can mitigate a one-cell
    displacement in the Northeast, but that is a partial mitigation, not proof
    of zero precision loss.
  - `data_pipeline/test_finalize_northeast_coordinates.py` (21 tests, fully
    offline/mocked, never contacts iNaturalist) covers deterministic
    finalization, every fault-injection scenario above, overwrite refusal,
    atomic-failure safety, and byte-identical verification mode.
  - **Integrated into the generator**: `build_northeast_training_catalog.py`
    now takes the sidecar as a versioned, hash-bound input
    (`EXPECTED_COORDINATES_SIDECAR_SHA256`) and joins coordinates by
    `observation_uuid`, requiring all 3,600 Northeast rows to resolve one
    (fail-closed otherwise). `manifest_all_northeast_v1.csv` was regenerated —
    its hash changed (expected: lat/lon are no longer blank for the 15 new
    species); `northeast_taxonomy_v1.json` is unaffected and stayed
    byte-identical, as expected. `training/test_geo_split.py` gained an
    integration test proving `training/data.py` resolves all 65 classes from
    the updated manifest and a train-only `build_geo_index` produces usable
    cells for every one of the 15 new species, with validation rows
    structurally excluded (train\_s never contains more than the 200 pinned
    train rows per new species — proven directly, not just asserted).
- Eight legacy rows across **6 species** were excluded from the usable merged
  manifest — `clean.py` had already rejected them (`too_small` ×7,
  `duplicate` ×1) and no cleaned file exists for them. Usable legacy total is
  9,981, not 9,989. The 6 species: Linepithema humile, Tetramorium immigrans,
  Paratrechina longicornis, Crematogaster scutellaris, Dolichoderus
  thoracicus, and Dorymyrmex bureni.
- **Known limitation: train-count imbalance.** The 15 new species get exactly
  200 train images each; the original 50 mostly get ~160 (most sit at 158–160
  train / 39–40 val, a handful lower after the 8 exclusions above) — a
  roughly 25% per-class train-count gap between old and new species that has
  not been corrected or balanced. This is accepted, not remediated, for the
  first expanded-catalog training run; note it when interpreting per-class
  results.
- Metadata readiness and post-exclusion availability are complete, and the
  train/development/final-test photo download is frozen. **Still pending
  before training: a manual labeled-photo quality review (diagnostic
  visibility, label plausibility, lookalike risk) and a perceptual
  near-duplicate review** (sha256 dedup only catches byte-identical files, not
  recompressed/resized duplicates). Neither has been done. No training or
  evaluation has started.
- A model trained on CC BY-NC / CC BY-NC-SA imagery inherits the non-commercial restriction. The restriction applies to the trained weights and any derived artifacts (`model.pth`, `backbone.onnx`, `prototypes.npy`), not only to the source images. A full licensing review is mandatory before any commercial use, public deployment, or redistribution of the model artifacts.
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
