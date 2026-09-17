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

  **Correction (Phase 5F1, not rewriting the above -- it is left as written
  because it is what was true at the time):** the 65-species training
  described immediately above, and every Gate v2 phase since, **already
  occurred** without this review ever having been done. It was not
  performed before training as this section originally required; it was
  deferred. It is **not optional** and it is **not retroactively
  satisfied** by anything that has happened since. It is now **mandatory
  before any model inference is run against `northeast_final_test_v1`** --
  see "Phase 5F1: manual-quality/perceptual-duplicate review machinery
  (preparation only)" below for the frozen contract, the deterministic
  600-row review queue, and the perceptual-hash scan tooling this
  correction prepares (not yet run).
- A model trained on CC BY-NC / CC BY-NC-SA imagery inherits the non-commercial restriction. The restriction applies to the trained weights and any derived artifacts (`model.pth`, `backbone.onnx`, `prototypes.npy`), not only to the source images. A full licensing review is mandatory before any commercial use, public deployment, or redistribution of the model artifacts.
- Keep personal local history and optional accounts on the roadmap. Public
  maps/social sharing remain deferred; no paid infrastructure without approval.

## Phase 4A: hardened training harness for the first 65-species B4 run

**Status: harness hardened, corrected across two review passes, and
preflight-verified; no training has run.** This section describes the
harness as it is actually implemented today -- earlier drafts of this
section (and of the harness itself) described a simpler single-file
`checkpoint_best.pth` design and a narrower resume check; both were revised
after review found real gaps, listed under "Corrections found by review"
below. Do not describe a design other than the one in this section.

- **Explicit, fail-closed, hash-and-image-byte-verified data source.**
  `train.py` gained `--manifest-csv`/`--local-data-dir`/`--taxonomy-json`/
  `--expected-manifest-sha256`/`--expected-taxonomy-sha256` (all required
  together; shared verification logic lives in `training/data_provenance.py`
  so `evaluate.py` can reuse it without a circular import). `DATABASE_URL`
  present in the environment is a hard failure for this path, never silently
  unset or bypassed; there is no bare-directory-walk fallback. Both hashes
  are verified before the manifest is parsed; the manifest-derived taxonomy
  must exactly equal the committed taxonomy object (full-object equality);
  every one of the 13,581 resolved image files is independently re-hashed
  against its manifest row's own recorded SHA-256 (~1.96 GB, verified once,
  never per epoch) before model initialization. For the Northeast run this
  asserts the frozen postconditions: 13,581 samples, 65 classes, 10,985
  train, 2,596 val, all 15 Northeast species at 200/40, every legacy species
  at 158-160 train with a nonempty val set.
- **Full-FP32 numerical policy, pinned and logged** (`training/numerics.py`,
  shared by `train.py`, `evaluate.py`, `eval_benchmark.py`):
  `cudnn.allow_tf32=False`, `cuda.matmul.allow_tf32=False`,
  `float32_matmul_precision("highest")`, `cudnn.benchmark=False`,
  `cudnn.deterministic=True`, no AMP/autocast/GradScaler anywhere. This
  removes an ambient reduced-precision variable and the run-to-run cudnn
  algorithm-selection variable; it does **not** make PyTorch-CUDA
  numerically bit-identical to the ONNX-CPU serving path -- ONNX parity for
  the new model remains a separate, later required step.
- **Deterministic RNG, with correct resume ordering.** Python `random`,
  NumPy, torch CPU, and torch CUDA/all-devices are seeded together
  (`numerics.seed_everything`); an explicit `torch.Generator` is passed to
  the training `DataLoader`; a top-level, Windows-picklable `worker_init_fn`
  seeds each spawned worker's `random`/NumPy state from torch's own
  per-worker seed. On resume: model (`pretrained=False` -- the checkpoint
  supplies every weight, never re-consulted or re-downloaded) -> optimizer ->
  `load_state_dict` (both) -> DataLoaders -> RNG/generator state restored
  LAST, immediately before the resumed epoch loop -- restoring RNG state
  before model construction (an earlier, incorrect draft of this harness did
  exactly that) would let model-init randomness silently desync the stream
  from what an uninterrupted run would have consumed.
- **Dataset-selection seed kept separate from the training seed.**
  `dataset_selection_seed=20260905` (frozen Northeast candidate-selection/
  download provenance) is a `train.py` module constant, recorded into
  `run_manifest.json` only -- never used to seed model
  init/shuffling/augmentation, and deliberately NOT stored in `config.yaml`:
  `config.yaml`'s own byte hash is pinned by frozen calibration/policy
  evidence (`test_policy_generator.py`'s `hashes.config_yaml` checks against
  `data/calibration_v1/calibration_v1_scores.json`), so editing it for an
  unrelated reason risks silently invalidating that evidence binding -- this
  was caught by a real test failure and reverted. `seed: 42` (`config.yaml`,
  unchanged) is the training seed for both this B4 control run and the later
  EfficientNetV2-S comparison.
- **Per-epoch, serving-mirror validation with a frozen selection rule.**
  After every epoch: augmentation-free, L2-normalized prototypes are
  recomputed from the pinned train split, and raw (no-geo) cosine top-1/top-3
  is evaluated on the pinned val split -- the same `topk_accuracy` path
  `evaluate.py` uses, never the classifier-head logits, never geo re-ranking.
  Selection rule, frozen: (1) highest val top-1; (2) tie -> highest top-3;
  (3) tie -> earliest epoch. All configured epochs run; no early stopping,
  so the later B4-vs-EfficientNetV2-S comparison has a stable control.
  Recomputing prototypes every epoch adds real wall-clock cost (a full,
  augmentation-free forward pass over all 10,985 train images per epoch, on
  top of training and val); actual per-epoch train/prototype/validation
  durations are recorded in `history.jsonl` and will be reported after the
  first real completed epoch, not assumed in advance.
- **Crash-consistent, versioned checkpointing** (`training/checkpoint.py`).
  `checkpoint_last.pth` is the single canonical, resumable commit marker,
  written atomically after every epoch: model + optimizer + RNG/generator
  state + completed epoch + a reference to the current best checkpoint
  (epoch/metrics/filename/sha256) + the full CANONICAL per-epoch history
  embedded directly. Whenever the selection rule improves, an immutable,
  versioned `checkpoint_best_epoch_NNN.pth` (model + metrics + resolved
  config + provenance only -- no optimizer/RNG state, so it never duplicates
  AdamW's two extra per-parameter moment tensors) is written *before*
  `checkpoint_last.pth` is updated to reference it; a superseded version is
  removed only *after* that reference safely commits, so a crash can only
  ever leave a harmless, reported-not-trusted orphan file, never a dangling
  reference. `history.jsonl` is a pure, always-rederivable CACHE of
  `checkpoint_last`'s canonical history, rewritten only after
  `checkpoint_last` itself commits; a resume that finds it stale, missing, or
  divergent repairs it deterministically from the checkpoint rather than
  trusting it. The unversioned `checkpoint_best.pth` is materialized only
  once, at successful finalization, as a copy of whichever versioned file
  `checkpoint_last` references at that point -- and finalization additionally
  asserts the restored best model's recomputed top1/top3 exactly match the
  metrics recorded at training time, refusing to finalize on any divergence.
  `--resume` fails closed unless manifest/taxonomy/pinned-split hashes,
  resolved config/hyperparameters, backbone/class count, git commit, the
  numerical policy, `run_kind`, `limit_batches`, and `wandb_enabled` all
  still match the checkpoint's saved provenance -- every mismatched field is
  individually reported. `run_manifest.json` status must be one of
  `initialized`/`running`/`paused_for_smoke`/`failed` to resume;
  `completed` always refuses (re-finalizing would silently overwrite a
  finished run's artifacts). A fresh run refuses to start over a nonempty
  `--artifacts-dir` unless `--resume` is given; this experiment must never
  default to `training/artifacts/` and never has in any command shown for
  it.
- **One shared run_manifest.json schema** (`training/run_manifest_schema.py`,
  `run_manifest_schema_version=1`), imported by both `train.py` (writer) and
  `evaluate.py` (reader) so the two can never silently drift. Fields are
  staged to match the real lifecycle -- `initialized` -> `data_verified`
  (adds `manifest`/`taxonomy_source`/`val_split`, each either a hash-bound
  object or, for a non-explicit data source, `null`) -> `epoch_committed`
  (adds `last_completed_epoch`/`best`) -> `completed` (adds
  `final_artifact_hashes`, all seven serving/bookkeeping artifacts including
  `val_split.json`). `train.py` validates the object immediately before every
  persisted state transition; `evaluate.py` validates it before reading any
  nested field, so a malformed or hand-edited `run_manifest.json` produces a
  specific, field-named error, never a raw `KeyError`. `manifest`/
  `taxonomy_source`/`val_split` are persisted once, right after data/image
  verification and strictly before model initialization; on resume they are
  never silently overwritten -- the freshly re-derived records must exactly
  equal what is already on disk, or the run aborts naming the disagreeing
  field. Bool is explicitly rejected wherever an integer is required (a real
  gap in early ad-hoc validation, since `bool` is an `int` subclass in
  Python). `paused_for_smoke` is a resumable status: it is written only after
  a fully committed epoch, so resuming from it carries exactly the same
  guarantees as resuming from `running`.
- **No `inference_policy.json`.** This phase never creates or updates one --
  a new backbone/prototypes/taxonomy invalidates the current gate's evidence
  scope regardless, and the old 0.60 evidence cannot be reused for a 65-class
  candidate.
- **`--preflight-only`.** Performs every data/hash/split/taxonomy/image-byte/
  output-safety/git-state check (and, under `--resume`, the existing
  checkpoint's provenance, its referenced best file's integrity, and whether
  `history.jsonl` currently matches the canonical history -- read-only,
  never repairing) with no model initialized, nothing downloaded, and
  nothing written; reports the pinned numerical policy, environment/GPU
  info, and ESTIMATED (not measured) artifact/checkpoint sizes. Verified
  PASS against the real frozen 65-species catalog with `--artifacts-dir`
  pointed at the proposed `training/artifacts/northeast_v1_b4_dev` (never
  created) -- except for git-state, which correctly FAILs whenever the
  working tree is dirty, exactly as the real run would refuse to start.
- **Git-clean gate.** A real full run or resume refuses to start unless
  `git rev-parse HEAD` resolves and `git status --porcelain` is empty --
  this run's provenance must reflect exactly the code that produced it.
  Preflight reports (never silently allows past) the same failure.
- **Weights & Biases is opt-in only** (`--wandb`), never enabled merely
  because `WANDB_API_KEY` happens to be set in the environment; the choice
  is bound into provenance (`wandb_enabled`), so a run cannot resume with a
  different logging choice than it started with.
- **`--limit-batches` is bound into provenance**, and any run using it is
  marked `run_kind=smoke` (vs. `full`) -- both fields are part of the
  resume-compatibility check, so a smoke run can never silently resume (or
  be mistaken for) a full production run.
- **`--pause-after-epoch N`** (one-based; requires `--limit-batches`; not
  part of the immutable provenance, so the resuming invocation may omit it)
  exists solely to smoke-test the real resume path: it exits successfully,
  without generating any final artifact, only after a fully committed epoch
  -- never mid-epoch or mid-commit -- and sets `run_manifest.status =
  "paused_for_smoke"`.
- **Known, accepted, un-remediated limitation carried into this control
  run:** the 15 new Northeast species get exactly 200 train images each; the
  original 50 mostly sit at 158-160 (a handful lower after 8 prior
  exclusions) -- roughly a 25% per-class train-count gap. No class
  weighting, resampling, or subsampling was added to address it in this
  phase, deliberately: the goal is a stable, unmodified control recipe for
  the later B4-vs-EfficientNetV2-S comparison, not a rebalanced training
  run. Both architectures must use the same seed, data, split, epoch budget,
  optimizer, augmentations, selection rule, and numerical policy.
- **Development model selection may use only the pinned 2,596-image val
  split.** `benchmark_v1`, `calibration_v1`, `unknown_test_v1`,
  `northeast_final_test_v1`, and the live 50-species `training/artifacts/`
  (plus its local `v1_50species/` backup) must remain untouched during this
  development/tuning phase.
- **Standalone evaluation bound to its own run.** `evaluate.py` branches on
  whether `model.pth` carries embedded provenance. If so: it reads
  `run_manifest.json` (schema-validated first), refuses `DATABASE_URL`,
  resolves the manifest/taxonomy/val_split it recorded (re-verifying every
  hash and every image byte), cross-checks `run_manifest.json`'s own
  recorded hashes against the checkpoint's embedded provenance (two
  independent records of the same facts that must agree), checks
  `taxonomy.json` against the freshly-verified taxonomy source as a full
  object (not merely by length), and -- when `status == "completed"` --
  cross-checks `model.pth`/`prototypes.npy`/`taxonomy.json`/`val_split.json`
  against `run_manifest.final_artifact_hashes`. A `--config` override must
  hash-match the run's own `resolved_config_sha256` or is refused. A clearly
  labeled legacy fallback (old `--config` + environment-driven
  `load_manifest()` behavior) is preserved ONLY for checkpoints with no
  embedded provenance (e.g. the live 50-species one) and is never reachable
  for a provenance-aware artifact. `eval_benchmark.py` got the same
  checkpoint-config binding (no manifest needed there, since it reads
  `benchmark_v1` directly) and never reports an unused `config.yaml`'s hash
  as if it were the configuration actually used -- it reports the embedded
  `resolved_config_sha256`, and a supplied `--config`'s hash only when it was
  explicitly given and verified to match.
- **Corrections found by review, after the harness first looked complete:**
  (1) resume restored RNG state *before* constructing the model/optimizer,
  silently breaking the random stream a resumed run should continue -- fixed
  by moving RNG restore to immediately before the resumed epoch loop; (2) the
  original single `checkpoint_best.pth`/append-only `history.jsonl` design
  had no defined crash-consistent commit order -- replaced by the versioned/
  canonical-history design above, with dedicated crash-boundary fault-
  injection tests; (3) `paused_for_smoke` was writable but not resumable (a
  real bug that would have blocked the planned smoke-test sequence) -- fixed
  by adding it to the resumable statuses, with a regression test exercising
  the real CLI/run_manifest gate, not just the epoch-orchestration function;
  (4) `train.py` never persisted the `manifest`/`taxonomy_source`/`val_split`
  fields `evaluate.py`'s provenance-aware path actually reads, which would
  have failed a completed run's evaluation with a raw `KeyError` -- fixed,
  with a shared schema module preventing the two from drifting again;
  (5) `eval_benchmark.py` reported the default `config.yaml`'s hash as if it
  were authoritative even when a provenance-aware checkpoint's embedded
  config was what actually got used -- fixed to report
  `resolved_config_sha256` instead. `torch.load()` without
  `weights_only=False` also failed outright under this environment's PyTorch
  2.14 (which now defaults `weights_only=True`) on every checkpoint's
  RNG-state/provenance metadata -- a real bug the orchestration integration
  test itself caught on first run, fixed via `checkpoint.torch_load_trusted`.
- Tests, by category: `training/test_train_harness.py` (helper/unit,
  including one real-catalog preflight test), `training/
  test_train_orchestration.py` (real orchestration/integration --
  `TestInterruptedVsUninterrupted` is what makes "resume tested" a true
  claim: it runs two epochs straight through, then separately pauses after
  one, tears down every in-memory object, and resumes into a brand-new
  model/optimizer/DataLoader, asserting identical sample order,
  augmentation-affected losses, final model parameters, full optimizer
  state, selected best epoch/metrics, and history rows), and `training/
  test_run_manifest_schema.py` (shared writer/reader schema contract tests).
  Plus a clean re-run of every previously-existing training and
  data_pipeline suite. `python train.py --preflight-only` against the real
  catalog: every check PASSes except (expectedly, pre-commit) git-state.
- **Real smoke test (commit `13636f5`), first attempt: pause succeeded,
  resume failed on a real CUDA bug the CPU-only synthetic orchestration test
  could not have caught.** Epoch 1 (2 capped training batches, full
  10,985-image prototype pass, full 2,596-image validation pass) trained,
  committed, and paused cleanly -- `checkpoint_last.pth`/
  `checkpoint_best_epoch_000.pth` verified schema-valid and integrity-checked
  before resume was attempted. Resuming in a fresh process crashed:
  `TypeError: RNG state must be a torch.ByteTensor`. Root cause: `main()`
  loaded `checkpoint_last.pth` with `map_location=device` (`"cuda"`), which
  remaps every tensor in the file to CUDA -- including the RNG-state tensors
  (`torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, a bare
  `torch.Generator()`'s state) that are always CPU-resident by definition,
  regardless of which device training runs on. `torch.set_rng_state()` /
  `torch.cuda.set_rng_state_all()` / `Generator.set_state()` all reject a
  non-CPU tensor. The harness's own failure handling worked exactly as
  designed despite the bug: `run_manifest.json` was updated to
  `status: "failed"` with the exact error recorded, `last_completed_epoch`
  and `best` were preserved untouched, and neither `checkpoint_last.pth` nor
  `checkpoint_best_epoch_000.pth` was corrupted (the crash occurred before
  any epoch-2 write began). All six live 50-species artifacts were confirmed
  byte-identical to their pre-run baseline afterward.
- **Correction:** `checkpoint.load_resume_checkpoint()` (new, in
  `checkpoint.py`) is now the single resume-loading path, always via
  `map_location="cpu"` -- never the target training device. Model/optimizer
  tensors still end up on the right device for free: `model.load_state_dict()`
  and `optimizer.load_state_dict()` already copy/cast CPU-loaded state onto
  whichever device the live model/optimizer parameters are already
  constructed on (AdamW's per-parameter `exp_avg`/`exp_avg_sq` moments move
  to CUDA automatically; a scalar step counter may intentionally stay on
  CPU). No blanket "move every tensor to CUDA" step was added. The existing
  resume ordering is unchanged: CPU checkpoint load -> construct CUDA
  model/optimizer -> load model state -> load optimizer state -> construct
  DataLoaders -> restore CPU/CUDA RNG and the DataLoader generator
  immediately before `run_epochs`. Covered by
  `test_train_harness.py::TestLoadResumeCheckpoint` (4 tests, including one
  gated on real CUDA that ran on this machine during this correction pass:
  loading a CPU checkpoint into a CUDA model/AdamW optimizer places
  `exp_avg`/`exp_avg_sq` on CUDA and one optimizer step succeeds).
- **The failed smoke evidence is preserved, not reused:**
  `training/artifacts/northeast_v1_b4_smoke_resume` (status `failed`, bound
  to commit `13636f5`) was not deleted, modified, resumed, or repurposed.
  The next attempt uses a fresh directory
  (`northeast_v1_b4_smoke_resume_v2`) against the corrected code.
- **Timing recorded from the failed run's completed epoch 1 (not yet
  re-measured against the fix):** training 40.8s for only the 2 capped
  batches -- **not a full-epoch number and must never be extrapolated as
  one**; prototype construction 989.5s over the full 10,985 train images;
  validation 234.4s over the full 2,596 val images. Prototype+validation
  together are genuine full-dataset measurements: at face value that is
  ~1,224s/epoch of evaluation overhead, or **~10.2 hours across a 30-epoch
  run if performed every epoch**, before any full-epoch training cost is
  added. No change to validation frequency, prototype sampling, or
  checkpoint selection has been made in response to this -- three options
  (validate every N epochs under one frozen shared protocol; a deterministic
  class-balanced prototype subset for development selection with full
  prototypes only at finalization; or keep full validation every epoch) are
  on the table but **none is authorized yet**. The corrected `_v2` smoke
  run's timing will be reported separately before that decision is made.

## Phase 4B: B4 development report and EfficientNetV2-S preparation

- **B4 candidate result, completed and selected.**
  `training/artifacts/northeast_v1_b4_dev_v2` finished all 30 configured
  epochs (`run_manifest.status == "completed"`, git commit `c622ce9`).
  Selected checkpoint under the frozen rule (highest val raw-cosine top-1,
  tie → highest top-3, tie → earliest epoch): **internal epoch 26 / human
  epoch 27**, val top-1 `0.6802773497688752` (1,766/2,596), val top-3
  `0.8285824345146379` (2,151/2,596), on the pinned 2,596-image development
  split only (65 species). By group: new 15 species n=600 top-1=397/600
  (0.6617), top-3=483/600 (0.805); legacy 50 species n=1,996
  top-1=1,369/1,996 (0.6859), top-3=1,668/1,996 (0.8357). **This is
  pinned development-split analysis; not benchmark or final-test evidence** —
  `benchmark_v1`/`calibration_v1`/`unknown_test_v1`/`northeast_final_test_v1`
  were not touched.
- **One-prediction margin, human epoch 6 vs. 27.** Human epoch 6 (internal
  epoch 5) scored val top-1 `0.6798921417565486` (1,765/2,596), top-3
  `0.8416795069337443` — actually higher top-3 than the selected epoch.
  Epoch 27 won the frozen selection rule (which ranks by top-1 first) by
  exactly **one** top-1 prediction (1,766 vs. 1,765). This is a mechanical
  tie-break outcome, **not evidence of a meaningful statistical
  improvement** between the two checkpoints.
- **Genus-oriented metrics are development-split analysis only**, computed
  from the same raw top-3 predictions used above (never geo re-ranked,
  never gate-filtered): genus_top1 (does the top-1 predicted species share
  the true genus), genus_top3_any (any of the top-3 shares it),
  wrong_species_but_correct_genus_top1 (conditional on a top-1 species
  miss), top3_unanimous_true_genus (all three top-3 predictions share the
  true genus). Reported overall, new-15-vs-legacy-50, per genus, and per
  species, each with numerator/denominator — see
  `training/reports/northeast_v1_b4_dev_report.json`
  (`training/report_dev_metrics.py`, `training/test_report_dev_metrics.py`).
  **Never cite these as benchmark, calibration, or final-test evidence, and
  never as a basis for a new confidence/abstention threshold.**
- **V2-S candidate config prepared** (`training/config.efficientnetv2_s.yaml`,
  `training/config.yaml` untouched): `tf_efficientnetv2_s.in21k_ft_in1k`
  (exact timm identifier confirmed by recon), embedding_dim 1280, native
  300×300 input, bicubic interpolation, mean/std `[0.5,0.5,0.5]` — V2-S's
  own pretrained preprocessing, deliberately different from B4's ImageNet
  contract. Every other recipe field (dataset, pinned split, optimizer, LR,
  weight decay, dropout, augmentations, loss, sampler, seed 42, validation
  cadence, numerical policy) is copied byte-for-byte from `config.yaml` and
  covered by `training/test_efficientnetv2s_config.py`, which also proves
  both configs resolve the identical 13,581-row manifest, 10,985/2,596
  split, and val_split.json sha256
  (`1039518efb33e43f2f97c66e11c1e24947c4a379fc54d6394ac5984c79d5ac7e`).
  `training/data.py` gained an optional `interpolation` config field
  (`resolve_interpolation`, torchvision named `InterpolationMode`, fail-closed
  on an unsupported value); an absent field preserves B4's historical
  bilinear behavior exactly (proven equal to torchvision's own Resize
  default, not just asserted).
- **V2-S pretrained weights were not cached locally as of the 2026-09-11
  recon** (only `tf_efficientnet_b4.ns_jft_in1k` was present in the HF hub
  cache) — the bounded smoke run is the first point at which downloading
  `tf_efficientnetv2_s.in21k_ft_in1k`'s official weights is authorized.
- **Next: a bounded V2-S smoke** (2 epochs, `--limit-batches`, pause/resume,
  its own ignored `northeast_v1_v2s_smoke_resume` artifact directory —
  never the eventual full-run directory). If the smoke and its review pass,
  **one fresh 30-epoch V2-S run** follows, in a fresh directory. There is
  **no 12-epoch screening run**, and a shorter run is never resumed as a
  30-epoch run by changing `--epochs`.
- **No frozen evaluation set may be touched during candidate selection.**
  Development model selection (B4 or V2-S) uses only the pinned 2,596-image
  val split; `benchmark_v1`/`calibration_v1`/`unknown_test_v1`/
  `northeast_final_test_v1` and the live 50-species `training/artifacts/`
  remain untouched throughout this phase.
- **A single-seed B4-vs-V2-S comparison is exploratory, not causal proof of
  architectural superiority.** Per the roadmap's own predeclared standard,
  at least two matched seeds per architecture are needed for a comparative
  claim; with local capacity permitting only one run each, any observed
  difference must be labeled exploratory and never attributed to the
  backbone choice alone.
- **V2-S candidate result (completed, development-split only).**
  `training/artifacts/northeast_v1_v2s_dev` finished all 30 configured
  epochs (`run_manifest.status == "completed"`, git commit `ae2486f`, same
  manifest/taxonomy/val_split hashes as B4). Selected under the identical
  frozen rule: **internal epoch 5 / human epoch 6**, val top-1
  `0.6660246533127889` (1,729/2,596), val top-3 `0.8278120184899846`
  (2,149/2,596). Runner-up: human epoch 15 (1,706/2,596 top-1, derived from
  `history.jsonl`) — a 23-prediction margin, mechanically larger than B4's
  1-prediction margin, but still a single seed. By group: new 15 species
  394/600 top-1 (0.6567), 491/600 top-3 (0.8183); legacy 50 species
  1,335/1,996 top-1 (0.6688), 1,658/1,996 top-3 (0.8307). Report:
  `training/reports/northeast_v1_v2s_dev_report.json`, generated and
  `--verify`-confirmed byte-identical the same way as B4's.
- **B4 vs. V2-S on the identical pinned 2,596-image development split
  (same manifest/taxonomy/val_split bindings; same numerical policy —
  full FP32, TF32 disabled, cuDNN deterministic, no AMP, both sides):**

  | metric | B4 (northeast_v1_b4_dev_v2) | V2-S (northeast_v1_v2s_dev) |
  | --- | --- | --- |
  | selected epoch (internal/human) | 26 / 27 | 5 / 6 |
  | all-65 top-1 | 1,766/2,596 (0.6803) | 1,729/2,596 (0.6660) |
  | all-65 top-3 | 2,151/2,596 (0.8286) | 2,149/2,596 (0.8278) |
  | new-15 top-1 | 397/600 (0.6617) | 394/600 (0.6567) |
  | new-15 top-3 | 483/600 (0.8050) | 491/600 (0.8183) |
  | legacy-50 top-1 | 1,369/1,996 (0.6859) | 1,335/1,996 (0.6688) |
  | legacy-50 top-3 | 1,668/1,996 (0.8357) | 1,658/1,996 (0.8307) |
  | genus_top1 (overall) | 1,854/2,596 (0.7142) | 1,809/2,596 (0.6968) |
  | genus_top3_any (overall) | 2,210/2,596 (0.8513) | 2,200/2,596 (0.8475) |
  | top3_unanimous_true_genus | 123/2,596 (0.0474) | 120/2,596 (0.0462) |
  | wrong-species-but-correct-genus top1 | 88/830 (0.1060) | 80/867 (0.0923) |
  | training+prototype+val wall-clock (30 epochs) | ~219,011 s (~60.8 h) | ~4,838 s (~1.34 h) |

  B4 leads on all-65 and legacy-50 top-1/top-3 and on every genus metric;
  V2-S's new-15 top-3 (0.8183) is higher than B4's (0.8050) -- a mixed,
  metric-dependent signal, not a uniform win for either side. V2-S trained
  **far** faster in wall-clock terms on this machine in this one run
  (~45x); this is a real, measured observation, not a controlled
  throughput benchmark (no isolation from other machine activity).
  **Under the frozen development selection rule, B4 remains the selected
  serving candidate** -- this is a single-seed comparison on development
  data only, exploratory, and not causal proof of architectural
  superiority; no frozen evaluation set was touched, no serving artifact
  was modified, and no retraining, recalibration, or parity run followed
  from it.

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

## Gate v2: status as of Phase 5C1

- **Phase 5B2 datasets are frozen and committed**: `data/calibration_v2/`
  (1,250 rows) and `data/unknown_test_v2/` (790 rows), manifest hashes
  bound in `training/gate_v2_selection_contract.json`.
- **Phase 5C threshold-selection contract is approved and prepared, not yet
  executed.** `docs/plans/gate-v2-threshold-selection.md` +
  `training/gate_v2_selection_contract.json`
  (`content_sha256: 991f7a0b8e83654e45575566eb0648ddcd69866e29c94b8a2030cc6f4bc19f77`,
  `status: frozen_before_calibration_scoring`) define a new, fully
  deterministic v2 algorithm -- not a reconstruction of v1's undocumented
  0.60 selection. The contract additionally binds canonical-LF hashes of
  its own four implementation sources (`api/inference.py`,
  `training/gate_v2_contract.py`, `training/score_calibration_v2.py`,
  `training/select_gate_v2_threshold.py`), an explicit `approved_outputs`
  destination pair, and a canonical identity-order hash over
  calibration_v2.csv's `(photo_id, observation_uuid, sha256)` row sequence
  (computed from CSV metadata only, never an image) that the scorer and the
  shared validator both recompute from score records -- so a reordered,
  missing, or substituted row is rejected even after a fresh
  content_sha256 recompute. `validate_contract` now checks every frozen
  semantic field (policy_name, dataset_quotas, binding paths,
  approved_outputs, runtime, decision_shape, selection policy,
  single_use_rule, closed_sources, low_quality_known, gate_framing) for
  EXACT equality against constants defined once in `gate_v2_contract.py`
  (and consumed, not duplicated, by `freeze_gate_v2_contract.py`) and
  rejects unexpected top-level content keys -- an internally-consistent but
  altered contract (e.g. a smaller quota, a redirected output path) is
  rejected, not just a rehashed one. So real execution verifies code
  identity, output-path safety, and full semantic equality to the one
  approved contract before doing any work.
  `training/score_calibration_v2.py` (ONNX-CPU scorer, --artifacts-dir
  verified by byte hash),
  `training/select_gate_v2_threshold.py` (image-free selector, writes
  its output for both `candidate_selected` and `no_useful_gate_found`),
  and `training/gate_v2_contract.py` (shared schema/arithmetic/score-file
  validator, now also checking runtime/provenance shape, exactly-65/exactly-10
  known_holdout distribution, top-3 index/slug distinctness and agreement,
  non-increasing similarities, a bounded top-level contract key set
  including a type/shape-checked `generation` block, a required
  `dataset_quotas.unknown_test_v2.purpose`, and totality over malformed
  score JSON -- unhashable/wrong-typed `category`, identity fields, or
  `top3_indices` entries produce validation problems, never an incidental
  KeyError/TypeError) are prepared and covered by
  `training/test_gate_v2_calibration.py` (165 synthetic/offline tests,
  including an end-to-end run against a from-scratch tiny ONNX model) --
  no code has been run against a real calibration_v2 image yet.
- **The 65-species candidate's parity prerequisite is satisfied** by the
  already-committed `training/reports/northeast_v1_b4_dev_v2_parity.json`
  (0/260 top-1 and top-3-set disagreements, 5/5 ONNX repeats bit-identical,
  100% CPUExecutionProvider node placement) -- no new parity run is required
  before scoring calibration_v2.

## Gate v2: Phase 5C2 calibration result (candidate selected; independent unknown_test_v2 validation pending)

Real calibration_v2 scoring and mechanical threshold selection have been run
**once**, against the frozen contract (`content_sha256:
991f7a0b8e83654e45575566eb0648ddcd69866e29c94b8a2030cc6f4bc19f77`) at commit
`2fc9261f6c8ca704a2fd220b3b6dc5ff4415f482`, and the results are committed as
data:

- `data/calibration_v2/calibration_v2_scores.json` --
  byte sha256 `4fec18a938ef22e06d6073016d1692512e8fb1e4e41ec645d9aeb33afbebe46b`,
  `content_sha256 35c7bf36042470dde8fc922106c6528fcb6a06e3bac454c4d196880796e778f4`.
- `data/calibration_v2/calibration_v2_selection.json` --
  byte sha256 `00e56d2e64941086ee1d1c663890cb95d3daa4dcce1e729ef972879376f1489b`,
  `content_sha256 82bc754dbf46643862504916f4c273922e9ac3cefc45bfa28e5eb4a15dfdc5b5`.

**Status: candidate selected; independent unknown_test_v2 validation
pending.** This is calibration evidence only -- it has NOT been validated
against the single-use, independent `unknown_test_v2` set, and it is NOT a
serving policy (no `inference_policy.json` exists or was generated from it).

- **Selected threshold: 0.61** (grid integer 61 of 201), compared with the
  contract's frozen `strict_less_than` rule -- a raw max-cosine of exactly
  0.61 is **accepted**, never rejected (equality accepted, not rejected).
- **No-abstention baseline** (all 650 known_holdout rows, no gate): top-1
  **59.85%** (389/650), top-3 **77.23%** (502/650).
- **Accepted-known coverage: 66.46%** (432/650 accepted, 218 rejected).
- **Accepted-known accuracy:** top-1 **78.70%** (340/432), top-3 **91.44%**.
- **Improvement over baseline: +18.86 percentage points** top-1 (78.70% -
  59.85%), comfortably clearing the contract's 5.0pp usefulness floor.
- **Rejection quality:** correct-prediction rejection rate **12.60%**
  (49/389 correct predictions rejected) vs. incorrect-prediction rejection
  rate **64.75%** (169/261 incorrect predictions rejected) -- ratio
  **5.14x**, i.e. the gate rejects incorrect predictions roughly 5x more
  often than correct ones.
- **Diagnostic-only OOD false-acceptance rate at 0.61:** `out_of_scope_ant`
  **50.33%**, `non_ant_insect` **17.33%**, `unrelated` **7.33%**. These
  numbers are diagnostic only, computed AFTER threshold selection, and per
  the frozen contract's `ood_never_affects_selection: true` policy, they
  played **no role whatsoever** in choosing 0.61 -- the selector only ever
  reads known_holdout accuracy/coverage. Given how permissive the
  out_of_scope_ant false-acceptance rate is at this threshold, this
  confirms (as documented in the top-level project instructions) that this
  remains a **selective confidence gate, not an unknown-species detector**.
- **unknown_test_v2 remains completely unopened by inference.** Scoring
  touched only `data/calibration_v2/` images; no unknown_test_v2 image has
  ever been read, decoded, or passed through the model. Its single
  permitted evaluation (per the contract's `single_use_rule`) remains
  available and has not been used.
- **Next gate:** review of this calibration result, then a separately
  authorized turn for the single, independent `unknown_test_v2` evaluation
  -- not run, implemented, or scheduled in this turn.

## Gate v2: Phase 5D1 -- unknown_test_v2 evaluator prepared, NOT run

The single-use, independent evaluator that will mechanically apply the
frozen 0.61 candidate to `unknown_test_v2` is prepared and frozen, but has
**not been executed** -- `data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json`
(the one-shot attempt marker) and `data/unknown_test_v2/unknown_test_v2_eval.json`
(the result) both remain absent, and no unknown_test_v2 image has been opened.

- `training/gate_v2_evaluation_contract.py` defines the frozen evaluation
  contract's schema/constants (mirroring `gate_v2_contract.py`'s pattern):
  the selection contract's content hash, calibration_v2 scores/selection
  byte+content hashes, the selected threshold (61 / 0.61, strict `<`,
  equality accepted), all four candidate artifact hashes, the
  unknown_test_v2 CSV/JSON hashes and exact quotas (790 total; 390
  known_holdout, 65 species x 6; 200/100/100 OOD), a row identity-order
  hash over unknown_test_v2.csv (metadata-only, no image access), and the
  three precommitted pass/fail criteria. `validate_evaluation_contract`
  checks every field for EXACT equality and rejects unexpected top-level or
  content keys (including inside the `generation` block).
- `training/freeze_gate_v2_evaluation_contract.py` (`--check`/`--write`,
  same discipline as the selection contract's freeze script) produced
  `training/gate_v2_evaluation_contract.json`
  (`content_sha256: 49bb0c4ee5749513b62e7bdfa7f50a7619fe16e2afc75a47d60bda25b4bdb10c`).
- `training/eval_unknown_test_v2.py` is the one-shot evaluator: `--preflight`
  (metadata-only -- verifies the evaluation contract, implementation-source
  hashes, a clean git tree, the frozen selection contract, and MECHANICALLY
  RECOMPUTES the calibration_v2 selection from the frozen scores, requiring
  exact equality with the stored result, before ever touching
  unknown_test_v2 metadata; never opens an image, never creates an ONNX
  session, never imports PIL, never writes any file) and `--evaluate`
  (repeats every preflight check; THEN, in order, verifies the explicit
  `--artifacts-dir`, imports every required runtime module, loads and
  validates taxonomy/prototypes, constructs the ONNX session and confirms
  `session.get_providers() == ["CPUExecutionProvider"]` -- all before the
  attempt marker exists, so a session-construction or taxonomy/prototype
  load failure never consumes the single-use budget -- and only then
  atomically creates and fsyncs the immutable attempt marker immediately
  before opening the first unknown_test_v2 image; a crash after that point
  leaves the marker in place and writes no partial output; there is no
  automatic retry). There is no `--threshold`, `--operator`, `--out`, or
  sweep/search mode, and (following a Codex correction) no
  `--contract`/`--selection-contract`/`--scores`/`--selection`/
  `--unknown-test-csv`/`--unknown-test-json` override either -- every input
  path is derived from `--repo` alone, so a byte-identical copy of any input
  file placed in another directory can never redirect image resolution.
  The precommitted validation rule (coverage >= 65%, accepted top-1
  accuracy improves on baseline by >= 5pp, and incorrect predictions are
  rejected at a strictly higher rate than correct ones as a health check,
  not a performance floor) is computed by `gate_v2_evaluation_contract.
  compute_validation()` -- the ONE shared function the evaluator calls to
  write the result and `validate_eval_content()` calls again, independently,
  to fully RECOMPUTE the entire stored `validation` block (status, all
  three criteria, every metric, per-species entries, diagnostic OOD FAR/AUC)
  from `records` and require exact equality; the stored block is never
  trusted on its own, so altering any of it and rehashing `content_sha256`
  fresh is still rejected. OOD categories are diagnostic-only and never
  affect the pass/fail status. The attempt marker itself is now validated
  for exact agreement with the contract (implementation-source hashes,
  scores/selection bindings, every candidate/dataset binding, a strict UTC
  timestamp) both before it is written and again immediately after, from
  the bytes actually persisted to disk; loading a finished evaluation output
  additionally re-locates and re-validates that marker and requires its
  actual byte sha256 to equal the output's own recorded
  `attempt_marker_sha256` -- a merely well-formed 64-hex-character value is
  not sufficient. Reuses (and binds as implementation sources)
  `score_calibration_v2.py`'s image-decode/ONNX-session helpers and
  `select_gate_v2_threshold.py`'s pure per-species/AUC/rejection-metric
  helpers rather than reimplementing them -- the OLD v1
  `eval_unknown_test.py` (PyTorch-based) is not reused.
- `training/test_gate_v2_evaluation.py` (83 synthetic/offline tests) covers
  contract-mutation rejection, exact threshold-boundary arithmetic (0.609999
  rejected / 0.610000 and 0.610001 accepted), the health-check direction,
  OOD-cannot-affect-status, malformed-JSON totality, mutation tests against
  a REAL evaluation output proving every metric/criterion/status change is
  rejected after a fresh content_sha256 recompute, attempt-marker mutation
  tests (altered timestamp, altered bindings, tampered implementation-source
  hash), a redirected-input-path regression (a byte-identical
  `unknown_test_v2.csv` copy beside an unrelated/empty image tree does not
  affect which images are read), and an end-to-end one-shot mechanism run
  (real tiny ONNX model + real tiny images) proving: a session-construction
  failure and a taxonomy/prototype-load failure each leave no marker and no
  result; the marker exists (asserted *inside* the first `Image.open`
  callback, not merely afterward) before any image is opened; a second
  invocation is rejected; a simulated post-marker failure leaves the marker
  in place with no partial output; and the final output refuses overwrite.
- **Final mechanical correction pass** (two remaining issues found by
  review): (1) `numpy` was previously imported lazily inside the per-row
  scoring loop -- i.e. AFTER the attempt marker already existed. Every
  required runtime import (`numpy`, `PIL`, `inference`, `onnxruntime`) now
  happens before `load_candidate_session_and_prototypes()` and before the
  marker is created, proven by a source-order regression test (every
  image-free init token precedes `_create_attempt_marker(`; every
  image-touching call -- `resolve_one_image`, `.read_bytes()`,
  `Image.open(` -- follows it). (2) Final-result publication no longer uses
  `os.replace()`, which silently overwrites an existing destination --
  `publish_eval_output()` writes a unique temp file in the destination
  directory, fsyncs it, then publishes via `os.link()` (which itself fails
  with `FileExistsError` if the destination exists, so exclusivity is
  structural, not a prior check racing the write) and always removes the
  temp file afterward. Covered directly (not merely via the marker-based
  second-invocation test, which stops earlier): a pre-existing destination
  with sentinel bytes is left byte-for-byte untouched and no temp file
  remains after a refused publish; a successful publish writes the complete
  bytes once and leaves no temp file.
- Real `--preflight` was run once against the actual repository (frozen
  metadata and candidate artifacts only, tracked tree verified clean via a
  non-destructive `git stash`/`git stash pop` around the two pending
  `TODO.md`/plan-doc edits, confirmed not left behind afterward): confirmed
  790 unknown_test_v2 rows, 65 known species, the mechanically recomputed
  selection exactly matches the committed `calibration_v2_selection.json`,
  threshold 61/0.61 strict-less-than with equality accepted, all four
  candidate artifact hashes verified, and CPUExecutionProvider available.
  No unknown_test_v2 image was opened; both approved output paths remain
  absent.
- **Next gate:** review of this preparation, then a separately authorized
  turn to run `eval_unknown_test_v2.py --evaluate` for the one, single
  permitted unknown_test_v2 evaluation -- not run in this turn.

## Gate v2: Phase 5D2/5D3 -- unknown_test_v2 consumed exactly once; independently VALIDATED

`unknown_test_v2` has now been evaluated -- **exactly once, as the frozen
`single_use_rule` requires** -- via `eval_unknown_test_v2.py --evaluate`
against the frozen evaluation contract
(`content_sha256: 49bb0c4ee5749513b62e7bdfa7f50a7619fe16e2afc75a47d60bda25b4bdb10c`)
at commit `ecd9cd4d23e4e55519a94247aff689ae868d1c8f`. **unknown_test_v2 must
never be evaluated again** -- the attempt marker below makes any further
`--evaluate` invocation refuse by construction.

Frozen evidence artifacts:

- `data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json` (the
  one-shot attempt marker) -- byte sha256
  `fd790252ffbe4b20a6f52c26fb902e0cd7fbfedbd88b5613da048992b1a062bf`,
  2,645 bytes.
- `data/unknown_test_v2/unknown_test_v2_eval.json` (the result) -- byte
  sha256 `b695e44ecd902b3763b6f202695e0495308a9f10d43400c47a46253c6029447e`,
  718,760 bytes, `content_sha256
  e7c1d565c7ecd0999e00c65cc238296cf78525fc8efd00da8af91b3a0dbf7616`.
  Re-loaded and fully re-verified (including the marker's actual byte hash
  against the value the result itself records) via
  `gate_v2_evaluation_contract.load_and_verify_eval_file` before this
  record was written.

**Result: `validation_status: validation_passed`.** All three precommitted
criteria passed. The frozen threshold, raw unrounded pre-geo max cosine
**< 0.61** (equality at exactly 0.61 is accepted, never rejected), was
**not adjusted** after seeing this result and never will be for this
candidate.

- **Known_holdout baseline** (n=390, no gate): top-1 **238/390 = 61.03%**
  (`0.6102564102564103`), top-3 **307/390 = 78.72%**
  (`0.7871794871794872`).
- **After the gate:** accepted **256**, rejected **134**, coverage
  **256/390 = 65.64%** (`0.6564102564102564`) -- clears the 65% floor.
- **Accepted accuracy:** top-1 **79.6875%**, top-3 **91.40625%**.
- **Top-1 improvement over baseline: +18.661858974359 percentage points**
  -- comfortably clears the 5.0pp usefulness floor.
- **Rejection behavior:** correct-prediction rejection rate
  **34/238 = 14.29%** (`0.14285714285714285`) vs. incorrect-prediction
  rejection rate **100/152 = 65.79%** (`0.6578947368421053`) -- incorrect
  predictions rejected **4.605263157894737x** more often than correct ones
  (health check passes; this only confirms the gate is not operating
  backwards, it is not itself a performance floor).
- **Diagnostic-only OOD false-acceptance rate at 0.61** (never used in any
  criterion): `out_of_scope_ant` **0.51** (n=200), `non_ant_insect`
  **0.11** (n=100), `unrelated` **0.19** (n=100).
- **Diagnostic-only known-vs-OOD AUC:** `out_of_scope_ant`
  **0.6516410256410257**, `non_ant_insect` **0.8430512820512821**,
  `unrelated` **0.8107692307692308**.
- Full per-species breakdown (65 species, **n=6 rows each** -- descriptive
  at this sample size, not a stable population-level per-species estimate)
  is preserved verbatim in `unknown_test_v2_eval.json`'s
  `content.validation.metrics.per_species_known`.

**Interpretation, stated explicitly:**

- OOD metrics above are **diagnostic only** and did not affect, and cannot
  affect, `validation_status` -- the criteria read only known_holdout
  accuracy/coverage, exactly as the frozen contract specifies.
- The permissive **51% out_of_scope_ant false-acceptance rate**
  reconfirms, on independent data, the project's existing framing: this
  remains a **selective confidence gate, not an unknown-species
  detector**.
- `calibration_v2_selection.json` **remains immutable**, still recording
  `status: candidate_selected` from Phase 5C2 -- this independent
  validation is represented entirely by the separate
  `unknown_test_v2_eval.json` evidence artifact above, never by editing
  the selection artifact.
- **Gate v2 is independently validated but NOT YET DEPLOYED.** No v2
  `inference_policy.json` has been generated; the live serving
  `inference_policy.json` and all serving artifacts are untouched.
- **Next gate:** a separately authorized decision/turn to actually
  generate and review a v2 `inference_policy.json` (or defer/decline) --
  not done in this turn.

## Gate v2: Phase 5E1 -- policy schema v2 / generator prepared, NOT run to a write

The independent validation above (`unknown_test_v2_eval.json`,
`validation_status: validation_passed`) stays frozen and byte-unchanged.
This phase added only schema/generator machinery:

- `policy_schema.py` (byte-identical in `training/` and `api/`) gained
  schema v2 alongside the untouched v1 default: `SCHEMA_VERSION_V2 = 2`,
  `FROZEN_THRESHOLD_V2 = 0.61`; `SCHEMA_VERSION = 1` / `FROZEN_THRESHOLD =
  0.6` are unmodified. A v2 policy additionally requires a
  `validation_evidence` block, and `EXPECTED_V2_VALIDATION_EVIDENCE`
  freezes the EXACT hash/status/FAR value of every field (not merely its
  format) -- `validate()` requires exact equality, so a rehashed policy
  that swaps in a different, individually well-formed hash or a different
  in-range FAR is rejected, not only one with the block missing/malformed.
- `api/inference_policy.py`'s loader now selects the permitted threshold
  from the policy's own `policy_schema_version` before comparing, so a v1
  policy carrying 0.61 or a v2 policy carrying 0.60 both fail closed as
  `unsupported_rule`.
- New, separate generator `training/generate_inference_policy_v2.py`
  reads the frozen Gate v2 evidence chain through the already-committed
  shared validators, recomputes and cross-checks the calibration
  selection, requires the independent evaluation's passed status, binds
  the candidate's three artifact hashes plus the parity report's byte
  hash (`bec08235...`), and requires a real ONNX Runtime session on
  `CPUExecutionProvider` exclusively. It loads the evaluation contract
  EARLY and, immediately after reading `calibration_v2_selection.json`,
  checks its actual byte hash against
  `evaluation_contract.content.selection_binding.byte_sha256` BEFORE
  touching status/result/diagnostics; every required selection field
  (`schema_version`, `content_sha256`, `generation`,
  `content.status/result/diagnostics`) is then checked via `.get()` --
  never direct indexing -- so each is a controlled `GeneratorV2Error`
  when missing/malformed, never a traceback (self-consistency and
  recomputation checks are retained afterward as defense in depth). The
  scores/selection cross-check against `scores_binding` and
  controlled-JSON-parse handling apply throughout. The two
  `policy_schema.py` copies must be byte-identical, checked via raw
  `read_bytes()` equality (catches CRLF-vs-LF-only divergence, which
  canonical-LF-normalized hashing would miss); canonical-LF hashes are
  retained separately for the diagnostic-only `content.provenance`
  recording of six source files. `--write` requires a clean tracked
  tree, records git HEAD + provenance, pre-validates the built policy
  through the REAL API loader in an isolated temp directory BEFORE any
  publication, and only then publishes atomically. `--check` reruns the
  complete preflight, rebuilds the expected deterministic content, and
  requires the stored top-level key set, schema version, content, and
  `content_sha256` to match EXACTLY, plus separately the stored
  `generation` key set and `generator_version` -- only `generated_at` may
  differ -- before a final loader check. Schema v2's envelope is now
  checked exactly (v1 unconstrained, unchanged): top-level keys exactly
  `policy_schema_version, content, content_sha256, generation`;
  `generation` exactly `generated_at, generator_version`; `generated_at`
  a strict `YYYY-MM-DDTHH:MM:SSZ`; `generator_version` equal to the ONE
  frozen `policy_schema.V2_GENERATOR_VERSION`. The original V1 generator
  (`training/inference_policy_generator.py`) is untouched and still reads
  only the unrenamed v1 constant names, so it cannot accidentally emit
  schema v2.
- New test files: `training/test_policy_schema_v2.py` (35 tests --
  v1/v2 coexistence, cross-version rejection, schema-version type
  strictness, exact-value `validation_evidence` mutation tests including
  hash-swap and FAR-change cases, schema-v2 envelope mutation tests
  (extra/missing top-level and generation keys, changed
  generator_version, malformed timestamps), each with a freshly
  recomputed `content_sha256`), `training/test_generate_policy_v2.py`
  (44 tests -- generator failure paths incl. malformed-JSON controlled
  failures, scores/selection binding mismatches, every missing required
  selection field, clean-tree gating, provenance recording, raw-byte
  schema-copy-divergence refusal including a CRLF/LF-only case,
  isolated-loader pre-validation with nothing left behind on failure,
  `--check` reconstruct-and-reject-every-mutation tests including
  generation/top-level-key mutations that never touch content_sha256,
  atomic-exclusive publish, write/check round-trip). `api/test_inference_
  policy.py` gained `TestSchemaV2Loading` (valid v2 loads active at 0.61;
  v1+0.61 and v2+0.60 both `unsupported_rule`; missing
  `validation_evidence` is `invalid_schema`) and
  `TestLiveV1PolicyStillLoadsAgainstRealArtifacts` (the real live v1
  policy still loads active at 0.60 against the real serving artifacts,
  read-only).
- **Only `--preflight` was run against real evidence**, against
  `training/artifacts/northeast_v1_b4_dev_v2`: `"ok": true`, every
  evidence hash bound correctly, git HEAD and all six provenance source
  hashes recorded, both `policy_schema.py` copies confirmed
  byte-identical (raw bytes), zero files written anywhere. No v2
  `inference_policy.json` was generated, no artifact was copied or
  promoted, and **the live 50-species v1 gate (threshold 0.60) remains the
  only active gate**. `northeast_final_test_v1` was not accessed.
  `--write`/`--check` were exercised only against synthetic fixtures.
- **Next gate:** a separately authorized `--write` run against the
  candidate directory, review of its output, then a separate atomic
  promotion step -- neither done in this phase.

## Gate v2: Phase 5E1 correction -- `--check`'s recorded git_head is an immutable generation commit

Found after Phase 5E2 generated a real candidate policy: the original
`cmd_check()` rebuilt `content.provenance.git_head` from the repo's
**current** HEAD, so any later commit -- even one wholly unrelated to Gate
v2 -- would falsely stale-out an unchanged policy on the next `--check`.
This made durable freezing/deployment incompatible with later
re-verification.

Fix: `run_preflight()` gained a `git_head_override` parameter
(`--preflight`/`--write` still use current HEAD, unchanged). `cmd_check()`
now calls new `validate_recorded_generation_commit()` against the policy's
OWN recorded `content.provenance.git_head`/`source_hashes` FIRST, fully
fail-closed: strict 40-hex format; resolves to a real commit object
(`git cat-file -t`); is an ancestor of current HEAD (`git merge-base
--is-ancestor`); every recorded source hash matches that file's
canonical-LF hash **as it existed at that commit** (`git show
<commit>:<path>`, never the working tree); and every recorded source hash
ALSO matches the CURRENT working-tree file. Only then does it rerun the
complete current preflight with that validated commit as
`git_head_override`, so unrelated later commits are tolerated but any
change to a provenance-bound source (this generator included) still makes
the policy stale. `api/inference_policy.py`'s loader is untouched --
provenance stays diagnostic-only; the API never invokes git.

New tests in `training/test_generate_policy_v2.py` (7 added, now 51
total in that file, 371 training tests overall): `--check` passes after
an unrelated later commit; `--check` fails when a provenance-bound source
changes afterward; controlled failure for a malformed recorded
`git_head`, a well-formed-but-nonexistent one, and one that resolves but
whose recorded source hashes disagree with that commit's real content;
a content mutation with freshly recomputed `content_sha256` still fails;
`--check` writes zero bytes. The fixture in that file now builds a REAL
small git repository (two commits: fixture setup, then generation
evidence) instead of mocking git away, since `--check` performs real git
operations.

**The Phase 5E2 candidate policy was deliberately left untouched by this
correction** (`training/artifacts/northeast_v1_b4_dev_v2/inference_policy.json`,
byte sha256 `9e9d0ea4447555170bec40902fd2f19582d44102c8047d516bedb969d3e93171`,
content_sha256 `2b6679e95ff2fe270965e93c3c3bbdb5eae0444b3f3b5eaaaad9700e451afbd8`).
Because `generate_inference_policy_v2.py` is itself one of the six
provenance-bound sources, this code change makes that candidate's recorded
provenance **stale** -- a `--check` against it will now correctly report
that this generator's source has changed since its generation commit.
Regenerating it under the corrected, committed code is a separately
authorized, one-write phase; this correction did not regenerate, stage, or
commit it.

## Gate v2: Phase 5E2 regeneration -- candidate policy regenerated and frozen as evidence

The stale Phase 5E2 candidate (generation commit `c20d68f...`) was
reverified via `--check` at HEAD `918422c1237d7847b8e963502416abeff1cd0eae`
and, as expected, failed in the exact controlled way the correction above
predicts: `training/generate_inference_policy_v2.py has changed since
commit c20d68f...` -- zero bytes written. It was then archived byte-for-byte
(never deleted) to
`training/artifacts/northeast_v1_b4_dev_v2/inference_policy.phase5e2-c20d68f.json`
(byte sha256 `9e9d0ea4447555170bec40902fd2f19582d44102c8047d516bedb969d3e93171`,
unchanged) -- **a local recovery copy only, never committed** (it lives
under the gitignored `training/artifacts/` tree, and no rule stages it).

A single authorized `--write` then regenerated the candidate at the
corrected code's own HEAD:

- **byte sha256:** `9feeae83013ecf72266084421fc74bbfe21757de528ad7ed258c01dcffc9422d`
- **size:** 6683 bytes
- **content_sha256:** `35aef7446b47df4c521a612a3d73db44a35e85b1bd0577dac2aa4330bf55b7b5`
- **generation.generated_at:** `2026-09-13T16:34:07Z`
- **generation.generator_version:** `1.0.0`
- **content.provenance.git_head (generation commit):** `918422c1237d7847b8e963502416abeff1cd0eae`
- **policy_schema_version:** 2, **threshold:** 0.61, **validation_status:** `validation_passed`

Independently reverified through `training/policy_schema.py`,
`api/policy_schema.py`, and `api/inference_policy.load_inference_policy()`
(active, reason `active`, threshold 0.61, strict raw/unrounded/pre-geo
comparison, equality at 0.61 accepted, `CPUExecutionProvider` exclusive,
exact candidate artifact hashes, exact frozen `validation_evidence`, all
six provenance source hashes matched against both the historical commit
and the current tree, `workspace_git_dirty: true` preserved as the
unchanged historical parity fact), and through `--check` (twice,
post-freeze-commit -- see below).

**This is frozen candidate evidence, not a deployment.** The live
50-species V1 gate (threshold 0.60) remains the only active serving
policy; `training/artifacts/inference_policy.json` and all six live
serving artifacts are untouched. The candidate model artifacts
(`backbone.onnx`, `prototypes.npy`, `taxonomy.json`, `model.pth`,
`geo_index.json`, `run_manifest.json` under
`training/artifacts/northeast_v1_b4_dev_v2/`) remain local, immutable, and
hash-bound by the candidate policy -- none were copied, promoted, or
otherwise modified by this phase.

This candidate policy's recorded generation commit **intentionally remains
`918422c...`** even after the evidence-freeze commit that follows adds it
to the repository at a later commit -- the corrected `--check`
(`validate_recorded_generation_commit()`, see the correction above)
validates that recorded commit historically (format, real commit object,
ancestor of current HEAD, source hashes matching both that commit's actual
history and the current working tree) and explicitly permits unrelated
descendant commits, rather than requiring the recorded commit to equal
whatever HEAD currently is.

**Next gate:** an isolated API smoke test of the candidate policy -- not
promotion, not deployment.

## Phase 5F1: manual-quality/perceptual-duplicate review machinery (preparation only)

**Preparation only -- no actual review, scan, adjudication, or finalization
has been run.** No dataset image was opened. `northeast_final_test_v1`
remains closed to model inference; nothing here promotes or modifies any
artifact. This section reflects the machinery after six Codex-reviewed
correction passes (working-tree state, not yet staged/committed).

- New shared, pure schema/constants module `training/manual_review_contract.py`:
  the frozen seed (`20260905`), the 600-row manual-review queue composition
  (450 `northeast_final_test_v1` rows + 15 new species x 5 train x 5
  development = 150 sampled rows from `northeast_expansion_v1`), the
  SEPARATE 9,259-row perceptual-scan population across all 8 domain parts
  (the full 3,000 train + 600 development rows, never just the review
  sample, plus 450 final_test + the 5 other-evidence sets' full row counts:
  benchmark_v1 1,591 / calibration_v1 1,005 / unknown_test_v1 573 /
  calibration_v2 1,250 / unknown_test_v2 790), the per-domain-part image
  directory layout binding (`domain_part_image_layouts` -- expansion/
  final_test go through a `clean` curation subdirectory, the 5
  other-evidence sets store images directly under `{dataset}/{slug}/`, with
  a shared deterministic `.jpg`/`.jpeg`/`.png` extension resolver that
  fails closed on zero or ambiguous matches), the two digest formulas
  (`compute_selection_digest` for bucket selection, a separately salted
  `compute_review_order_digest` for intermixing the 600 rows into two
  300-row suggested sessions), the exact review-state/label-plausibility/
  duplicate-suspicion enums, the remediation rule verbatim, the
  perceptual-hash algorithm parameters (32x32 pHash resize, top-left 8x8
  DCT-II coefficients, median threshold; 9x8 dHash; 8 dihedral
  orientations; thresholds pHash<=10 / dHash<=8), the four adjudication
  labels, the 36 canonical comparison domains (8 within-part + 28
  cross-part, mechanically generated so A-vs-B and B-vs-A can never both
  appear), the 7 domains that mandate a stop before
  `northeast_final_test_v1` model inference, and the shared strict schema
  validators for all five scan reports plus the post-adjudication
  finalization artifact (exact keys, cross-report hash bindings, and
  semantic derivation -- a mutated-then-rehashed report is still rejected).
- New generic `training/append_only_ledger.py`: a hash-chained, append-only
  JSONL ledger (genesis hash, `prev_record_hash`/`record_hash` chaining,
  exact required-field-set enforcement, duplicate-identity rejection)
  shared by both the manual-review tool and the pair-adjudication tool.
  Tail recovery is governed by the raw trailing-newline byte, not merely
  whether the last line parses: a newline-terminated but malformed final
  line is real corruption and fails closed (never auto-deleted); a
  non-newline-terminated invalid fragment is treated as an interrupted
  write and physically truncated (atomic, fsynced); a non-newline-
  terminated but *valid* final record (interrupted after the JSON bytes
  landed but before the trailing newline) is preserved and newline-
  terminated, never discarded.
- New `training/generate_manual_review_queue.py`: builds the deterministic
  600-row manual-review queue from
  `data/northeast_final_test_v1/northeast_final_test_v1.csv` and
  `data/northeast_expansion_v1/northeast_train_dev_v1.csv` -- **CSV
  metadata only, never an image path**. `--preflight`/`--write`/`--check`
  modes; fails closed on a source-manifest row-count/column mismatch, a
  bucket with fewer than 5 eligible rows, or a duplicate/missing row
  identity.
- New `training/perceptual_hash.py`: pHash/dHash using only Pillow and
  NumPy (no new dependency) -- verify source bytes against the manifest
  sha256, EXIF-transpose, deterministic grayscale, all 8 dihedral
  orientations, an explicit orthonormal float64 DCT-II cosine matrix (no
  FFT shortcut) for pHash, strict-greater-than pixel comparison for dHash,
  plus a BK-tree exact radius-search index proven equal to a brute-force
  reference on randomized and real-image synthetic fixtures.
- `training/manual_review_tool.py`: pausable, append-only, hash-chained
  recording of one reviewer decision per frozen queue row. Every working
  mode (`--preflight`/`--next`/`--status`/`--record`) now goes through
  `load_verified_ledger`, ONE shared semantic verification path that --
  beyond generic hash-chain replay -- rejects an unknown/extra or
  duplicate `queue_index` and re-runs full decision-field validation
  against the frozen queue row and current contract for every EXISTING
  record, not merely the one being appended; a record whose identity-bound
  field was altered and the hash chain locally recomputed is rejected by
  every mode, and progress/the next index are derived from that one
  verified snapshot rather than a second raw read. `--preflight` can never
  report `ok: true` with an outstanding chain or semantic problem. Enforces
  the 300-decisions-per-session cap (additional session IDs remain usable
  beyond it); **never imports onnxruntime or any inference/model module**
  (statically checked by its own test, via AST import inspection, not a
  substring grep).
- `training/scan_perceptual_duplicates.py`: metadata-only manifest loading
  and independent metadata-leakage detection (matching `observation_uuid`
  across datasets/splits, regardless of whether the photographs differ)
  are pure/no-image-access; `load_and_verify_full_scan_state` is the ONE
  shared full-verification entry point (contract, all five reports, schema/
  binding, reconstructed 8-part population, genuine re-derivation) reused
  by `--check` here and by every mode of `adjudicate_pairs.py` and
  `finalize_stop_status.py`. Within-domain candidate search
  (`generate_candidate_pairs_within_indexed`/`_bruteforce`) examines every
  unordered pair exactly once, in one canonical sorted identity order, so
  A-vs-B and B-vs-A can never both appear under the same `pair_id`.
  `compute_hashes_for_rows` is the ONE function that opens image bytes (not
  run this phase). `cmd_scan` validates the freshly assembled five-report
  bundle BOTH structurally (`mrc.validate_scan_report_bundle`) AND against
  the real frozen population (`validate_reports_against_population`) while
  everything is still in memory, strictly before the exclusive five-file
  publish -- either layer failing publishes nothing, leaves no temp files,
  and a corrected later `--scan` remains possible. A real, metadata/
  filesystem-only `--resolve-check` mode resolves every one of the 9,259
  scan rows' image paths (zero bytes opened) and reports per-part extension
  counts. `evaluate_stop_conditions` implements the mandatory-stop rule: a
  CONFIRMED `same_source_image` adjudication or a metadata-leakage finding
  in a stop-mandating domain stops before inference; a bare perceptual
  candidate never does on its own -- but the scan-time `stop_status_report`
  itself can never carry a confirmed-duplicate reason, since it runs before
  any adjudication.
- `training/adjudicate_pairs.py`: same append-only/hash-chain discipline,
  keyed by an order-independent `pair_id`
  (`manual_review_contract.compute_pair_id`), recording one of the four
  exact adjudication labels per candidate pair. Every working mode
  (`--preflight`/`--next`/`--status`/`--record`, and
  `confirmed_same_source_pairs`) goes through `load_verified_state` (the
  shared scan-state loader above -- never a candidate-report-only envelope
  check) and `load_verified_ledger`, ONE shared semantic ledger-
  verification path reused as-is by `finalize_stop_status.py`: beyond
  generic chain replay, it rejects an unknown/extra or duplicate `pair_id`
  and re-runs full field validation (identity_a/b, domain, pHash/dHash
  distance, both report-hash bindings, enums, timestamp) against the fully
  verified candidate row for every EXISTING record.
- New `training/finalize_stop_status.py`: a SEPARATE, contract-bound
  post-adjudication finalization artifact and gate
  (`--preflight`/`--finalize`/`--check`) -- the scan-time
  `stop_status_report` is produced before adjudication and can never carry
  a confirmed-duplicate stop; this artifact is produced only once every
  candidate pair has an adjudication record, via
  `adjudicate_pairs.load_verified_ledger(..., require_canonical_tail=
  True)` -- the same shared ledger check every other mode uses, plus the
  additional requirement that the raw ledger bytes are newline-terminated
  with no silently-dropped trailing fragment. It folds confirmed
  same-source findings in by domain, preserves every scan-time
  metadata-leakage stop verbatim, activates the mandatory stop only in the
  7 frozen critical domains, refuses to complete until the ledger's
  `pair_id` set exactly equals the candidate set, and publishes
  exclusively/atomically -- never modifying the five scan reports, the
  ledger, or any dataset. It is the actual required gate before
  `northeast_final_test_v1` model inference; no new remediation policy is
  invented, only the already-frozen `STOP_BEFORE_INFERENCE_DOMAINS`/stop
  reasons/`remediation_rule` are acted on.
- Frozen JSON contract `training/manual_review_contract.json` (written by
  `training/freeze_manual_review_contract.py --write`, byte-for-byte
  reproducible via `--check`, binding all 9 implementation-source file
  hashes including `finalize_stop_status.py`): `content_sha256`
  `f32133d6c532b89890f5ec026e338eb71d029c1a60cedce0ef9bc00ef852e4a9`.
- Deterministic queue `training/manual_review_queue.csv` (600 rows; byte
  sha256 `99a0cbb191f5e3eb1dcb68a84e6018020aeb89be6f4335c06ca2eb5537cae7f4`
  -- unchanged since the first round, since queue selection itself never
  changed) and its summary `training/manual_review_queue_summary.json`
  (byte-for-byte reproducible via `--check`; embeds source-manifest hashes/
  counts, not the 600 rows themselves): `content_sha256`
  `888db2eaa217f23e5073461be652191fa174fc8606ff6acef7c7df9251cc073b`
  (byte sha256 `58335df32b91e0f4ce57758c4f293d3427bc7ba2e0243d9a06836107f7751c67`).
  Composition verified exactly: 450 `final_test` + 75 `train` + 75
  `development` = 600 review-queue rows (the separate 9,259-row perceptual-
  scan population is described above); every one of the 15 new species
  contributes exactly 5 train + 5 development review-queue rows; suggested
  sessions split exactly 300/300.
- Test files (272 tests total, all offline/synthetic, all passing):
  `training/test_manual_review_contract.py` (52),
  `training/test_append_only_ledger.py` (17 -- including all three raw
  trailing-newline tail-recovery cases),
  `training/test_perceptual_hash.py` (25 -- exact pHash/dHash boundary
  cases at 10/11 and 8/9, resize/recompression/rotation/mirror synthetic
  candidates, BK-tree-vs-brute-force equality, all against synthetic
  in-memory images only),
  `training/test_generate_manual_review_queue.py` (26 -- composition,
  determinism, source-manifest mismatch rejection, duplicate/missing
  identity rejection, no-real-image-access proof via a `Path.open` spy),
  `training/test_manual_review_tool.py` (36 -- schema rejection, append-
  only/session-cap/resume, the no-model-import guard, other-evidence
  manifest-substitution rejection, and mutate-and-rehash ledger-tamper
  regressions proving every mode rejects a semantically altered existing
  record and the ledger stays byte-identical),
  `training/test_scan_and_adjudicate.py` (59 -- metadata leakage,
  canonical within-set candidate-pair generation, indexed-vs-brute-force
  equality under unordered semantics, pre-publication population
  validation with zero-files-left-behind proofs, real end-to-end `--scan`/
  `--check` re-derivation failure injection, and adjudication-ledger
  mutate-and-rehash regressions), `training/test_freeze_manual_review_contract.py`
  (32 -- contract determinism plus a static no-dataset-mutation guard),
  `training/test_finalize_stop_status.py` (25 -- completeness gating,
  leakage-preservation, confirmed-duplicate stop derivation, and ledger
  semantic-validation failure injection including a trailing-fragment
  rejection even with every real record present).
- **Only preparation was run**: `py_compile` on every Phase 5F1 module, the
  offline/synthetic test suites above plus the full existing regression
  discovery (933 tests total, 13 skipped/GPU-only, 0 failures),
  `freeze_manual_review_contract.py --write`/`--check` (twice),
  `generate_manual_review_queue.py --check` (twice, unchanged), a real
  metadata-only `--preflight` for the queue tool/scanner/manual-review
  tool, and a real metadata/filesystem-only `--resolve-check` (9,259/9,259
  rows resolved, 0 missing, 0 ambiguous). `--review`/`--record`, `--scan`,
  `--adjudicate`, and `--finalize` were never invoked. No dataset image was
  opened; no report, ledger, or finalization artifact exists on disk; git
  status/diff confirm no `data/` file changed and nothing is staged.
- **Next gate:** an explicitly separate, later-authorized phase to actually
  run the manual review (`--record`) against the frozen queue, then the
  perceptual scan (`--scan`), pair adjudication (`--record`), and finally
  `finalize_stop_status.py --finalize` -- none of which are authorized yet.

## Phase 5F2: manual review completed; single-target correction mechanism prepared

The real 600-row manual review (`manual_review_tool.py --record`, the gate
Phase 5F1 left explicitly unauthorized) has since been run to completion:
`training/manual_review_ledger.jsonl` now holds 600 semantically verified
records (two sessions of exactly 300), passing
`manual_review_tool.load_verified_ledger` end to end -- generic hash-chain
replay AND full re-validation of every record against its frozen queue row
and the frozen contract. Byte sha256 of the ledger as it stands:
`75a2ef01058d978ddec19daf4e9eb4d599ed497cc03a506ac23fd3ed0e06bec4`. This
file, like the real review/scan/adjudication/finalization steps themselves,
remains uncommitted pending its own separate authorization -- nothing about
it is staged by this entry.

**A reviewer needed to correct one already-recorded judgment** (queue_index
430: `unusable_no_visible_ant` -> `poor_quality_usable`, a clarification
that the ant was present but small/out of focus, not truly absent) after
the review had already completed. The original ledger is append-only and
closed by design; editing it after the fact would break the guarantee
Phase 5F1's own hardening pass established (every working mode refuses a
tampered-but-rehashed record). So the correction is a **wholly separate,
additive mechanism, authorizing EXACTLY this one correction and no other**
-- not a general-purpose corrections tool:

- `training/correct_manual_review.py` (+ `training/manual_review_
  corrections.jsonl`, not yet written): an append-only, hash-chained
  ledger for the correction, same `append_only_ledger.py` discipline as
  the original review/adjudication ledgers, but a write-ONCE artifact --
  `--write` refuses outright if the file already exists, rather than
  appending a second time. A SEPARATE file the original ledger never
  touches; deliberately NOT added to `manual_review_contract.
  APPROVED_OUTPUT_PATHS` (embedded verbatim in the already-frozen
  `manual_review_contract.json`; adding a key there would change its
  `content_sha256`). None of the 9 already-frozen Phase 5F1 implementation
  sources were edited, so the frozen contract/queue/summary remain
  byte-for-byte unchanged (`--check` reconfirmed twice: contract
  `f32133d6c532b89890f5ec026e338eb71d029c1a60cedce0ef9bc00ef852e4a9`,
  queue `99a0cbb191f5e3eb1dcb68a84e6018020aeb89be6f4335c06ca2eb5537cae7f4`,
  summary `888db2eaa217f23e5073461be652191fa174fc8606ff6acef7c7df9251cc073b`).
- **Every value that matters is a frozen module constant, not CLI input.**
  `load_verified_original()` gates the ORIGINAL ledger's byte sha256,
  record count (600), and per-session counts (300/300) against frozen
  `APPROVED_*` constants BEFORE a single correction is even considered --
  a ledger with a different byte hash (even one that is itself perfectly
  chain-valid, or a full rechain of the identical 600 records in a
  different order) is rejected outright. `validate_correction_fields()`
  then binds every field to the ONE approved target (`queue_index` 430,
  its exact row identity, its exact original `record_hash`/judgment, the
  exact corrected judgment, and a fixed `reason`) -- not merely "internally
  consistent with itself", but equal to the frozen constants. `--write`'s
  CLI accepts ONLY operational metadata (`--reviewer-id`, `--session-id`,
  `--corrected-at-utc`); there is no flag, and no parameter on
  `build_approved_correction_fields()`, through which a different
  `queue_index` or judgment combination could ever be supplied. A future
  correction requires a separately reviewed protocol/source update, not a
  flag on this one.
- **Each correction record also binds its own generator's provenance**,
  following the exact immutable-generation-commit pattern already
  established for Gate v2's `generate_inference_policy_v2.py`: the git
  commit `correct_manual_review.py` was committed at, that file's
  canonical-LF sha256 AT that commit (read via `git show`, never the
  working tree), and a fixed protocol version
  (`phase5f2-correction-v1-single-target-430`). `--write` requires the
  correction source to be committed and the tracked tree clean before it
  will run at all. `--check` validates the recorded commit as a REAL
  ancestor of current HEAD (an unrelated later commit does not invalidate
  existing evidence) and re-verifies the source hash both at that commit
  and in the current working tree (a later edit to the generator itself
  -- even a well-intentioned one -- makes the evidence stale and forces
  regeneration).
- **Lifecycle is deliberately asymmetric**, not a copy of the
  preflight/write/check trio used elsewhere: `--preflight` reports the
  approved correction as pending (file absent) or applied (file present)
  without requiring either state; `--write` refuses if any correction
  artifact already exists; `--check` REQUIRES the artifact to exist and
  contain exactly the one approved record -- zero, extra, duplicate, or
  substituted corrections all fail closed, even after being rehashed to
  look self-consistent. The resulting effective counts are checked against
  a frozen `APPROVED_EFFECTIVE_COUNTS` constant as an independent, final
  assertion, not merely derived and trusted.
- **Publication is a structurally exclusive, atomic stage/fsync/hard-link
  publisher, local to `correct_manual_review.py`** -- NOT
  `append_only_ledger.append_record()`, which opens its destination in
  plain append mode and is a TOCTOU race (two concurrent callers can both
  observe the destination absent and both then write). Since this
  protocol permits exactly one record for this artifact, ever,
  `_publish_single_correction_record()` instead: builds the complete
  record in memory (`prev_record_hash` = the ledger genesis hash,
  `record_hash` via `append_only_ledger`'s own canonical hashing
  functions -- the artifact stays fully hash-chain-compatible); serializes
  it as exactly one newline-terminated JSONL line; stages that into a
  UNIQUE temp file (pid + a random uuid4 suffix) in the destination's own
  directory; the ENTIRE open/write/flush/fsync/prevalidate/link lifecycle
  runs inside one `try/finally` so a failure at ANY of those steps --
  not merely during prevalidation or the link attempt -- still guarantees
  the temp file is removed; prevalidates the staged bytes (exact
  round-trip, exactly one parsed record, a full `verify_chain` replay)
  BEFORE attempting to publish; publishes via `os.link(tmp, dest)`, which
  fails atomically at the filesystem level (`FileExistsError`) if `dest`
  already exists -- the same same-directory hard-link pattern
  `scan_perceptual_duplicates.publish_reports_atomically` and
  `finalize_stop_status.cmd_finalize` already use, never `os.replace`,
  never plain append, and never trusting a prior `.exists()` check by
  itself. `append_only_ledger.py` (a frozen Phase 5F1 implementation
  source) was NOT modified to add this -- the publisher is local, so the
  frozen contract's implementation-source hashes stay unchanged.
- **Publication success is never trusted on comparing bytes alone.**
  Before `record_correction`/`--write` reports success, it reloads the
  PUBLISHED artifact from disk through the full real verification path
  (`load_verified_original` + `load_verified_corrections`, freshly
  re-invoked against the file that now actually exists on disk -- never
  reusing the pre-publish state or the in-memory dict the publisher
  returned) and independently re-confirms: exactly one correction exists;
  the hash chain is valid; every approved-target/original-ledger binding
  and generator-provenance check passes; the resulting effective counts
  exactly equal `APPROVED_EFFECTIVE_COUNTS`; and the persisted bytes equal
  the prevalidated staged bytes. Any post-publication verification failure
  raises `CorrectionError` WITHOUT deleting or rewriting the (already
  correctly published) artifact -- it is left exactly as it is, for manual
  review, never silently repaired or discarded.
- `effective_records`/`effective_counts` remain pure, in-memory-only
  functions: the original 600 records, with the one matching correction's
  judgment fields overlaid; row count is always exactly 600.
- Simulated in memory (never written) against the real repo's frozen
  contract/queue and the real 600-record ledger, the approved correction
  reproduces exactly: `usable` 542, `poor_quality_usable` 53,
  `unusable_no_visible_ant` 1, `unusable_wrong_organism` 4, `implausible`
  5, `plausible` 542, `uncertain` 53, `duplicate_suspicion: none` 600,
  final-test `poor_quality_usable` 41, final-test unusable total 3 -- and
  the real ledger's byte hash was confirmed unchanged before and after.
- `training/test_correct_manual_review.py` (45 tests, offline/synthetic,
  all passing, including a REAL throwaway git repository per fixture --
  `git init` + a real commit containing a copy of the actual
  `correct_manual_review.py`, mirroring `test_generate_policy_v2.py`'s
  established pattern -- so the generator-provenance checks are exercised
  against real git history, not mocked away): frozen-original-ledger-
  authority rejection (byte-different, and separately a fully rechained-
  but-still-byte-different ledger, both rejected BEFORE any correction is
  considered; wrong record count; wrong session counts), single-approved-
  target enforcement (no CLI flags exist for target/judgment; a direct
  Python call with a tampered `queue_index` or an alternate but otherwise
  *valid* judgment combination is rejected), lifecycle tests (`--check` on
  an absent artifact fails; write-then-check succeeds; a second `--write`
  is refused; extra/substituted corrections fail even after rehashing),
  generator-provenance tests (an uncommitted source edit or a dirty tree
  blocks `--write`; the written record's commit/source-hash/version are
  exactly right; an unrelated LATER commit leaves existing evidence valid;
  a later edit to the generator source itself invalidates it; mutated-and-
  rehashed commit/source-hash fields are rejected), exact-effective-count
  enforcement, mutate-and-rehash rejection of row identity/reason/contract
  binding, exclusive-atomic-publication tests (successful publication is
  exactly one valid record with no temp file left; a pre-existing sentinel
  destination is never touched; a destination created mid-publish -- via a
  wrapped `os.link` -- wins the race, stays byte-identical, and `--write`
  fails cleanly; two REAL `threading.Barrier`-synchronized concurrent
  `--write` calls -- exactly one succeeds, exactly one fails, one record
  on disk, no temp files; an `os.fsync` failure AFTER the temp file is
  created still leaves no destination and no temp file; an unexpected
  `os.link` error and a forced prevalidation failure both leave nothing
  behind), and post-publication-verification tests (the reload/re-verify
  step is proven to actually execute, via call-counting; a forced
  post-publication count mismatch, a forced reload/re-verification
  failure, and a real on-disk byte corruption injected right after a
  genuine successful publish all block success while leaving the already-
  published artifact untouched, never deleted or rewritten) -- every path
  proven to leave the original ledger byte-identical throughout.
- **Only preparation was run this pass**: `py_compile`, the new offline
  test suite plus the full Phase 5F1 suite (317 tests, 1 skip, 0
  failures), `--check` (twice, unchanged) for both the queue and the
  contract, and a real read-only `--preflight` against the actual repo
  (reporting the current, uncorrected baseline: `unusable_no_visible_ant`
  2, `poor_quality_usable` 52, `plausible` 543, `uncertain` 52,
  `approved_correction_pending: true`). The real `--write` was
  deliberately NOT run -- `training/manual_review_corrections.jsonl` does
  not exist yet. No image was opened; no perceptual scan, adjudication, or
  finalization step was touched.
- **Next gate -- an explicit TWO-commit lifecycle, each separately
  authorized:** (1) this preparation commit (source, tests, and this TODO
  entry -- no ledger, no correction artifact); (2) the real,
  separately-authorized `correct_manual_review.py --write`, run only after
  commit (1) is itself committed (so the generator-provenance binding has
  a real commit to point at) and the tracked tree is clean; then a further
  evidence commit adding the resulting `training/manual_review_
  corrections.jsonl` (and, separately, a decision on whether/when to also
  commit the completed Phase 5F2 `manual_review_ledger.jsonl` itself).

## Phase 5F2: manual review completed and frozen as evidence

**The real Phase 5F2 manual review is complete and its evidence is now
committed.** `training/manual_review_ledger.jsonl` holds 600 semantically
verified records (two sessions of exactly 300: `phase5f2-review-session-1`
= 300, `phase5f2-review-session-2` = 300), passing
`manual_review_tool.py --preflight`'s full shared verification path (chain
+ per-record re-validation against the frozen queue/contract) with zero
`chain_problems` and `remaining_count: 0`. Byte sha256: `75a2ef010
58d978ddec19daf4e9eb4d599ed497cc03a506ac23fd3ed0e06bec4` (549,954 bytes).
**This file is immutable append-only evidence -- it is committed exactly
as it was written and is never edited, rechained, or replaced again.**

The one approved queue_index-430 correction (see the section above for the
full provenance-binding design) has been written for real, exactly once:

- **First attempt (shell-mangled, zero writes, not a real attempt):**
  invoking with an unquoted `--repo C:\dev\antid` let the shell strip
  every backslash, producing the literal (invalid) path `C:devantid`.
  `correct_manual_review.py`'s very first step
  (`_require_clean_tracked_tree`) failed trying to `git status` in that
  nonexistent directory and exited 1 BEFORE contract/ledger loading,
  record construction, temp-file creation, or publication -- confirmed
  by re-checking every prior invariant (ledger hash, correction-artifact
  absence, zero temp files, untouched git status) immediately afterward.
  This did not count as the artifact-generation attempt.
- **Second, separately authorized attempt (the real one-shot write):**
  invoked with quoted forward-slash paths
  (`--repo 'C:/dev/antid'`, argument-round-trip proven identical via a
  direct `pathlib.Path` equality assertion first), run exactly once,
  directly, no pipe/wrapper/retry. Exit 0.
  `training/manual_review_corrections.jsonl`: byte sha256
  `bfc1d7614e0e3964773bc6e4aa98680992d1971c9f3255d1550cbbdb7af165c8`
  (2,152 bytes, exactly 1 record, newline-terminated), `record_hash`
  `335e5759663937fcc782ddf8d06a43ff89bc326d21496be8a3f6397b20b6a5bb`,
  `prev_record_hash` = the ledger genesis hash, `corrected_at_utc`
  `2026-09-14T19:07:28Z`, generator provenance `generator_git_commit`
  `b485ea0ebae64664c7516d8aed72b93c9e7ac3f8` (the preparation commit),
  `generator_source_sha256`
  `676b4bdbf8c179649e8ea1aff85e9d521ea5ad11d07c277fc0f92448e75daec7`,
  `generator_protocol_version` `phase5f2-correction-v1-single-target-430`.
  `--check` immediately afterward, and again before this evidence commit,
  both confirm `total_corrections: 1`, `approved_correction_pending:
  false`, and the exact effective counts below -- byte-identical, chain-
  valid, every approved-target/original-ledger/generator-provenance
  binding intact.

**Effective counts (original ledger + the one correction, in-memory
overlay only -- neither file is mutated to produce this view):**
`usable` 542, `poor_quality_usable` 53, `unusable_no_visible_ant` 1,
`unusable_wrong_organism` 4, `plausible` 542, `implausible` 5, `uncertain`
53, `duplicate_suspicion: none` 600, final-test `poor_quality_usable` 41,
final-test unusable total 3.

**The original ledger remains the sole immutable record of what was
actually decided at review time.** Any consumer that wants the corrected
(effective) view -- rather than the as-originally-recorded one -- MUST
separately load and apply `training/manual_review_corrections.jsonl` via
`correct_manual_review.effective_records()`/`effective_counts()`; reading
`manual_review_ledger.jsonl` alone yields the pre-correction state (queue_
index 430 still shows `unusable_no_visible_ant`/`plausible`), which is
correct and expected -- that file is what the reviewer actually wrote, not
what was later clarified.

**Perceptual scanning (`scan_perceptual_duplicates.py --scan`) is described
in the Phase 5F3 section below. Pair adjudication and finalization have NOT
started.** No adjudication ledger or finalization artifact exists. No image
was opened at any point in Phase 5F1 or 5F2.

## Phase 5F3: perceptual-duplicate/metadata-leakage scan run exactly once

**`scan_perceptual_duplicates.py --scan` has been run exactly once, for
real, against the full frozen 9,259-row scan population (the separate,
larger population described in the Phase 5F1 section above -- never the
600-row manual-review sample).** All five scan reports were produced,
published atomically, and re-verified once with `--check` (full
re-derivation against the real population, not merely self-consistency).
Both manual-review ledgers (`manual_review_ledger.jsonl`,
`manual_review_corrections.jsonl`) are byte-identical to their committed
state before and after -- the scanner never touches them.

- **9,259 images hashed**, verified against their manifest sha256 before
  decoding, under **Pillow 12.3.0 / NumPy 2.4.6** (recorded in
  `perceptual_duplicate_hashes.json`'s content).
- **435 metadata-leakage findings** (matching `observation_uuid` across
  datasets, independent of the photographs) -- **every one of them is
  `calibration_v1` <-> `calibration_v2`; none involve `final_test`** or
  any other domain pair.
- **16,539 perceptual candidate pairs across all 36 domains -- ALL OF THEM
  UNCONFIRMED.** A candidate pair is a pHash/dHash proximity match only; it
  becomes a duplicate ONLY after a human adjudicator records
  `same_source_image` for it via `adjudicate_pairs.py --record`, which has
  not run. Candidate-rule decomposition (mutually exclusive, verified
  directly from the published report): pairs satisfying BOTH the
  pHash<=10 AND dHash<=8 rules simultaneously: **517**; pHash-rule-only
  (dHash>8): **11,805**; dHash-rule-only (pHash>10): **4,217**; of the 517
  satisfying both, those with EXACT pHash=0 and dHash=0 (byte-identical or
  near-identical-content images): **435** (a coincidental match in COUNT,
  not identity, with the 435 metadata-leakage findings above -- the two are
  independent signals over different domain pairs and are not claimed to
  be the same underlying pairs).
- Of the 16,539 candidates, **1,681** fall in the 7 frozen critical
  (`STOP_BEFORE_INFERENCE_DOMAINS`) domains that pair `final_test` against
  train/development/the 5 other-evidence sets:
  `benchmark_v1_vs_final_test` 332, `calibration_v1_vs_final_test` 253,
  `calibration_v2_vs_final_test` 188, `expansion_development_vs_final_test`
  104, `expansion_train_vs_final_test` 569, `final_test_vs_unknown_test_v1`
  133, `final_test_vs_unknown_test_v2` 102. `within_final_test` holds
  another **53** candidates but is NOT one of the 7 stop domains (it is a
  within-set domain, not a final_test-vs-something cross domain).
- **Scan-time `overall_stop_before_inference: false`** -- correct and
  expected: the scan-time `stop_status_report` can never carry a confirmed-
  duplicate reason (it runs before any adjudication exists), and none of
  the 435 leakage findings touch a stop domain, so nothing here currently
  mandates a stop. This says nothing about what a completed adjudication
  might later find among the 1,681 + 53 final-test-adjacent candidates.
- Report hashes (byte sha256 / content_sha256), all reconfirmed via a
  second real `--check` before this evidence commit:
  `perceptual_duplicate_hashes.json`
  `d0fd1f6ec081d3387a01228b409d80cc439ff002774a7efb318e930b7f37b58d` /
  `c5e5af6e10349c926e2098b358b8c7a41988d7d530189288f5248d9e60d0ad6b`;
  `perceptual_duplicate_metadata_leakage.json`
  `4c962c36a7f60ed5e7305162ab430f7860262ef972145fd9a3a6dd0b836342cb` /
  `52c824eac24b20a2deaf3c71dc2353dae5c8513e3017ac908d5d145f82d31066`;
  `perceptual_duplicate_candidate_pairs.json`
  `3ce05cf2cdb91d77d6a5b153e4ee4d1f6be1fbd29de1788e2a1bac3964c9f93b` /
  `aa565447b7aa5c9ac19f997eaa314d9ebeaa5f562a12ebb35f0baec02332a769`;
  `perceptual_duplicate_domain_summary.json`
  `38075e7c9203750274bc85394a4488062eb50a0ba101eaeb4718756162865082` /
  `cf673de4d409c070bcf814bcf6ba834378d044997ce2b18bb403a17886e9b27d`;
  `perceptual_duplicate_stop_status.json`
  `91dfc220ce82949990a2f18896b6e6d53bf6d0f3db8a8b2dc0dea49bf4a52deb` /
  `b6cce48c5156a97bef504b60d670758ef13c42b2edf966c7f9ba667583492947`.
- **Pair adjudication has NOT started.** `training/pair_adjudication_
  ledger.jsonl` does not exist. **The frozen finalizer
  (`finalize_stop_status.py`) currently requires every one of the 16,539
  candidates to have an adjudication record before `--finalize` can
  succeed.** Adjudicating 16,539 pairs is a substantial, separate reviewer
  workload -- whether/how to reduce, batch, or otherwise scope that work
  (and the finalizer's all-or-nothing completeness rule itself) requires
  its own separately reviewed decision before adjudication begins. No
  image beyond the ones already hashed in this scan has been or will be
  opened without that decision.

## Phase 5F4A: perceptual-pair adjudication v2 protocol prepared, NOT run

**The "separately reviewed decision" flagged at the end of the Phase 5F3
section above has been made: a separate, additive, versioned v2
adjudication protocol (`docs/plans/perceptual-adjudication-v2.md`) scopes
the 16,539-candidate workload down to a 3,972-row mandatory queue, split
into three independently-tracked workstreams.** This revises finalization
SCOPE only -- it does not alter the Phase 5F1 scan, the frozen pHash<=10/
dHash<=8 candidate rule, the frozen 0.61 Gate v2 threshold, or the four
adjudication labels' own meanings. No candidate image has been opened; no
real adjudication has occurred.

**Correction pass applied (same phase, before any evidence generation):**
the initial preparation recorded implementation-source provenance against
commit `44061a4` even though none of the six new source paths existed at
that commit -- invalid provenance. The two real generated artifacts from
that initial pass (`pair_adjudication_v2_queue.json`,
`pair_adjudication_v2_contract.json`) were deleted (never committed;
byte sha256 `0dcac037b02deebfc72811a0589582455d1e73ac535fac1133f1db5ebbd31a5a`
and `a5d24185a8e255653f8f7343b85410b97822723682a708cf40be00bc0c2428c1`
respectively, recorded here only for the record) and are **not**
regenerated in this phase. Generation now requires a real two-commit
lifecycle (source-preparation commit, then a separate evidence-generation
phase) -- see below and the plan doc.

- Three workstreams, mechanically re-derived and verified against the real
  committed `perceptual_duplicate_candidate_pairs.json` (never merely
  copied): `final_test_independence` (blocking, 1,681 candidates across the
  7 frozen `final_test`-vs-everything domains), `final_test_internal_
  repetition` (diagnostic, 53 candidates, `within_final_test` only --
  confirmed same-source pairs here are reported as an effective-sample-
  size limitation, NOT leakage, and never alone trigger a stop), and
  `gate_evidence_independence` (blocking, 2,238 candidates across 5 domains
  binding `calibration_v2`/`expansion_development`/`expansion_train`
  against each other and `unknown_test_v2`). Scoped total **3,972** +
  outside-scope **12,567** = the full frozen scan's **16,539** -- verified,
  not assumed. Every outside-scope candidate is reported as
  `not_adjudicated`, never `not_duplicate`, and the excluded population is
  now itself hash-bound (`outside_scope_identity_order_sha256`, verified
  fresh against the real candidate report at every runtime load).
- The two blocking channels stop **independently** the moment their own
  first `same_source_image` or `uncertain` decision lands; the diagnostic
  channel never stops early. `uncertain` makes its channel inconclusive and
  blocking -- never cleared or passed.
- **Priority order is now literal** (corrected from the initial pass's
  within-domain interpretation): ALL 18 both-pHash-and-dHash-rule
  candidates queue first, globally (queue_index 0-17), across the 5
  domains that contain them, each forming its own small single-domain
  priority block; then every remaining candidate queues exactly as if
  those 18 rows never existed, in the originally specified domain
  sequence. Split into **25 session blocks** (5 new small priority blocks
  + the 20 standard blocks) -- one domain AND one priority tier per
  session, <=300 rows, numbered blocks for large domains. Real,
  mechanically-verified `identity_order_sha256 =
  7e906cc2a66017940a8aef586e0eb9be7743c36f1dddec4ec22d46341562d14f`
  (preflight-only; not yet bound into a generated queue artifact).
- **Source provenance is now a real two-commit lifecycle.** Every one of
  the six approved implementation sources must be bound to a commit that
  actually, verifiably contains that exact path/content --
  `freeze_pair_adjudication_v2_contract.py --write` refuses unless the
  tracked tree is clean and every source is tracked at HEAD;
  `verify_implementation_source_provenance` (in
  `pair_adjudication_v2_contract.py`) checks the recorded commit resolves
  and is an ancestor of current HEAD, the source's content at that commit
  hashes to the recorded value, AND the current working tree still
  matches (authoritative, not diagnostic -- unlike Gate v2's provenance,
  a bound-source edit fails `--check`, not merely marks it stale). `--check`
  never rebuilds against a fresh current HEAD (an unrelated later commit
  must not invalidate already-generated evidence); it reuses the
  published contract's own recorded sources verbatim and re-verifies them
  separately. `adjudicate_pairs_v2.load_verified_v2_state` runs this same
  check before trusting any contract; every stored ledger record binds
  generator provenance to the CONTRACT's own recorded `adjudicate_pairs_v2`
  entry, never an independently computed current-HEAD/working-tree value.
- **`session_id` is now mechanically bound**, never an arbitrary caller
  value: it must exactly equal the current eligible queue row's own
  `suggested_session`, enforced in `record_adjudication` itself (not only
  the UI) and re-checked for every stored ledger record; the review UI
  refuses to even launch/construct for a mismatched session.
- **Runtime state loading now rederives the queue from the verified
  candidate report** (not just self-consistency of the queue file against
  the contract's recorded hashes) -- a jointly-tampered queue+contract
  pair that recomputes every internal hash consistently is still rejected
  because the rederived rows/outside-scope population no longer match.
- **Contract generation independently rederives the exact canonical queue
  too.** `freeze_pair_adjudication_v2_contract.py` no longer trusts only the
  queue artifact's row count/identity hash: it rebuilds the expected queue
  from the verified candidate report and requires both exact parsed content
  and exact canonical bytes before a contract can be published. Its
  `generation` envelope is also schema-checked with exact keys and frozen
  values during `--check`, so modifying generation metadata while preserving
  `content_sha256` cannot evade validation.
- **The review UI cannot cross a frozen session boundary.** After every
  decision (including a blocking channel's early-stop decision), the session
  re-reads the globally eligible row but exposes/opens it only when its
  `suggested_session` still equals the UI's current session. A next-session
  row is reported for handoff and is never opened implicitly.
- New requirement versus v1: `adjudicate_pairs_v2.py --record` now acquires
  a structurally exclusive per-write lock (`training/pair_adjudication_v2.
  lock`, atomic `O_CREAT|O_EXCL`) before touching the ledger -- a
  pre-existing/stale lock is NEVER silently removed, only reported for
  review; the append is reverified-then-fsynced-then-reverified inside the
  lock, and only the lock this invocation created is ever released;
  `--record` also requires a clean tracked tree before touching the ledger.
- Files (all new, none of the frozen Phase 5F1 sources modified):
  `pair_adjudication_v2_contract.py`, `build_pair_adjudication_v2_queue.py`,
  `freeze_pair_adjudication_v2_contract.py`, `adjudicate_pairs_v2.py`,
  `pair_adjudication_v2_review.py` (all PIL/Tkinter imports deferred to
  `launch_review_ui()`, never invoked here), `finalize_stop_status_v2.py`,
  and six test modules (98 tests, all passing, offline/synthetic fixtures
  including real throwaway-git-repo provenance fixtures). Neither
  `pair_adjudication_v2_queue.json` nor `pair_adjudication_v2_contract.json`
  exists -- generation is deferred to the two-commit lifecycle above.
- All 317 pre-existing Phase 5F1/5F2/5F3 tests still pass unchanged; the
  five committed scan reports and both existing ledgers are byte-identical
  to their committed state before and after this work.
- A real, metadata-only `adjudicate_pairs_v2.py --next` invocation was run
  once during the initial preparation pass, before this correction pass
  identified the provenance defect -- it was read-only against the (since
  deleted) real queue/contract: no image was opened, no ledger was
  written, and no adjudication was consumed. It is recorded here as a
  procedural deviation (the phase's own instructions say "do not run
  --next") and is **not** repeated in this or any later preparation pass.
- **No real pair has been adjudicated. `pair_adjudication_v2_queue.json`,
  `pair_adjudication_v2_contract.json`, `pair_adjudication_v2_ledger.jsonl`,
  `pair_adjudication_v2.lock`, and
  `perceptual_duplicate_post_adjudication_stop_status_v2.json` do not
  exist.** A real source-preparation commit, then a separate evidence-
  generation phase, then running `--next`, opening the review UI,
  adjudicating any pair, and finalizing all require their own separately
  authorized phases.

- Final verification after these corrections: all **98** Phase 5F4A tests
  and all **317** pre-existing Phase 5F tests pass (**415 total, 1 expected
  skip**); all six new implementation modules compile cleanly; the real
  metadata-only queue preflight still derives 3,972 rows, 18 globally-first
  both-rule rows, 25 session blocks, and identity-order sha256
  `7e906cc2a66017940a8aef586e0eb9be7743c36f1dddec4ec22d46341562d14f`.

## Phase 5F4B: v2 adjudication queue and contract generated, NOT adjudicated

The required two-commit lifecycle is now in progress. The six implementation
sources, tests, plan, and preparation notes were committed first as
`7e847379bfadc378f1fa1b57c6ebd3f7390e81c1`. Against that clean committed
state, the queue and contract were each generated exactly once, then checked
twice byte-for-byte. No `--next`, reviewer UI, `--record`, or finalizer path
was run.

- `training/pair_adjudication_v2_queue.json`: 2,500,250 bytes; byte sha256
  `28f9d3f984d14d88905cee5357e39802d384d87586fb10cab4e078e9057ca70d`;
  3,972 rows; identity-order sha256
  `7e906cc2a66017940a8aef586e0eb9be7743c36f1dddec4ec22d46341562d14f`;
  18 both-rule rows globally first; 25 session blocks.
- `training/pair_adjudication_v2_contract.json`: 9,579 bytes; byte sha256
  `27d9f66949a0ec2b9cb22a8993646a3c8a301955d7e3b664d1281ec44c230fb3`;
  content sha256
  `852275c4a6d3107ece2fef9a2f97b3834d0ca21078c8cfb164ef015c21b415f0`;
  all six implementation-source entries bind commit `7e847379...`.
- Scope remains exactly 3,972 mandatory + 12,567 outside scope = 16,539;
  outside-scope identity-order sha256 is
  `d43fb8839beb641df258385d1b83c291e68d68bf1732dc5bc701db93470d92db`.
- Both real metadata-only runtime preflights pass: adjudicated 0, remaining
  3,972, every workstream pending, `ready_to_finalize: false`.
- `pair_adjudication_v2_ledger.jsonl`, `pair_adjudication_v2.lock`, and
  `perceptual_duplicate_post_adjudication_stop_status_v2.json` remain absent.
  No real image has been opened.

## Phase 5F4C: reviewer handoff after 309 completed decisions

Reviewer `huso1` completed queue indexes 0--308 (309 decisions): the 18
globally-first both-rule candidates, all 238
`calibration_v2_vs_unknown_test_v2` candidates, and all 53
`within_final_test` candidates. Results at handoff: 304 `different_image`,
5 `different_photo_same_observation`, 0 `same_source_image`, 0 `uncertain`;
all workstreams remain unstopped. The ledger is valid and no lock remains.

The user then delegated the remaining visual comparison work to Codex. Before
opening queue index 309, the method is fixed as follows: preserve all existing
`huso1` records unchanged; write future records with reviewer id
`codex-visual-v1`; visually inspect both images for every decision; never use
pHash/dHash or metadata alone as the label; record `uncertain` rather than
guessing; retain all existing early-stop rules; and stratify the final report
by reviewer id. This uses the contract's existing required `reviewer_id`
provenance field and does not alter the frozen queue, contract, labels, scope,
or finalization semantics.

## Phase 5F4 v3: stop and final-test handoff

Pair review was permanently stopped and the 762-decision ledger frozen in
commit `c488b8ff39b7e30210d7b42a6b0a1890b63ee7e3`. There are 3,210—not
3,307—remaining in-scope candidates, permanently `not_adjudicated`. The v2
contract has **not** passed. The v3 decision records the measured rationale
and the incomplete perceptual-independence label. Before opening any final-test
image for inference: commit the v3 decision, then separately commit a one-shot
evaluator and reporting rule binding this label and all 450 frozen rows. Run
the final test only once; preserve the unusable/poor-quality rows and disclose
their manual-review counts alongside the result. No model or gate tuning from
the final-test result.

For the next catalog expansion, measure B4 throughput under
`cudnn_deterministic=False` and `cudnn_benchmark=True` before assuming another
~60-hour run is necessary. The earlier B4 epoch (~6,876 s) versus V2-S
(~128 s) was about 54× slower under the shared deterministic policy; the
full-run ratio was closer to 45×. A deterministic-cuDNN interaction with
depthwise convolutions is a plausible primary explanation, **not a proven
causal attribution** without a controlled B4 ablation. Determinism was used
to keep the original comparison fair; it need not be imposed on a new
throughput experiment. B4 remains the better selected model and already fits
the serving contract; do not switch to V2-S solely for speed.

The separate `training/final_test_v1_reporting_rule.json` and
`training/eval_northeast_final_test_v1_once.py` preparation commit freezes the
450-row, 15×30 denominator and raw ONNX-CPU 0.61 gate report. Its exclusive
attempt marker is created before the first image read; a failed attempt after
that point is consumed, not retried. The result must carry
`perceptual_independence_incomplete_by_decision`, never a v2 screening pass.
Preflight is metadata-only. The actual one-shot run and subsequent evidence
commit are separate operations.
