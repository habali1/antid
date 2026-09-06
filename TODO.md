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
