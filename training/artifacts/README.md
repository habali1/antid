# Artifacts

These are the **real trained weights**: a 30-epoch fine-tune of
`tf_efficientnet_b4` over 50 ant species (~9,990 iNaturalist images, ~200 per
species), final training loss 0.056.

## Reproducible baseline: benchmark_v1

    top-1 (raw cosine)   60.8% micro   (macro 62.4% -- see caveat below)
    top-3 (raw cosine)   79.3% micro   (macro 80.4%)
    top-1 (+ geo)        64.2% micro   (macro 65.9%)
    top-3 (+ geo)        82.2% micro   (macro 83.0%)
    (n = 1591, 50 species, 100% with usable coordinates)

**Micro (image-weighted) is the number to cite.** Macro (unweighted mean
across species) is close, but two of the 50 species have exactly 1 benchmark
image each and both happened to land on the correct answer, pulling macro up
by chance — a different pair of photos could just as easily have pulled it
down. Macro here is informational, not a more-precise secondary metric.

This comes from `data/benchmark_v1/` — an entirely fresh, independently-
scraped, frozen evaluation set built specifically because the original
training-time split (below) turned out to be unrecoverable. Full methodology,
exclusion criteria, and a three-way overlap verification (photo_id /
observation_uuid / image sha256, all zero) against the training set are in
`data/benchmark_v1/benchmark_v1.json`. The full per-species breakdown and
artifact hashes for this run are in `data/benchmark_v1/benchmark_v1_eval.json`.

**This measures accuracy among these 50 species only.** An ant outside them is
still forced into whichever of the 50 scores highest -- there is no "none of
these" outcome, and nothing about model architecture changes that. See
`benchmark_v1.json`'s `scope` field.

Reproduce with:

    cd ..
    python eval_benchmark.py

It **refuses to run** unless all 1,591 rows resolve to exactly one local image
each with a verified sha256 match — never a silent partial evaluation. On a
fresh clone, or if local images are missing/corrupted, restore the exact
frozen set first:

    cd ../data_pipeline
    python scrape_benchmark.py --restore --out ../data/benchmark_v1

`--restore` re-downloads only the specific `observation_uuid`/`photo_id`
records `benchmark_v1.csv` already lists and verifies each against its
recorded hash; it never selects new candidates and never modifies the CSV.
Running `scrape_benchmark.py` *without* `--restore` builds a different, new
benchmark version instead — do not use it to try to reproduce this one.

Two species (`anoplolepis-custodiens`, `polyrhachis-schistacea`) have only 1
benchmark image each — not enough recent iNat observations existed to reach
the target sample size. Their individual per-species figures are not
statistically meaningful; see `low_sample_species` in the eval JSON. They are
kept in the benchmark rather than dropped, so they do contribute to the
overall micro/macro numbers above like any other class.

**Do not use benchmark_v1 as a tuning signal.** It exists to report one number
per finished model, not to pick among candidates during development.
Repeatedly evaluating architecture/hyperparameter variants against it and
keeping the best score would slowly turn this into a validation set the model
is indirectly fit to, defeating the point. Any future architecture work
(e.g. EfficientNetV2) should tune against a newly pinned validation split
(`val_split.json` from that run) and touch benchmark_v1 exactly once, on the
finalized candidate.

## Unknown/out-of-scope rejection — frozen candidate, independently validated

A candidate abstention rule — **abstain when max_sim < 0.60, raw cosine, no
geo** — was selected using `data/calibration_v1/` (1,005 images: known-species
holdout, out-of-scope ants, non-ant insects, unrelated photos) and then graded
**exactly once**, unmodified, on `data/unknown_test_v1/` (573 images,
disjoint from training/benchmark_v1/calibration_v1 on photo_id/observation_uuid
/sha256, and additionally **species-disjoint** from calibration_v1's 165
out-of-scope-ant species). Full methodology, verification, and results in
`data/calibration_v1/calibration_v1.json` (`frozen_candidate_abstention_threshold`,
`phase_c_validation`) and `data/unknown_test_v1/unknown_test_v1.json`.

**Result: validated as a selective confidence/quality gate, not as an
unknown-species detector.** It measurably improves known-species accuracy
among what it accepts (69.8% → 85.5%) and rejects incorrect predictions ~4x
more often than correct ones — but 91/200 out-of-scope ant photographs (45.5%)
in the independent test still pass the gate and receive one of the 50
supported species as their closest match. It is now implemented as an optional,
hash-bound serving policy and surfaced by the mobile client. **This remains a
confidence gate, not an unknown-species detector**, and 0.60 must not be
re-tuned against unknown_test_v1's results.

## Parity evidence and limitations

Three scripts investigated whether the serving path (ONNX, CPU) produces the
same results as the training/evaluation path (PyTorch, CUDA, batch 32) used
to select and validate the 0.60 abstention threshold above:
`parity_check.py`, `parity_diagnostic.py`, `parity_flag_ablation.py`, writing
`parity_report.json`, `parity_diagnostic.json`, `parity_flag_ablation.json`
respectively. **All six files are frozen and were not rerun for this
release.** Their exact sha256 hashes, recorded here so any future diff is
detectable:

    training/parity_check.py                    2ae11808a6c557a3a0cd283799bd4b5cbaf60e379e4bdf626bda93b0a080af0a
    training/parity_diagnostic.py                b4e3501bedf2e0a907bbf79d6bb5775ffc61ace92430098c19304b2753cdd335
    training/parity_flag_ablation.py             f719fe2c53482ddd3cb09fce773f18b2907375bb1e7ef5a6cc4ee7d18bc640ce
    training/artifacts/parity_report.json        dfe6030f1b8b7399923064a182dc0212963526322225465ca0f4c018892d245b
    training/artifacts/parity_diagnostic.json    59c9818b852a1dc53a5876b9aeb2634eea6e4d1cd92c82783d3f2a955dc30335
    training/artifacts/parity_flag_ablation.json 9be162144afadaf00770330ca4f2e26b45f605b95a06ecd1bfef1dfeb143acf6

Each report names its associated script, and the file timestamps are
consistent with that association, but no report embeds a hash of the script
bytes. **Treat this as contemporaneous preserved provenance, not
cryptographic proof that these exact script bytes produced the reports.**

**These scripts must not be reused to produce new evidence** until the
deferred audit items in the repository root's `TODO.md` are fixed. A future
parity run must write versioned `v2` reports (e.g. `parity_report_v2.json`)
rather than overwrite these frozen `v1` files.

Known limitations of the frozen evidence, carried forward rather than
resolved:

- `parity_flag_ablation.json`'s `part_b_cross_check_vs_parity_report
  .max_diff_reference_path` is **0.0401538908**, even though that report's own
  note expected an exact reproduction of `parity_report.json`'s reference
  path. `parity_flag_ablation.json`'s top-level `warnings` list is **empty**
  — that emptiness must not be read as resolving this discrepancy; it simply
  means the script's own warning checks didn't happen to flag it.
- Because of the above, `calibration_mirror_cuda_batch32` (the PyTorch-CUDA
  path used throughout `parity_flag_ablation.json`) is a **diagnostic
  reproduction** of the historical calibration/Phase C evaluator, not an
  exact byte-for-byte replay of its entire scoring execution.
- `exploratory_spearman_norm_vs_divergence.spearman_rho` ≈ **-0.107** is weak
  exploratory evidence (explicitly labeled `EXPLORATORY ONLY` in the report)
  and supports no causal or population claim about embedding norm predicting
  divergence.
- The 200-image stratified sample (4 images/species × 50 species) used
  throughout is diagnostic evidence about this fixed sample, not a
  population bound over unseen images.
- The measured, production-relevant result that stands despite the caveats
  above: **0/200 gate disagreements** between the `calibration_mirror_cuda_batch32`
  path and the production ONNX-CPU path (`part_b_pairwise_distributions
  .calibration_mirror_cuda_batch32__vs__production_onnx_cpu_batch1
  .gate_disagreement_count`).

## Historical number — do not treat as reproducible

At the end of the original training run, this model measured:

    top-1  66.6%      top-3  81.9%      (n = 1996, training-time val split)

`eval.json` still holds this result. **It cannot be regenerated and should not
be quoted as current model performance** — see the reproducibility warning
below for why. It's kept here only as a historical record of what that run
reported at the time, alongside `benchmark_v1` above as the number that
actually replaces it.

## Files

    model.pth              training checkpoint {model: state_dict, config: cfg}
    backbone.onnx           EfficientNet-B4: (batch,3,380,380) -> (batch,1792), opset 17
    prototypes.npy          (num_classes, 1792) unit-norm mean train embeddings
    taxonomy.json           class_idx -> {species_name, common_name, taxon_id, slug}
    geo_index.json          {cell_size_deg, cells: {slug: [[lat_cell, lon_cell], ...]}}
    inference_policy.json   optional hash-bound confidence-gate policy
    eval.json               historical top-1/top-3, overall and per species (see above)
    val_split.json          the pinned held-out set ({slug}/{photo_id} keys) -- did not
                            exist for this run; present for future retrains

**Serving has three mandatory model artifacts: `backbone.onnx`,
`prototypes.npy`, and `taxonomy.json`.** `model.pth` is a *training* checkpoint
— the full model including the classifier head that gets discarded at export
time; the API (`api/inference.py`) never loads it and doesn't need it present.
`geo_index.json` is optional and enables geo re-ranking; the API reports
`geo_index_loaded: false` plus a reason when it is unavailable or unusable.
`inference_policy.json` is also optional. It activates the selective confidence
gate only when its schema, hashes of all three mandatory artifacts,
preprocessing contract, and CPU-only provider policy verify; it deliberately
does not bind the independently refreshable geo index. If the policy is absent
or invalid, closest-match inference remains available with
`gate_active: false` and `low_confidence: null`, while `/health` reports
`inference_policy_loaded: false` and `inference_policy_reason`.

Row order in `prototypes.npy` **is** the class index, and must match
`taxonomy.json` — `AntIdentifier.__init__` raises on a length mismatch, but a
same-length reordering would fail silently, so never regenerate one without the
other.

## Reproducibility warning (why benchmark_v1 exists)

`val_split.json` pins the held-out set going forward, for retrains. This run
predates it, and for this run the reported accuracy **cannot be regenerated**:
the val split was implied by the manifest's `split` column and row order, and
`data/manifest_all.csv` was regenerated a minute after training finished,
rewriting both. Once that happens every reconstructable split silently mixes
training images back in.

That is not hypothetical — it happened to this run. Re-evaluating these
weights against any split derivable from the current manifest scores about
**93%** top-1. The gap is memorization: the model scores ~99.6% on images it
trained on, so any 80/20 slice of a dataset it has fully seen averages out near
93%. **That 93% figure must never be reported as model performance** — it is
the model grading its own homework, not a measurement of generalization.
`benchmark_v1` exists specifically to give this model a number that isn't
either of these two: not the unrecoverable 66.6% and not the contaminated 93%.

If you retrain, `val_split.json` will be written automatically and

    python evaluate.py --geo          # reads val_split.json automatically

becomes reproducible again for that run. `benchmark_v1` still applies
independently of retraining — it's a fixed test set, not tied to any one
checkpoint.

## Regenerating

    cd ..
    python train.py --config config.yaml     # on a CUDA GPU; overwrites everything here
    python export.py                         # re-export ONNX from model.pth alone
    python evaluate.py --geo                 # recompute eval.json (+ geo re-ranking) against val_split.json
    python eval_benchmark.py                 # recompute against the frozen, unseen benchmark_v1

`geo_index.json` is written automatically when the image manifest carries
observation coordinates (iNaturalist provides them). Note that `train.py` builds
it from **all** samples, train and val — for a leak-free geo measurement on
val_split.json use `python evaluate.py --geo --geo-source train`, which
rebuilds the index from the training split only. `eval_benchmark.py` doesn't
need this care: `benchmark_v1` is entirely disjoint from the training set by
construction, so there's no leak-free variant to choose between.
