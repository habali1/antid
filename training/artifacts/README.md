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
    eval.json               historical top-1/top-3, overall and per species (see above)
    val_split.json          the pinned held-out set ({slug}/{photo_id} keys) -- did not
                            exist for this run; present for future retrains

**Serving needs exactly three of these: `backbone.onnx`, `prototypes.npy`, and
`taxonomy.json`.** `model.pth` is a *training* checkpoint — the full model
including the classifier head that gets discarded at export time; the API
(`api/inference.py`) never loads it and doesn't need it present.
`geo_index.json` is optional and enables geo re-ranking; the API reports
`geo_index_loaded: false` without it.

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
