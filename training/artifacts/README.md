# Artifacts

These are the **real trained weights**: a 30-epoch fine-tune of
`tf_efficientnet_b4` over 50 ant species (~9,990 iNaturalist images, ~200 per
species), final training loss 0.056.

## Reproducible baseline: benchmark_v1

    top-1 (raw cosine)   60.8% micro / 62.4% macro
    top-3 (raw cosine)   79.3% micro / 80.4% macro
    top-1 (+ geo)        64.2% micro / 65.9% macro
    top-3 (+ geo)        82.2% micro / 83.0% macro
    (n = 1591, 50 species, 100% with usable coordinates)

This is the number to cite for this model. It comes from
`data/benchmark_v1/` — an entirely fresh, independently-scraped, frozen
evaluation set built specifically because the original training-time split
(below) turned out to be unrecoverable. Full methodology, exclusion criteria,
and a three-way overlap verification (photo_id / observation_uuid / image
sha256, all zero) against the training set are in
`data/benchmark_v1/benchmark_v1.json`. The full per-species breakdown and
artifact hashes for this run are in `data/benchmark_v1/benchmark_v1_eval.json`.
Reproduce with:

    cd ..
    python eval_benchmark.py

Two species (`anoplolepis-custodiens`, `polyrhachis-schistacea`) have only 1
benchmark image each — not enough recent iNat observations existed to reach
the target sample size. Their individual per-species figures are not
statistically meaningful; see `low_sample_species` in the eval JSON. They are
kept in the benchmark rather than dropped, so they do contribute to the
overall micro/macro numbers above like any other class.

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

Only the first four are needed to serve. `geo_index.json` is optional and
enables geo re-ranking; the API reports `geo_index_loaded: false` without it.

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
