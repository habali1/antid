# AntID

**Compare an ant photo with supported species.** Point a phone at an ant, get
the three closest visual matches — optionally re-ranked by where you're
standing — plus a separately validated low-confidence flag when its hash-bound
serving policy is active.

AntID is an end-to-end system, not just a model: it scrapes and cleans its own
training set from public biodiversity data, fine-tunes an EfficientNet-B4
backbone, exports a portable ONNX artifact, serves it behind a FastAPI endpoint,
and ships a React Native client for iOS, Android, and web.

<sub>Python · PyTorch · timm · ONNX Runtime · FastAPI · PostgreSQL · React Native · TypeScript</sub>

---

## Results

50 species, evaluated on **benchmark_v1** — 1,591 images that are not just
held out, but drawn from an entirely separate scrape, verified to share zero
`photo_id`, zero `observation_uuid`, and zero image hash with anything the
model trained on (methodology and verification in
`data/benchmark_v1/benchmark_v1.json`). **Micro accuracy (image-weighted) is
the headline number** — it's what "point the app at a random ant" actually
measures:

| Inference setup | Top-1 (micro) | Top-3 (micro) |
|---|---:|---:|
| Cosine similarity over class prototypes | **60.8%** | **79.3%** |
| + geographic re-ranking (100% of images had coordinates) | **64.2%** | **82.2%** |

The unweighted per-species (macro) average is close — 62.4%/65.9% top-1,
80.4%/83.0% top-3 — but **is not a reliable secondary number here**: two of the
50 species have exactly 1 benchmark image each (see below), and both happened
to score 100% on that single trial, pulling macro up by chance. A different
draw of those two photos could just as easily have pulled it down. Treat macro
as informational, not as evidence the model does better per-species than the
micro number suggests.

Random chance on 50 classes is 2%. Reproduce with `python eval_benchmark.py`
from `training/` — it **refuses to run** unless all 1,591 rows resolve to
exactly one local image apiece with a verified sha256 match against
`benchmark_v1.csv`; it never silently evaluates a subset. If images are
missing or don't match (e.g. on a fresh clone), restore the exact frozen set
first — see **Restoring the benchmark locally**, below. Full per-species
breakdown, the coordinate-bearing-subset figures, and sha256 hashes of both
the benchmark manifest and the evaluated artifacts are written to
`data/benchmark_v1/benchmark_v1_eval.json` each run.

> **Why a second, independently-scraped benchmark exists at all.** The
> original training run measured 66.6% top-1 / 81.9% top-3 on its own
> held-out split — a real result at the time, but the split was only implied
> by the training manifest's `split` column and row order, and the manifest
> was regenerated a minute after training finished, silently rewriting both.
> That run's number is **not reproducible** and is kept only as a historical
> note (`training/artifacts/eval.json`, `training/artifacts/README.md`).
> Critically: re-evaluating those same weights against any split
> reconstructable from today's manifest reports **~93% top-1 — do not read
> that as model performance.** The model scores ~99.6% on images it trained
> on, so any 80/20 slice of a dataset it has fully seen averages out near
> 93%; it's the model grading its own homework, not a measurement of
> generalization. `train.py` now writes `artifacts/val_split.json` to pin the
> split for future retrains, but for *this* checkpoint the only trustworthy,
> reproducible number is benchmark_v1 above.

**Several species score meaningfully lower on this fresh benchmark than the
old (unreproducible) split reported** — e.g. *Camponotus pennsylvanicus*
53.5% → 31.4%, *Linepithema humile* 50.0% → 31.4%. That gap is itself
informative: the original split was drawn from the same scrape, observers, and
time window as training, which flatters generalization even when it isn't
contaminated in the strict sense benchmark_v1 rules out. Per-species spread
stays wide either way — weak classes here (*C. pennsylvanicus*, *L. humile*,
*Atta mexicana*, all ≤35% top-1) are small-to-medium, near-uniform ants whose
diagnostic characters (petiole shape, propodeal spines, antennal segment
counts) are hard to resolve in a handheld phone photo; strong classes
(*Cephalotes atratus*, *Ectatomma tuberculatum*, both ≥85%) are large,
high-contrast, or distinctively sculptured. That pattern is *consistent with*
a data-resolution ceiling, but this benchmark can't actually prove that's the
whole story — the weak classes could equally reflect distribution shift
between the two scrapes (different photographers, seasons, camera phones),
genuine model limitations, or label noise in the source observations. Training
loss did converge cleanly to 0.056, which at least rules out "still
underfitting" as the explanation. (Two species, *Anoplolepis custodiens* and
*Polyrhachis schistacea*, have only 1 benchmark image each — too few recent
iNat observations existed post-cutoff to reach the target sample size — so
their individual figures aren't statistically meaningful; they're still
counted in the overall numbers above, and were deliberately kept rather than
dropped or backfilled by loosening the freeze criteria.)

**This measures accuracy among the 50 trained species only.** An ant that
isn't one of them still gets forced into whichever of the 50 scores highest —
the system has no "none of these" outcome. The shipped confidence gate can
abstain on weaker matches, but it is not an unknown-species detector: 45.5% of
out-of-scope ant photographs still passed it in the independent test. Broader
coverage or a genuinely open-set approach is still needed. See
`data/benchmark_v1/benchmark_v1.json`'s `scope` field for the same caveat
machine-readable.

**benchmark_v1 is a report card, not a tuning signal.** Future architecture
work (an EfficientNetV2 swap, say) should iterate against a freshly pinned
`val_split.json` from that run, then touch benchmark_v1 exactly once, on the
finalized candidate. Repeatedly evaluating candidates against it and keeping
the best score would slowly turn this benchmark into something the model is
indirectly fit to, the same problem it exists to avoid.

Training dataset: 9,989 images across 50 species (~200 each), 7,990 train /
1,999 val, sourced from the public iNaturalist open-data bucket.

---

## The core design decision: embeddings, not softmax

The model trains as a classifier but **does not serve as one.**

`training/model.py` puts a `Linear` head on the backbone and trains with
cross-entropy. At export time that head is *thrown away*. What ships is the
1792-dim embedding trunk, plus one **prototype** vector per species — the
L2-normalized mean training embedding for that class. Inference is a cosine
similarity between the query embedding and the prototype matrix, ranked.

Why this is worth the extra machinery:

- **Adding a species costs one matrix row, not a retraining run.** Embed that
  species' images, average, normalize, append to `prototypes.npy`, add a
  taxonomy entry. No head to resize, no catastrophic forgetting, no redeploy of
  weights. For a domain with ~14,000 described ant species and a long tail that
  grows every year, a fixed-width softmax head is the wrong shape.
- **Scores are comparable and interpretable.** A cosine similarity means "how
  close to the class centroid," which degrades gracefully on out-of-distribution
  input. A softmax over 50 classes will confidently report a species for a photo
  of a beetle.
- **It makes geo re-ranking a clean additive operation** on a bounded score,
  rather than a hack on top of normalized probabilities.

---

## Architecture

```
   iNaturalist open-data S3  (public, --no-sign-request)
              │
              │  scrape_inat.py ── clean.py ── upload.py
              ▼
   ┌──────────────────────┐        ┌──────────────────────┐
   │  GCS / S3 bucket     │◄──────►│  PostgreSQL          │
   │  cleaned images      │        │  species · images    │
   └──────────┬───────────┘        └──────────┬───────────┘
              │                               │
              └───────────┬───────────────────┘
                          ▼
              train.py  (EfficientNet-B4 @ 380px, timm, ImageNet init)
                          │
                          │  export.py → ONNX opset 17, dynamic batch
                          ▼
   ┌──────────────────────────────────────────────────────┐
   │  backbone.onnx    (batch,3,380,380) → (batch,1792)    │
   │  prototypes.npy   (num_classes, 1792) unit-norm       │
   │  taxonomy.json    class_idx → species metadata        │
   │  geo_index.json   slug → occupied 1° grid cells       │  ← optional
   │  inference_policy.json  hash-bound confidence gate     │  ← optional
   └───────────────────────────┬──────────────────────────┘
                               ▼
              FastAPI  ·  POST /identify?lat=&lon=  ·  ONNX Runtime
                               │
                               ▼
              React Native app  (iOS · Android · react-native-web)
```

The four stages are deliberately decoupled — each runs on a different machine
and they share **no application code**. They communicate only through
PostgreSQL, a cloud bucket, and the artifact files above.

### Contract 1 — the artifact interface

Training emits three mandatory model artifacts and two independently optional
sidecars. Together they are the file-level API between the two halves:

| File | Shape / schema | Notes |
|---|---|---|
| `backbone.onnx` | `input (batch,3,380,380) f32` → `embedding (batch,1792)` | opset 17, dynamic batch |
| `prototypes.npy` | `(num_classes, 1792)`, L2-normalized | **row order == class index** |
| `taxonomy.json` | `{class_idx: {species_name, common_name, taxon_id, slug}}` | length must equal prototype rows |
| `geo_index.json` | `{cell_size_deg, cells: {slug: [[lat,lon], …]}}` | optional |
| `inference_policy.json` | schema-versioned policy + sha256 bindings | optional; enables the confidence gate only when valid |

The API cannot start without the first three files. A missing or invalid
sidecar disables only its own feature. `inference_policy.json` hash-binds the
three mandatory artifacts (not the independently refreshable geo index),
requires CPU-only ONNX execution, and evaluates the raw, unrounded, pre-geo
maximum cosine. If it cannot be activated, `/identify` returns
`gate_active: false` and `low_confidence: null`; `/health` reports the policy's
functional loaded flag and reason.

Class indices are contiguous `0..N-1` **sorted by species slug**, enforced
identically in the database loader, the local-directory loader, and
`taxonomy.json`. That ordering is the reproducibility guarantee keeping
prototype rows aligned with taxonomy keys — and `AntIdentifier.__init__` raises
on a length mismatch rather than serving silently misaligned predictions.

### Contract 2 — preprocessing parity

Training's validation transform and the API's `preprocess` must produce a
byte-identical tensor: resize to 380×380, scale to `[0,1]`, normalize with
ImageNet mean/std. These constants live in three places (`config.yaml`,
`training/data.py`, `api/inference.py`) and drift between them degrades accuracy
*silently* — no crash, no error, just quietly worse predictions.

This is exactly the failure mode that's expensive to catch in production, so
`training/evaluate.py` deliberately **re-implements the serving cosine-ranking
path from scratch** rather than importing it. If the two implementations
diverge, evaluation metrics move and the drift surfaces immediately.

---

## Geographic re-ranking

Ant species have real ranges, and a photo usually comes with GPS. When
`geo_index.json` is present *and* a request supplies both `lat` and `lon`,
`identify` adds `GEO_BOOST` (env var, default `0.05`) to the cosine score of any
species recorded in the user's 1° grid cell or its 8 neighbours, then re-ranks.

The details that keep this honest:

- The reported `similarity` is always the **raw** cosine — the boost changes
  ordering, never the number shown to the user.
- `geo_boosted` is set only when a result's rank *strictly improved* because of
  the boost, so the app can explain *why* something moved up.
- `geo_filtered` reports whether location was applied at all.
- Location is **always optional**. Every screen in the app works with
  `lat: null, lon: null`, and if only one coordinate arrives the API discards
  both rather than guessing.

Nothing is ever filtered *out* by geography — an off-range species can still win
on image evidence alone, which matters for the introduced and
human-transported species that make up much of applied ant identification.

---

## Selective confidence gate

The optional `inference_policy.json` enables a frozen abstention rule:
`low_confidence` when the raw, unrounded, pre-geo maximum cosine is strictly
below 0.60. It was selected on `calibration_v1` and evaluated exactly once on
the species-disjoint `unknown_test_v1`. Among known-species photographs,
accuracy rose from 69.8% overall to 85.5% among accepted results. But 45.5% of
out-of-scope ant photographs still passed, so this is a **confidence gate, not
an unknown-species detector**.

The policy is bound to the exact ONNX backbone, prototypes, and taxonomy and is
validated only for CPU execution. Missing, malformed, stale, or mismatched
policy state disables the gate without disabling closest-match inference;
clients receive `gate_active: false` and `low_confidence: null`, while
`/health` exposes the reason.

---

## Layout

| Path | What |
|---|---|
| `data_pipeline/` | scrapers (iNat primary; AntWeb + GBIF gap-fill), `scrape_benchmark.py` (frozen benchmark builder), cleaning, upload, SQL schema, geo-index builder |
| `training/` | `train.py`, `export.py` (→ONNX), `evaluate.py`, `eval_benchmark.py`, `model.py`, `data.py`, configs |
| `api/` | FastAPI server — `main.py` (routes), `inference.py` (`AntIdentifier`) |
| `mobile/` | React Native + react-native-web client |

---

## Quickstart

### 0 · Setup
```bash
cp .env.example .env                      # DATABASE_URL + STORAGE_BUCKET
psql "$DATABASE_URL" -f data_pipeline/db_schema.sql
```

### 1 · Data pipeline
```bash
cd data_pipeline
python scrape_inat.py --dry-run                                  # prints targets, downloads nothing
python scrape_inat.py --species-limit 50 --images-per-species 200 --out ../data/raw
python scrape_inat.py --auto-discover --species-limit 50         # top-N Formicidae by observation count
python clean.py  --input ../data/raw --output ../data/clean      # dedupe, drop corrupt/tiny images
python upload.py --src ../data/clean --bucket "$STORAGE_BUCKET"
python build_geo_index.py --metadata-dir ../data/inat_metadata   # optional: geo re-ranking index
```

iNat imagery comes from the public `s3://inaturalist-open-data` bucket with
`--no-sign-request` — no AWS credentials required. The manifests are
TAB-separated despite `.csv` filenames and run to multiple GB, so they're
streamed in chunks rather than loaded.

Database writes are idempotent via `ON CONFLICT` on `species.slug` and
`images.storage_path`, so an interrupted scrape can simply be re-run.

### 2 · Training  *(CUDA GPU; falls back CUDA → MPS → CPU)*
```bash
cd training
pip install -r requirements.txt
python train.py --config config.yaml      # any config key is CLI-overridable
python export.py                          # re-export ONNX from artifacts/model.pth
python evaluate.py                        # top-1/top-3 under the serving cosine path
python evaluate.py --geo                  # + geographic re-ranking, side by side
python eval_benchmark.py                  # the reproducible baseline -- see Results above
```

`evaluate.py` reads `artifacts/val_split.json` to recover the exact held-out set
and warns loudly if it is missing. `--geo` uses the shipped `geo_index.json`,
which `train.py` builds from train **and** val — so each val image's own
coordinate votes for its own answer. For a leak-free measurement:

```bash
python evaluate.py --geo --geo-source train   # rebuild the index from train only
```

`eval_benchmark.py` is different: it doesn't touch `val_split.json` or the
training manifest at all. It scores the current artifacts against
`data/benchmark_v1/` — the frozen, independently-scraped set described in
Results — so its number stays meaningful across retrains without needing any
split bookkeeping.

Two-minute CPU smoke run that exercises the full interface end to end — set
`LOCAL_DATA_DIR` to any `{species_slug}/{image}.jpg` folder first:

```bash
python train.py --config config.smoke.yaml --epochs 1 --batch-size 2 \
  --limit-batches 2 --artifacts-dir ../scratch/smoke
```

> **Pass `--artifacts-dir`.** `config.smoke.yaml` otherwise writes to the same
> `artifacts/` directory as a real run and will overwrite trained weights.

### 3 · API
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000     # reads ../training/artifacts (override: ARTIFACTS_DIR)

curl -X POST "http://localhost:8000/identify" -F "file=@ant.jpg"
curl -X POST "http://localhost:8000/identify?lat=42.27&lon=-83.07" -F "file=@ant.jpg"
```

`GET /health` · `GET /species` · `POST /identify` (multipart `file`, optional
`lat`/`lon`). The three mandatory model artifacts load once at startup;
missing any of them crashes the process deliberately rather than serving a
half-initialized model. Optional geo/policy sidecars fail independently.
`/health` reports `geo_index_loaded`/`geo_index_reason` and
`inference_policy_loaded`/`inference_policy_reason`. `/identify` adds
`gate_active` and nullable `low_confidence`; an inactive policy is represented
as `false`/`null`, never as a reassuring `low_confidence: false`.

### 4 · Mobile
```bash
cd mobile
npm install
cp .env.example .env                      # API_BASE_URL, default http://localhost:8000
npm run web                               # → http://localhost:3000
npm run ios                               # macOS + Xcode (cd ios && pod install first)
npm run android                           # emulator; auto-uses 10.0.2.2:8000
npm run typecheck                         # strict tsc --noEmit
```

Web is a **Metro target**, not a separate build: `metro.config.js` maps bare
`react-native` → `react-native-web` and falls RN-core deep imports back to their
`.ios` pure-JS variants. See `mobile/README.md` for platform notes.

---

## Where each stage runs

| Stage | Requirement |
|---|---|
| Data pipeline | Any machine with network access; no cloud credentials needed for iNat |
| Training | A real CUDA GPU — EfficientNet-B4 at 380px is not a laptop workload |
| API | Anywhere the artifacts are present; ONNX Runtime is explicitly CPU-only for the validated gate |
| iOS build | macOS + Xcode. Web and Android build anywhere with Node 18+ |

**Requirements:** Python 3.11+ · Node 18+ · PostgreSQL · a GCS or S3 bucket.

---

## Testing

There is no single centralized test runner. Correctness checks map directly
onto the failure modes that actually occur here:

| Check | Catches |
|---|---|
| `python train.py --config config.smoke.yaml …` | interface breaks across the whole training→artifact path |
| the ONNX checker inside `export.py` | shape/opset regressions in the exported graph |
| `python evaluate.py --geo` | preprocessing drift, prototype misalignment, geo-ranking drift |
| `python eval_benchmark.py` | the reproducible baseline itself — the only number in Results not tied to a training-manifest split |
| `python training/test_policy_generator.py` | policy evidence, schema, hashing, and generator failure paths |
| `python api/test_inference_policy.py`, `python api/test_inference.py`, and `python api/test_main.py` | fail-safe policy loading, boundary semantics, geo independence, and API/health response behavior |
| `npm run typecheck` | client/API contract drift (strict TypeScript) |

Trained weights and datasets are **not** committed — `training/artifacts/` and
`data/` are gitignored. Run the pipeline and training steps above to regenerate
them. The one exception is `data/benchmark_v1/`'s metadata: `benchmark_v1.csv`,
`benchmark_v1.json`, and `benchmark_v1_eval.json` are force-added despite the
`data/` ignore rule, because the benchmark's identity — which exact photos,
what was verified, what the model scored — needs to travel with the repo even
though the ~1,591 downloaded images themselves don't. The generated
`inference_policy.json` and frozen parity reports are also force-added as
provenance artifacts; trained weights remain external.

### Restoring the benchmark locally

On a fresh clone (or after deleting `data/benchmark_v1/*/`), the frozen
manifest is present but the images aren't. Fetch exactly the images it
describes — same `observation_uuid`/`photo_id` records, verified against the
sha256 already recorded in the CSV:

```bash
cd data_pipeline
python scrape_benchmark.py --restore --out ../data/benchmark_v1
```

This is read-only with respect to `benchmark_v1.csv`: it never selects new
candidates and never rewrites the file, only re-populates the image files it
already lists. `eval_benchmark.py` (see Results) enforces this by refusing to
evaluate unless every row resolves to exactly one hash-verified image, so a
partial or stale local copy fails loudly instead of silently scoring a
different, smaller benchmark.

Running `scrape_benchmark.py` **without** `--restore` does something different
on purpose — it selects a *new* set of candidate observations (for building
`benchmark_v2` or similar) and will not reproduce `benchmark_v1`.

---

## Limitations

- **50 species out of ~14,000 described, and no reliable unknown-species
  detector.** Coverage is
  skewed toward well-photographed North American, European, and Australian
  taxa. The shipped confidence gate can withhold a reliable-match claim, but
  it does not produce a species-independent "none of these" classification.
  It improves accuracy among accepted known-species matches, yet still allows
  roughly half of out-of-scope ant photographs to pass and receive a supported-
  species closest match. A better backbone does not fix this by itself;
  broader coverage or an open-set approach is still required. See
  `training/artifacts/README.md` for the full independent validation.
- **Confusable small dark ants are near the resolution limit of the input.**
  Several genera cannot be separated to species from a field photo at all; a
  genus-level answer would be more honest for those, and grouping the weakest
  classes into genus-level buckets is the clearest next improvement.
- **Prototypes assume one visual mode per species.** Strongly polymorphic
  species (major/minor castes, alates) are poorly served by a single centroid;
  multiple prototypes per class would fit the biology better.
- **Not a substitute for a specialist.** For anything medically or
  agriculturally consequential, confirm with a myrmecologist and a microscope.
