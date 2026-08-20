# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

End-to-end ant species identification across four loosely-coupled stages:
`data_pipeline/` (scrape → clean → upload) → `training/` (fine-tune
EfficientNet-B4 → export artifacts) → `api/` (FastAPI inference) → `mobile/`
(React Native app for iOS / Android / Web). Each stage runs on a different
machine and they communicate only through PostgreSQL, a cloud bucket, and a
handful of artifact files — not through shared code (the one exception:
`training/data.py` imports `StorageClient` from `data_pipeline/common.py` via a
`sys.path` insert).

## The two contracts that hold everything together

Almost every non-obvious bug in this codebase comes from breaking one of these.

**1. The artifact interface (training → serving).** Training emits exactly three
required files, and the API consumes exactly those three:
- `backbone.onnx` — input `"input"` `(batch,3,380,380)` float32 → output
  `"embedding"` `(batch,1792)`, opset 17, dynamic batch.
- `prototypes.npy` — `(num_classes, 1792)` L2-normalized mean train embedding
  per class. **Row order == class index.**
- `taxonomy.json` — `{class_idx: {species_name, common_name, taxon_id, slug}}`.
  Length must equal `prototypes.npy` row count or `AntIdentifier.__init__`
  raises.
- `geo_index.json` (optional 4th) — `{cell_size_deg, cells: {slug: [[lat,lon],…]}}`.

Classification is **embedding + cosine similarity, not softmax**. The model
(`training/model.py`) trains a `Linear` classifier head with cross-entropy, but
that head is *discarded* at serve time — only the backbone (→ 1792-dim
embedding) is exported. Adding a species means computing one new prototype row,
not retraining a head.

**2. Preprocessing parity (must be byte-identical on both sides).** The val
transform in `training/data.py` (`build_transforms(train=False)`) and
`AntIdentifier.preprocess` in `api/inference.py` must produce the same tensor:
resize to `(380, 380)`, scale to `[0,1]`, normalize with ImageNet
mean/std (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`). These constants are
duplicated in `config.yaml` (`image_size`, `normalize`), `data.py`, and
`inference.py` (`IMAGE_SIZE`, `MEAN`, `STD`). Change one → change all three, or
accuracy silently collapses. `training/evaluate.py` deliberately re-implements
the cosine-ranking path so training and serving can't diverge unnoticed.

**Class index ordering** is also load-bearing: indices are contiguous `0..N-1`
**sorted by species slug**, in both the DB loader and the directory loader
(`data.py`), and in `taxonomy.json`. This is the reproducibility guarantee that
keeps prototype rows aligned with taxonomy keys.

## Geo re-ranking (optional)

When `geo_index.json` is present *and* a request supplies both `lat` and `lon`,
`AntIdentifier.identify` adds `GEO_BOOST` (env, default `0.05`) to the cosine
score of any species observed in the user's 1° grid cell or its 8 neighbors,
then re-ranks. The reported `similarity` stays the **raw** cosine; `geo_boosted`
is true only when a result's rank *strictly improved* because of the boost.
`geo_filtered` in the response means location was applied at all. If only one of
lat/lon arrives, `main.py` drops both. The index is written two ways: `train.py`
auto-emits it when the image manifest carries coordinates, or
`data_pipeline/build_geo_index.py` builds it standalone from the iNat manifests.

## Commands

There is **no unit-test runner** (no pytest/jest). "Tests" are: the training
smoke run, `tsc --noEmit`, the ONNX checker inside `export.py`, and
`evaluate.py`'s serving-mirror metrics. Verify changes with those.

**Setup (once):**
```bash
cp .env.example .env                                  # DATABASE_URL + STORAGE_BUCKET
psql "$DATABASE_URL" -f data_pipeline/db_schema.sql
```

**Data pipeline** (`cd data_pipeline`) — note actual CLI flags differ from the
README quickstart in places; trust the code:
```bash
python scrape_inat.py --dry-run                       # prints targets, downloads nothing
python scrape_inat.py --species-limit 30 --images-per-species 200 --out ../data/raw
python scrape_inat.py --auto-discover --species-limit 50   # top-N Formicidae from iNat metadata
python clean.py --input ../data/raw --output ../data/clean # NOT --src/--out
python upload.py --src ../data/clean --bucket "$STORAGE_BUCKET"
python build_geo_index.py --metadata-dir ../data/inat_metadata
```
iNat data comes from the public `s3://inaturalist-open-data` bucket
(`--no-sign-request`, no AWS creds). Manifests are TAB-separated despite `.csv`
names and multi-GB, so they're streamed in chunks.

**Training** (`cd training`, `pip install -r requirements.txt`, wants a CUDA GPU
— falls back CUDA → MPS → CPU):
```bash
python train.py --config config.yaml                  # any key overridable: --epochs --batch-size --lr --image-size --artifacts-dir
python export.py                                       # re-export ONNX from artifacts/model.pth
python evaluate.py                                     # top-1/top-3 under cosine inference
```
CPU smoke run that regenerates placeholder artifacts in ~2 min (set
`LOCAL_DATA_DIR` to any `{slug}/{img}.jpg` folder first):
```bash
python train.py --config config.smoke.yaml --epochs 1 --batch-size 2 --limit-batches 2
```
The files in `training/artifacts/` are **synthetic placeholders** from a 1-epoch
run — they verify the interface, not real weights. Real training overwrites them.

**API** (`cd api`, `pip install -r requirements.txt`):
```bash
uvicorn main:app --reload --port 8000                 # expects ../training/artifacts/ (override with ARTIFACTS_DIR)
```
Artifacts load once at startup; missing artifacts crash the process by design.
Endpoints: `GET /health`, `GET /species`, `POST /identify?lat=&lon=` (multipart
`file`, lat/lon optional).

**Mobile** (`cd mobile`):
```bash
npm install
cp .env.example .env                                  # API_BASE_URL (default http://localhost:8000)
npm run web        # → http://localhost:3000 (Metro + static page server via scripts/web.js)
npm run ios        # macOS + Xcode; run `cd ios && pod install` first
npm run android    # emulator; auto-uses http://10.0.2.2:8000 when API_BASE_URL unset
npm run typecheck  # strict tsc --noEmit — the closest thing to a test suite here
```

## Data sourcing precedence

`training/data.py` `load_manifest` picks its source in order: **(1)** the
PostgreSQL `images`/`species` tables if `DATABASE_URL` is set (canonical;
streams images from `gs://`/`s3://`, caching under `LOCAL_DATA_DIR`); **(2)** a
local `LOCAL_DATA_DIR` laid out as `{species_slug}/{image}.jpg` for offline
work. The pipeline writes to both DB and a manifest CSV so either path works.
DB writes are idempotent via `ON CONFLICT` on `species.slug` and
`images.storage_path` — both UNIQUE constraints are required for re-runnable
scrapes.

## Mobile specifics worth knowing

- **Web is a Metro target**, not a separate build: `metro.config.js` maps bare
  `react-native` → `react-native-web` and falls back RN-core deep imports
  (platform-split internals with no web variant) to their `.ios` pure-JS
  variants. Touching that resolver can break the web bundle silently.
- **NativeWind v4 is wired but dormant.** Everything uses `StyleSheet.create`.
  The tailwind/`global.css`/Metro plumbing exists; to actually use `className`,
  re-add `'nativewind/babel'` to `babel.config.js` presets (it's off because its
  runtime drags reanimated's RN-core imports into the web bundle).
- **Location is always optional.** Every screen works with `lat:null, lon:null`.
  `client.ts` appends lat/lon query params only when *both* are non-null.
- `cp` of `ios/`+`android/` is checked in *with permission patches applied*;
  `scripts/bootstrap_native.sh` regenerates them from the RN 0.76.5 template if
  deleted (and you'd lose those patches).
