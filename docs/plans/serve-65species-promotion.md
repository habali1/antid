# 65-species B4 serving promotion — executed file-level release

Status (2026-09-17): six candidate files were promoted to default `training/artifacts/`, their destination hashes verified, and a loopback API smoke passed. The smoke process was then stopped; this records the default-serving bundle, **not** an always-running API service. No frozen dataset or evidence artifact was touched. `training/artifacts/v1_50species/` remains the verified rollback bundle.

## Verified rollback authority (2026-09-17)

Before promotion, all six `v1_50species/` files were byte-identical to the then-default 50-species files, and the three mandatory model hashes matched its own `inference_policy.json` bindings. The actual API policy loader accepted that bundle (`active=True`, `reason=active`, threshold `0.60`); its taxonomy has 50 entries, prototypes have shape `(50,1792)`, and its geo index has 50 recognized species. It was preserved unchanged during promotion and is an independently usable rollback bundle.

| File | Verified v1 rollback SHA-256 | Current 65-species candidate SHA-256 |
|---|---|---|
| `backbone.onnx` | `7856e2b035a31704c23db36c70aee96ebcd8cfb7a0711e5c878c90044d7e2cdb` | `fc22d26ae5c73d20613dafcef02e75291c8779ec8a1296ebc9a72f0e7d7f826b` |
| `prototypes.npy` | `b2aec98b94f1a48a0b2170b25cefd774a4ac0a7a0535217299075f4d25a0bae2` | `0e52a7f350996a48f4129e1924e9b0af79517e4c3273dd9ecc7e851121d43825` |
| `taxonomy.json` | `057819601f47613a0b3fa099982547d4234e4f28f0cd45170d50774695538cf9` | `c8672287d59b5b9d3fb9aec5fd85a008ad30441f2ce3159713a3752b37a04767` |
| `geo_index.json` | `56fbdcac943c74bd184b5b8ab12953452f76b6ba42882bd6c5695e887a0dd5e5` | `1772c97baa58aebeb46f2110885530a0cfa8599c45b18fe999cb362750989ee7` |
| `model.pth` | `43732525b498a5b4e64221c39fc77c700db4259a25d8f672e1a39a358e794eec` | `523e6aa8e7db16be8f88285a71db0a4800e2629d3c906eb7f7135adf51a5061c` |
| `inference_policy.json` | `2ac739e153f08af0363e7a291260700f3d24af00000b4b62a382310207a97c05` | `9feeae83013ecf72266084421fc74bbfe21757de528ad7ed258c01dcffc9422d` **promoted unchanged** |

Before promotion, three default-serving model files were hard links to an old scratch directory. The release avoided `Copy-Item -Force` and other in-place writes: it staged independent copies, renamed each old live entry aside, then renamed the staged file into place. Keep the previous entries for recovery. The v1 backup files themselves were not those hard links and must not be edited.

## Executed release gates and deferred diagnostics

1. Rollback authority was verified before copying: all six v1 files and candidate files matched the table, the v1 policy bindings and loader passed, and no API was running.
2. The existing candidate generator `--check` passed under ORT 1.30.0; the 0.61 policy was neither modified nor regenerated. The threshold and three bound model hashes are unchanged.
3. The six candidate files were copied to `.promotion-stage-849f20b38b634283bfc246a62f68c4e8/incoming`, hash-verified, and published by renaming each old default entry into that stage's retained `previous/` and each incoming entry into the default root. The six destination hashes then matched the candidate column above. This avoided in-place writes to the three previously hard-linked live files. The v1 backup was not moved or deleted.
4. An isolated API smoke used `api/.venv/Scripts/python.exe`, `ARTIFACTS_DIR` unset, and loopback port 8011. `/health` reported 65 species with policy and geo active; `/species` matched all 65 taxonomy entries in index order. Three `/identify` requests used the same synthetic in-memory JPEG (no coordinates, latitude alone, both coordinates): HTTP 200, three distinct matches, finite similarities, gate active, and a pre-geo low-confidence decision unchanged by geo application. The independently computed raw pre-geo maximum for this image was `-0.034030985087156296`, below 0.61, consistent with `low_confidence=true`. The smoke PID was stopped and port 8011 was free afterward. This checks current endpoints, not controlled threshold scores; strict boundary behavior is already covered by committed `should_abstain` tests.
5. `README.md` was updated only after the file-level promotion and smoke. It labels all-65 development (1,766/2,596 top-1; 2,151/2,596 top-3) separately from the one-shot 15-species final test (296/450; 356/450), includes gate coverage 317/450 and accepted top-1 252/317, and carries `perceptual_independence_incomplete_by_decision`. The old 50-species `benchmark_v1` results are historical, and the non-commercial license limitation is explicit.

**Deferred to a separate reviewed task:** `/health` threshold, gate-active, and recognized-cell diagnostics; `/identify` unrounded `raw_pre_geo_max_cosine`; informational current-runtime policy metadata; and any resulting policy regeneration. These are diagnostic visibility improvements, not serving requirements. Coupling them to file-level promotion would turn a file move into a code-change-plus-policy-regeneration cycle. The immutable parity report's ORT 1.29.0 / CUDA Torch environment is a historical fact and was not rewritten. No threshold retuning, model retraining, frozen-set inference, or frozen-evidence modification was part of promotion.

## Exact file-copy/publish operations used for this promotion

The commands below show the reviewed stage-then-rename operation with the existing policy hash. This release has already been executed; **do not rerun the block against the live bundle**. It stages all six files before changing the root and never writes into a hard-linked live file in place.

```powershell
$repo = (Resolve-Path -LiteralPath 'C:\dev\antid').Path
$live = Join-Path $repo 'training\artifacts'
$source = Join-Path $live 'northeast_v1_b4_dev_v2'
$names = @('backbone.onnx','prototypes.npy','taxonomy.json','geo_index.json','model.pth','inference_policy.json')
$expected = @{
  'backbone.onnx' = 'fc22d26ae5c73d20613dafcef02e75291c8779ec8a1296ebc9a72f0e7d7f826b'
  'prototypes.npy' = '0e52a7f350996a48f4129e1924e9b0af79517e4c3273dd9ecc7e851121d43825'
  'taxonomy.json' = 'c8672287d59b5b9d3fb9aec5fd85a008ad30441f2ce3159713a3752b37a04767'
  'geo_index.json' = '1772c97baa58aebeb46f2110885530a0cfa8599c45b18fe999cb362750989ee7'
  'model.pth' = '523e6aa8e7db16be8f88285a71db0a4800e2629d3c906eb7f7135adf51a5061c'
  'inference_policy.json' = '9feeae83013ecf72266084421fc74bbfe21757de528ad7ed258c01dcffc9422d'
}
foreach ($n in $names) {
  $p = Join-Path $source $n
  if (!(Test-Path -LiteralPath $p -PathType Leaf) -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne $expected[$n]) { throw "Candidate invalid: $n" }
  if (!(Test-Path -LiteralPath (Join-Path $live $n) -PathType Leaf)) { throw "Live artifact missing before promotion: $n" }
}
$stage = Join-Path $live ('.promotion-stage-' + [guid]::NewGuid().ToString('N'))
$incoming = New-Item -ItemType Directory -Path (Join-Path $stage 'incoming')
$previous = New-Item -ItemType Directory -Path (Join-Path $stage 'previous')
foreach ($n in $names) {
  Copy-Item -LiteralPath (Join-Path $source $n) -Destination (Join-Path $incoming.FullName $n)
  if ((Get-FileHash -LiteralPath (Join-Path $incoming.FullName $n) -Algorithm SHA256).Hash.ToLower() -ne $expected[$n]) { throw "Stage hash mismatch: $n" }
}
foreach ($n in $names) {
  Move-Item -LiteralPath (Join-Path $live $n) -Destination (Join-Path $previous.FullName $n)
  Move-Item -LiteralPath (Join-Path $incoming.FullName $n) -Destination (Join-Path $live $n)
}
foreach ($n in $names) {
  if ((Get-FileHash -LiteralPath (Join-Path $live $n) -Algorithm SHA256).Hash.ToLower() -ne $expected[$n]) { throw "Published hash mismatch: $n; do not start API" }
}
Write-Host "All six 65-species files verified; previous live entries retained at $($previous.FullName)"
```

## Exact rollback procedure: restore the verified 50-species bundle

Stop the API first and keep it stopped until every check succeeds. This procedure copies from the untouched `v1_50species/` directory, never moves or deletes it. It preserves the previous live entries in a unique stage directory. It is valid even if the promotion stopped halfway, provided the backup still passes the preflight checks.

```powershell
$repo = (Resolve-Path -LiteralPath 'C:\dev\antid').Path
$live = Join-Path $repo 'training\artifacts'
$backup = Join-Path $live 'v1_50species'
$names = @('backbone.onnx','prototypes.npy','taxonomy.json','geo_index.json','model.pth','inference_policy.json')
$expected = @{
  'backbone.onnx' = '7856e2b035a31704c23db36c70aee96ebcd8cfb7a0711e5c878c90044d7e2cdb'
  'prototypes.npy' = 'b2aec98b94f1a48a0b2170b25cefd774a4ac0a7a0535217299075f4d25a0bae2'
  'taxonomy.json' = '057819601f47613a0b3fa099982547d4234e4f28f0cd45170d50774695538cf9'
  'geo_index.json' = '56fbdcac943c74bd184b5b8ab12953452f76b6ba42882bd6c5695e887a0dd5e5'
  'model.pth' = '43732525b498a5b4e64221c39fc77c700db4259a25d8f672e1a39a358e794eec'
  'inference_policy.json' = '2ac739e153f08af0363e7a291260700f3d24af00000b4b62a382310207a97c05'
}
foreach ($n in $names) {
  $p = Join-Path $backup $n
  if (!(Test-Path -LiteralPath $p -PathType Leaf) -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne $expected[$n]) { throw "Rollback source invalid: $n; no live file changed" }
}
$policy = Get-Content -LiteralPath (Join-Path $backup 'inference_policy.json') -Raw | ConvertFrom-Json
foreach ($n in @('backbone.onnx','prototypes.npy','taxonomy.json')) {
  if ($policy.content.artifact_hashes.$n -ne $expected[$n]) { throw "Rollback policy binding mismatch: $n" }
}
$stage = Join-Path $live ('.rollback-stage-' + [guid]::NewGuid().ToString('N'))
$incoming = New-Item -ItemType Directory -Path (Join-Path $stage 'incoming')
$previous = New-Item -ItemType Directory -Path (Join-Path $stage 'previous')
foreach ($n in $names) {
  Copy-Item -LiteralPath (Join-Path $backup $n) -Destination (Join-Path $incoming.FullName $n)
  if ((Get-FileHash -LiteralPath (Join-Path $incoming.FullName $n) -Algorithm SHA256).Hash.ToLower() -ne $expected[$n]) { throw "Rollback stage hash mismatch: $n" }
}
foreach ($n in $names) {
  $dest = Join-Path $live $n
  if (Test-Path -LiteralPath $dest -PathType Leaf) { Move-Item -LiteralPath $dest -Destination (Join-Path $previous.FullName $n) }
  Move-Item -LiteralPath (Join-Path $incoming.FullName $n) -Destination $dest
}
foreach ($n in $names) {
  if ((Get-FileHash -LiteralPath (Join-Path $live $n) -Algorithm SHA256).Hash.ToLower() -ne $expected[$n]) { throw "Rollback publish mismatch: $n; do not start API" }
}
& (Join-Path $repo 'api\.venv\Scripts\python.exe') -c "import sys; from pathlib import Path; sys.path.insert(0,'api'); import inference,inference_policy; s=inference_policy.load_inference_policy(Path('training/artifacts'),['CPUExecutionProvider'],inference.PREPROCESSING_CONTRACT); assert s.active and s.threshold == 0.60 and s.reason == 'active'"
if ($LASTEXITCODE -ne 0) { throw 'Rollback API policy loader did not accept restored v1 bundle' }
Write-Host "50-species rollback verified; previous live entries retained at $($previous.FullName)"
```

After rollback, start the API only after `/health` reports 50 species and active v1 policy at 0.60 (using the policy loader for the threshold if `/health` has not yet gained that field), `/species` matches the v1 taxonomy, and a synthetic `/identify` request succeeds. Never repurpose the 65-species one-shot final test for a rollback check.
