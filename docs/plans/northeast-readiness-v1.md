# AntID Northeast candidate readiness — metadata audit v1

Generated 2026-09-05T20:27:48.864672Z. This is a metadata snapshot, not a frozen dataset.
No photos were downloaded, no split membership was assigned, and no model artifact was changed.

## Predeclared capacity target

Per provisional species: **160 train + 40 development + 30 untouched final-test = 230 observations**. Readiness uses the approved personal/non-commercial pool (CC0, CC BY, CC BY-SA, CC BY-NC, or CC BY-NC-SA) and excludes every prior photo/observation match found.
This target was fixed before any model score was inspected; the displayed allocation is capacity only.

| Candidate | Genus (iNat ID) | Core eligible | Approved personal/NC | No license | Observers | States | Date span | Prior overlaps | Approved shortfall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Nylanderia flavipes (372712) | Nylanderia (69405) | 367 | 2,030 | 438 (17.6%) | 73 | 5 | 2015-05-07 to 2026-08-27 | 1 | 0 |
| Lasius neoniger (69311) | Lasius (69086) | 193 | 1,074 | 311 (22.2%) | 75 | 8 | 2013-05-12 to 2026-09-04 | 1 | 0 |
| Lasius claviger (222712) | Lasius (69086) | 156 | 1,003 | 330 (24.5%) | 72 | 8 | 2016-01-28 to 2026-04-17 | 0 | 0 |
| Camponotus novaeboracensis (143252) | Camponotus (62781) | 174 | 955 | 291 (23.0%) | 80 | 7 | 2007-08-02 to 2026-08-27 | 1 | 0 |
| Lasius emarginatus (341143) | Lasius (69086) | 77 | 798 | 236 (22.6%) | 21 | 1 | 2017-03-08 to 2026-08-06 | 2 | 0 |
| Camponotus americanus (215970) | Camponotus (62781) | 117 | 818 | 198 (19.3%) | 58 | 7 | 2014-08-03 to 2026-08-28 | 2 | 0 |
| Camponotus subbarbatus (143505) | Camponotus (62781) | 161 | 777 | 196 (19.9%) | 48 | 3 | 2012-07-22 to 2026-08-20 | 1 | 0 |
| Camponotus nearcticus (146942) | Camponotus (62781) | 140 | 683 | 180 (20.5%) | 60 | 7 | 2014-08-11 to 2026-07-28 | 0 | 0 |
| Lasius americanus (966095) | Lasius (69086) | 99 | 537 | 146 (20.5%) | 50 | 9 | 2016-06-12 to 2026-08-12 | 2 | 0 |
| Lasius aphidicola (1032788) | Lasius (69086) | 93 | 532 | 125 (18.7%) | 52 | 8 | 2016-12-01 to 2026-08-31 | 0 | 0 |
| Aphaenogaster rudis (213893) | Aphaenogaster (68931) | 64 | 428 | 145 (25.0%) | 28 | 4 | 2016-07-04 to 2026-08-24 | 0 | 0 |
| Formica exsectoides (133654) | Formica (47339) | 114 | 459 | 119 (20.5%) | 45 | 9 | 2013-05-20 to 2026-07-18 | 0 | 0 |
| Ponera pennsylvanica (203419) | Ponera (203420) | 86 | 399 | 92 (18.6%) | 32 | 5 | 2014-04-09 to 2026-09-03 | 0 | 0 |
| Temnothorax curvispinosus (232366) | Temnothorax (424607) | 88 | 393 | 84 (17.4%) | 29 | 6 | 2017-05-11 to 2026-07-12 | 1 | 0 |
| Lasius interjectus (222713) | Lasius (69086) | 44 | 345 | 122 (25.7%) | 32 | 5 | 2013-06-12 to 2026-06-27 | 2 | 0 |

## Reading the result

- 15/15 candidates meet the numeric 230-observation target under the approved personal/non-commercial license pool.
- The previously agreed 150-image admission floor is not binding because every current candidate clears 230. It remains mandatory for any later replacement candidate.
- Source-unlicensed share ranges from 17.4% to 25.7%. Lasius interjectus has both the highest share and the smallest approved pool, but its 345 eligible rows still exceed the 230 quota by 115.
- Numeric readiness does not approve a species. A labeled-photo review must still check label quality and whether diagnostic ant features are visible, especially for Lasius, Camponotus, and Aphaenogaster lookalikes.
- Exact active species IDs and authoritative genus ancestors were verified from iNaturalist; no fuzzy name fallback was used.
- Coordinates were intentionally not written to the audit CSV. State membership and open/obscured status are retained; exact locations are unnecessary for this decision.

## Exclusions and remaining gate

Every observation was checked against training, benchmark_v1, calibration_v1, and unknown_test_v1 using all photo IDs on the observation plus its observation UUID. Training has 69 photos whose parent UUID could not be reconstructed; those remain protected by photo ID only.

**Image SHA-256 and perceptual near-duplicate exclusion have not been performed.** The API metadata does not contain the downloaded bytes needed for those checks. After download, the pipeline must hash every file, compare it with all frozen hashes, and review perceptual near-duplicates before any split is frozen.

## License policy used for planning

- Core reporting tier: CC0, CC BY, CC BY-SA.
- Approved personal/non-commercial pool as of 2026-09-05: core licenses plus CC BY-NC and CC BY-NC-SA.
- CC licenses with NoDerivatives and unlicensed/all-rights-reserved photos are not counted as eligible.
- An empty `photo_license` records that iNaturalist returned no license code; it is preserved as source truth and the row is ineligible.
- Attribution and the source URL are preserved per candidate row. This is conservative project policy, not legal advice; review obligations before redistribution or commercial use.

## Bounded download/storage proposal

Maximum if all 15 candidates are retained: **3,450 medium images**. Existing medium downloads average 147.0 KiB and have a 95th-percentile size of 284.5 KiB. That projects to 495 MiB at the mean or 958 MiB using the p95-per-file bound. Reserve **2.0 GiB** for raw and cleaned working copies plus metadata; no paid storage is required.

## Reproducibility

The row-level snapshot is `data/northeast_readiness_v1/candidates.csv` (14,431 rows, SHA-256 `082522f359585db13319378e14a1066c4fb7ffdda98194a9fae829e531cdc5ed`).
The companion JSON references that CSV by path, row count, and SHA-256 rather than duplicating its rows, and records the required future download-manifest fields.
Query parameters, source hashes, API retrieval window, per-species counts, license counts, observer concentration, geographic spread, and date coverage are in the companion JSON.
The script paces requests at about one per second in line with iNaturalist's published API practices.

## Stop point

Do not download images or substitute/freeze species yet. The license/quota policy is approved; define and review the small manual photo-review sample next.
