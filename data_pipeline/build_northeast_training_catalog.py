#!/usr/bin/env python3
"""build_northeast_training_catalog.py — deterministically rebuild the
versioned 65-species Northeast training catalog from authoritative inputs.

Outputs (data/northeast_expansion_v1/, the only paths this script ever
writes to):
    manifest_all_northeast_v1.csv   13,581-row versioned training manifest
    northeast_taxonomy_v1.json      65-entry versioned training taxonomy

Authoritative inputs (read-only; this script NEVER writes to any of these):
    data/northeast_expansion_v1/catalog_inputs_v1/base_manifest_50_v1.csv
                                                              frozen, committed byte-for-byte
                                                              snapshot of the original 50-species
                                                              data/manifest_all.csv
    data/northeast_expansion_v1/catalog_inputs_v1/base_taxonomy_50_v1.json
                                                              frozen, committed byte-for-byte
                                                              snapshot of the verified original
                                                              50-species taxonomy (species_name,
                                                              common_name, taxon_id per slug) --
                                                              NOT the live training/artifacts/
                                                              taxonomy.json, which a future retrain
                                                              will overwrite, and not the local
                                                              training/artifacts/v1_50species/ backup,
                                                              which is gitignored and not portable
    data/northeast_expansion_v1/catalog_inputs_v1/legacy_photo_observation_map_v1.json
                                                              frozen, committed byte-for-byte
                                                              snapshot of the legacy
                                                              photo_id -> observation_uuid map
    data/clean/{slug}/{photo_id}.{ext}                       cleaned image files -- LOCAL ONLY,
                                                              not committed, not reproducible from
                                                              this repo alone (see note below)
    data/northeast_expansion_v1/northeast_train_dev_v1.csv   frozen Northeast train+dev rows
    data/benchmark_v1/benchmark_v1.csv, data/calibration_v1/calibration_v1.csv,
    data/unknown_test_v1/unknown_test_v1.csv, data/northeast_final_test_v1/northeast_final_test_v1.csv
                                                              frozen eval sets, for the zero-overlap check

Every count this script checks (13,581 rows, 65 species, 10,985 train /
2,596 val, 200/40 per new species, 9,912/69 recovered/unavailable legacy
UUIDs) is verified as a POSTCONDITION of joining the real inputs above -- none
of them are used to construct the output. In particular, which legacy rows
are "usable" is never a hardcoded list: a row is usable if and only if
{slug}/{photo_id} actually resolves under data/clean/ right now. This script
also never reads its own prior output as an input.

Portability, precisely stated: the three small metadata inputs under
catalog_inputs_v1/ are committed to the repo, so the MANIFEST/TAXONOMY
DERIVATION LOGIC is reproducible from a fresh clone in isolation (e.g. by unit
tests that supply their own synthetic clean_root). A full, real check-mode run
additionally requires the actual `data/clean/` image tree -- for the original
50 species AND the 15 Northeast species copied in from
`data/northeast_expansion_v1/clean/` -- which this script does NOT restore,
verify, or otherwise prove reproducible. The Northeast portion of that image
tree has a documented restore path
(`data_pipeline/scrape_northeast_expansion.py --restore`, per
northeast_expansion_v1.json's own "restore" field) followed by copying into
`data/clean/`; the original 50 species' `data/clean/` has no such committed
restore path at all. Do NOT describe this script as "clean-clone
reproducible" -- only its metadata derivation is; the image tree is a
separate, unproven, local prerequisite.

Usage:
    python build_northeast_training_catalog.py            # check mode (default): rebuild
                                                            # into a temp dir, diff byte-for-byte
                                                            # against the frozen outputs, write nothing
    python build_northeast_training_catalog.py --write     # rebuild and overwrite the frozen
                                                            # outputs in data/northeast_expansion_v1/
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

EXT_PROBE_ORDER = (".jpg", ".jpeg", ".png", ".webp")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SPLIT_MAP = {"train": "train", "development": "val"}
MANIFEST_FIELDS = ["species", "slug", "taxon_id", "photo_id", "source", "lat", "lon", "split"]
# "common_name" is additive: training/data.py's _manifest_from_csv already
# reads an optional common_name column (it just has never been populated by
# any manifest CSV before this one). Without it, retraining from this CSV
# would silently produce common_name=None for all 65 species, discarding the
# curated common names the original 50 already have -- see
# test_taxonomy_full_object_equality_with_versioned_snapshot.
RICH_FIELDS = ["common_name", "photo_license", "photo_attribution", "observation_uuid",
              "source_url", "sha256", "provenance_status"]
ALL_FIELDS = MANIFEST_FIELDS + RICH_FIELDS
# Every northeast_v1_complete row must have COMPLETE provenance -- that's the
# entire distinction from legacy_partial rows, which are explicitly allowed
# blanks. A blank in any of these on a Northeast row is a data problem, not a
# documented limitation, so it fails closed rather than silently shipping.
REQUIRED_NORTHEAST_PROVENANCE_FIELDS = (
    "photo_license", "photo_attribution", "observation_uuid", "photo_id", "source_url", "sha256",
)
NEW_SPECIES_SLUGS = frozenset({
    "aphaenogaster-rudis", "camponotus-americanus", "camponotus-nearcticus", "camponotus-novaeboracensis",
    "camponotus-subbarbatus", "formica-exsectoides", "lasius-americanus", "lasius-aphidicola",
    "lasius-claviger", "lasius-emarginatus", "lasius-interjectus", "lasius-neoniger",
    "nylanderia-flavipes", "ponera-pennsylvanica", "temnothorax-curvispinosus",
})

# The original 50-species taxonomy.json's exact frozen byte hash (also
# inference_policy.json's recorded artifact_hashes["taxonomy.json"]). Used to
# refuse building from an original-taxonomy input that isn't actually the
# untouched original -- e.g. one already overwritten by a retrain.
EXPECTED_ORIGINAL_TAXONOMY_SHA256 = "057819601f47613a0b3fa099982547d4234e4f28f0cd45170d50774695538cf9"

# The reviewed, offline-finalized Northeast coordinates sidecar's exact frozen
# byte hash (finalize_northeast_coordinates.py's output). Binding this exact
# hash means a future edit to that sidecar can't silently change what
# lat/lon values flow into the manifest without this generator refusing to
# proceed against it.
EXPECTED_COORDINATES_SIDECAR_SHA256 = "17680f64ab81573969e3994f202a01ab9dad89f7aa8467d56a857f88e0cd98aa"

# Independently-derived postconditions the real inputs must satisfy. None of
# these numbers are used to build the output -- they gate it after the fact.
EXPECTED_TOTAL_ROWS = 13581
EXPECTED_TOTAL_SPECIES = 65
EXPECTED_TOTAL_TRAIN = 10985
EXPECTED_TOTAL_VAL = 2596
EXPECTED_NORTHEAST_ROWS = 3600
EXPECTED_NEW_SPECIES_TRAIN = 200
EXPECTED_NEW_SPECIES_VAL = 40
EXPECTED_LEGACY_UUID_MATCHED = 9912
EXPECTED_LEGACY_UUID_UNMATCHED = 69


class CatalogIntegrityError(RuntimeError):
    """A required invariant did not hold. Always fail closed -- never write
    partial or best-effort output."""


def _genus_from_species_name(species_name: str) -> str:
    """First token of the binomial species name -- the genus.

    Intentionally duplicated from training/data.py's identical helper rather
    than imported: data_pipeline/ and training/ stay independently runnable
    (see CLAUDE.md's "loosely-coupled stages"), and this is a two-line pure
    function, not worth a new cross-stage dependency.
    """
    parts = (species_name or "").split()
    return parts[0] if parts else ""


def _resolve(root: Path, slug: str, photo_id: str) -> Path | None:
    for ext in EXT_PROBE_ORDER:
        p = root / slug / f"{photo_id}{ext}"
        if p.exists():
            return p
    return None


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Inputs:
    manifest_all: Path
    clean_root: Path
    northeast_train_dev: Path
    uuid_helper: Path
    original_taxonomy: Path
    coordinates_sidecar: Path | None = None
    frozen_eval_sets: dict[str, Path] = field(default_factory=dict)
    # Set to None to skip (only ever done by tests against synthetic fixtures
    # that intentionally don't match the real frozen hash).
    expected_original_taxonomy_sha256: str | None = EXPECTED_ORIGINAL_TAXONOMY_SHA256
    # Same idea for the reviewed coordinates sidecar. None disables the
    # lat/lon join entirely (rows keep blank lat/lon, as before its existence).
    expected_coordinates_sidecar_sha256: str | None = None

    @classmethod
    def real(cls) -> "Inputs":
        # The three small metadata inputs are read from the committed
        # catalog_inputs_v1/ snapshots -- NOT data/manifest_all.csv,
        # data/benchmark_v1/_training_obs_uuids.json, or the gitignored
        # local training/artifacts/v1_50species/ backup -- so a committed
        # generator never depends on uncommitted local state for its metadata
        # derivation. See the module docstring's portability note: this does
        # NOT make the image tree (data/clean/) reproducible.
        inputs_dir = REPO / "data" / "northeast_expansion_v1" / "catalog_inputs_v1"
        return cls(
            manifest_all=inputs_dir / "base_manifest_50_v1.csv",
            clean_root=REPO / "data" / "clean",
            northeast_train_dev=REPO / "data" / "northeast_expansion_v1" / "northeast_train_dev_v1.csv",
            uuid_helper=inputs_dir / "legacy_photo_observation_map_v1.json",
            original_taxonomy=inputs_dir / "base_taxonomy_50_v1.json",
            coordinates_sidecar=REPO / "data" / "northeast_expansion_v1" / "northeast_coordinates_v1.json",
            frozen_eval_sets={
                "benchmark_v1": REPO / "data" / "benchmark_v1" / "benchmark_v1.csv",
                "calibration_v1": REPO / "data" / "calibration_v1" / "calibration_v1.csv",
                "unknown_test_v1": REPO / "data" / "unknown_test_v1" / "unknown_test_v1.csv",
                "northeast_final_test_v1": REPO / "data" / "northeast_final_test_v1" / "northeast_final_test_v1.csv",
            },
            expected_coordinates_sidecar_sha256=EXPECTED_COORDINATES_SIDECAR_SHA256,
        )


@dataclass
class BuildResult:
    manifest_rows: list[dict]
    taxonomy: dict[str, dict]
    excluded_legacy: list[tuple[str, str]]
    stats: dict


def _load_and_verify_original_taxonomy(inputs: Inputs) -> dict[str, dict]:
    if not inputs.original_taxonomy.exists():
        raise CatalogIntegrityError(
            f"original taxonomy input does not exist: {inputs.original_taxonomy}")
    raw = inputs.original_taxonomy.read_bytes()
    if inputs.expected_original_taxonomy_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != inputs.expected_original_taxonomy_sha256:
            raise CatalogIntegrityError(
                f"{inputs.original_taxonomy} does not match the expected original-taxonomy "
                f"sha256 (expected {inputs.expected_original_taxonomy_sha256}, got {actual}) -- "
                f"has it already been overwritten by a retrain? Refusing to build from it."
            )
    tax = json.loads(raw)
    if sorted(int(k) for k in tax) != list(range(len(tax))):
        raise CatalogIntegrityError(f"{inputs.original_taxonomy} keys are not contiguous from 0")
    by_slug = {v["slug"]: v for v in tax.values()}
    if len(by_slug) != len(tax):
        raise CatalogIntegrityError(f"{inputs.original_taxonomy} has duplicate slugs")
    return by_slug


def _load_and_verify_coordinates_sidecar(
        inputs: Inputs, ne_source_rows: list[dict]) -> dict[str, tuple[float, float]]:
    """{observation_uuid: (lat, lon)} from the reviewed, offline-finalized
    coordinates sidecar -- or {} if no sidecar is configured, in which case
    Northeast rows keep blank lat/lon exactly as before this integration.

    Beyond the outer whole-file sha256 binding, this validates the sidecar's
    internal content against ne_source_rows (the already-read
    northeast_train_dev rows) rather than trusting it at face value -- an
    unmodified but stale or mismatched sidecar (e.g. finalized against a
    different source manifest, or missing/adding observations) must still
    fail closed even if somehow its outer bytes matched the pinned hash."""
    if inputs.coordinates_sidecar is None:
        return {}
    if not inputs.coordinates_sidecar.exists():
        raise CatalogIntegrityError(
            f"coordinates sidecar input does not exist: {inputs.coordinates_sidecar}")
    raw = inputs.coordinates_sidecar.read_bytes()
    if inputs.expected_coordinates_sidecar_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != inputs.expected_coordinates_sidecar_sha256:
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar} does not match the expected coordinates-"
                f"sidecar sha256 (expected {inputs.expected_coordinates_sidecar_sha256}, "
                f"got {actual}) -- refusing to build lat/lon from an unverified sidecar."
            )
    sidecar = json.loads(raw)

    schema_version = sidecar.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise CatalogIntegrityError(
            f"{inputs.coordinates_sidecar}: schema_version must be exactly integer 1, "
            f"got {schema_version!r}")

    source_manifest = sidecar.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise CatalogIntegrityError(f"{inputs.coordinates_sidecar} has no source_manifest object")
    actual_source_sha256 = hashlib.sha256(inputs.northeast_train_dev.read_bytes()).hexdigest()
    if source_manifest.get("sha256") != actual_source_sha256:
        raise CatalogIntegrityError(
            f"{inputs.coordinates_sidecar}'s source_manifest.sha256 "
            f"{source_manifest.get('sha256')!r} does not match the actual sha256 "
            f"{actual_source_sha256} of {inputs.northeast_train_dev} -- refusing to trust "
            f"coordinates finalized against a different source manifest."
        )
    if source_manifest.get("rows") != len(ne_source_rows):
        raise CatalogIntegrityError(
            f"{inputs.coordinates_sidecar}'s source_manifest.rows "
            f"{source_manifest.get('rows')!r} does not match {len(ne_source_rows)} actual "
            f"rows in {inputs.northeast_train_dev}"
        )

    coverage = sidecar.get("coverage")
    if not isinstance(coverage, dict):
        raise CatalogIntegrityError(f"{inputs.coordinates_sidecar} has no coverage object")
    if (coverage.get("rows_total") != len(ne_source_rows)
            or coverage.get("rows_with_coordinate") != len(ne_source_rows)):
        raise CatalogIntegrityError(
            f"{inputs.coordinates_sidecar}'s coverage is not complete: {coverage!r} "
            f"(expected rows_total == rows_with_coordinate == {len(ne_source_rows)})"
        )
    if coverage.get("coverage_rate") != 1.0:
        raise CatalogIntegrityError(
            f"{inputs.coordinates_sidecar}'s coverage_rate {coverage.get('coverage_rate')!r} "
            f"does not represent complete coverage (expected 1.0)"
        )

    observations = sidecar.get("observations")
    if not isinstance(observations, dict):
        raise CatalogIntegrityError(f"{inputs.coordinates_sidecar} has no observations object")

    source_by_uuid: dict[str, dict] = {}
    for r in ne_source_rows:
        uuid = r.get("observation_uuid")
        if not uuid or not uuid.strip():
            raise CatalogIntegrityError(
                f"{inputs.northeast_train_dev} has a row with a blank observation_uuid "
                f"(slug={r.get('slug')!r}, photo_id={r.get('photo_id')!r})"
            )
        if uuid in source_by_uuid:
            raise CatalogIntegrityError(
                f"{inputs.northeast_train_dev} has duplicate observation_uuid {uuid!r}"
            )
        source_by_uuid[uuid] = r

    sidecar_uuids = set(observations)
    source_uuids = set(source_by_uuid)
    missing = source_uuids - sidecar_uuids
    unexpected = sidecar_uuids - source_uuids
    if missing:
        raise CatalogIntegrityError(
            f"{inputs.coordinates_sidecar} is missing {len(missing)} observation_uuid(s) "
            f"present in {inputs.northeast_train_dev}: {sorted(missing)[:5]}"
        )
    if unexpected:
        raise CatalogIntegrityError(
            f"{inputs.coordinates_sidecar} has {len(unexpected)} observation_uuid(s) not "
            f"present in {inputs.northeast_train_dev}: {sorted(unexpected)[:5]}"
        )

    out: dict[str, tuple[float, float]] = {}
    for uuid, entry in observations.items():
        if not isinstance(entry, dict):
            raise CatalogIntegrityError(f"{inputs.coordinates_sidecar}: malformed entry for {uuid}")
        source_row = source_by_uuid[uuid]

        if entry.get("geoprivacy") == "private":
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: observation {uuid} is geoprivacy=private -- "
                f"a private coordinate must never be used"
            )

        if entry.get("slug") != source_row["slug"]:
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: entry for {uuid} has slug "
                f"{entry.get('slug')!r}, source manifest says {source_row['slug']!r}"
            )
        try:
            entry_taxon_id = int(entry.get("taxon_id"))
        except (TypeError, ValueError):
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: non-integer taxon_id for {uuid}: "
                f"{entry.get('taxon_id')!r}"
            )
        if isinstance(entry.get("taxon_id"), bool) or entry_taxon_id != int(source_row["taxon_id"]):
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: entry for {uuid} has taxon_id "
                f"{entry.get('taxon_id')!r}, source manifest says {source_row['taxon_id']!r}"
            )

        lat, lon = entry.get("lat"), entry.get("lon")
        if (isinstance(lat, bool) or isinstance(lon, bool)
                or not isinstance(lat, (int, float)) or not isinstance(lon, (int, float))):
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: non-numeric coordinate for {uuid}: "
                f"lat={lat!r} lon={lon!r}"
            )
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: non-finite coordinate for {uuid}: "
                f"lat={lat!r} lon={lon!r}"
            )
        if not (-90.0 <= lat <= 90.0):
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: latitude out of range for {uuid}: {lat!r}")
        if not (-180.0 <= lon <= 180.0):
            raise CatalogIntegrityError(
                f"{inputs.coordinates_sidecar}: longitude out of range for {uuid}: {lon!r}")

        out[uuid] = (float(lat), float(lon))
    return out


def build_catalog(inputs: Inputs) -> BuildResult:
    """Pure function: (inputs) -> (manifest_rows, taxonomy). Fails closed
    (raises CatalogIntegrityError) rather than returning partial output.
    Never reads or writes the generated output files themselves."""
    original_taxonomy_by_slug = _load_and_verify_original_taxonomy(inputs)

    if not inputs.manifest_all.exists():
        raise CatalogIntegrityError(f"missing input: {inputs.manifest_all}")
    orig_rows = _read_csv(inputs.manifest_all)
    orig_slugs = {r["slug"] for r in orig_rows}
    if orig_slugs != set(original_taxonomy_by_slug):
        raise CatalogIntegrityError(
            f"{inputs.manifest_all}'s slug set does not match {inputs.original_taxonomy}'s: "
            f"only in manifest: {orig_slugs - set(original_taxonomy_by_slug)}; "
            f"only in taxonomy: {set(original_taxonomy_by_slug) - orig_slugs}"
        )

    uuid_map: dict[str, str] = {}
    if inputs.uuid_helper.exists():
        uuid_map = json.loads(inputs.uuid_helper.read_text())["photo_to_observation_uuid"]

    if not inputs.northeast_train_dev.exists():
        raise CatalogIntegrityError(f"missing input: {inputs.northeast_train_dev}")
    ne_source_rows = _read_csv(inputs.northeast_train_dev)

    coords_by_uuid = _load_and_verify_coordinates_sidecar(inputs, ne_source_rows)

    # ---- legacy rows: usable iff the cleaned file actually resolves NOW ----
    legacy_rows: list[dict] = []
    excluded_legacy: list[tuple[str, str]] = []
    for r in orig_rows:
        tax_entry = original_taxonomy_by_slug.get(r["slug"])
        if tax_entry is not None:
            if r["species"] != tax_entry["species_name"]:
                raise CatalogIntegrityError(
                    f"{inputs.manifest_all} row species {r['species']!r} for slug {r['slug']!r} "
                    f"disagrees with {inputs.original_taxonomy}'s species_name "
                    f"{tax_entry['species_name']!r}"
                )
            if tax_entry.get("taxon_id") is not None and int(r["taxon_id"]) != int(tax_entry["taxon_id"]):
                raise CatalogIntegrityError(
                    f"{inputs.manifest_all} row taxon_id {r['taxon_id']!r} for slug {r['slug']!r} "
                    f"disagrees with {inputs.original_taxonomy}'s taxon_id {tax_entry['taxon_id']!r}"
                )
        path = _resolve(inputs.clean_root, r["slug"], r["photo_id"])
        if path is None:
            excluded_legacy.append((r["slug"], r["photo_id"]))
            continue
        row = {k: r[k] for k in MANIFEST_FIELDS}
        row["common_name"] = tax_entry.get("common_name") or ""
        row["photo_license"] = ""
        row["photo_attribution"] = ""
        row["observation_uuid"] = uuid_map.get(r["photo_id"], "")
        row["source_url"] = ""
        row["sha256"] = _sha256_file(path)
        row["provenance_status"] = "legacy_partial"
        legacy_rows.append(row)

    # ---- northeast rows: must ALL resolve (this dataset is supposed to be complete) ----
    northeast_rows: list[dict] = []
    for r in ne_source_rows:
        split = SPLIT_MAP.get(r["split"])
        if split is None:
            raise CatalogIntegrityError(
                f"{inputs.northeast_train_dev}: unexpected split value {r['split']!r} "
                f"for photo_id {r['photo_id']!r} (expected 'train' or 'development')"
            )
        path = _resolve(inputs.clean_root, r["slug"], r["photo_id"])
        if path is None:
            raise CatalogIntegrityError(
                f"Northeast row {r['slug']}/{r['photo_id']} does not resolve under "
                f"{inputs.clean_root} -- the Northeast copy step must run before this generator."
            )
        actual_sha = _sha256_file(path)
        if actual_sha != r["sha256"]:
            raise CatalogIntegrityError(
                f"Northeast row {r['slug']}/{r['photo_id']}: resolved file hash {actual_sha} "
                f"does not match {inputs.northeast_train_dev}'s recorded sha256 {r['sha256']}"
            )
        if inputs.coordinates_sidecar is not None:
            coord = coords_by_uuid.get(r["observation_uuid"])
            if coord is None:
                raise CatalogIntegrityError(
                    f"Northeast row {r['slug']}/{r['photo_id']} (observation "
                    f"{r['observation_uuid']}) has no entry in the coordinates sidecar "
                    f"{inputs.coordinates_sidecar} -- every Northeast row must resolve a "
                    f"coordinate when a sidecar is configured."
                )
            lat, lon = str(coord[0]), str(coord[1])
        else:
            lat, lon = "", ""

        row = {
            "species": r["species"], "slug": r["slug"], "taxon_id": r["taxon_id"],
            "photo_id": r["photo_id"], "source": "inat_api", "lat": lat, "lon": lon,
            "split": split, "common_name": "",
            "photo_license": r["photo_license"], "photo_attribution": r["photo_attribution"],
            "observation_uuid": r["observation_uuid"], "source_url": r["source_url"],
            "sha256": actual_sha, "provenance_status": "northeast_v1_complete",
        }
        for f in REQUIRED_NORTHEAST_PROVENANCE_FIELDS:
            if not (row[f] and row[f].strip()):
                raise CatalogIntegrityError(
                    f"Northeast row {r['slug']}/{r['photo_id']}: required provenance field "
                    f"{f!r} is blank -- every northeast_v1_complete row must have complete "
                    f"provenance ({', '.join(REQUIRED_NORTHEAST_PROVENANCE_FIELDS)})."
                )
        northeast_rows.append(row)

    all_rows = legacy_rows + northeast_rows

    # ---- taxonomy ----
    ne_meta_by_slug: dict[str, dict] = {}
    for r in ne_source_rows:
        ne_meta_by_slug.setdefault(r["slug"], r)
    all_slugs = sorted({r["slug"] for r in all_rows})
    taxonomy: dict[str, dict] = {}
    for i, slug in enumerate(all_slugs):
        if slug in original_taxonomy_by_slug:
            src = original_taxonomy_by_slug[slug]
            species_name = src["species_name"]
            common_name = src.get("common_name")
            taxon_id = src["taxon_id"]
        else:
            src = ne_meta_by_slug[slug]
            species_name = src["species"]
            common_name = None
            taxon_id = int(src["taxon_id"])
        taxonomy[str(i)] = {
            "species_name": species_name,
            "common_name": common_name,
            "taxon_id": taxon_id,
            "slug": slug,
            "genus": _genus_from_species_name(species_name),
        }

    stats = _verify_postconditions(all_rows, taxonomy, inputs)
    return BuildResult(manifest_rows=all_rows, taxonomy=taxonomy,
                       excluded_legacy=excluded_legacy, stats=stats)


def _verify_postconditions(all_rows: list[dict], taxonomy: dict[str, dict], inputs: Inputs) -> dict:
    """Every check here is a postcondition of the joined real data, never an
    input used to build it. Raises CatalogIntegrityError on any violation."""

    def fail(msg: str):
        raise CatalogIntegrityError(msg)

    if len(all_rows) != EXPECTED_TOTAL_ROWS:
        fail(f"expected {EXPECTED_TOTAL_ROWS} usable rows, got {len(all_rows)}")

    slugs = {r["slug"] for r in all_rows}
    if len(slugs) != EXPECTED_TOTAL_SPECIES:
        fail(f"expected {EXPECTED_TOTAL_SPECIES} species, got {len(slugs)}")

    pair_counts = Counter((r["slug"], r["photo_id"]) for r in all_rows)
    dupes = {k: v for k, v in pair_counts.items() if v > 1}
    if dupes:
        fail(f"duplicate (slug, photo_id) rows: {list(dupes.items())[:5]}")

    bad_format = [r for r in all_rows if not SHA256_RE.match(r["sha256"])]
    if bad_format:
        fail(f"{len(bad_format)} row(s) have a malformed sha256, e.g. {bad_format[0]}")

    hash_counts = Counter(r["sha256"] for r in all_rows)
    dupe_hashes = {h: c for h, c in hash_counts.items() if c > 1}
    if dupe_hashes:
        fail(f"duplicate sha256 values in the merged set: {list(dupe_hashes.items())[:5]}")

    all_hash_set = set(hash_counts)
    for name, path in inputs.frozen_eval_sets.items():
        if not path.exists():
            fail(f"missing frozen eval set for overlap check: {path}")
        eval_hashes = {r["sha256"] for r in _read_csv(path)}
        overlap = all_hash_set & eval_hashes
        if overlap:
            fail(f"{len(overlap)} sha256 overlap(s) with {name}: {list(overlap)[:5]}")

    for f in ("common_name", "photo_license", "photo_attribution", "observation_uuid",
             "source_url", "sha256"):
        if any(r[f] == "legacy_provenance_unavailable" for r in all_rows):
            fail(f"sentinel string leaked into structured field {f!r}")

    legacy = [r for r in all_rows if r["provenance_status"] == "legacy_partial"]
    northeast = [r for r in all_rows if r["provenance_status"] == "northeast_v1_complete"]
    if len(northeast) != EXPECTED_NORTHEAST_ROWS:
        fail(f"expected {EXPECTED_NORTHEAST_ROWS} northeast rows, got {len(northeast)}")

    if inputs.coordinates_sidecar is not None:
        with_coords = sum(1 for r in northeast if r["lat"] and r["lon"])
        if with_coords != EXPECTED_NORTHEAST_ROWS:
            fail(f"expected all {EXPECTED_NORTHEAST_ROWS} northeast rows to have a "
                f"coordinate when a sidecar is configured, got {with_coords}")

    matched_uuid = sum(1 for r in legacy if r["observation_uuid"])
    unmatched_uuid = sum(1 for r in legacy if not r["observation_uuid"])
    if matched_uuid != EXPECTED_LEGACY_UUID_MATCHED:
        fail(f"expected {EXPECTED_LEGACY_UUID_MATCHED} legacy rows with a recovered "
            f"observation_uuid, got {matched_uuid}")
    if unmatched_uuid != EXPECTED_LEGACY_UUID_UNMATCHED:
        fail(f"expected {EXPECTED_LEGACY_UUID_UNMATCHED} legacy rows with unavailable "
            f"observation_uuid, got {unmatched_uuid}")

    legacy_split = Counter(r["split"] for r in legacy)
    northeast_split = Counter(r["split"] for r in northeast)
    total_train = legacy_split["train"] + northeast_split["train"]
    total_val = legacy_split["val"] + northeast_split["val"]
    if (total_train, total_val) != (EXPECTED_TOTAL_TRAIN, EXPECTED_TOTAL_VAL):
        fail(f"train/val decomposition mismatch: legacy_train={legacy_split['train']} "
            f"legacy_val={legacy_split['val']} northeast_train={northeast_split['train']} "
            f"northeast_val={northeast_split['val']} -> total train={total_train} val={total_val}, "
            f"expected train={EXPECTED_TOTAL_TRAIN} val={EXPECTED_TOTAL_VAL}")

    per_new_species: dict[str, Counter] = {}
    for r in northeast:
        per_new_species.setdefault(r["slug"], Counter())[r["split"]] += 1
    if set(per_new_species) != NEW_SPECIES_SLUGS:
        fail(f"northeast species set mismatch: {set(per_new_species)} != {NEW_SPECIES_SLUGS}")
    bad_new = {s: dict(c) for s, c in per_new_species.items()
              if c.get("train") != EXPECTED_NEW_SPECIES_TRAIN or c.get("val") != EXPECTED_NEW_SPECIES_VAL}
    if bad_new:
        fail(f"new species not at {EXPECTED_NEW_SPECIES_TRAIN}/{EXPECTED_NEW_SPECIES_VAL} "
            f"train/val: {bad_new}")

    if sorted(int(k) for k in taxonomy) != list(range(EXPECTED_TOTAL_SPECIES)):
        fail("taxonomy keys are not contiguous 0..N-1")
    tax_slugs = [taxonomy[str(i)]["slug"] for i in range(EXPECTED_TOTAL_SPECIES)]
    if tax_slugs != sorted(tax_slugs):
        fail("taxonomy is not slug-sorted")
    missing_genus = [v["slug"] for v in taxonomy.values() if not v.get("genus")]
    if missing_genus:
        fail(f"taxonomy entries missing genus: {missing_genus}")

    return {
        "total_rows": len(all_rows), "species": len(slugs),
        "legacy_usable": len(legacy), "northeast_rows": len(northeast),
        "legacy_train": legacy_split["train"], "legacy_val": legacy_split["val"],
        "northeast_train": northeast_split["train"], "northeast_val": northeast_split["val"],
        "total_train": total_train, "total_val": total_val,
        "legacy_uuid_matched": matched_uuid, "legacy_uuid_unmatched": unmatched_uuid,
    }


def serialize_manifest_csv(rows: list[dict]) -> bytes:
    import io
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=ALL_FIELDS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def serialize_taxonomy_json(taxonomy: dict) -> bytes:
    # CRLF to match the established convention: train.py's own taxonomy.json
    # writes go through Path.write_text() on Windows, which translates "\n"
    # to "\r\n" -- both the live training/artifacts/taxonomy.json and the
    # previously-frozen northeast_taxonomy_v1.json are CRLF throughout.
    return json.dumps(taxonomy, indent=2).replace("\n", "\r\n").encode("utf-8")


def _write_outputs(result: BuildResult, manifest_out: Path, taxonomy_out: Path) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_bytes(serialize_manifest_csv(result.manifest_rows))
    taxonomy_out.write_bytes(serialize_taxonomy_json(result.taxonomy))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Overwrite the frozen outputs in data/northeast_expansion_v1/. "
                         "Default is check mode: rebuild into a temp dir and diff "
                         "byte-for-byte against the frozen outputs, writing nothing.")
    args = ap.parse_args()

    inputs = Inputs.real()
    manifest_out = REPO / "data" / "northeast_expansion_v1" / "manifest_all_northeast_v1.csv"
    taxonomy_out = REPO / "data" / "northeast_expansion_v1" / "northeast_taxonomy_v1.json"

    try:
        result = build_catalog(inputs)
    except CatalogIntegrityError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"built catalog: {result.stats}")
    print(f"excluded {len(result.excluded_legacy)} legacy rows (no resolved file under "
         f"{inputs.clean_root}), e.g. {result.excluded_legacy[:5]}")

    if args.write:
        _write_outputs(result, manifest_out, taxonomy_out)
        print(f"wrote {manifest_out}")
        print(f"wrote {taxonomy_out}")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp_manifest = Path(td) / "manifest_all_northeast_v1.csv"
        tmp_taxonomy = Path(td) / "northeast_taxonomy_v1.json"
        _write_outputs(result, tmp_manifest, tmp_taxonomy)

        ok = True
        for tmp, frozen in ((tmp_manifest, manifest_out), (tmp_taxonomy, taxonomy_out)):
            if not frozen.exists():
                print(f"FAIL: frozen output missing: {frozen}", file=sys.stderr)
                ok = False
                continue
            same = filecmp.cmp(tmp, frozen, shallow=False)
            print(("OK  " if same else "FAIL") + f" {frozen.name} matches freshly rebuilt output byte-for-byte")
            ok = ok and same
        print("\nCHECK MODE:", "PASS" if ok else "FAIL", "-- nothing written")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
