#!/usr/bin/env python3
"""manual_review_contract.py — shared exact-schema contract for Phase 5F1's
manual-quality / perceptual-duplicate review machinery.

Pure schema/constants module: no filesystem access happens at IMPORT time.
`load_and_verify_contract(repo)` is the one function that does I/O -- every
consumer (the queue generator's --check, manual_review_tool.py,
scan_perceptual_duplicates.py, adjudicate_pairs.py) calls it FIRST, before
anything else, so an arbitrary same-shaped-but-substituted queue/summary/
contract is rejected rather than silently accepted.

Nothing in this module opens an image or touches a model.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT_STATUS_FROZEN = "frozen_before_manual_review"
CONTRACT_REL_PATH = "training/manual_review_contract.json"

# ---------------------------------------------------------------- generic --
_HEX = frozenset("0123456789abcdef")


def is_sha256_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s) <= _HEX


def is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_content_sha256(content: dict) -> str:
    return sha256_bytes(canonical_json_bytes(content))


def canonical_lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def canonical_lf_sha256_file(path) -> str:
    return sha256_bytes(canonical_lf_bytes(Path(path).read_bytes()))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


TIMESTAMP_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def is_strict_utc_timestamp(value: Any) -> bool:
    return isinstance(value, str) and bool(TIMESTAMP_UTC_RE.match(value))


class ContractError(RuntimeError):
    """A contract/queue/summary verification problem. Always fails closed."""


# ------------------------------------------------------ queue composition --
SEED = 20260905

FINAL_TEST_DATASET = "northeast_final_test_v1"
FINAL_TEST_MANIFEST_REL_PATH = "data/northeast_final_test_v1/northeast_final_test_v1.csv"
FINAL_TEST_SPLIT = "final_test"
N_FINAL_TEST_ROWS = 450

EXPANSION_DATASET = "northeast_expansion_v1"
EXPANSION_MANIFEST_REL_PATH = "data/northeast_expansion_v1/northeast_train_dev_v1.csv"
EXPANSION_TRAIN_SPLIT = "train"
EXPANSION_DEV_SPLIT = "development"
EXPANSION_SPLITS = (EXPANSION_TRAIN_SPLIT, EXPANSION_DEV_SPLIT)
N_NEW_SPECIES = 15
PER_SLUG_PER_SPLIT = 5
N_SAMPLED_ROWS = N_NEW_SPECIES * PER_SLUG_PER_SPLIT * len(EXPANSION_SPLITS)  # 150

N_QUEUE_ROWS = N_FINAL_TEST_ROWS + N_SAMPLED_ROWS  # 600

N_SUGGESTED_SESSIONS = 2
ROWS_PER_SUGGESTED_SESSION = N_QUEUE_ROWS // N_SUGGESTED_SESSIONS  # 300
MAX_DECISIONS_PER_SESSION = 300

# The EXACT approved 15 Northeast-expansion species slugs -- not merely "15
# distinct slugs". A source manifest containing any different 15 slugs (or
# any of these slugs missing) fails closed. Sorted for a single canonical
# order everywhere this tuple is serialized.
APPROVED_SPECIES_SLUGS = (
    "aphaenogaster-rudis",
    "camponotus-americanus",
    "camponotus-nearcticus",
    "camponotus-novaeboracensis",
    "camponotus-subbarbatus",
    "formica-exsectoides",
    "lasius-americanus",
    "lasius-aphidicola",
    "lasius-claviger",
    "lasius-emarginatus",
    "lasius-interjectus",
    "lasius-neoniger",
    "nylanderia-flavipes",
    "ponera-pennsylvanica",
    "temnothorax-curvispinosus",
)
assert len(APPROVED_SPECIES_SLUGS) == N_NEW_SPECIES
assert list(APPROVED_SPECIES_SLUGS) == sorted(APPROVED_SPECIES_SLUGS)

# Exact frozen byte hash + row count of each source manifest, as they stood
# when this queue was designed. Verified byte-for-byte before either
# manifest's rows are trusted for selection.
FROZEN_SOURCE_MANIFESTS = {
    FINAL_TEST_DATASET: {
        "path": FINAL_TEST_MANIFEST_REL_PATH,
        "byte_sha256": "0c230bb51f8f2d074a796fce4b1d35a50595a0843d8955291a9829fc4f56f992",
        "row_count": N_FINAL_TEST_ROWS,
    },
    EXPANSION_DATASET: {
        "path": EXPANSION_MANIFEST_REL_PATH,
        "byte_sha256": "9840997f7121907eabc9c5675244749f9d0eaa3908844bfd9db781d2424215f7",
        "row_count": 3600,
    },
}

# The FULL expansion population used by the perceptual scan -- deliberately
# NOT the queue's 5-per-slug-per-split manual-review SAMPLE. The 600-row
# queue exists only to bound manual review to a tractable size; the scan
# must see every train/development row in the frozen manifest so a
# duplicate involving an UNSAMPLED row is never silently missed.
EXPANSION_SCAN_SPLIT_ROW_COUNTS = {
    EXPANSION_TRAIN_SPLIT: 3000,
    EXPANSION_DEV_SPLIT: 600,
}
assert sum(EXPANSION_SCAN_SPLIT_ROW_COUNTS.values()) == FROZEN_SOURCE_MANIFESTS[EXPANSION_DATASET]["row_count"]

# Exact frozen path/byte hash/row count for the five other previously-used
# evidence sets the scan also compares against, PLUS -- where the dataset
# has one -- the byte hash of its own JSON provenance sidecar (checked as
# an additional, independent binding; a changed/replaced manifest OR its
# provenance JSON fails closed before any image opens).
OTHER_EVIDENCE_MANIFESTS = {
    "benchmark_v1": {
        "path": "data/benchmark_v1/benchmark_v1.csv",
        "byte_sha256": "61b3439850da9100b4ae8393f4f9cd90d45b4d5b381f20eaa1ea63d8fb1ec38c",
        "row_count": 1591,
        "provenance_json_path": "data/benchmark_v1/benchmark_v1.json",
        "provenance_json_byte_sha256": "137260dc56e4a24115c71df37383d600ed97d83cbc021080f9d0290a0c311ce9",
    },
    "calibration_v1": {
        "path": "data/calibration_v1/calibration_v1.csv",
        "byte_sha256": "83a80351f8ac65e89ea6124ac7de6d2cb6a3c68efc1f33eb3ecc2eca8c5f63c0",
        "row_count": 1005,
        "provenance_json_path": "data/calibration_v1/calibration_v1.json",
        "provenance_json_byte_sha256": "2e4c4be09656ab001312334641d3306323a8e07ab729634a90a0af1d13af35ed",
    },
    "unknown_test_v1": {
        "path": "data/unknown_test_v1/unknown_test_v1.csv",
        "byte_sha256": "b010af9a66db79fd6b40cd808d666f5f6dd5ebe376321c139707d05dd1849b07",
        "row_count": 573,
        "provenance_json_path": "data/unknown_test_v1/unknown_test_v1.json",
        "provenance_json_byte_sha256": "90e0a388691829fa5e49f5779a24bf5838d8ee85ab4fa9f4f02435fccad4f50d",
    },
    "calibration_v2": {
        "path": "data/calibration_v2/calibration_v2.csv",
        "byte_sha256": "8201200f4c5d7c869a71c3c927ccee0dfb83d6b91a595241dd49996deee1d46e",
        "row_count": 1250,
        "provenance_json_path": "data/calibration_v2/calibration_v2.json",
        "provenance_json_byte_sha256": "d7d96d321c7ebb4a5b3cd55ded2f2bb72628b6ec38b82ef135c5be088369ed9e",
    },
    "unknown_test_v2": {
        "path": "data/unknown_test_v2/unknown_test_v2.csv",
        "byte_sha256": "d706cb6dbda7a672594fc425ce5c8f8f0f6f868662c3d52c5f9c88083163b09f",
        "row_count": 790,
        "provenance_json_path": "data/unknown_test_v2/unknown_test_v2.json",
        "provenance_json_byte_sha256": "eea1cbb3118dfc05f42f477ef58d6ccbc5458df1e8a5496845a146d161a89e6d",
    },
}
OTHER_EVIDENCE_MANIFEST_ENTRY_KEYS = frozenset(
    {"path", "byte_sha256", "row_count", "provenance_json_path", "provenance_json_byte_sha256"}
)


# --------------------------------------------------------- image resolver --
# Every image file under every dataset uses one of exactly these three
# extensions (verified against the real repository); tried in this fixed
# order for determinism, but ALL matching extensions on disk are enumerated
# before deciding -- resolution fails closed on zero matches (missing file)
# and on more than one match (ambiguous: two files claiming the same
# photo_id). Existence checks only -- never opens/reads a file.
IMAGE_EXTENSIONS_ORDERED = (".jpg", ".jpeg", ".png")


class ImageResolutionError(RuntimeError):
    """Zero or ambiguous-multiple candidate image files for one row.
    Always fails closed."""


QUEUE_ROW_REQUIRED_FIELDS = (
    "queue_index", "suggested_session", "dataset", "slug", "split",
    "species", "taxon_id", "observation_uuid", "photo_id", "sha256",
    "selection_digest", "review_order_digest",
)
QUEUE_ROW_FIELD_SET = frozenset(QUEUE_ROW_REQUIRED_FIELDS)

SOURCE_MANIFEST_REQUIRED_COLUMNS = (
    "species", "slug", "taxon_id", "split", "observation_uuid", "photo_id", "sha256",
)

QUEUE_ARTIFACT_REL_PATH = "training/manual_review_queue.csv"
SUMMARY_ARTIFACT_REL_PATH = "training/manual_review_queue_summary.json"

# Approved output paths for artifacts NOT YET produced (real --scan/--review/
# --adjudicate are reserved for a later phase) -- fixed here so no tool can
# be pointed at an arbitrary output location.
APPROVED_OUTPUT_PATHS = {
    "manual_review_ledger": "training/manual_review_ledger.jsonl",
    "pair_adjudication_ledger": "training/pair_adjudication_ledger.jsonl",
    "perceptual_hashes_report": "training/perceptual_duplicate_hashes.json",
    "metadata_leakage_report": "training/perceptual_duplicate_metadata_leakage.json",
    "candidate_pairs_report": "training/perceptual_duplicate_candidate_pairs.json",
    "domain_summary_report": "training/perceptual_duplicate_domain_summary.json",
    "stop_status_report": "training/perceptual_duplicate_stop_status.json",
    "post_adjudication_stop_status_report": "training/perceptual_duplicate_post_adjudication_stop_status.json",
}


def compute_selection_digest(seed: int, slug: str, split: str, observation_uuid: str, photo_id: Any) -> str:
    numeric_photo_id = int(photo_id)
    payload = f"{seed}\0{slug}\0{split}\0{observation_uuid}\0{numeric_photo_id}".encode("utf-8")
    return sha256_bytes(payload)


def compute_review_order_digest(seed: int, dataset: str, slug: str, split: str,
                                observation_uuid: str, photo_id: Any) -> str:
    numeric_photo_id = int(photo_id)
    payload = (f"{seed}\0review_order\0{dataset}\0{slug}\0{split}\0{observation_uuid}"
              f"\0{numeric_photo_id}").encode("utf-8")
    return sha256_bytes(payload)


def compute_queue_identity_order_sha256(rows: list[dict]) -> str:
    """The ONE ordered-identity hash binding the queue's exact row order AND
    content: for each row (in file order), the tuple (queue_index, dataset,
    split, slug, observation_uuid, numeric photo_id, image sha256), joined
    with unit/record separators so no field-boundary ambiguity is possible.
    Any reorder, insertion, deletion, or field substitution changes this
    hash."""
    parts = []
    for row in rows:
        triple = (
            str(int(row["queue_index"])), str(row["dataset"]), str(row["split"]), str(row["slug"]),
            str(row["observation_uuid"]), str(int(row["photo_id"])), str(row["sha256"]),
        )
        parts.append("\x1f".join(triple))
    payload = "\x1e".join(parts).encode("utf-8")
    return sha256_bytes(payload)


# ---------------------------------------------------------- review schema --
REVIEW_STATES = frozenset({
    "usable",
    "poor_quality_usable",
    "unusable_no_visible_ant",
    "unusable_corrupt",
    "unusable_wrong_organism",
    "uncertain",
})
LABEL_PLAUSIBILITY_VALUES = frozenset({"plausible", "implausible", "uncertain"})
DUPLICATE_SUSPICION_VALUES = frozenset({"none", "suspected", "strong"})

# Every manual-review decision binds the frozen contract/queue identity it
# was made against, in addition to the row's own identity fields -- a
# decision recorded against a DIFFERENT (substituted) queue/contract is
# structurally distinguishable from one recorded against the real one.
REVIEW_RECORD_REQUIRED_FIELDS = frozenset({
    "queue_index", "dataset", "split", "slug", "observation_uuid", "photo_id", "sha256",
    "contract_content_sha256", "queue_byte_sha256", "queue_identity_order_sha256",
    "review_state", "label_plausibility", "duplicate_suspicion",
    "notes", "reviewer_id", "session_id", "reviewed_at_utc",
    "prev_record_hash", "record_hash",
})

REMEDIATION_RULE = (
    "Findings from the manual review do not modify northeast_final_test_v1 under "
    "any circumstance. Unusable images (no visible ant, corrupted file, clearly "
    "different organism) are counted and reported alongside the final-test result "
    "as a known limitation. Poor-quality but usable images remain in scope by "
    "design: they reflect real user input, and the confidence gate exists precisely "
    "to handle them. No image is removed, replaced, or relabeled."
)


def review_record_identity(record: dict) -> Any:
    return record.get("queue_index")


# -------------------------------------------------------- perceptual hash --
PHASH_BITS = 64
DHASH_BITS = 64
PHASH_MAX_HAMMING_DISTANCE = 10
DHASH_MAX_HAMMING_DISTANCE = 8
PHASH_RESIZE = (32, 32)
PHASH_KEEP = 8
DHASH_RESIZE = (9, 8)
N_DIHEDRAL_ORIENTATIONS = 8

CANDIDATE_RULE_DESCRIPTION = (
    "A pair is flagged when the minimum Hamming distance across all "
    f"orientation-hash combinations satisfies pHash <= {PHASH_MAX_HAMMING_DISTANCE} OR "
    f"dHash <= {DHASH_MAX_HAMMING_DISTANCE} (boundary equality included). These "
    "thresholds generate CANDIDATES only -- they never automatically declare "
    "duplication, and a negative scan does not prove that every crop or "
    "derivative is absent."
)

ADJUDICATION_LABELS = frozenset({
    "same_source_image",
    "different_photo_same_observation",
    "different_image",
    "uncertain",
})
# Adjudication records bind the exact candidate row (identity/domain/
# distances), not merely a pair_id -- so a record cannot be replayed against
# a substituted candidate-pairs report that reuses the same pair_id for a
# different underlying pair or different measured distances.
ADJUDICATION_RECORD_REQUIRED_FIELDS = frozenset({
    "pair_id", "identity_a", "identity_b", "domain", "phash_distance", "dhash_distance",
    "candidate_pairs_report_sha256", "scan_report_content_sha256",
    "label", "reviewer_id", "session_id", "reviewed_at_utc", "notes",
    "prev_record_hash", "record_hash",
})


def adjudication_record_identity(record: dict) -> Any:
    return record.get("pair_id")


def compute_pair_id(identity_a: str, identity_b: str) -> str:
    a, b = sorted([identity_a, identity_b])
    return sha256_bytes(f"{a}\0{b}".encode("utf-8"))


# ------------------------------------------------------ comparison domains --
# Every "part" this contract's comparison domains draw rows from. Kept as
# a flat list of 8 named parts (rather than nested dataset/split structure)
# so within-part and every cross-part pair can be generated mechanically and
# canonically, with no risk of A/B vs B/A double-counting.
DOMAIN_PART_EXPANSION_TRAIN = "expansion_train"
DOMAIN_PART_EXPANSION_DEVELOPMENT = "expansion_development"
DOMAIN_PART_FINAL_TEST = "final_test"
DOMAIN_PART_BENCHMARK_V1 = "benchmark_v1"
DOMAIN_PART_CALIBRATION_V1 = "calibration_v1"
DOMAIN_PART_UNKNOWN_TEST_V1 = "unknown_test_v1"
DOMAIN_PART_CALIBRATION_V2 = "calibration_v2"
DOMAIN_PART_UNKNOWN_TEST_V2 = "unknown_test_v2"

DOMAIN_PARTS = (
    DOMAIN_PART_EXPANSION_TRAIN,
    DOMAIN_PART_EXPANSION_DEVELOPMENT,
    DOMAIN_PART_FINAL_TEST,
    DOMAIN_PART_BENCHMARK_V1,
    DOMAIN_PART_CALIBRATION_V1,
    DOMAIN_PART_UNKNOWN_TEST_V1,
    DOMAIN_PART_CALIBRATION_V2,
    DOMAIN_PART_UNKNOWN_TEST_V2,
)
OTHER_EVIDENCE_SETS = (
    DOMAIN_PART_BENCHMARK_V1, DOMAIN_PART_CALIBRATION_V1, DOMAIN_PART_UNKNOWN_TEST_V1,
    DOMAIN_PART_CALIBRATION_V2, DOMAIN_PART_UNKNOWN_TEST_V2,
)


def canonical_domain_pair_key(part_a: str, part_b: str) -> tuple[str, str]:
    """Sorting the two part names is the ONE canonicalization rule: (a, b)
    and (b, a) always produce the identical key, so a cross domain can never
    be generated -- or looked up -- twice under two different names."""
    return tuple(sorted([part_a, part_b]))


def build_comparison_domains() -> list[dict]:
    """Mechanically generates: one WITHIN-set domain per part (8), plus
    every unordered CROSS-set pair of distinct parts (C(8,2) = 28) -- 36
    domains total, each reported separately, each named canonically so A/B
    and B/A can never both appear. This single generator is what
    "train vs development", "train/development vs each frozen evidence
    set", "final_test vs each previously used evidence set", and "every
    pairwise cross-evaluation-set comparison" all reduce to -- there is
    nothing left to enumerate by hand."""
    domains = []
    for part in DOMAIN_PARTS:
        domains.append({
            "name": f"within_{part}",
            "kind": "within",
            "parts": [part],
        })
    for part_a, part_b in itertools.combinations(DOMAIN_PARTS, 2):
        a, b = canonical_domain_pair_key(part_a, part_b)
        domains.append({
            "name": f"{a}_vs_{b}",
            "kind": "cross",
            "parts": [a, b],
        })
    domains.sort(key=lambda d: d["name"])
    return domains


COMPARISON_DOMAINS = tuple(build_comparison_domains())
assert len(COMPARISON_DOMAINS) == len(DOMAIN_PARTS) + (len(DOMAIN_PARTS) * (len(DOMAIN_PARTS) - 1)) // 2  # 8 + 28 = 36


def _stop_domain_names() -> frozenset[str]:
    """Every CROSS domain involving final_test paired with train,
    development, or any of the 5 other evidence sets mandates a stop before
    northeast_final_test_v1 model inference on a confirmed same-source-image
    adjudication or a metadata-leakage finding. within_final_test itself,
    and every other cross/within domain, does not."""
    stop_partners = (DOMAIN_PART_EXPANSION_TRAIN, DOMAIN_PART_EXPANSION_DEVELOPMENT) + OTHER_EVIDENCE_SETS
    names = set()
    for partner in stop_partners:
        a, b = canonical_domain_pair_key(DOMAIN_PART_FINAL_TEST, partner)
        names.add(f"{a}_vs_{b}")
    return frozenset(names)


STOP_BEFORE_INFERENCE_DOMAINS = _stop_domain_names()
assert len(STOP_BEFORE_INFERENCE_DOMAINS) == 7
assert STOP_BEFORE_INFERENCE_DOMAINS <= {d["name"] for d in COMPARISON_DOMAINS}

STOP_REASON_SAME_SOURCE_IMAGE = "confirmed_same_source_image"
STOP_REASON_MATCHING_OBSERVATION_UUID = "matching_observation_uuid"


# ------------------------------------------------- image layout, per part --
# The on-disk image directory layout is NOT uniform across all 8 domain
# parts, verified against the real repository:
#   expansion_train / expansion_development / final_test:
#     data/{dataset}/clean/{slug}/{photo_id}.{ext}
#   every one of the 5 other-evidence parts:
#     data/{dataset}/{slug}/{photo_id}.{ext}          (no "clean" subdir)
# Bound here per DOMAIN PART (not merely per dataset) so both consumers
# resolve every row's path from this one frozen mapping -- never from a
# hardcoded "clean" assumption -- and a changed layout fails closed via
# validate_contract_structure()/load_and_verify_contract() rather than
# silently mis-resolving (or ambiguously matching) an image path.
IMAGE_LAYOUT_CLEAN_SUBDIR = "clean"
IMAGE_LAYOUT_DIRECT = None  # no intermediate subdirectory

DOMAIN_PART_IMAGE_LAYOUTS = {
    DOMAIN_PART_EXPANSION_TRAIN: {"dataset": EXPANSION_DATASET, "subdir": IMAGE_LAYOUT_CLEAN_SUBDIR},
    DOMAIN_PART_EXPANSION_DEVELOPMENT: {"dataset": EXPANSION_DATASET, "subdir": IMAGE_LAYOUT_CLEAN_SUBDIR},
    DOMAIN_PART_FINAL_TEST: {"dataset": FINAL_TEST_DATASET, "subdir": IMAGE_LAYOUT_CLEAN_SUBDIR},
    DOMAIN_PART_BENCHMARK_V1: {"dataset": DOMAIN_PART_BENCHMARK_V1, "subdir": IMAGE_LAYOUT_DIRECT},
    DOMAIN_PART_CALIBRATION_V1: {"dataset": DOMAIN_PART_CALIBRATION_V1, "subdir": IMAGE_LAYOUT_DIRECT},
    DOMAIN_PART_UNKNOWN_TEST_V1: {"dataset": DOMAIN_PART_UNKNOWN_TEST_V1, "subdir": IMAGE_LAYOUT_DIRECT},
    DOMAIN_PART_CALIBRATION_V2: {"dataset": DOMAIN_PART_CALIBRATION_V2, "subdir": IMAGE_LAYOUT_DIRECT},
    DOMAIN_PART_UNKNOWN_TEST_V2: {"dataset": DOMAIN_PART_UNKNOWN_TEST_V2, "subdir": IMAGE_LAYOUT_DIRECT},
}
assert set(DOMAIN_PART_IMAGE_LAYOUTS) == set(DOMAIN_PARTS)
DOMAIN_PART_IMAGE_LAYOUT_ENTRY_KEYS = frozenset({"dataset", "subdir"})


def domain_part_for_dataset_and_split(dataset: str, split: str) -> str:
    """The 3 queue-backed domain parts (expansion_train/development,
    final_test) are only distinguishable from a queue row's (dataset,
    split) pair -- this is the one place that mapping lives, so both
    manual_review_tool.py and scan_perceptual_duplicates.py resolve a
    queue row's image the same way."""
    if dataset == FINAL_TEST_DATASET:
        return DOMAIN_PART_FINAL_TEST
    if dataset == EXPANSION_DATASET:
        if split == EXPANSION_TRAIN_SPLIT:
            return DOMAIN_PART_EXPANSION_TRAIN
        if split == EXPANSION_DEV_SPLIT:
            return DOMAIN_PART_EXPANSION_DEVELOPMENT
    raise ImageResolutionError(f"no domain part for dataset={dataset!r} split={split!r}")


def resolve_image_path(image_root: Path, image_layouts: dict, domain_part: str, slug: str,
                       photo_id: Any) -> Path:
    """The ONE shared deterministic image-path resolver used by both
    manual_review_tool.py and scan_perceptual_duplicates.py, so the two
    tools can never disagree about which file a queue/scan row refers to.

    `image_layouts` MUST be the verified contract's own
    `domain_part_image_layouts` mapping (e.g.
    `verified["contract"]["content"]["domain_part_image_layouts"]`) --
    never DOMAIN_PART_IMAGE_LAYOUTS directly -- so a changed/substituted
    contract's layout rule is what actually governs resolution, not a
    hardcoded module constant. Never opens the file -- `Path.exists()`
    only."""
    layout = image_layouts.get(domain_part)
    if not isinstance(layout, dict) or set(layout) != DOMAIN_PART_IMAGE_LAYOUT_ENTRY_KEYS:
        raise ImageResolutionError(f"no verified image layout for domain part {domain_part!r}")
    dataset, subdir = layout["dataset"], layout["subdir"]
    numeric_photo_id = str(int(photo_id))
    base_dir = Path(image_root) / dataset / (subdir if subdir else "") / slug
    candidates = [base_dir / f"{numeric_photo_id}{ext}" for ext in IMAGE_EXTENSIONS_ORDERED]
    matches = [c for c in candidates if c.exists()]
    if not matches:
        raise ImageResolutionError(
            f"no image file found for {domain_part}/{slug}/{numeric_photo_id} -- tried "
            f"{[str(c) for c in candidates]}"
        )
    if len(matches) > 1:
        raise ImageResolutionError(
            f"ambiguous image file for {domain_part}/{slug}/{numeric_photo_id}: multiple "
            f"candidates exist: {[str(m) for m in matches]}"
        )
    return matches[0]


# --------------------------------------------------------- contract build --
def build_contract_content(*, queue_artifact: dict, summary_artifact: dict,
                           implementation_sources: dict) -> dict:
    """Pure function, no filesystem access: `queue_artifact`,
    `summary_artifact`, and `implementation_sources` are computed by the
    CALLER (freeze_manual_review_contract.py) from the actual files on disk
    -- this function only assembles them alongside the constants above into
    the one frozen content dict."""
    return {
        "seed": SEED,
        "approved_species_slugs": list(APPROVED_SPECIES_SLUGS),
        "source_manifests": {k: dict(v) for k, v in FROZEN_SOURCE_MANIFESTS.items()},
        "expansion_scan_split_row_counts": dict(EXPANSION_SCAN_SPLIT_ROW_COUNTS),
        "other_evidence_manifests": {k: dict(v) for k, v in OTHER_EVIDENCE_MANIFESTS.items()},
        "domain_part_image_layouts": {k: dict(v) for k, v in DOMAIN_PART_IMAGE_LAYOUTS.items()},
        "queue_composition": {
            "final_test_dataset": FINAL_TEST_DATASET,
            "final_test_manifest_path": FINAL_TEST_MANIFEST_REL_PATH,
            "final_test_split": FINAL_TEST_SPLIT,
            "n_final_test_rows": N_FINAL_TEST_ROWS,
            "expansion_dataset": EXPANSION_DATASET,
            "expansion_manifest_path": EXPANSION_MANIFEST_REL_PATH,
            "expansion_splits": list(EXPANSION_SPLITS),
            "n_new_species": N_NEW_SPECIES,
            "per_slug_per_split": PER_SLUG_PER_SPLIT,
            "n_sampled_rows": N_SAMPLED_ROWS,
            "n_queue_rows": N_QUEUE_ROWS,
        },
        "session_policy": {
            "n_suggested_sessions": N_SUGGESTED_SESSIONS,
            "rows_per_suggested_session": ROWS_PER_SUGGESTED_SESSION,
            "max_decisions_per_session": MAX_DECISIONS_PER_SESSION,
            "note": "The tool remains pausable and may use additional session IDs during "
                    "actual review, but no single session may record more than "
                    f"{MAX_DECISIONS_PER_SESSION} decisions.",
        },
        "queue_artifact": dict(queue_artifact),
        "summary_artifact": dict(summary_artifact),
        "queue_row_required_fields": list(QUEUE_ROW_REQUIRED_FIELDS),
        "review_states": sorted(REVIEW_STATES),
        "label_plausibility_values": sorted(LABEL_PLAUSIBILITY_VALUES),
        "duplicate_suspicion_values": sorted(DUPLICATE_SUSPICION_VALUES),
        "review_record_required_fields": sorted(REVIEW_RECORD_REQUIRED_FIELDS),
        "remediation_rule": REMEDIATION_RULE,
        "perceptual_hash": {
            "phash": {
                "resize": list(PHASH_RESIZE), "resample": "Pillow LANCZOS",
                "transform": "explicit orthonormal 2-D DCT-II, float64 cosine matrix",
                "coefficients": "top-left 8x8 (row-major)",
                "threshold": "median of the 64 coefficients; bit=1 iff coefficient > median",
                "bits": PHASH_BITS,
            },
            "dhash": {
                "resize": list(DHASH_RESIZE), "resample": "Pillow LANCZOS",
                "rule": "bit=1 iff left pixel > right pixel",
                "bits": DHASH_BITS,
            },
            "preprocessing": "verify source bytes against the manifest sha256; apply EXIF "
                             "transpose; convert deterministically to grayscale; generate all "
                             f"{N_DIHEDRAL_ORIENTATIONS} dihedral orientations (4 rotations and "
                             "their horizontal mirrors) before hashing",
            "orientations": N_DIHEDRAL_ORIENTATIONS,
            "thresholds": {
                "phash_max_hamming_distance": PHASH_MAX_HAMMING_DISTANCE,
                "dhash_max_hamming_distance": DHASH_MAX_HAMMING_DISTANCE,
            },
            "candidate_rule": CANDIDATE_RULE_DESCRIPTION,
            "search": "exact radius-search index (BK-tree), one per hash type, over every "
                      "orientation hash of the query side, queried against an index built "
                      "from every orientation hash of the indexed side; a brute-force "
                      "reference implementation is retained for small synthetic tests only "
                      "and proven equal to the indexed output.",
        },
        "adjudication_labels": sorted(ADJUDICATION_LABELS),
        "adjudication_label_meanings": {
            "same_source_image": "resized, recompressed, cropped, rotated, mirrored, or "
                                 "otherwise re-uploaded derivatives of the same source "
                                 "photograph are duplicates",
            "different_photo_same_observation": "a genuinely different photograph from the "
                                                "same observation is NOT a perceptual duplicate",
            "different_image": "unrelated images that merely landed as a candidate",
            "uncertain": "adjudicator could not confidently decide",
        },
        "adjudication_record_required_fields": sorted(ADJUDICATION_RECORD_REQUIRED_FIELDS),
        "metadata_leakage": "detected/reported independently of perceptual duplication: "
                            "identical observation_uuid values across datasets/splits, even "
                            "when their photographs differ",
        "comparison_domains": [dict(d) for d in COMPARISON_DOMAINS],
        "stop_before_inference_domains": sorted(STOP_BEFORE_INFERENCE_DOMAINS),
        "stop_before_inference_reasons": [STOP_REASON_SAME_SOURCE_IMAGE, STOP_REASON_MATCHING_OBSERVATION_UUID],
        "mutation_policy": "findings never mutate a manifest or an image -- no deletion, "
                           "replacement, relabeling, or silent substitution. "
                           "unusable/poor-quality findings are reported and final evaluation "
                           "may proceed. A confirmed train/development <-> "
                           "northeast_final_test_v1 same-source-image collision, a matching "
                           "observation_uuid across those sets, or any final-test collision "
                           "with another previously used evidence set each mandates a stop "
                           "for reviewed interpretation before northeast_final_test_v1 model "
                           "inference.",
        "implementation_sources": {k: dict(v) for k, v in implementation_sources.items()},
        "approved_output_paths": dict(APPROVED_OUTPUT_PATHS),
    }


# ------------------------------------------------------------- validation --
CONTRACT_REQUIRED_TOP_KEYS = frozenset({"schema_version", "status", "content", "content_sha256", "generation"})
CONTRACT_GENERATION_ALLOWED_KEYS = frozenset({"note", "generator"})

CONTENT_REQUIRED_KEYS = frozenset({
    "seed", "approved_species_slugs", "source_manifests", "expansion_scan_split_row_counts",
    "other_evidence_manifests", "domain_part_image_layouts", "queue_composition", "session_policy",
    "queue_artifact", "summary_artifact", "queue_row_required_fields", "review_states",
    "label_plausibility_values", "duplicate_suspicion_values", "review_record_required_fields",
    "remediation_rule", "perceptual_hash", "adjudication_labels", "adjudication_label_meanings",
    "adjudication_record_required_fields", "metadata_leakage", "comparison_domains",
    "stop_before_inference_domains", "stop_before_inference_reasons", "mutation_policy",
    "implementation_sources", "approved_output_paths",
})

SOURCE_MANIFEST_ENTRY_KEYS = frozenset({"path", "byte_sha256", "row_count"})
QUEUE_ARTIFACT_ENTRY_KEYS = frozenset({"path", "byte_sha256", "row_count", "identity_order_sha256"})
SUMMARY_ARTIFACT_ENTRY_KEYS = frozenset({"path", "byte_sha256", "content_sha256"})


def _check_exact_keys(problems: list[str], label: str, actual: Any, required: frozenset) -> bool:
    if not isinstance(actual, dict):
        problems.append(f"{label} is not a JSON object")
        return False
    actual_keys = set(actual)
    missing = required - actual_keys
    extra = actual_keys - required
    ok = True
    if missing:
        problems.append(f"{label} missing key(s): {sorted(missing)}")
        ok = False
    if extra:
        problems.append(f"{label} has unexpected key(s): {sorted(extra)}")
        ok = False
    return ok


def validate_contract_structure(contract: Any) -> list[str]:
    """Pure, no filesystem access: rejects missing/extra keys at every
    level, bool-as-int, malformed hashes, invalid enums, and incorrect
    counts. Returns a list of problems (empty == structurally valid).
    NEVER raises -- arbitrary JSON (wrong types nested anywhere) always
    produces a controlled problem list, not a TypeError/KeyError."""
    problems: list[str] = []
    if not isinstance(contract, dict):
        return ["contract is not a JSON object"]

    _check_exact_keys(problems, "contract", contract, CONTRACT_REQUIRED_TOP_KEYS)

    sv = contract.get("schema_version")
    if not is_strict_int(sv) or sv != SCHEMA_VERSION:
        problems.append(f"schema_version must be strict int {SCHEMA_VERSION}, got {sv!r}")

    if contract.get("status") != CONTRACT_STATUS_FROZEN:
        problems.append(f"status must be exactly {CONTRACT_STATUS_FROZEN!r}, got {contract.get('status')!r}")

    generation = contract.get("generation")
    if not isinstance(generation, dict):
        problems.append("generation is not a JSON object")
    else:
        extra_gen = set(generation) - CONTRACT_GENERATION_ALLOWED_KEYS
        if extra_gen:
            problems.append(f"generation has unexpected key(s): {sorted(extra_gen)}")

    content = contract.get("content")
    if not isinstance(content, dict):
        problems.append("content is not a JSON object")
        return problems

    recorded_hash = contract.get("content_sha256")
    if not is_sha256_hex(recorded_hash):
        problems.append(f"content_sha256 is not a 64-char lowercase hex string: {recorded_hash!r}")
    else:
        try:
            recomputed = compute_content_sha256(content)
        except (TypeError, ValueError) as exc:
            problems.append(f"content is not canonicalizable: {exc}")
        else:
            if recorded_hash != recomputed:
                problems.append(f"content_sha256 mismatch: recorded={recorded_hash} recomputed={recomputed}")

    _check_exact_keys(problems, "content", content, CONTENT_REQUIRED_KEYS)

    if content.get("seed") != SEED or not is_strict_int(content.get("seed")):
        problems.append(f"content.seed must be strict int {SEED}, got {content.get('seed')!r}")

    slugs = content.get("approved_species_slugs")
    if not isinstance(slugs, list) or any(not isinstance(s, str) for s in slugs):
        problems.append("content.approved_species_slugs must be a list of strings")
    elif tuple(slugs) != APPROVED_SPECIES_SLUGS:
        problems.append(f"content.approved_species_slugs must be exactly {list(APPROVED_SPECIES_SLUGS)!r}, "
                        f"got {slugs!r}")

    source_manifests = content.get("source_manifests")
    if not isinstance(source_manifests, dict) or set(source_manifests) != set(FROZEN_SOURCE_MANIFESTS):
        problems.append(f"content.source_manifests keys must be exactly "
                        f"{sorted(FROZEN_SOURCE_MANIFESTS)}, got "
                        f"{sorted(source_manifests) if isinstance(source_manifests, dict) else source_manifests!r}")
    else:
        for name, expected in FROZEN_SOURCE_MANIFESTS.items():
            entry = source_manifests.get(name)
            if not _check_exact_keys(problems, f"content.source_manifests[{name!r}]", entry, SOURCE_MANIFEST_ENTRY_KEYS):
                continue
            if entry.get("path") != expected["path"]:
                problems.append(f"content.source_manifests[{name!r}].path must be {expected['path']!r}")
            if not is_sha256_hex(entry.get("byte_sha256")) or entry.get("byte_sha256") != expected["byte_sha256"]:
                problems.append(f"content.source_manifests[{name!r}].byte_sha256 must be {expected['byte_sha256']!r}")
            if not is_strict_int(entry.get("row_count")) or entry.get("row_count") != expected["row_count"]:
                problems.append(f"content.source_manifests[{name!r}].row_count must be {expected['row_count']!r}")

    if content.get("expansion_scan_split_row_counts") != dict(EXPANSION_SCAN_SPLIT_ROW_COUNTS):
        problems.append(f"content.expansion_scan_split_row_counts must be exactly "
                        f"{dict(EXPANSION_SCAN_SPLIT_ROW_COUNTS)!r}")

    other_manifests = content.get("other_evidence_manifests")
    if not isinstance(other_manifests, dict) or set(other_manifests) != set(OTHER_EVIDENCE_MANIFESTS):
        problems.append(f"content.other_evidence_manifests keys must be exactly "
                        f"{sorted(OTHER_EVIDENCE_MANIFESTS)}, got "
                        f"{sorted(other_manifests) if isinstance(other_manifests, dict) else other_manifests!r}")
    else:
        for name, expected in OTHER_EVIDENCE_MANIFESTS.items():
            entry = other_manifests.get(name)
            if not _check_exact_keys(problems, f"content.other_evidence_manifests[{name!r}]", entry,
                                     OTHER_EVIDENCE_MANIFEST_ENTRY_KEYS):
                continue
            for field in ("path", "provenance_json_path"):
                if entry.get(field) != expected[field]:
                    problems.append(f"content.other_evidence_manifests[{name!r}].{field} must be "
                                    f"{expected[field]!r}")
            for field in ("byte_sha256", "provenance_json_byte_sha256"):
                if not is_sha256_hex(entry.get(field)) or entry.get(field) != expected[field]:
                    problems.append(f"content.other_evidence_manifests[{name!r}].{field} must be "
                                    f"{expected[field]!r}")
            if not is_strict_int(entry.get("row_count")) or entry.get("row_count") != expected["row_count"]:
                problems.append(f"content.other_evidence_manifests[{name!r}].row_count must be "
                                f"{expected['row_count']!r}")

    layouts = content.get("domain_part_image_layouts")
    if not isinstance(layouts, dict) or set(layouts) != set(DOMAIN_PART_IMAGE_LAYOUTS):
        problems.append(f"content.domain_part_image_layouts keys must be exactly "
                        f"{sorted(DOMAIN_PART_IMAGE_LAYOUTS)}, got "
                        f"{sorted(layouts) if isinstance(layouts, dict) else layouts!r}")
    else:
        for part, expected in DOMAIN_PART_IMAGE_LAYOUTS.items():
            entry = layouts.get(part)
            if not _check_exact_keys(problems, f"content.domain_part_image_layouts[{part!r}]", entry,
                                     DOMAIN_PART_IMAGE_LAYOUT_ENTRY_KEYS):
                continue
            if entry.get("dataset") != expected["dataset"]:
                problems.append(f"content.domain_part_image_layouts[{part!r}].dataset must be "
                                f"{expected['dataset']!r}")
            if entry.get("subdir") != expected["subdir"]:
                problems.append(f"content.domain_part_image_layouts[{part!r}].subdir must be "
                                f"{expected['subdir']!r}")

    queue_artifact = content.get("queue_artifact")
    if _check_exact_keys(problems, "content.queue_artifact", queue_artifact, QUEUE_ARTIFACT_ENTRY_KEYS):
        if queue_artifact.get("path") != QUEUE_ARTIFACT_REL_PATH:
            problems.append(f"content.queue_artifact.path must be {QUEUE_ARTIFACT_REL_PATH!r}")
        if not is_sha256_hex(queue_artifact.get("byte_sha256")):
            problems.append("content.queue_artifact.byte_sha256 is not a 64-char lowercase hex string")
        if not is_strict_int(queue_artifact.get("row_count")) or queue_artifact.get("row_count") != N_QUEUE_ROWS:
            problems.append(f"content.queue_artifact.row_count must be strict int {N_QUEUE_ROWS}")
        if not is_sha256_hex(queue_artifact.get("identity_order_sha256")):
            problems.append("content.queue_artifact.identity_order_sha256 is not a 64-char lowercase hex string")

    summary_artifact = content.get("summary_artifact")
    if _check_exact_keys(problems, "content.summary_artifact", summary_artifact, SUMMARY_ARTIFACT_ENTRY_KEYS):
        if summary_artifact.get("path") != SUMMARY_ARTIFACT_REL_PATH:
            problems.append(f"content.summary_artifact.path must be {SUMMARY_ARTIFACT_REL_PATH!r}")
        if not is_sha256_hex(summary_artifact.get("byte_sha256")):
            problems.append("content.summary_artifact.byte_sha256 is not a 64-char lowercase hex string")
        if not is_sha256_hex(summary_artifact.get("content_sha256")):
            problems.append("content.summary_artifact.content_sha256 is not a 64-char lowercase hex string")

    if content.get("review_states") != sorted(REVIEW_STATES):
        problems.append("content.review_states must be exactly the frozen review-state enum")
    if content.get("label_plausibility_values") != sorted(LABEL_PLAUSIBILITY_VALUES):
        problems.append("content.label_plausibility_values must be exactly the frozen enum")
    if content.get("duplicate_suspicion_values") != sorted(DUPLICATE_SUSPICION_VALUES):
        problems.append("content.duplicate_suspicion_values must be exactly the frozen enum")
    if content.get("adjudication_labels") != sorted(ADJUDICATION_LABELS):
        problems.append("content.adjudication_labels must be exactly the frozen enum")
    if content.get("remediation_rule") != REMEDIATION_RULE:
        problems.append("content.remediation_rule must match the frozen text verbatim")

    domains = content.get("comparison_domains")
    expected_domains = [dict(d) for d in COMPARISON_DOMAINS]
    if domains != expected_domains:
        problems.append("content.comparison_domains does not exactly match the frozen 36-domain set")
    if content.get("stop_before_inference_domains") != sorted(STOP_BEFORE_INFERENCE_DOMAINS):
        problems.append("content.stop_before_inference_domains does not match the frozen set")

    impl_sources = content.get("implementation_sources")
    if not isinstance(impl_sources, dict):
        problems.append("content.implementation_sources is not a JSON object")
    else:
        for name, entry in impl_sources.items():
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                problems.append(f"content.implementation_sources[{name!r}] must have exactly path and sha256")
            elif not is_sha256_hex(entry.get("sha256")):
                problems.append(f"content.implementation_sources[{name!r}].sha256 is malformed")

    if content.get("approved_output_paths") != dict(APPROVED_OUTPUT_PATHS):
        problems.append("content.approved_output_paths must match the frozen approved paths exactly")

    return problems


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_and_verify_contract(repo: Path) -> dict:
    """The ONE entry point every consumer calls FIRST. Loads
    training/manual_review_contract.json, validates its structure, then
    cross-checks it against the CURRENT source manifests, queue CSV, and
    summary JSON on disk -- byte-for-byte where the contract records a byte
    hash. Raises ContractError (never an uncaught exception) on any
    mismatch, including a substituted same-shaped queue/summary/contract."""
    repo = Path(repo)
    contract_path = repo / CONTRACT_REL_PATH
    if not contract_path.exists():
        raise ContractError(f"contract does not exist: {contract_path}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{contract_path} is not valid JSON: {exc}") from exc

    problems = validate_contract_structure(contract)
    if problems:
        raise ContractError(f"{contract_path} failed structural validation: {problems}")

    content = contract["content"]

    for name, expected in FROZEN_SOURCE_MANIFESTS.items():
        path = repo / expected["path"]
        if not path.exists():
            raise ContractError(f"source manifest does not exist: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected["byte_sha256"]:
            raise ContractError(f"{path} byte sha256 {actual_hash} != contract-bound {expected['byte_sha256']!r}")
        rows = _read_csv_rows(path)
        if len(rows) != expected["row_count"]:
            raise ContractError(f"{path} has {len(rows)} rows, contract expects {expected['row_count']}")

    for name, expected in OTHER_EVIDENCE_MANIFESTS.items():
        path = repo / expected["path"]
        if not path.exists():
            raise ContractError(f"other-evidence manifest does not exist: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected["byte_sha256"]:
            raise ContractError(f"{path} byte sha256 {actual_hash} != contract-bound "
                                f"{expected['byte_sha256']!r} -- this is not the approved manifest")
        rows = _read_csv_rows(path)
        if len(rows) != expected["row_count"]:
            raise ContractError(f"{path} has {len(rows)} rows, contract expects {expected['row_count']}")

        json_path = repo / expected["provenance_json_path"]
        if not json_path.exists():
            raise ContractError(f"provenance JSON does not exist: {json_path}")
        actual_json_hash = sha256_file(json_path)
        if actual_json_hash != expected["provenance_json_byte_sha256"]:
            raise ContractError(f"{json_path} byte sha256 {actual_json_hash} != contract-bound "
                                f"{expected['provenance_json_byte_sha256']!r} -- this is not the "
                                f"approved provenance file")

    queue_path = repo / QUEUE_ARTIFACT_REL_PATH
    if not queue_path.exists():
        raise ContractError(f"queue CSV does not exist: {queue_path}")
    queue_bytes = queue_path.read_bytes()
    queue_hash = sha256_bytes(queue_bytes)
    bound_queue = content["queue_artifact"]
    if queue_hash != bound_queue["byte_sha256"]:
        raise ContractError(f"{queue_path} byte sha256 {queue_hash} != contract-bound "
                            f"{bound_queue['byte_sha256']!r} -- this is not the approved queue")
    queue_rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    if len(queue_rows) != bound_queue["row_count"]:
        raise ContractError(f"{queue_path} has {len(queue_rows)} rows, contract expects {bound_queue['row_count']}")
    for row in queue_rows:
        row["queue_index"] = int(row["queue_index"])
        row["suggested_session"] = int(row["suggested_session"])
    queue_rows.sort(key=lambda r: r["queue_index"])
    identity_hash = compute_queue_identity_order_sha256(queue_rows)
    if identity_hash != bound_queue["identity_order_sha256"]:
        raise ContractError(f"{queue_path} identity-order sha256 {identity_hash} != contract-bound "
                            f"{bound_queue['identity_order_sha256']!r} -- rows have been reordered, "
                            f"substituted, or altered")

    summary_path = repo / SUMMARY_ARTIFACT_REL_PATH
    if not summary_path.exists():
        raise ContractError(f"summary JSON does not exist: {summary_path}")
    summary_bytes = summary_path.read_bytes()
    summary_byte_hash = sha256_bytes(summary_bytes)
    bound_summary = content["summary_artifact"]
    if summary_byte_hash != bound_summary["byte_sha256"]:
        raise ContractError(f"{summary_path} byte sha256 {summary_byte_hash} != contract-bound "
                            f"{bound_summary['byte_sha256']!r}")
    try:
        summary = json.loads(summary_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{summary_path} is not valid JSON: {exc}") from exc
    if summary.get("content_sha256") != bound_summary["content_sha256"]:
        raise ContractError(f"{summary_path} content_sha256 {summary.get('content_sha256')!r} != "
                            f"contract-bound {bound_summary['content_sha256']!r}")

    return {"contract": contract, "queue_rows": queue_rows, "summary": summary,
            "queue_path": queue_path, "summary_path": summary_path, "contract_path": contract_path}


# ------------------------------------------------------------ scan reports --
# Shared strict schema validators for the five --scan output reports, used
# by BOTH scan_perceptual_duplicates.py --check and adjudicate_pairs.py --
# a self-consistent content_sha256 alone is never sufficient: every
# validator below also checks exact keys, the contract-hash binding, the
# relevant cross-report hash binding(s), and (for candidate pairs and stop
# status) the semantic derivation, not merely the shape.
REPORT_TOP_KEYS = frozenset({"schema_version", "content", "content_sha256"})

HASHES_REPORT_CONTENT_KEYS = frozenset({"contract_content_sha256", "pillow_version", "numpy_version", "hashes"})
HASH_ENTRY_KEYS = frozenset({"phashes", "dhashes", "sha256"})

LEAKAGE_REPORT_CONTENT_KEYS = frozenset({"contract_content_sha256", "findings"})
LEAKAGE_FINDING_KEYS = frozenset({"observation_uuid", "domains", "entries"})
LEAKAGE_ENTRY_KEYS = frozenset({"domain", "dataset", "slug", "split", "photo_id", "identity"})

CANDIDATE_PAIRS_REPORT_CONTENT_KEYS = frozenset(
    {"contract_content_sha256", "hashes_report_content_sha256", "domains"}
)
CANDIDATE_PAIR_KEYS = frozenset({"pair_id", "identity_a", "identity_b", "phash_distance", "dhash_distance"})

DOMAIN_SUMMARY_REPORT_CONTENT_KEYS = frozenset({
    "contract_content_sha256", "candidate_pairs_report_content_sha256",
    "metadata_leakage_report_content_sha256", "domains",
})
DOMAIN_SUMMARY_ENTRY_KEYS = frozenset({"a_count", "b_count", "candidate_count", "leakage_count"})

STOP_STATUS_REPORT_CONTENT_KEYS = frozenset({
    "contract_content_sha256", "domain_summary_report_content_sha256", "domains",
    "overall_stop_before_inference",
})
STOP_STATUS_ENTRY_KEYS = frozenset({"stop_before_inference", "reasons"})

_ALL_DOMAIN_NAMES = frozenset(d["name"] for d in COMPARISON_DOMAINS)


def _validate_report_envelope(report: Any, label: str, content_keys: frozenset) -> tuple[list[str], dict | None]:
    problems: list[str] = []
    if not isinstance(report, dict):
        return [f"{label} is not a JSON object"], None
    _check_exact_keys(problems, label, report, REPORT_TOP_KEYS)
    sv = report.get("schema_version")
    if not is_strict_int(sv) or sv != SCHEMA_VERSION:
        problems.append(f"{label}.schema_version must be strict int {SCHEMA_VERSION}")
    content = report.get("content")
    if not isinstance(content, dict):
        problems.append(f"{label}.content is not a JSON object")
        return problems, None
    recorded = report.get("content_sha256")
    if not is_sha256_hex(recorded):
        problems.append(f"{label}.content_sha256 is not a 64-char lowercase hex string")
    else:
        try:
            recomputed = compute_content_sha256(content)
        except (TypeError, ValueError) as exc:
            problems.append(f"{label}.content is not canonicalizable: {exc}")
        else:
            if recomputed != recorded:
                problems.append(f"{label}.content_sha256 mismatch: recorded={recorded} recomputed={recomputed}")
    _check_exact_keys(problems, f"{label}.content", content, content_keys)
    return problems, content


def validate_hashes_report(report: Any, contract: dict) -> list[str]:
    problems, content = _validate_report_envelope(report, "perceptual_hashes_report", HASHES_REPORT_CONTENT_KEYS)
    if content is None:
        return problems
    if content.get("contract_content_sha256") != contract.get("content_sha256"):
        problems.append("perceptual_hashes_report.content.contract_content_sha256 does not match "
                        "the verified contract")
    hashes = content.get("hashes")
    if not isinstance(hashes, dict):
        problems.append("perceptual_hashes_report.content.hashes is not a JSON object")
    else:
        for identity, entry in hashes.items():
            if not isinstance(identity, str) or not identity:
                problems.append(f"perceptual_hashes_report.content.hashes has a non-string/blank "
                                f"key: {identity!r}")
            if not _check_exact_keys(problems, f"perceptual_hashes_report.content.hashes[{identity!r}]",
                                     entry, HASH_ENTRY_KEYS):
                continue
            for hk in ("phashes", "dhashes"):
                values = entry.get(hk)
                if not (isinstance(values, list) and len(values) == N_DIHEDRAL_ORIENTATIONS
                        and all(is_strict_int(v) and 0 <= v < (1 << PHASH_BITS) for v in values)):
                    problems.append(f"perceptual_hashes_report.content.hashes[{identity!r}].{hk} must "
                                    f"be a list of {N_DIHEDRAL_ORIENTATIONS} strict non-negative ints "
                                    f"< 2**{PHASH_BITS}")
            if not is_sha256_hex(entry.get("sha256")):
                problems.append(f"perceptual_hashes_report.content.hashes[{identity!r}].sha256 is malformed")
    return problems


def validate_metadata_leakage_report(report: Any, contract: dict) -> list[str]:
    problems, content = _validate_report_envelope(report, "metadata_leakage_report", LEAKAGE_REPORT_CONTENT_KEYS)
    if content is None:
        return problems
    if content.get("contract_content_sha256") != contract.get("content_sha256"):
        problems.append("metadata_leakage_report.content.contract_content_sha256 does not match "
                        "the verified contract")
    findings = content.get("findings")
    if not isinstance(findings, list):
        problems.append("metadata_leakage_report.content.findings is not a list")
    else:
        for i, finding in enumerate(findings):
            if not _check_exact_keys(problems, f"metadata_leakage_report.content.findings[{i}]",
                                     finding, LEAKAGE_FINDING_KEYS):
                continue
            if not isinstance(finding.get("observation_uuid"), str) or not finding["observation_uuid"]:
                problems.append(f"metadata_leakage_report.content.findings[{i}].observation_uuid invalid")
            domains, entries = finding.get("domains"), finding.get("entries")
            if not (isinstance(domains, list) and len(domains) >= 2 and domains == sorted(domains)):
                problems.append(f"metadata_leakage_report.content.findings[{i}].domains must be a "
                                f"sorted list of >= 2 domain names")
            if not isinstance(entries, list) or len(entries) < 2:
                problems.append(f"metadata_leakage_report.content.findings[{i}].entries must have >= 2 entries")
            else:
                for entry in entries:
                    if not isinstance(entry, dict) or set(entry) != LEAKAGE_ENTRY_KEYS:
                        problems.append(f"metadata_leakage_report.content.findings[{i}] has a malformed entry")
    return problems


def validate_candidate_pairs_report(report: Any, contract: dict, hashes_report_content_sha256: Any) -> list[str]:
    problems, content = _validate_report_envelope(report, "candidate_pairs_report",
                                                  CANDIDATE_PAIRS_REPORT_CONTENT_KEYS)
    if content is None:
        return problems
    if content.get("contract_content_sha256") != contract.get("content_sha256"):
        problems.append("candidate_pairs_report.content.contract_content_sha256 does not match "
                        "the verified contract")
    if content.get("hashes_report_content_sha256") != hashes_report_content_sha256:
        problems.append("candidate_pairs_report.content.hashes_report_content_sha256 does not match "
                        "the verified hashes report")
    domains = content.get("domains")
    if not isinstance(domains, dict) or set(domains) != _ALL_DOMAIN_NAMES:
        problems.append(f"candidate_pairs_report.content.domains keys must be exactly the frozen "
                        f"{len(_ALL_DOMAIN_NAMES)} domain names")
        return problems
    domain_by_part_kind = {d["name"]: d for d in COMPARISON_DOMAINS}
    global_pair_ids: dict[str, str] = {}
    for domain_name, pairs in domains.items():
        if not isinstance(pairs, list):
            problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}] is not a list")
            continue
        is_within = domain_by_part_kind.get(domain_name, {}).get("kind") == "within"
        seen_in_domain: set = set()
        for j, pair in enumerate(pairs):
            if not isinstance(pair, dict) or set(pair) != CANDIDATE_PAIR_KEYS:
                problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}][{j}] malformed")
                continue
            pid = pair.get("pair_id")
            if pid != compute_pair_id(str(pair.get("identity_a", "")), str(pair.get("identity_b", ""))):
                problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}][{j}].pair_id "
                                f"does not match compute_pair_id(identity_a, identity_b)")
            if pid in seen_in_domain:
                problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}] has duplicate "
                                f"pair_id {pid!r} within the same domain")
            seen_in_domain.add(pid)
            if pid in global_pair_ids and global_pair_ids[pid] != domain_name:
                problems.append(f"candidate_pairs_report pair_id {pid!r} appears in both "
                                f"{global_pair_ids[pid]!r} and {domain_name!r} -- duplicate across the "
                                f"complete report")
            global_pair_ids.setdefault(pid, domain_name)
            ia, ib = pair.get("identity_a"), pair.get("identity_b")
            if is_within and isinstance(ia, str) and isinstance(ib, str) and ia > ib:
                problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}][{j}] "
                                f"identity_a/identity_b are not in canonical sorted order")
            p, d = pair.get("phash_distance"), pair.get("dhash_distance")
            if not is_strict_int(p) or not (0 <= p <= PHASH_BITS):
                problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}][{j}].phash_distance invalid")
                p = None
            if not is_strict_int(d) or not (0 <= d <= DHASH_BITS):
                problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}][{j}].dhash_distance invalid")
                d = None
            if p is not None and d is not None and not (p <= PHASH_MAX_HAMMING_DISTANCE or d <= DHASH_MAX_HAMMING_DISTANCE):
                problems.append(f"candidate_pairs_report.content.domains[{domain_name!r}][{j}] does "
                                f"not satisfy the candidate rule (pHash<={PHASH_MAX_HAMMING_DISTANCE} "
                                f"OR dHash<={DHASH_MAX_HAMMING_DISTANCE})")
    return problems


def validate_domain_summary_report(report: Any, contract: dict, candidate_pairs_report: Any,
                                   leakage_report: Any) -> list[str]:
    problems, content = _validate_report_envelope(report, "domain_summary_report",
                                                  DOMAIN_SUMMARY_REPORT_CONTENT_KEYS)
    if content is None:
        return problems
    if content.get("contract_content_sha256") != contract.get("content_sha256"):
        problems.append("domain_summary_report.content.contract_content_sha256 does not match "
                        "the verified contract")
    cp_content_sha256 = candidate_pairs_report.get("content_sha256") if isinstance(candidate_pairs_report, dict) else None
    if content.get("candidate_pairs_report_content_sha256") != cp_content_sha256:
        problems.append("domain_summary_report.content.candidate_pairs_report_content_sha256 does "
                        "not match the verified candidate-pairs report")
    lk_content_sha256 = leakage_report.get("content_sha256") if isinstance(leakage_report, dict) else None
    if content.get("metadata_leakage_report_content_sha256") != lk_content_sha256:
        problems.append("domain_summary_report.content.metadata_leakage_report_content_sha256 does "
                        "not match the verified metadata-leakage report")

    domains = content.get("domains")
    if not isinstance(domains, dict) or set(domains) != _ALL_DOMAIN_NAMES:
        problems.append("domain_summary_report.content.domains keys must be exactly the frozen domain names")
        return problems

    cp_content = candidate_pairs_report.get("content") if isinstance(candidate_pairs_report, dict) else None
    cp_domains = cp_content.get("domains") if isinstance(cp_content, dict) else None
    cp_domains = cp_domains if isinstance(cp_domains, dict) else {}
    lk_content = leakage_report.get("content") if isinstance(leakage_report, dict) else None
    lk_findings = lk_content.get("findings") if isinstance(lk_content, dict) else None
    lk_findings = lk_findings if isinstance(lk_findings, list) else []

    for domain in COMPARISON_DOMAINS:
        name = domain["name"]
        entry = domains.get(name)
        if not isinstance(entry, dict) or set(entry) != DOMAIN_SUMMARY_ENTRY_KEYS:
            problems.append(f"domain_summary_report.content.domains[{name!r}] malformed")
            continue
        cp_list = cp_domains.get(name)
        expected_candidate_count = len(cp_list) if isinstance(cp_list, list) else None
        if expected_candidate_count is None or entry.get("candidate_count") != expected_candidate_count:
            problems.append(f"domain_summary_report.content.domains[{name!r}].candidate_count does "
                            f"not match the candidate-pairs report")
        if domain["kind"] == "within":
            expected_leakage_count = 0
        else:
            a, b = domain["parts"]
            expected_leakage_count = sum(
                1 for f in lk_findings
                if isinstance(f, dict) and {a, b} <= set(f.get("domains", []) if isinstance(f.get("domains"), list) else [])
            )
        if entry.get("leakage_count") != expected_leakage_count:
            problems.append(f"domain_summary_report.content.domains[{name!r}].leakage_count does not "
                            f"match the metadata-leakage report")
        for count_field in ("a_count", "b_count"):
            if not is_strict_int(entry.get(count_field)) or entry.get(count_field) < 0:
                problems.append(f"domain_summary_report.content.domains[{name!r}].{count_field} must "
                                f"be a non-negative strict int")
    return problems


def validate_stop_status_report(report: Any, contract: dict, domain_summary_report: Any) -> list[str]:
    problems, content = _validate_report_envelope(report, "stop_status_report", STOP_STATUS_REPORT_CONTENT_KEYS)
    if content is None:
        return problems
    if content.get("contract_content_sha256") != contract.get("content_sha256"):
        problems.append("stop_status_report.content.contract_content_sha256 does not match the verified contract")
    ds_content_sha256 = domain_summary_report.get("content_sha256") if isinstance(domain_summary_report, dict) else None
    if content.get("domain_summary_report_content_sha256") != ds_content_sha256:
        problems.append("stop_status_report.content.domain_summary_report_content_sha256 does not "
                        "match the verified domain-summary report")

    domains = content.get("domains")
    if not isinstance(domains, dict) or set(domains) != _ALL_DOMAIN_NAMES:
        problems.append("stop_status_report.content.domains keys must be exactly the frozen domain names")
        return problems

    ds_content = domain_summary_report.get("content") if isinstance(domain_summary_report, dict) else None
    ds_domains = ds_content.get("domains") if isinstance(ds_content, dict) else None
    ds_domains = ds_domains if isinstance(ds_domains, dict) else {}

    any_stop = False
    for name, entry in domains.items():
        if not isinstance(entry, dict) or set(entry) != STOP_STATUS_ENTRY_KEYS:
            problems.append(f"stop_status_report.content.domains[{name!r}] malformed")
            continue
        reasons = entry.get("reasons")
        valid_reason_set = {STOP_REASON_SAME_SOURCE_IMAGE, STOP_REASON_MATCHING_OBSERVATION_UUID}
        if not isinstance(reasons, list) or any(r not in valid_reason_set for r in reasons):
            problems.append(f"stop_status_report.content.domains[{name!r}].reasons contains an invalid reason")
            reasons = []
        if STOP_REASON_SAME_SOURCE_IMAGE in reasons:
            problems.append(f"stop_status_report.content.domains[{name!r}] claims a confirmed "
                            f"same_source_image stop, but --scan runs before any adjudication -- "
                            f"this reason can only be added later, by adjudicate_pairs.py, never by "
                            f"a freshly generated scan report")
        ds_entry = ds_domains.get(name)
        leakage_count = ds_entry.get("leakage_count") if isinstance(ds_entry, dict) else None
        expected_leakage_reason = bool(leakage_count) and name in STOP_BEFORE_INFERENCE_DOMAINS
        has_leakage_reason = STOP_REASON_MATCHING_OBSERVATION_UUID in reasons
        if expected_leakage_reason != has_leakage_reason:
            problems.append(f"stop_status_report.content.domains[{name!r}] leakage-based stop reason "
                            f"does not match the domain-summary report")
        expected_stop = bool(reasons)
        if entry.get("stop_before_inference") != expected_stop:
            problems.append(f"stop_status_report.content.domains[{name!r}].stop_before_inference "
                            f"does not match its own reasons")
        if expected_stop:
            any_stop = True

    if content.get("overall_stop_before_inference") != any_stop:
        problems.append("stop_status_report.content.overall_stop_before_inference does not match "
                        "the per-domain results")
    return problems


SCAN_REPORT_NAMES = (
    "perceptual_hashes_report", "metadata_leakage_report", "candidate_pairs_report",
    "domain_summary_report", "stop_status_report",
)


def validate_scan_report_bundle(reports: dict, contract: dict) -> list[str]:
    """Validates all five scan reports together, including every
    cross-report hash binding, in dependency order:
    hashes -> candidate_pairs -> domain_summary -> stop_status, with
    metadata_leakage feeding domain_summary directly. `reports` must have
    keys exactly SCAN_REPORT_NAMES."""
    if set(reports) != set(SCAN_REPORT_NAMES):
        return [f"reports bundle keys must be exactly {sorted(SCAN_REPORT_NAMES)}, got {sorted(reports)}"]

    problems: list[str] = []
    problems += validate_hashes_report(reports["perceptual_hashes_report"], contract)
    problems += validate_metadata_leakage_report(reports["metadata_leakage_report"], contract)
    hashes_report = reports["perceptual_hashes_report"]
    hashes_content_sha256 = hashes_report.get("content_sha256") if isinstance(hashes_report, dict) else None
    problems += validate_candidate_pairs_report(reports["candidate_pairs_report"], contract, hashes_content_sha256)
    problems += validate_domain_summary_report(reports["domain_summary_report"], contract,
                                               reports["candidate_pairs_report"],
                                               reports["metadata_leakage_report"])
    problems += validate_stop_status_report(reports["stop_status_report"], contract,
                                            reports["domain_summary_report"])
    return problems


# ----------------------------------------------- post-adjudication gate --
# A separate, contract-bound artifact from the scan-time stop_status_report:
# that report is produced BEFORE any human adjudication and must never claim
# a confirmed same_source_image stop. This one is produced AFTER the full
# adjudication ledger is complete, and is the actual required gate before
# northeast_final_test_v1 model inference -- it is the only artifact allowed
# to fold confirmed same_source_image decisions into the stop decision.
POST_ADJUDICATION_REPORT_CONTENT_KEYS = frozenset({
    "contract_content_sha256", "stop_status_report_content_sha256",
    "candidate_pairs_report_content_sha256", "adjudication_ledger_byte_sha256",
    "domains", "overall_stop_before_inference", "total_candidates", "adjudicated_count", "complete",
})
POST_ADJUDICATION_DOMAIN_ENTRY_KEYS = STOP_STATUS_ENTRY_KEYS


def validate_post_adjudication_report(report: Any, contract: dict, stop_status_report: Any,
                                      candidate_pairs_report: Any, adjudication_ledger_byte_sha256: Any) -> list[str]:
    """Strict schema + cross-binding validator for the post-adjudication
    finalization artifact. Does NOT re-derive confirmed-same-source
    findings from the ledger itself (that recomputation lives in
    finalize_stop_status.py, which has the ledger's parsed records) --
    this checks shape, bindings, `complete`, and every per-domain
    stop_before_inference/reasons value is internally consistent."""
    problems, content = _validate_report_envelope(report, "post_adjudication_stop_status_report",
                                                   POST_ADJUDICATION_REPORT_CONTENT_KEYS)
    if content is None:
        return problems
    if content.get("contract_content_sha256") != contract.get("content_sha256"):
        problems.append("post_adjudication_stop_status_report.content.contract_content_sha256 does not "
                        "match the verified contract")
    ss_content_sha256 = stop_status_report.get("content_sha256") if isinstance(stop_status_report, dict) else None
    if content.get("stop_status_report_content_sha256") != ss_content_sha256:
        problems.append("post_adjudication_stop_status_report.content.stop_status_report_content_sha256 "
                        "does not match the verified scan-time stop status report")
    cp_content_sha256 = candidate_pairs_report.get("content_sha256") if isinstance(candidate_pairs_report, dict) else None
    if content.get("candidate_pairs_report_content_sha256") != cp_content_sha256:
        problems.append("post_adjudication_stop_status_report.content.candidate_pairs_report_content_sha256 "
                        "does not match the verified candidate-pairs report")
    if content.get("adjudication_ledger_byte_sha256") != adjudication_ledger_byte_sha256:
        problems.append("post_adjudication_stop_status_report.content.adjudication_ledger_byte_sha256 does "
                        "not match the verified adjudication ledger")

    total = content.get("total_candidates")
    adjudicated = content.get("adjudicated_count")
    if not is_strict_int(total) or total < 0:
        problems.append("post_adjudication_stop_status_report.content.total_candidates must be a "
                        "non-negative strict int")
        total = None
    if not is_strict_int(adjudicated) or adjudicated < 0:
        problems.append("post_adjudication_stop_status_report.content.adjudicated_count must be a "
                        "non-negative strict int")
        adjudicated = None
    if total is not None and adjudicated is not None and adjudicated != total:
        problems.append(f"post_adjudication_stop_status_report.content.adjudicated_count {adjudicated} "
                        f"!= total_candidates {total} -- finalization requires every candidate adjudicated")
    if content.get("complete") is not True:
        problems.append("post_adjudication_stop_status_report.content.complete must be exactly true -- "
                        "this artifact is only ever published once adjudication is complete")

    domains = content.get("domains")
    if not isinstance(domains, dict) or set(domains) != _ALL_DOMAIN_NAMES:
        problems.append("post_adjudication_stop_status_report.content.domains keys must be exactly the "
                        "frozen domain names")
        return problems

    ss_content = stop_status_report.get("content") if isinstance(stop_status_report, dict) else None
    ss_domains = ss_content.get("domains") if isinstance(ss_content, dict) else None
    ss_domains = ss_domains if isinstance(ss_domains, dict) else {}

    any_stop = False
    for name, entry in domains.items():
        if not isinstance(entry, dict) or set(entry) != POST_ADJUDICATION_DOMAIN_ENTRY_KEYS:
            problems.append(f"post_adjudication_stop_status_report.content.domains[{name!r}] malformed")
            continue
        reasons = entry.get("reasons")
        valid_reason_set = {STOP_REASON_SAME_SOURCE_IMAGE, STOP_REASON_MATCHING_OBSERVATION_UUID}
        if not isinstance(reasons, list) or any(r not in valid_reason_set for r in reasons):
            problems.append(f"post_adjudication_stop_status_report.content.domains[{name!r}].reasons "
                            f"contains an invalid reason")
            reasons = []
        # The metadata-leakage stop from the scan-time report must be
        # PRESERVED -- never dropped when folding in confirmed decisions.
        ss_entry = ss_domains.get(name)
        ss_reasons = ss_entry.get("reasons") if isinstance(ss_entry, dict) else []
        ss_reasons = ss_reasons if isinstance(ss_reasons, list) else []
        if STOP_REASON_MATCHING_OBSERVATION_UUID in ss_reasons and STOP_REASON_MATCHING_OBSERVATION_UUID not in reasons:
            problems.append(f"post_adjudication_stop_status_report.content.domains[{name!r}] dropped "
                            f"the scan-time metadata-leakage stop reason")
        expected_stop = bool(reasons)
        if entry.get("stop_before_inference") != expected_stop:
            problems.append(f"post_adjudication_stop_status_report.content.domains[{name!r}]."
                            f"stop_before_inference does not match its own reasons")
        if entry.get("stop_before_inference") and name not in STOP_BEFORE_INFERENCE_DOMAINS:
            problems.append(f"post_adjudication_stop_status_report.content.domains[{name!r}] is not a "
                            f"frozen critical domain but claims a stop")
        if expected_stop:
            any_stop = True

    if content.get("overall_stop_before_inference") != any_stop:
        problems.append("post_adjudication_stop_status_report.content.overall_stop_before_inference does "
                        "not match the per-domain results")
    return problems
