#!/usr/bin/env python3
"""audit_gate_v2_readiness.py — Phase 5A: calibration_v2 / unknown_test_v2
metadata readiness audit.

METADATA ONLY. This script never downloads an image, never runs a model or
ONNX session, never selects a threshold, never evaluates anything, and never
creates calibration_v2/unknown_test_v2 as frozen directories or manifests.
It writes a non-frozen readiness snapshot under data/gate_v2_readiness/ that
a later, separate downloader/freezer consumes.

Correction pass (this version): the original stopping condition compared
`target` against the TOTAL number of rows collected (eligible + ineligible),
so a fetch loop could stop having found only a handful of genuinely eligible
rows once enough ANY rows (mostly license-ineligible) had accumulated. Every
fetch loop below now stops on the count of ELIGIBLE rows only; ineligible
rows are still collected, kept in the output with their rejection reason,
and reported, but never satisfy a target or consume per-species diversity
capacity.

This pass also adopts **license policy B**: every row entering calibration_v2
or unknown_test_v2 -- including rows reused from calibration_v1 -- must carry
an approved license (CC0, CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA) plus a
complete provenance record (source URL, attribution). Reused rows are hash-
verified against calibration_v1's own recorded SHA-256 for ALL 572 candidate
rows (not just the out-of-scope-ant ones) before any eligibility judgment is
made; any missing/ambiguous/mismatched file is a frozen-data integrity
failure that stops the audit, never a "shortfall" to quietly backfill around.

No confidence threshold exists yet -- 0.60 is never imported, applied, or
assumed anywhere in this file.

Network behavior: metadata-only GET requests to the public iNaturalist API,
paced at ~1/second, identical policy to the existing scrapers. No image URL
is ever fetched.

Usage:
    python audit_gate_v2_readiness.py                # full audit incl. API calls
    python audit_gate_v2_readiness.py --offline       # exclusions + reuse verification only,
                                                       # no network calls (for review/tests)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from common import slugify  # noqa: E402

API = "https://api.inaturalist.org/v1"
USER_AGENT = "AntID-gate-v2-readiness/1.0 (educational project, metadata-only)"

TAXON_FORMICIDAE = 47336
TAXON_INSECTA = 47158
UNRELATED_TAXA = {"aves": 3, "plantae": 47126, "mammalia": 40151}

MASTER_SEED = 20260905
DOMAIN_LABELS = (
    "calibration_v2_known", "calibration_v2_ood",
    "calibration_v2_non_ant_insect", "calibration_v2_unrelated",
    "unknown_test_v2_known", "unknown_test_v2_ood",
    "unknown_test_v2_non_ant_insect", "unknown_test_v2_unrelated",
)

# Approved personal/non-commercial license pool -- policy B: applies to EVERY
# row entering either v2 dataset, reused or fresh. Never CC-ND/CC-BY-NC-ND,
# never blank, never unknown.
APPROVED_LICENSES = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}

NEW_15_SPECIES_SLUGS = frozenset({
    "aphaenogaster-rudis", "camponotus-americanus", "camponotus-nearcticus",
    "camponotus-novaeboracensis", "camponotus-subbarbatus", "formica-exsectoides",
    "lasius-americanus", "lasius-aphidicola", "lasius-claviger", "lasius-emarginatus",
    "lasius-interjectus", "lasius-neoniger", "nylanderia-flavipes",
    "ponera-pennsylvanica", "temnothorax-curvispinosus",
})

# Frozen per-plan quotas -- unchanged by this correction pass.
CALIBRATION_KNOWN_PER_SPECIES = 10
UNKNOWN_TEST_KNOWN_PER_SPECIES = 6
KNOWN_MINIMUM_PER_SPECIES = CALIBRATION_KNOWN_PER_SPECIES + UNKNOWN_TEST_KNOWN_PER_SPECIES  # 16
KNOWN_RECOMMENDED_FLOOR = 24
CALIBRATION_OOD_TARGET_TOTAL = 300
CALIBRATION_CONTROL_TARGET = {"non_ant_insect": 150, "unrelated": 150}
UNKNOWN_TEST_OOD_TARGET_TOTAL = 200
UNKNOWN_TEST_FRESH_CONTROL_TARGET = {"non_ant_insect": 100, "unrelated": 100}

# Northeast metadata-audit reference eligible rate (11231/14431 candidates
# survived that audit's own eligibility filter). Used only as a descriptive
# comparison point here -- never a quota, never a gate on its own.
NORTHEAST_REFERENCE_ELIGIBLE_RATE = 11231 / 14431  # 0.778255...
REFERENCE_DELTA_FLAG_THRESHOLD = 0.10  # 10 percentage points

EXPECTED_V1_INTERSECTIONS = {
    "calibration_v1_out_of_scope_ant_rows": 300,
    "calibration_v1_ood_rows_now_supported": 28,
    "calibration_v1_ood_species_now_supported": 8,
    "calibration_v1_ood_reusable_rows": 272,
    "calibration_v1_ood_reusable_species": 157,
    "calibration_v1_known_holdout_new15_species": 0,
    "calibration_v1_low_quality_known_new15_species": 0,
    "unknown_test_v1_out_of_scope_ant_rows": 200,
    "unknown_test_v1_ood_rows_now_supported": 5,
    "unknown_test_v1_ood_species_now_supported": 3,
    "unknown_test_v1_known_holdout_new15_species": 0,
}

CANDIDATE_CSV_FIELDS = [
    "category", "intended_dataset", "selection_status", "species", "slug", "taxon_id",
    "observation_id", "observation_uuid", "photo_id", "source_url", "sha256", "created_at",
    "photo_license", "photo_attribution", "lat", "lon", "provenance_source",
    "origin_dataset", "origin_csv_sha256", "origin_row_number", "origin_photo_id",
    "origin_observation_uuid", "origin_sha256", "reuse_eligibility_reason",
    "rejection_reason",
]

EXCLUSION_SOURCES = {
    "northeast_manifest": "data/northeast_expansion_v1/manifest_all_northeast_v1.csv",
    "benchmark_v1": "data/benchmark_v1/benchmark_v1.csv",
    "calibration_v1": "data/calibration_v1/calibration_v1.csv",
    "unknown_test_v1": "data/unknown_test_v1/unknown_test_v1.csv",
    "northeast_final_test_v1": "data/northeast_final_test_v1/northeast_final_test_v1.csv",
}


class AuditError(RuntimeError):
    """A required invariant did not hold. Always fail closed."""


# ----------------------------------------------------------------- utilities
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def license_eligible(code: str | None) -> bool:
    return (code or "").strip().lower() in APPROVED_LICENSES


# ------------------------------------------------------------------- taxonomy
def load_supported_taxonomy(repo: Path) -> dict[str, dict[str, Any]]:
    path = repo / "data/northeast_expansion_v1/northeast_taxonomy_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_slug = {v["slug"]: v for v in raw.values()}
    if len(by_slug) != 65:
        raise AuditError(f"expected exactly 65 supported species, found {len(by_slug)}")
    return by_slug


# ------------------------------------------------------------------ exclusion
@dataclass
class ExclusionIndex:
    photo_ids: set[str] = field(default_factory=set)
    observation_uuids: set[str] = field(default_factory=set)
    observation_ids: set[str] = field(default_factory=set)
    sha256s: set[str] = field(default_factory=set)
    taxon_ids: set[str] = field(default_factory=set)
    slugs: set[str] = field(default_factory=set)

    def overlaps(self, *, photo_id: str | None = None, observation_uuid: str | None = None,
                observation_id: str | None = None, sha256: str | None = None) -> str | None:
        if photo_id and photo_id in self.photo_ids:
            return "photo_id"
        if observation_uuid and observation_uuid in self.observation_uuids:
            return "observation_uuid"
        if observation_id and observation_id in self.observation_ids:
            return "observation_id"
        if sha256 and sha256 in self.sha256s:
            return "sha256"
        return None


def build_exclusion_indexes(repo: Path) -> tuple[dict[str, ExclusionIndex], dict[str, Any]]:
    indexes: dict[str, ExclusionIndex] = {}
    provenance: dict[str, Any] = {}
    combined = ExclusionIndex()

    for name, rel_path in EXCLUSION_SOURCES.items():
        path = repo / rel_path
        if not path.exists():
            raise AuditError(f"required exclusion source missing: {path}")
        rows = read_csv(path)
        idx = ExclusionIndex()
        for r in rows:
            if r.get("photo_id"):
                idx.photo_ids.add(str(r["photo_id"]))
            if r.get("observation_uuid"):
                idx.observation_uuids.add(str(r["observation_uuid"]))
            if r.get("observation_id"):
                idx.observation_ids.add(str(r["observation_id"]))
            if r.get("sha256"):
                idx.sha256s.add(str(r["sha256"]))
            if r.get("taxon_id"):
                idx.taxon_ids.add(str(r["taxon_id"]))
            if r.get("slug"):
                idx.slugs.add(str(r["slug"]))
        indexes[name] = idx
        for attr in ("photo_ids", "observation_uuids", "observation_ids", "sha256s",
                    "taxon_ids", "slugs"):
            getattr(combined, attr).update(getattr(idx, attr))
        provenance[name] = {
            "path": rel_path, "sha256": sha256_file(path), "rows": len(rows),
            "unique_photo_ids": len(idx.photo_ids),
            "unique_observation_uuids": len(idx.observation_uuids),
        }
    indexes["all"] = combined
    return indexes, provenance


# ------------------------------------------------------- v1 intersection audit
def reproduce_v1_intersections(repo: Path, supported: dict[str, dict[str, Any]]) -> dict[str, Any]:
    supported_slugs = set(supported)

    cal_rows = read_csv(repo / EXCLUSION_SOURCES["calibration_v1"])
    ut_rows = read_csv(repo / EXCLUSION_SOURCES["unknown_test_v1"])

    cal_ood = [r for r in cal_rows if r.get("category") == "out_of_scope_ant"]
    cal_ood_now_supported = [r for r in cal_ood if r["slug"] in supported_slugs]
    cal_ood_reusable = [r for r in cal_ood if r["slug"] not in supported_slugs]
    cal_known_holdout = [r for r in cal_rows if r.get("category") == "known_holdout"]
    cal_low_quality = [r for r in cal_rows if r.get("category") == "low_quality_known"]

    ut_ood = [r for r in ut_rows if r.get("category") == "out_of_scope_ant"]
    ut_ood_now_supported = [r for r in ut_ood if r["slug"] in supported_slugs]
    ut_known_holdout = [r for r in ut_rows if r.get("category") == "known_holdout"]

    actual = {
        "calibration_v1_out_of_scope_ant_rows": len(cal_ood),
        "calibration_v1_ood_rows_now_supported": len(cal_ood_now_supported),
        "calibration_v1_ood_species_now_supported": len({r["slug"] for r in cal_ood_now_supported}),
        "calibration_v1_ood_reusable_rows": len(cal_ood_reusable),
        "calibration_v1_ood_reusable_species": len({r["slug"] for r in cal_ood_reusable}),
        "calibration_v1_known_holdout_new15_species":
            len({r["slug"] for r in cal_known_holdout} & NEW_15_SPECIES_SLUGS),
        "calibration_v1_low_quality_known_new15_species":
            len({r["slug"] for r in cal_low_quality} & NEW_15_SPECIES_SLUGS),
        "unknown_test_v1_out_of_scope_ant_rows": len(ut_ood),
        "unknown_test_v1_ood_rows_now_supported": len(ut_ood_now_supported),
        "unknown_test_v1_ood_species_now_supported": len({r["slug"] for r in ut_ood_now_supported}),
        "unknown_test_v1_known_holdout_new15_species":
            len({r["slug"] for r in ut_known_holdout} & NEW_15_SPECIES_SLUGS),
    }

    disagreements = {
        key: {"expected": EXPECTED_V1_INTERSECTIONS[key], "actual": actual[key]}
        for key in EXPECTED_V1_INTERSECTIONS
        if EXPECTED_V1_INTERSECTIONS[key] != actual[key]
    }
    if disagreements:
        raise AuditError(f"v1 intersection reproduction disagreement: {disagreements}")

    return {
        "expected": dict(EXPECTED_V1_INTERSECTIONS), "actual": actual, "agrees": True,
        "calibration_v1_ood_now_supported_species": sorted({r["slug"] for r in cal_ood_now_supported}),
        "calibration_v1_ood_reusable_rows_detail": cal_ood_reusable,
        "unknown_test_v1_ood_now_supported_species": sorted({r["slug"] for r in ut_ood_now_supported}),
    }


# --------------------------------------------------------- reuse row identity
def extract_reuse_candidate_rows(cal_all_rows: list[dict[str, str]],
                                 supported: dict[str, dict[str, Any]],
                                 cal_csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    """The three pools of calibration_v1 rows PROPOSED for calibration_v2
    reuse: 272 still-out-of-scope-ant rows, 150 non_ant_insect, 150 unrelated.
    Canonical fields are populated directly from origin (photo_id ==
    origin_photo_id, etc.) -- never left blank pending a later join."""
    cal_csv_sha256 = sha256_file(cal_csv_path)
    pools: dict[str, list[dict[str, Any]]] = {"out_of_scope_ant": [], "non_ant_insect": [],
                                              "unrelated": []}
    for i, r in enumerate(cal_all_rows):
        cat = r["category"]
        if cat == "out_of_scope_ant" and r["slug"] in supported:
            continue  # one of the 28 now-supported rows -- not a reuse candidate
        if cat not in pools:
            continue
        taxon_id_raw = r.get("taxon_id")
        try:
            taxon_id = int(taxon_id_raw)
        except (TypeError, ValueError):
            taxon_id = None
        pools[cat].append({
            "category": cat, "intended_dataset": "calibration_v2",
            "selection_status": "pending", "species": r["species"], "slug": r["slug"],
            "taxon_id": taxon_id,
            "observation_id": "", "observation_uuid": r["observation_uuid"],
            "photo_id": r["photo_id"], "source_url": "", "sha256": "",
            "created_at": r.get("created_at", ""),
            "photo_license": "", "photo_attribution": "",
            "lat": r.get("lat", ""), "lon": r.get("lon", ""),
            "provenance_source": "calibration_v1_reuse",
            "origin_dataset": "calibration_v1", "origin_csv_sha256": cal_csv_sha256,
            "origin_row_number": i + 2,  # 1-based + header row
            "origin_photo_id": r["photo_id"], "origin_observation_uuid": r["observation_uuid"],
            "origin_sha256": r["sha256"], "reuse_eligibility_reason": "",
            "rejection_reason": "", "hash_verification_status": "unchecked",
        })
    return pools


def verify_all_reuse_candidate_hashes(repo: Path, pools: dict[str, list[dict[str, Any]]]
                                      ) -> dict[str, Any]:
    """Hash-verify EVERY row in all three reuse pools (572 total) against its
    calibration_v1-recorded SHA-256. Any missing file, ambiguous extension,
    or hash mismatch is a frozen-data integrity failure -- raises AuditError
    immediately, naming every problem found, rather than treating it as an
    ordinary shortfall to backfill around."""
    root = repo / "data/calibration_v1"
    report: dict[str, Any] = {}
    all_problems: list[dict[str, Any]] = []
    for cat, rows in pools.items():
        problems = []
        for row in rows:
            slug, photo_id = row["slug"], row["photo_id"]
            expected = row.get("origin_sha256", "")
            existing = [p for ext in ("jpg", "jpeg", "png", "webp")
                       if (p := root / slug / f"{photo_id}.{ext}").exists()]
            if not existing:
                row["hash_verification_status"] = "file_missing"
                problems.append({"category": cat, "slug": slug, "photo_id": photo_id,
                                "problem": "file_missing"})
                continue
            if len(existing) > 1:
                row["hash_verification_status"] = "ambiguous_extension"
                problems.append({"category": cat, "slug": slug, "photo_id": photo_id,
                                "problem": "ambiguous_extension"})
                continue
            actual = sha256_file(existing[0])
            if actual != expected:
                row["hash_verification_status"] = "hash_mismatch"
                problems.append({"category": cat, "slug": slug, "photo_id": photo_id,
                                "problem": "hash_mismatch", "expected": expected, "actual": actual})
                continue
            row["hash_verification_status"] = "verified"
            row["sha256"] = actual
        n_verified = sum(1 for r in rows if r["hash_verification_status"] == "verified")
        report[cat] = {
            "rows_checked": len(rows), "rows_verified": n_verified,
            "rows_with_problems": len(rows) - n_verified,
            "problems": [p for p in problems],
        }
        all_problems.extend(problems)
    if all_problems:
        raise AuditError(
            f"frozen-data integrity failure: {len(all_problems)} calibration_v1 reuse "
            f"row(s) failed hash verification: {all_problems[:10]}"
            + (" ... (truncated)" if len(all_problems) > 10 else "")
        )
    return report


# --------------------------------------------------------------- eligibility
def is_fresh_row_eligible(row: dict[str, Any]) -> tuple[bool, str]:
    """A fresh candidate row is eligible only when every applicable field is
    valid and nonblank. observation_id is optional. taxon_id must be a real
    int, never a bool (bool is an int subclass in Python) and never blank."""
    if not row.get("species"):
        return False, "missing_species"
    if not row.get("slug"):
        return False, "missing_slug"
    taxon_id = row.get("taxon_id")
    if isinstance(taxon_id, bool) or not isinstance(taxon_id, int) or taxon_id <= 0:
        return False, "invalid_taxon_id"
    if not row.get("observation_uuid"):
        return False, "missing_observation_uuid"
    if not row.get("photo_id"):
        return False, "missing_photo_id"
    if not row.get("source_url"):
        return False, "missing_source_url"
    if not license_eligible(row.get("photo_license")):
        return False, "license_not_approved"
    if not row.get("photo_attribution"):
        return False, "missing_attribution"
    return True, ""


def is_reused_row_eligible(row: dict[str, Any]) -> tuple[bool, str]:
    """Policy B for a reused calibration_v1 row: must already be hash-
    verified (never assumed), plus approved license, source URL, and
    attribution -- the latter two typically arriving via enrichment, since
    calibration_v1's own CSV never recorded them."""
    status = row.get("hash_verification_status")
    if status != "verified":
        return False, f"hash_{status or 'unchecked'}"
    code = (row.get("photo_license") or "").strip()
    if not code:
        return False, "license_blank"
    if not license_eligible(code):
        return False, "license_not_approved"
    if not row.get("source_url"):
        return False, "missing_source_url"
    if not row.get("photo_attribution"):
        return False, "missing_attribution"
    return True, ""


def finalize_reused_rows(pools: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """After hash verification + license enrichment, classify every reused
    row's eligibility and set its final selection_status/reuse_eligibility_
    reason. Returns a per-category eligible-rate-style report."""
    reports = {}
    for cat, rows in pools.items():
        rejection_counts: Counter[str] = Counter()
        n_eligible = 0
        for row in rows:
            eligible, reason = is_reused_row_eligible(row)
            if eligible:
                row["selection_status"] = "candidate_eligible"
                row["reuse_eligibility_reason"] = "verified_by_sha256"
                n_eligible += 1
            else:
                row["selection_status"] = "ineligible"
                row["rejection_reason"] = reason
                rejection_counts[reason] += 1
        examined = len(rows)
        rate = n_eligible / examined if examined else None
        delta_pp = (rate - NORTHEAST_REFERENCE_ELIGIBLE_RATE) * 100 if rate is not None else None
        reports[cat] = {
            "examined": examined, "eligible": n_eligible, "eligible_rate": rate,
            "delta_from_reference_pp": delta_pp,
            "materially_below_northeast_reference":
                bool(rate is not None
                    and rate < NORTHEAST_REFERENCE_ELIGIBLE_RATE - REFERENCE_DELTA_FLAG_THRESHOLD),
            "rejection_breakdown": dict(sorted(rejection_counts.items())),
        }
    return reports


def eligible_rate_report(examined: int, eligible: int, rejection_counts: Counter) -> dict[str, Any]:
    rate = eligible / examined if examined else None
    delta_pp = (rate - NORTHEAST_REFERENCE_ELIGIBLE_RATE) * 100 if rate is not None else None
    return {
        "examined": examined, "eligible": eligible, "eligible_rate": rate,
        "delta_from_reference_pp": delta_pp,
        "materially_below_northeast_reference":
            bool(rate is not None
                and rate < NORTHEAST_REFERENCE_ELIGIBLE_RATE - REFERENCE_DELTA_FLAG_THRESHOLD),
        "rejection_breakdown": dict(sorted(rejection_counts.items())),
    }


def classify_known_species_readiness(n_available: int) -> str:
    if n_available < KNOWN_MINIMUM_PER_SPECIES:
        return "fails_readiness"
    if n_available < KNOWN_RECOMMENDED_FLOOR:
        return "fragile"
    return "ready"


# ------------------------------------------------------------ deterministic RNG
def domain_rng(domain_label: str, master_seed: int = MASTER_SEED) -> random.Random:
    if domain_label not in DOMAIN_LABELS:
        raise AuditError(f"unknown domain label: {domain_label!r}")
    derived = sha256_bytes(f"{master_seed}:{domain_label}".encode("utf-8"))
    return random.Random(int(derived[:16], 16))


def deterministic_select(candidates: list[dict[str, Any]], domain_label: str, n: int,
                         key: Callable[[dict], Any]) -> tuple[list[dict], list[dict]]:
    snapshot = sorted(candidates, key=key)
    rng = domain_rng(domain_label)
    shuffled = snapshot[:]
    rng.shuffle(shuffled)
    return shuffled[:n], shuffled[n:]


# --------------------------------------------------------------- HTTP client
class PacedJsonClient:
    def __init__(self, interval_seconds: float = 1.05, retries: int = 5) -> None:
        self.interval_seconds = interval_seconds
        self.retries = retries
        self._last_request = 0.0
        self.request_count = 0
        self.first_request_utc: str | None = None
        self.last_request_utc: str | None = None

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urllib.parse.urlencode(params or {})
        url = f"{API}{path}" + (f"?{query}" if query else "")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.interval_seconds:
                time.sleep(self.interval_seconds - elapsed)
            now = utc_now()
            if self.first_request_utc is None:
                self.first_request_utc = now
            self.last_request_utc = now
            self._last_request = time.monotonic()
            self.request_count += 1
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    body = response.read()
                    content_type = response.headers.get("Content-Type", "")
                    if "image/" in content_type:
                        raise AuditError(f"refusing an image-typed response for a metadata GET: {url}")
                    return json.loads(body.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"request failed after {self.retries} attempts: {url}") from last_error


# ----------------------------------------------------------- candidate fetch
def _parse_loc(obs: dict) -> tuple[float | None, float | None]:
    loc = obs.get("location")
    if loc and "," in loc:
        try:
            a, b = loc.split(",")
            return float(a), float(b)
        except ValueError:
            pass
    return None, None


def _photo_row(obs: dict, photo: dict, taxon_id: Any, species: str, slug: str,
              category: str, intended_dataset: str) -> dict[str, Any]:
    lat, lon = _parse_loc(obs)
    license_code = (photo.get("license_code") or "").strip().lower()
    return {
        "category": category, "intended_dataset": intended_dataset,
        "selection_status": "pending",
        "species": species, "slug": slug, "taxon_id": taxon_id,
        "observation_id": obs.get("id"), "observation_uuid": obs.get("uuid"),
        "photo_id": photo.get("id"), "source_url": photo.get("url") or "",
        "created_at": obs.get("created_at") or "",
        "photo_license": license_code, "photo_attribution": photo.get("attribution") or "",
        "lat": lat, "lon": lon, "provenance_source": "inat_api_gate_v2_readiness",
        "origin_dataset": "", "origin_csv_sha256": "", "origin_row_number": "",
        "origin_photo_id": "", "origin_observation_uuid": "", "origin_sha256": "",
        "reuse_eligibility_reason": "", "rejection_reason": "",
    }


def _new_fetch_stats() -> dict[str, Any]:
    return {
        "pages_fetched": 0, "api_results_examined": 0, "excluded_by_frozen_id": 0,
        "unique_candidates_examined": 0, "eligible_candidates": 0,
        "rejection_counts": Counter(), "termination_reason": None,
    }


def _finalize_stats(stats: dict[str, Any], eligible_count: int, target: int,
                    api_exhausted: bool) -> None:
    if eligible_count >= target:
        stats["termination_reason"] = "eligible_target_reached"
    elif api_exhausted:
        stats["termination_reason"] = "api_exhausted"
    else:
        stats["termination_reason"] = "max_pages_exhausted"
    stats["eligible_candidates"] = eligible_count
    stats["quota_short_and_bounded"] = bool(
        eligible_count < target and stats["termination_reason"] == "max_pages_exhausted")
    stats["result_label"] = ("bounded_query_inconclusive" if stats["quota_short_and_bounded"]
                             else "sufficient" if eligible_count >= target
                             else "genuine_data_scarcity")
    stats["rejection_counts"] = dict(sorted(stats["rejection_counts"].items()))


def fetch_species_candidates(client: PacedJsonClient, taxon_id: int, species: str, slug: str,
                             category: str, intended_dataset: str, exclusion: ExclusionIndex,
                             target: int, max_pages: int = 5) -> tuple[list[dict[str, Any]], dict]:
    """Metadata-only candidate rows for one taxon. `target` is a target
    ELIGIBLE-row count: pagination continues until `target` eligible rows
    are found, the API returns no further results, or max_pages is reached
    -- never merely until `target` total rows (eligible or not) accumulate.
    Ineligible rows are kept in the return list with their rejection reason,
    never silently dropped, and never count toward the target."""
    rows: list[dict[str, Any]] = []
    stats = _new_fetch_stats()
    eligible_count = 0
    page = 1
    api_exhausted = False
    while eligible_count < target and page <= max_pages:
        payload = client.get("/observations", {
            "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
            "captive": "false", "per_page": 200, "page": page,
            "order_by": "id", "order": "asc",
        })
        results = payload.get("results") or []
        stats["pages_fetched"] += 1
        stats["api_results_examined"] += len(results)
        if not results:
            api_exhausted = True
            break
        for obs in results:
            photos = obs.get("photos") or []
            if not photos:
                continue
            photo = photos[0]
            if photo.get("id") is None:
                continue
            pid, ouuid = str(photo["id"]), str(obs.get("uuid") or "")
            if exclusion.overlaps(photo_id=pid, observation_uuid=ouuid):
                stats["excluded_by_frozen_id"] += 1
                continue
            stats["unique_candidates_examined"] += 1
            obs_taxon = obs.get("taxon") or {}
            row_species = obs_taxon.get("name") or species
            row_slug = slug if category == "known_holdout" else slugify(row_species)
            row_taxon_id = obs_taxon.get("id") or taxon_id
            row = _photo_row(obs, photo, row_taxon_id, row_species, row_slug,
                             category, intended_dataset)
            eligible, reason = is_fresh_row_eligible(row)
            if eligible:
                row["selection_status"] = "candidate_eligible"
                eligible_count += 1
            else:
                row["selection_status"] = "ineligible"
                row["rejection_reason"] = reason
                stats["rejection_counts"][reason] += 1
            rows.append(row)
            if eligible_count >= target:
                break
        page += 1
    _finalize_stats(stats, eligible_count, target, api_exhausted)
    return rows, stats


def fetch_diverse_ood_candidates(client: PacedJsonClient, exclusion: ExclusionIndex,
                                 without_taxon_ids: str, target: int, category: str,
                                 intended_dataset: str, taxon_id: int = TAXON_FORMICIDAE,
                                 max_per_species: int = 15,
                                 max_pages: int = 30) -> tuple[list[dict[str, Any]], dict]:
    """Species-diverse candidates under `taxon_id` (Formicidae for
    out_of_scope_ant, Insecta for non_ant_insect, or one of Aves/Plantae/
    Mammalia for unrelated), excluding every taxon_id in `without_taxon_ids`.
    `target` is a target ELIGIBLE-row count. The per-species diversity cap
    (`max_per_species`) counts ELIGIBLE rows only -- an ineligible candidate
    for an already-capped species is recorded as ineligible for a distinct
    reason, never silently omitted and never treated as consuming that
    species' diversity slot."""
    rows: list[dict[str, Any]] = []
    per_species_eligible: dict[str, int] = defaultdict(int)
    stats = _new_fetch_stats()
    eligible_count = 0
    page = 1
    api_exhausted = False
    while eligible_count < target and page <= max_pages:
        params = {
            "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
            "captive": "false", "per_page": 200, "page": page,
            "order_by": "id", "order": "asc",
        }
        if without_taxon_ids:
            params["without_taxon_id"] = without_taxon_ids
        payload = client.get("/observations", params)
        results = payload.get("results") or []
        stats["pages_fetched"] += 1
        stats["api_results_examined"] += len(results)
        if not results:
            api_exhausted = True
            break
        for obs in results:
            taxon = obs.get("taxon") or {}
            if taxon.get("rank") != "species" or not taxon.get("name"):
                continue
            slug = slugify(taxon["name"])
            photos = obs.get("photos") or []
            if not photos:
                continue
            photo = photos[0]
            if photo.get("id") is None:
                continue
            pid, ouuid = str(photo["id"]), str(obs.get("uuid") or "")
            if exclusion.overlaps(photo_id=pid, observation_uuid=ouuid):
                stats["excluded_by_frozen_id"] += 1
                continue
            stats["unique_candidates_examined"] += 1
            row = _photo_row(obs, photo, taxon["id"], taxon["name"], slug,
                             category, intended_dataset)
            eligible, reason = is_fresh_row_eligible(row)
            if not eligible:
                row["selection_status"] = "ineligible"
                row["rejection_reason"] = reason
                stats["rejection_counts"][reason] += 1
                rows.append(row)
                continue
            if per_species_eligible[slug] >= max_per_species:
                row["selection_status"] = "ineligible"
                row["rejection_reason"] = "species_diversity_cap_reached"
                stats["rejection_counts"]["species_diversity_cap_reached"] += 1
                rows.append(row)
                continue
            row["selection_status"] = "candidate_eligible"
            per_species_eligible[slug] += 1
            eligible_count += 1
            rows.append(row)
            if eligible_count >= target:
                break
        page += 1
    _finalize_stats(stats, eligible_count, target, api_exhausted)
    return rows, stats


def enrich_reused_rows_license(client: PacedJsonClient, rows: list[dict[str, Any]],
                               batch_size: int = 100) -> dict[str, Any]:
    """Best-effort metadata-only enrichment of legacy calibration_v1 rows
    (license/attribution/source_url are never recorded there) via a batched
    GET /observations?uuid=... lookup. Never invents a value and never drops
    a row for missing enrichment."""
    enriched = 0
    missing: list[str] = []
    uuids = [r["origin_observation_uuid"] for r in rows if r.get("origin_observation_uuid")]
    by_uuid: dict[str, dict] = {}
    for i in range(0, len(uuids), batch_size):
        batch = uuids[i:i + batch_size]
        payload = client.get("/observations", {"uuid": ",".join(batch), "per_page": len(batch)})
        for obs in payload.get("results") or []:
            by_uuid[str(obs.get("uuid"))] = obs

    for row in rows:
        obs = by_uuid.get(row.get("origin_observation_uuid"))
        photo = None
        if obs:
            for p in obs.get("photos") or []:
                if str(p.get("id")) == str(row.get("origin_photo_id")):
                    photo = p
                    break
        if not photo:
            missing.append(row.get("origin_observation_uuid", ""))
            continue
        row["photo_license"] = (photo.get("license_code") or "").strip().lower()
        row["photo_attribution"] = photo.get("attribution") or ""
        row["source_url"] = photo.get("url") or ""
        enriched += 1

    return {"rows_considered": len(rows), "rows_enriched": enriched,
           "rows_missing_enrichment": len(missing),
           "missing_observation_uuids_sample": missing[:20]}


def dedupe_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove rows sharing a logical identity (photo_id, falling back to
    origin_photo_id for reused rows) among selected/reused rows only --
    ineligible/reserve rows from independent fetches may legitimately repeat
    across categories and are left alone."""
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["selection_status"] not in ("selected", "reused"):
            out.append(r)
            continue
        identity = str(r.get("photo_id") or r.get("origin_photo_id") or id(r))
        if identity in seen:
            continue
        seen.add(identity)
        out.append(r)
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out-dir", type=Path, default=Path("data/gate_v2_readiness"))
    ap.add_argument("--request-interval", type=float, default=1.05)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--known-target-per-species", type=int, default=KNOWN_RECOMMENDED_FLOOR)
    ap.add_argument("--known-max-pages", type=int, default=5)
    ap.add_argument("--calibration-ood-reserve-target", type=int, default=40)
    ap.add_argument("--calibration-control-reserve-target", type=int, default=20)
    ap.add_argument("--unknown-ood-reserve-target", type=int, default=30)
    ap.add_argument("--unknown-control-reserve-target", type=int, default=15)
    args = ap.parse_args()

    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir

    supported = load_supported_taxonomy(repo)
    exclusions, exclusion_provenance = build_exclusion_indexes(repo)
    intersections = reproduce_v1_intersections(repo, supported)

    cal_csv_path = repo / EXCLUSION_SOURCES["calibration_v1"]
    cal_all_rows = read_csv(cal_csv_path)
    reuse_pools = extract_reuse_candidate_rows(cal_all_rows, supported, cal_csv_path)
    if len(reuse_pools["out_of_scope_ant"]) != 272:
        raise AuditError(
            f"expected 272 out_of_scope_ant reuse candidates, got {len(reuse_pools['out_of_scope_ant'])}")
    for cat, expected_n in CALIBRATION_CONTROL_TARGET.items():
        if len(reuse_pools[cat]) != expected_n:
            raise AuditError(
                f"expected {expected_n} {cat} reuse candidates, got {len(reuse_pools[cat])}")

    hash_report = verify_all_reuse_candidate_hashes(repo, reuse_pools)
    print(f"[audit] v1 intersections reproduced exactly as expected")
    for cat, r in hash_report.items():
        print(f"[audit] hash-verified {r['rows_verified']}/{r['rows_checked']} "
             f"calibration_v1 {cat} reuse rows", flush=True)

    result: dict[str, Any] = {
        "schema_version": 2,
        "purpose": "Phase 5A metadata-only readiness audit for calibration_v2/unknown_test_v2; "
                  "not a frozen dataset, not a threshold, not an evaluation.",
        "license_policy": "B: approved license + complete provenance required for EVERY row, "
                          "reused or fresh.",
        "master_seed": MASTER_SEED,
        "domain_labels": list(DOMAIN_LABELS),
        "v1_intersections": {k: v for k, v in intersections.items()
                            if k not in ("calibration_v1_ood_reusable_rows_detail",)},
        "exclusion_sources": exclusion_provenance,
        "reuse_hash_verification": hash_report,
        "offline_mode": args.offline,
    }

    if args.offline:
        write_json(out_dir / "gate_v2_readiness.json", result)
        print(f"[audit] offline mode: wrote {out_dir / 'gate_v2_readiness.json'} "
             f"(no API calls made)")
        return 0

    run_full_audit(repo, out_dir, args, supported, exclusions, intersections, reuse_pools, result)
    return 0


def run_full_audit(repo: Path, out_dir: Path, args, supported: dict[str, dict[str, Any]],
                   exclusions: dict[str, ExclusionIndex], intersections: dict[str, Any],
                   reuse_pools: dict[str, list[dict[str, Any]]], result: dict[str, Any]) -> None:
    client = PacedJsonClient(interval_seconds=args.request_interval)
    all_rows: list[dict[str, Any]] = []
    seen_photo_ids: set[str] = set()
    seen_obs_uuids: set[str] = set()

    def register(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            pid, ouuid = str(r.get("photo_id")), str(r.get("observation_uuid") or "")
            if r["selection_status"] == "candidate_eligible":
                if pid in seen_photo_ids or (ouuid and ouuid in seen_obs_uuids):
                    r["selection_status"] = "ineligible"
                    r["rejection_reason"] = "duplicate_within_audit"
                else:
                    seen_photo_ids.add(pid)
                    if ouuid:
                        seen_obs_uuids.add(ouuid)
        return rows

    # ---- reused-row license enrichment, THEN policy-B eligibility ------------
    all_reused = reuse_pools["out_of_scope_ant"] + reuse_pools["non_ant_insect"] + reuse_pools["unrelated"]
    enrichment = enrich_reused_rows_license(client, all_reused)
    result["reused_row_license_enrichment"] = enrichment
    reused_eligibility_report = finalize_reused_rows(reuse_pools)
    result["reused_row_eligibility"] = reused_eligibility_report
    all_rows.extend(all_reused)

    reused_eligible = {cat: [r for r in rows if r["selection_status"] == "candidate_eligible"]
                       for cat, rows in reuse_pools.items()}
    for rows in reused_eligible.values():
        for r in rows:
            pid, ouuid = str(r["photo_id"]), str(r["observation_uuid"])
            seen_photo_ids.add(pid)
            if ouuid:
                seen_obs_uuids.add(ouuid)

    # ---- known species (calibration + unknown-test, per species) -----------
    per_species_report = []
    for slug in sorted(supported):
        info = supported[slug]
        target = args.known_target_per_species
        candidates, stats = fetch_species_candidates(
            client, info["taxon_id"], info["species_name"], slug, "known_holdout",
            "calibration_v2+unknown_test_v2", exclusions["all"], target=target,
            max_pages=args.known_max_pages)
        candidates = register(candidates)
        eligible = [c for c in candidates if c["selection_status"] == "candidate_eligible"]
        n_available = len(eligible)
        cal_selected, remainder = deterministic_select(
            eligible, "calibration_v2_known", min(CALIBRATION_KNOWN_PER_SPECIES, n_available),
            key=lambda r: (r["photo_id"], r["observation_uuid"]))
        ut_selected, reserve = deterministic_select(
            remainder, "unknown_test_v2_known",
            min(UNKNOWN_TEST_KNOWN_PER_SPECIES, len(remainder)),
            key=lambda r: (r["photo_id"], r["observation_uuid"]))
        for r in cal_selected:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        for r in ut_selected:
            r["selection_status"], r["intended_dataset"] = "selected", "unknown_test_v2"
        for r in reserve:
            r["selection_status"], r["intended_dataset"] = "reserve", "reserve"
        all_rows.extend(candidates)

        status = classify_known_species_readiness(n_available)
        rate_report = eligible_rate_report(stats["unique_candidates_examined"],
                                           stats["eligible_candidates"],
                                           Counter(stats["rejection_counts"]))
        per_species_report.append({
            "slug": slug, "taxon_id": info["taxon_id"], "available_eligible": n_available,
            "calibration_selected": len(cal_selected), "unknown_test_selected": len(ut_selected),
            "reserve": len(reserve), "status": status,
            "pages_fetched": stats["pages_fetched"],
            "api_results_examined": stats["api_results_examined"],
            "excluded_by_frozen_id": stats["excluded_by_frozen_id"],
            "unique_candidates_examined": stats["unique_candidates_examined"],
            "termination_reason": stats["termination_reason"],
            "result_label": stats["result_label"],
            "eligible_rate": rate_report["eligible_rate"],
            "delta_from_reference_pp": rate_report["delta_from_reference_pp"],
            "materially_below_northeast_reference": rate_report["materially_below_northeast_reference"],
            "rejection_breakdown": rate_report["rejection_breakdown"],
        })
        print(f"[audit] known/{slug}: available={n_available} cal={len(cal_selected)} "
             f"ut={len(ut_selected)} reserve={len(reserve)} status={status} "
             f"term={stats['termination_reason']}", flush=True)

    result["known_species_readiness"] = per_species_report
    result["known_species_summary"] = {
        "species_count": len(per_species_report),
        "ready": sum(1 for r in per_species_report if r["status"] == "ready"),
        "fragile": sum(1 for r in per_species_report if r["status"] == "fragile"),
        "fails_readiness": sum(1 for r in per_species_report if r["status"] == "fails_readiness"),
        "bounded_query_inconclusive":
            sum(1 for r in per_species_report if r["result_label"] == "bounded_query_inconclusive"),
    }

    # ---- calibration OOD: dynamically derived fresh deficit -----------------
    reusable_species = sorted({r["slug"] for r in intersections["calibration_v1_ood_reusable_rows_detail"]})
    reusable_taxon_by_slug = {r["slug"]: int(r["taxon_id"])
                              for r in intersections["calibration_v1_ood_reusable_rows_detail"]}
    cal_ood_deficit = CALIBRATION_OOD_TARGET_TOTAL - len(reused_eligible["out_of_scope_ant"])
    cal_ood_target = max(0, cal_ood_deficit) + args.calibration_ood_reserve_target
    cal_ood_fresh: list[dict[str, Any]] = []
    cal_ood_fresh_eligible_count = 0
    for slug in reusable_species:
        if cal_ood_fresh_eligible_count >= cal_ood_target:
            break
        taxon_id = reusable_taxon_by_slug[slug]
        got, stats = fetch_species_candidates(
            client, taxon_id, slug.replace("-", " ").capitalize(), slug, "out_of_scope_ant",
            "calibration_v2", exclusions["all"],
            target=max(1, cal_ood_target - cal_ood_fresh_eligible_count), max_pages=2)
        got = register(got)
        cal_ood_fresh.extend(got)
        cal_ood_fresh_eligible_count += sum(1 for r in got if r["selection_status"] == "candidate_eligible")
    cal_ood_fresh_eligible = [r for r in cal_ood_fresh if r["selection_status"] == "candidate_eligible"]
    cal_ood_selected, cal_ood_reserve = deterministic_select(
        cal_ood_fresh_eligible, "calibration_v2_ood",
        min(max(0, cal_ood_deficit), len(cal_ood_fresh_eligible)),
        key=lambda r: (r["photo_id"], r["observation_uuid"]))
    for r in cal_ood_selected:
        r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
    for r in cal_ood_reserve:
        r["selection_status"], r["intended_dataset"] = "reserve", "reserve"
    all_rows.extend(cal_ood_fresh)
    for r in reused_eligible["out_of_scope_ant"]:
        r["selection_status"] = "reused"

    calibration_v2_ood_taxon_ids = {reusable_taxon_by_slug[r["slug"]] for r in reused_eligible["out_of_scope_ant"]}
    calibration_v2_ood_taxon_ids.update(r["taxon_id"] for r in cal_ood_selected)
    calibration_v2_ood_species = {r["slug"] for r in reused_eligible["out_of_scope_ant"]} | \
        {r["slug"] for r in cal_ood_selected}

    result["calibration_ood_plan"] = {
        "target_total": CALIBRATION_OOD_TARGET_TOTAL,
        "reused_eligible_rows": len(reused_eligible["out_of_scope_ant"]),
        "dynamically_derived_fresh_deficit": cal_ood_deficit,
        "fresh_eligible_found": len(cal_ood_fresh_eligible),
        "fresh_selected": len(cal_ood_selected),
        "fresh_reserve": len(cal_ood_reserve),
        "fresh_rows_use_existing_157_species": True,
        "fresh_rows_introduce_new_species": False,
        "total_planned": len(reused_eligible["out_of_scope_ant"]) + len(cal_ood_selected),
        "species": sorted(calibration_v2_ood_species),
        "taxon_ids": sorted(calibration_v2_ood_taxon_ids),
        "readiness": "ready" if len(cal_ood_selected) >= max(0, cal_ood_deficit) else "fails_readiness",
    }

    # ---- calibration non_ant_insect / unrelated: dynamic deficits -----------
    cal_control_deficit = {
        cat: CALIBRATION_CONTROL_TARGET[cat] - len(reused_eligible[cat])
        for cat in ("non_ant_insect", "unrelated")
    }
    for cat in ("non_ant_insect", "unrelated"):
        for r in reused_eligible[cat]:
            r["selection_status"] = "reused"

    # ---- unknown-test OOD exclusion set built from ACTUAL planned calibration OOD --
    excluded_taxon_ids = {info["taxon_id"] for info in supported.values()}
    excluded_taxon_ids.update(calibration_v2_ood_taxon_ids)
    ut_ood_target = UNKNOWN_TEST_OOD_TARGET_TOTAL + args.unknown_ood_reserve_target
    ut_ood_rows, ut_ood_stats = fetch_diverse_ood_candidates(
        client, exclusions["all"], ",".join(str(t) for t in sorted(excluded_taxon_ids)),
        target=ut_ood_target, category="out_of_scope_ant", intended_dataset="unknown_test_v2")
    ut_ood_rows = register(ut_ood_rows)
    for r in ut_ood_rows:
        if r["selection_status"] == "candidate_eligible" and r["slug"] in calibration_v2_ood_species:
            r["selection_status"] = "ineligible"
            r["rejection_reason"] = "shares_species_with_calibration_v2_ood"
    ut_ood_eligible = [r for r in ut_ood_rows if r["selection_status"] == "candidate_eligible"]
    ut_ood_selected, ut_ood_reserve = deterministic_select(
        ut_ood_eligible, "unknown_test_v2_ood",
        min(UNKNOWN_TEST_OOD_TARGET_TOTAL, len(ut_ood_eligible)),
        key=lambda r: (r["photo_id"], r["observation_uuid"]))
    for r in ut_ood_selected:
        r["selection_status"] = "selected"
    for r in ut_ood_reserve:
        r["selection_status"] = "reserve"
    all_rows.extend(ut_ood_rows)

    species_counts = Counter(r["slug"] for r in ut_ood_selected)
    ut_ood_rate = eligible_rate_report(ut_ood_stats["unique_candidates_examined"],
                                       ut_ood_stats["eligible_candidates"],
                                       Counter(ut_ood_stats["rejection_counts"]))
    result["unknown_test_ood_plan"] = {
        "target_total": UNKNOWN_TEST_OOD_TARGET_TOTAL,
        "eligible_found": len(ut_ood_eligible),
        "selected": len(ut_ood_selected), "reserve": len(ut_ood_reserve),
        "represented_species_count": len(species_counts),
        "max_rows_per_species": max(species_counts.values(), default=0),
        "species_list": sorted(species_counts),
        "species_disjoint_from_supported": True,
        "species_disjoint_from_calibration_v2_ood": True,
        "termination_reason": ut_ood_stats["termination_reason"],
        "result_label": ut_ood_stats["result_label"],
        "eligible_rate": ut_ood_rate["eligible_rate"],
        "delta_from_reference_pp": ut_ood_rate["delta_from_reference_pp"],
        "materially_below_northeast_reference": ut_ood_rate["materially_below_northeast_reference"],
        "readiness": "ready" if len(ut_ood_selected) >= UNKNOWN_TEST_OOD_TARGET_TOTAL else "fails_readiness",
    }

    # ---- controls: calibration backfill THEN unknown-test fresh -------------
    control_root_taxon = {"non_ant_insect": TAXON_INSECTA, "unrelated": None}
    control_without = {"non_ant_insect": str(TAXON_FORMICIDAE), "unrelated": ""}
    calibration_control_plan = {}
    unknown_control_plan = {}
    for cat in ("non_ant_insect", "unrelated"):
        deficit = max(0, cal_control_deficit[cat])
        need = deficit + UNKNOWN_TEST_FRESH_CONTROL_TARGET[cat] + args.unknown_control_reserve_target
        if cat == "non_ant_insect":
            rows, stats = fetch_diverse_ood_candidates(
                client, exclusions["all"], control_without[cat], need, cat,
                "calibration_v2+unknown_test_v2", taxon_id=control_root_taxon[cat])
        else:
            rows = []
            stats_list = []
            per_taxon = -(-need // len(UNRELATED_TAXA))
            for name, tid in UNRELATED_TAXA.items():
                got, st = fetch_diverse_ood_candidates(
                    client, exclusions["all"], "", per_taxon, cat,
                    "calibration_v2+unknown_test_v2", taxon_id=tid)
                rows.extend(got)
                stats_list.append(st)
            stats = {
                "pages_fetched": sum(s["pages_fetched"] for s in stats_list),
                "api_results_examined": sum(s["api_results_examined"] for s in stats_list),
                "excluded_by_frozen_id": sum(s["excluded_by_frozen_id"] for s in stats_list),
                "unique_candidates_examined": sum(s["unique_candidates_examined"] for s in stats_list),
                "eligible_candidates": sum(s["eligible_candidates"] for s in stats_list),
                "termination_reason": stats_list[-1]["termination_reason"],
                "result_label": stats_list[-1]["result_label"],
                "rejection_counts": dict(Counter() ),
            }
            merged = Counter()
            for s in stats_list:
                merged.update(s["rejection_counts"])
            stats["rejection_counts"] = dict(sorted(merged.items()))
        rows = register(rows)
        eligible = [r for r in rows if r["selection_status"] == "candidate_eligible"]
        cal_selected, remainder = deterministic_select(
            eligible, f"calibration_v2_{cat}", min(deficit, len(eligible)),
            key=lambda r: (r["photo_id"], r["observation_uuid"]))
        ut_selected, reserve = deterministic_select(
            remainder, f"unknown_test_v2_{cat}",
            min(UNKNOWN_TEST_FRESH_CONTROL_TARGET[cat], len(remainder)),
            key=lambda r: (r["photo_id"], r["observation_uuid"]))
        for r in cal_selected:
            r["selection_status"], r["intended_dataset"] = "selected", "calibration_v2"
        for r in ut_selected:
            r["selection_status"], r["intended_dataset"] = "selected", "unknown_test_v2"
        for r in reserve:
            r["selection_status"], r["intended_dataset"] = "reserve", "reserve"
        all_rows.extend(rows)

        rate = eligible_rate_report(stats["unique_candidates_examined"], stats["eligible_candidates"],
                                    Counter(stats["rejection_counts"]))
        calibration_control_plan[cat] = {
            "reused_eligible_rows": len(reused_eligible[cat]),
            "dynamically_derived_fresh_deficit": cal_control_deficit[cat],
            "fresh_selected": len(cal_selected),
            "total_planned": len(reused_eligible[cat]) + len(cal_selected),
            "readiness": "ready" if len(cal_selected) >= deficit else "fails_readiness",
        }
        unknown_control_plan[cat] = {
            "target": UNKNOWN_TEST_FRESH_CONTROL_TARGET[cat],
            "eligible_found_in_shared_pool": len(eligible),
            "selected": len(ut_selected), "reserve": len(reserve),
            "termination_reason": stats["termination_reason"],
            "result_label": stats["result_label"],
            "eligible_rate": rate["eligible_rate"],
            "delta_from_reference_pp": rate["delta_from_reference_pp"],
            "materially_below_northeast_reference": rate["materially_below_northeast_reference"],
            "readiness": "ready" if len(ut_selected) >= UNKNOWN_TEST_FRESH_CONTROL_TARGET[cat]
                        else "fails_readiness",
        }
    result["calibration_control_plan"] = calibration_control_plan
    result["unknown_test_control_plan"] = unknown_control_plan

    # ---- output integrity -----------------------------------------------------
    all_rows = dedupe_candidate_rows(all_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_CSV_FIELDS, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    status_counts = Counter(r["selection_status"] for r in all_rows)
    rejection_counts = Counter(r["rejection_reason"] for r in all_rows if r.get("rejection_reason"))
    result["selection_status_totals"] = dict(sorted(status_counts.items()))
    result["rejection_summary"] = dict(sorted(rejection_counts.items()))
    result["fresh_vs_reused_totals"] = {
        "fresh_selected": sum(1 for r in all_rows if r["selection_status"] == "selected"
                             and r["provenance_source"] != "calibration_v1_reuse"),
        "fresh_reserve": sum(1 for r in all_rows if r["selection_status"] == "reserve"),
        "fresh_ineligible": sum(1 for r in all_rows if r["selection_status"] == "ineligible"
                               and r["provenance_source"] != "calibration_v1_reuse"),
        "reused_selected": sum(1 for r in all_rows if r["selection_status"] == "reused"),
        "reused_ineligible": sum(1 for r in all_rows if r["selection_status"] == "ineligible"
                                and r["provenance_source"] == "calibration_v1_reuse"),
    }
    result["network"] = {
        "api": API, "request_count": client.request_count,
        "first_request_utc": client.first_request_utc, "last_request_utc": client.last_request_utc,
        "request_interval_seconds": args.request_interval,
        "image_typed_responses_observed": 0,
        "proof": "PacedJsonClient.get() raises AuditError on any image/* Content-Type response; "
                "none occurred (see image_typed_responses_observed).",
    }
    result["candidates_csv"] = {
        "path": str(csv_path.relative_to(repo)).replace("\\", "/"),
        "sha256": sha256_file(csv_path), "rows": len(all_rows),
    }
    result["frozen_facts"] = {
        "no_threshold_applied": True, "no_image_downloaded": True,
        "no_model_or_onnx_session_loaded": True, "no_inference_or_evaluation": True,
        "calibration_v2_and_unknown_test_v2_not_created": True,
    }

    json_path = out_dir / "gate_v2_readiness.json"
    write_json(json_path, result)
    print(f"[audit] wrote {csv_path} ({len(all_rows)} rows)")
    print(f"[audit] wrote {json_path}")
    print(f"[audit] {client.request_count} API requests made; STOP -- metadata only")


if __name__ == "__main__":
    sys.exit(main())
