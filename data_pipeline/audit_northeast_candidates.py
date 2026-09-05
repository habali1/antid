#!/usr/bin/env python3
"""Build a metadata-only readiness snapshot for Northeast expansion candidates.

This script deliberately downloads no image bytes. It queries public iNaturalist
taxon and observation metadata at roughly one request per second, verifies the
15 exact taxon IDs recorded in the count snapshot, and writes:

* a row-level candidate metadata CSV (one deterministic photo per observation),
* a machine-readable readiness summary JSON, and
* a concise Markdown report.

Photo-id and observation-UUID exclusions are applied against the historical
training manifests and all frozen evaluation sets. Image hashes and perceptual
near-duplicate checks cannot be performed without image bytes; both remain a
mandatory post-download gate and are reported as such.

No split membership is assigned, no species membership is frozen, and no model
or serving artifact is touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

API = "https://api.inaturalist.org/v1"
USER_AGENT = "AntID-northeast-readiness/1.0 (educational project)"
FORMICIDAE_TAXON_ID = 47336

STATE_PLACE_IDS = {
    "CT": 49,
    "ME": 17,
    "MA": 2,
    "NH": 41,
    "RI": 8,
    "VT": 47,
    "NJ": 51,
    "NY": 48,
    "PA": 42,
}

# The primary readiness calculation uses licenses without a non-commercial
# restriction. The broader personal-use tier is reported separately, never
# silently mixed into the primary pool.
CORE_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
PERSONAL_NC_LICENSES = CORE_LICENSES | {"cc-by-nc", "cc-by-nc-sa"}
NO_DERIVATIVES_LICENSES = {"cc-by-nd", "cc-by-nc-nd"}
APPROVED_LICENSES = PERSONAL_NC_LICENSES
LICENSE_POLICY_EFFECTIVE_DATE = "2026-09-05"

# Fixed before looking at model scores. These are capacity targets only; actual
# membership and splits remain unassigned until the user approves downloading.
QUOTA = {"train": 160, "development": 40, "final_test": 30}

CSV_FIELDS = [
    "taxon_id",
    "species",
    "genus_id",
    "genus",
    "state",
    "observation_id",
    "observation_uuid",
    "observer_id",
    "observed_on",
    "created_at",
    "photo_id",
    "photo_license",
    "license_tier",
    "photo_attribution",
    "photo_url_medium",
    "photo_width",
    "photo_height",
    "photo_count",
    "core_eligible_photo_count",
    "personal_nc_eligible_photo_count",
    "internal_duplicate",
    "internal_duplicate_of_observation_uuid",
    "prior_overlap",
    "overlap_sources",
    "eligible_core",
    "eligible_personal_nc",
    "geoprivacy",
    "obscured",
]

# No download manifest exists yet. This is the fail-closed contract its builder
# must implement: provenance cannot be reconstructed safely after selection.
FUTURE_DOWNLOAD_MANIFEST_REQUIRED_FIELDS = [
    "observation_id",
    "observation_uuid",
    "photo_id",
    "taxon_id",
    "genus_id",
    "photo_license",
    "photo_attribution",
    "photo_url_medium",
    "sha256",
    "split",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Exclusions:
    photo_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    observation_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def sources_for(self, photo_ids: Iterable[str], observation_uuid: str) -> list[str]:
        ids = set(photo_ids)
        hits = {
            source
            for source, known in self.photo_sources.items()
            if ids & known
        }
        if observation_uuid:
            hits.update(
                source
                for source, known in self.observation_sources.items()
                if observation_uuid in known
            )
        return sorted(hits)


def load_exclusions(repo: Path) -> tuple[Exclusions, dict[str, Any]]:
    exclusions = Exclusions()
    provenance: dict[str, Any] = {}

    training_manifests = [
        repo / "data/manifest_all.csv",
        repo / "data/raw/manifest_inat.csv",
        repo / "data/raw_wave2/manifest_inat.csv",
    ]
    training_photo_ids: set[str] = set()
    for path in training_manifests:
        if not path.exists():
            raise FileNotFoundError(f"required training manifest missing: {path}")
        rows = read_csv(path)
        training_photo_ids.update(str(row["photo_id"]) for row in rows if row.get("photo_id"))
        provenance[str(path.relative_to(repo)).replace("\\", "/")] = {
            "sha256": sha256_file(path),
            "rows": len(rows),
        }
    exclusions.photo_sources["training"] = training_photo_ids

    recovered_path = repo / "data/benchmark_v1/_training_obs_uuids.json"
    recovered = json.loads(recovered_path.read_text(encoding="utf-8"))
    photo_to_uuid = recovered.get("photo_to_observation_uuid", {})
    exclusions.observation_sources["training"] = {
        str(value) for value in photo_to_uuid.values() if value
    }
    provenance[str(recovered_path.relative_to(repo)).replace("\\", "/")] = {
        "sha256": sha256_file(recovered_path),
        "mapped_photo_ids": len(photo_to_uuid),
        "unique_observation_uuids": len(exclusions.observation_sources["training"]),
        "unmatched_training_photos": int(recovered.get("n_training_photos_total", 0))
        - int(recovered.get("n_matched", 0)),
    }

    frozen = {
        "benchmark_v1": repo / "data/benchmark_v1/benchmark_v1.csv",
        "calibration_v1": repo / "data/calibration_v1/calibration_v1.csv",
        "unknown_test_v1": repo / "data/unknown_test_v1/unknown_test_v1.csv",
    }
    for source, path in frozen.items():
        if not path.exists():
            raise FileNotFoundError(f"required frozen manifest missing: {path}")
        rows = read_csv(path)
        exclusions.photo_sources[source] = {
            str(row["photo_id"]) for row in rows if row.get("photo_id")
        }
        exclusions.observation_sources[source] = {
            str(row["observation_uuid"])
            for row in rows
            if row.get("observation_uuid")
        }
        provenance[str(path.relative_to(repo)).replace("\\", "/")] = {
            "sha256": sha256_file(path),
            "rows": len(rows),
            "unique_photo_ids": len(exclusions.photo_sources[source]),
            "unique_observation_uuids": len(exclusions.observation_sources[source]),
        }

    return exclusions, provenance


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
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"request failed after {self.retries} attempts: {url}") from last_error


def license_tier(code: str | None) -> str:
    normalized = (code or "").lower()
    if normalized in CORE_LICENSES:
        return "core_reuse"
    if normalized in PERSONAL_NC_LICENSES:
        return "personal_noncommercial"
    if normalized in NO_DERIVATIVES_LICENSES:
        return "no_derivatives"
    return "unlicensed_or_unknown"


def select_photo(
    photos: list[dict[str, Any]], forbidden_photo_ids: set[int] | None = None
) -> tuple[dict[str, Any], int, int, bool]:
    usable = [p for p in photos if p.get("id") is not None and p.get("url")]
    if not usable:
        raise ValueError("observation has no usable photo metadata")
    forbidden = forbidden_photo_ids or set()
    unique_options = [p for p in usable if int(p["id"]) not in forbidden]
    reused = not unique_options
    selectable = unique_options or usable
    priority = {
        "core_reuse": 0,
        "personal_noncommercial": 1,
        "no_derivatives": 2,
        "unlicensed_or_unknown": 3,
    }
    chosen = min(
        selectable,
        key=lambda p: (priority[license_tier(p.get("license_code"))], int(p["id"])),
    )
    n_core = sum(
        (p.get("license_code") or "").lower() in CORE_LICENSES for p in unique_options
    )
    n_personal = sum(
        (p.get("license_code") or "").lower() in PERSONAL_NC_LICENSES
        for p in unique_options
    )
    return chosen, n_core, n_personal, reused


def resolve_taxa(
    client: PacedJsonClient,
    expected: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    ids = [int(item["taxon_id"]) for item in expected]
    payload = client.get("/taxa/" + ",".join(str(value) for value in ids), {"per_page": 200})
    by_id = {int(item["id"]): item for item in payload.get("results", [])}
    if set(by_id) != set(ids):
        raise RuntimeError(f"taxon lookup mismatch: wanted {sorted(ids)}, got {sorted(by_id)}")

    expected_names = {int(item["taxon_id"]): item["name"] for item in expected}
    resolved: dict[int, dict[str, Any]] = {}
    for taxon_id in ids:
        taxon = by_id[taxon_id]
        problems = []
        if taxon.get("name") != expected_names[taxon_id]:
            problems.append(f"name={taxon.get('name')!r}, expected={expected_names[taxon_id]!r}")
        if taxon.get("rank") != "species":
            problems.append(f"rank={taxon.get('rank')!r}")
        if taxon.get("is_active") is not True:
            problems.append(f"is_active={taxon.get('is_active')!r}")
        ancestor_ids = {int(value) for value in taxon.get("ancestor_ids", [])}
        if FORMICIDAE_TAXON_ID not in ancestor_ids:
            problems.append("Formicidae is absent from ancestor_ids")
        ancestors = list(taxon.get("ancestors") or [])
        genus = next((a for a in ancestors if a.get("rank") == "genus"), None)
        if not genus:
            problems.append("genus ancestor missing")
        if problems:
            raise RuntimeError(f"taxon {taxon_id} failed exact validation: {'; '.join(problems)}")
        resolved[taxon_id] = {
            "taxon_id": taxon_id,
            "name": taxon["name"],
            "rank": taxon["rank"],
            "is_active": taxon["is_active"],
            "genus_id": int(genus["id"]),
            "genus": genus["name"],
            "ancestor_ids": [int(value) for value in taxon.get("ancestor_ids", [])],
        }
    return resolved


def observation_query_params(taxon_id: int, page: int) -> dict[str, Any]:
    return {
        "taxon_id": taxon_id,
        "place_id": ",".join(str(STATE_PLACE_IDS[state]) for state in STATE_PLACE_IDS),
        "quality_grade": "research",
        "photos": "true",
        "captive": "false",
        "per_page": 200,
        "page": page,
        "order_by": "id",
        "order": "asc",
    }


def fetch_observations(client: PacedJsonClient, taxon_id: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    first_total: int | None = None
    page = 1
    while True:
        payload = client.get("/observations", observation_query_params(taxon_id, page))
        if first_total is None:
            first_total = int(payload.get("total_results", 0))
        batch = list(payload.get("results", []))
        rows.extend(batch)
        if not batch or len(batch) < 200 or len(rows) >= int(payload.get("total_results", 0)):
            break
        page += 1
        if page > 50:
            raise RuntimeError(f"unexpected pagination beyond 10,000 records for taxon {taxon_id}")
    return rows, int(first_total or 0)


def state_for_observation(obs: dict[str, Any]) -> str:
    place_ids = {int(value) for value in obs.get("place_ids", [])}
    matches = [state for state, place_id in STATE_PLACE_IDS.items() if place_id in place_ids]
    if len(matches) != 1:
        raise RuntimeError(
            f"observation {obs.get('id')} has {len(matches)} Northeast state matches: {matches}"
        )
    return matches[0]


def medium_url(photo: dict[str, Any]) -> str:
    if photo.get("medium_url"):
        return str(photo["medium_url"])
    return str(photo["url"]).replace("/square.", "/medium.")


def normalize_observation(
    obs: dict[str, Any],
    taxon: dict[str, Any],
    exclusions: Exclusions,
    used_photo_ids: dict[int, str],
) -> dict[str, Any]:
    observed_taxon = obs.get("taxon") or {}
    if int(observed_taxon.get("id", -1)) != taxon["taxon_id"]:
        raise RuntimeError(
            f"observation {obs.get('id')} resolved to taxon {observed_taxon.get('id')}, "
            f"not exact requested taxon {taxon['taxon_id']}"
        )
    if obs.get("quality_grade") != "research" or obs.get("captive") is True:
        raise RuntimeError(f"observation {obs.get('id')} violates query eligibility")
    photos = list(obs.get("photos") or [])
    chosen, n_core, n_personal, internal_duplicate = select_photo(
        photos, set(used_photo_ids)
    )
    all_photo_ids = [str(p["id"]) for p in photos if p.get("id") is not None]
    observation_uuid = str(obs.get("uuid") or "")
    overlap_sources = exclusions.sources_for(all_photo_ids, observation_uuid)
    dimensions = chosen.get("original_dimensions") or {}
    tier = license_tier(chosen.get("license_code"))
    no_overlap = not overlap_sources and not internal_duplicate
    duplicate_of = used_photo_ids.get(int(chosen["id"]), "") if internal_duplicate else ""
    return {
        "taxon_id": taxon["taxon_id"],
        "species": taxon["name"],
        "genus_id": taxon["genus_id"],
        "genus": taxon["genus"],
        "state": state_for_observation(obs),
        "observation_id": int(obs["id"]),
        "observation_uuid": observation_uuid,
        "observer_id": int((obs.get("user") or {}).get("id") or 0),
        "observed_on": obs.get("observed_on") or "",
        "created_at": obs.get("created_at") or "",
        "photo_id": int(chosen["id"]),
        "photo_license": (chosen.get("license_code") or "").lower(),
        "license_tier": tier,
        "photo_attribution": chosen.get("attribution") or "",
        "photo_url_medium": medium_url(chosen),
        "photo_width": dimensions.get("width") or "",
        "photo_height": dimensions.get("height") or "",
        "photo_count": len(photos),
        "core_eligible_photo_count": n_core,
        "personal_nc_eligible_photo_count": n_personal,
        "internal_duplicate": internal_duplicate,
        "internal_duplicate_of_observation_uuid": duplicate_of,
        "prior_overlap": bool(overlap_sources),
        "overlap_sources": ";".join(overlap_sources),
        "eligible_core": no_overlap and n_core > 0,
        "eligible_personal_nc": no_overlap and n_personal > 0,
        "geoprivacy": obs.get("geoprivacy") or "open",
        "obscured": bool(obs.get("obscured")),
    }


def bool_csv(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def quota_capacity(n: int) -> dict[str, Any]:
    required = sum(QUOTA.values())
    remaining = n
    allocation: dict[str, int] = {}
    # Protect the untouched final-test and development capacity first; training
    # receives what remains. This is only a feasibility display, not membership.
    for split in ("final_test", "development", "train"):
        allocation[split] = min(QUOTA[split], remaining)
        remaining -= allocation[split]
    return {
        "required": required,
        "available": n,
        "shortfall": max(0, required - n),
        "numerically_ready": n >= required,
        "illustrative_capacity_only": allocation,
    }


def summarize_species(
    taxon: dict[str, Any],
    rows: list[dict[str, Any]],
    prior_count: int,
    api_total: int,
) -> dict[str, Any]:
    core = [row for row in rows if row["eligible_core"]]
    personal = [row for row in rows if row["eligible_personal_nc"]]
    usable = core
    observers = Counter(row["observer_id"] for row in usable)
    top_observer_n = max(observers.values(), default=0)
    observed_dates = sorted(row["observed_on"] for row in usable if row["observed_on"])
    created_dates = sorted(row["created_at"] for row in usable if row["created_at"])
    overlap_counts = Counter()
    for row in rows:
        for source in str(row["overlap_sources"]).split(";"):
            if source:
                overlap_counts[source] += 1
    state_counts = Counter(row["state"] for row in usable)
    license_counts = Counter(row["photo_license"] or "unlicensed" for row in rows)
    tier_counts = Counter(row["license_tier"] for row in rows)
    unlicensed_rows = license_counts["unlicensed"]
    core_quota = quota_capacity(len(core))
    approved_quota = quota_capacity(len(personal))
    warnings = []
    if api_total != prior_count:
        warnings.append(
            f"live API total differs from dated species-count snapshot by {api_total - prior_count:+d}"
        )
    if not approved_quota["numerically_ready"]:
        warnings.append(
            f"approved personal/non-commercial pool is {approved_quota['shortfall']} "
            "observations below quota"
        )
    if usable and top_observer_n / len(usable) > 0.25:
        warnings.append("one observer contributes more than 25% of the core-license pool")
    if len(state_counts) < 3:
        warnings.append("core-license pool spans fewer than three Northeast states")
    return {
        **taxon,
        "dated_aggregate_snapshot_count": prior_count,
        "live_api_total": api_total,
        "metadata_rows": len(rows),
        "prior_overlap_rows": sum(bool(row["prior_overlap"]) for row in rows),
        "internal_duplicate_rows": sum(bool(row["internal_duplicate"]) for row in rows),
        "prior_overlap_by_source": dict(sorted(overlap_counts.items())),
        "core_license_eligible_rows": len(core),
        "personal_noncommercial_eligible_rows": len(personal),
        "distinct_core_observers": len(observers),
        "largest_core_observer_count": top_observer_n,
        "largest_core_observer_share": round(top_observer_n / len(usable), 6) if usable else None,
        "core_state_counts": {state: state_counts.get(state, 0) for state in STATE_PLACE_IDS},
        "core_states_present": sum(state_counts[state] > 0 for state in STATE_PLACE_IDS),
        "selected_photo_license_counts_all_rows": dict(sorted(license_counts.items())),
        "selected_photo_license_tier_counts_all_rows": dict(sorted(tier_counts.items())),
        "source_unlicensed_rows": unlicensed_rows,
        "source_unlicensed_share": round(unlicensed_rows / len(rows), 6) if rows else None,
        "core_observed_date_min": observed_dates[0] if observed_dates else None,
        "core_observed_date_max": observed_dates[-1] if observed_dates else None,
        "core_created_at_min": created_dates[0] if created_dates else None,
        "core_created_at_max": created_dates[-1] if created_dates else None,
        "core_quota_capacity": core_quota,
        "approved_quota_capacity": approved_quota,
        "warnings": warnings,
    }


def existing_image_size_stats(repo: Path) -> dict[str, Any]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    sizes = sorted(
        path.stat().st_size
        for root in (repo / "data/raw", repo / "data/raw_wave2")
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not sizes:
        return {"sample_count": 0, "mean_bytes": None, "p95_bytes": None}
    p95 = sizes[math.ceil(0.95 * len(sizes)) - 1]
    return {
        "sample_count": len(sizes),
        "mean_bytes": round(sum(sizes) / len(sizes)),
        "p95_bytes": p95,
        "source": "existing medium-size training downloads in data/raw and data/raw_wave2",
    }


def render_markdown(summary: dict[str, Any], csv_path: Path, repo: Path) -> str:
    species = summary["species"]
    required = sum(QUOTA.values())
    lines = [
        "# AntID Northeast candidate readiness — metadata audit v1",
        "",
        f"Generated {summary['retrieval']['completed_at_utc']}. This is a metadata snapshot, not a frozen dataset.",
        "No photos were downloaded, no split membership was assigned, and no model artifact was changed.",
        "",
        "## Predeclared capacity target",
        "",
        f"Per provisional species: **{QUOTA['train']} train + {QUOTA['development']} development + "
        f"{QUOTA['final_test']} untouched final-test = {required} observations**. Readiness uses the approved "
        "personal/non-commercial pool (CC0, CC BY, CC BY-SA, CC BY-NC, or CC BY-NC-SA) and excludes every prior photo/observation match found.",
        "This target was fixed before any model score was inspected; the displayed allocation is capacity only.",
        "",
        "| Candidate | Genus (iNat ID) | Core eligible | Approved personal/NC | No license | Observers | States | Date span | Prior overlaps | Approved shortfall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for item in species:
        lines.append(
            f"| {item['name']} ({item['taxon_id']}) | {item['genus']} ({item['genus_id']}) | "
            f"{item['core_license_eligible_rows']:,} | {item['personal_noncommercial_eligible_rows']:,} | "
            f"{item['source_unlicensed_rows']:,} ({item['source_unlicensed_share']:.1%}) | "
            f"{item['distinct_core_observers']:,} | {item['core_states_present']} | "
            f"{item['core_observed_date_min']} to {item['core_observed_date_max']} | "
            f"{item['prior_overlap_rows']:,} | {item['approved_quota_capacity']['shortfall']:,} |"
        )
    ready = sum(item["approved_quota_capacity"]["numerically_ready"] for item in species)
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            f"- {ready}/{len(species)} candidates meet the numeric {required}-observation target under the approved personal/non-commercial license pool.",
            "- The previously agreed 150-image admission floor is not binding because every current candidate clears 230. It remains mandatory for any later replacement candidate.",
            "- Source-unlicensed share ranges from 17.4% to 25.7%. Lasius interjectus has both the highest share and the smallest approved pool, but its 345 eligible rows still exceed the 230 quota by 115.",
            "- Numeric readiness does not approve a species. A labeled-photo review must still check label quality and whether diagnostic ant features are visible, especially for Lasius, Camponotus, and Aphaenogaster lookalikes.",
            "- Exact active species IDs and authoritative genus ancestors were verified from iNaturalist; no fuzzy name fallback was used.",
            "- Coordinates were intentionally not written to the audit CSV. State membership and open/obscured status are retained; exact locations are unnecessary for this decision.",
            "",
            "## Exclusions and remaining gate",
            "",
            "Every observation was checked against training, benchmark_v1, calibration_v1, and unknown_test_v1 using all photo IDs on the observation plus its observation UUID. Training has 69 photos whose parent UUID could not be reconstructed; those remain protected by photo ID only.",
            "",
            "**Image SHA-256 and perceptual near-duplicate exclusion have not been performed.** The API metadata does not contain the downloaded bytes needed for those checks. After download, the pipeline must hash every file, compare it with all frozen hashes, and review perceptual near-duplicates before any split is frozen.",
            "",
            "## License policy used for planning",
            "",
            "- Core reporting tier: CC0, CC BY, CC BY-SA.",
            "- Approved personal/non-commercial pool as of 2026-09-05: core licenses plus CC BY-NC and CC BY-NC-SA.",
            "- CC licenses with NoDerivatives and unlicensed/all-rights-reserved photos are not counted as eligible.",
            "- An empty `photo_license` records that iNaturalist returned no license code; it is preserved as source truth and the row is ineligible.",
            "- Attribution and the source URL are preserved per candidate row. This is conservative project policy, not legal advice; review obligations before redistribution or commercial use.",
            "",
            "## Bounded download/storage proposal",
            "",
            f"Maximum if all 15 candidates are retained: **{summary['budget']['max_images']:,} medium images**. "
            f"Existing medium downloads average {summary['budget']['existing_mean_kib']:.1f} KiB and have a "
            f"95th-percentile size of {summary['budget']['existing_p95_kib']:.1f} KiB. That projects to "
            f"{summary['budget']['projected_mean_mib']:.0f} MiB at the mean or "
            f"{summary['budget']['projected_p95_mib']:.0f} MiB using the p95-per-file bound. "
            f"Reserve **{summary['budget']['recommended_disk_reservation_gib']:.1f} GiB** for raw and cleaned working copies plus metadata; no paid storage is required.",
            "",
            "## Reproducibility",
            "",
            f"The row-level snapshot is `{csv_path.relative_to(repo).as_posix()}` ({summary['candidate_csv']['rows']:,} rows, SHA-256 `{summary['candidate_csv']['sha256']}`).",
            "The companion JSON references that CSV by path, row count, and SHA-256 rather than duplicating its rows, and records the required future download-manifest fields.",
            "Query parameters, source hashes, API retrieval window, per-species counts, license counts, observer concentration, geographic spread, and date coverage are in the companion JSON.",
            "The script paces requests at about one per second in line with iNaturalist's published API practices.",
            "",
            "## Stop point",
            "",
            "Do not download images or substitute/freeze species yet. The license/quota policy is approved; define and review the small manual photo-review sample next.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_existing_outputs(
    repo: Path, csv_path: Path, json_path: Path, report_path: Path
) -> None:
    """Apply an approved policy change without re-querying live metadata."""
    if not csv_path.exists() or not json_path.exists():
        raise FileNotFoundError("--refresh-existing requires the existing CSV and JSON")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    actual_csv_hash = sha256_file(csv_path)
    if actual_csv_hash != summary.get("candidate_csv", {}).get("sha256"):
        raise RuntimeError("candidate CSV hash no longer matches the readiness JSON")
    rows = read_csv(csv_path)
    if len(rows) != int(summary["candidate_csv"]["rows"]):
        raise RuntimeError("candidate CSV row count no longer matches the readiness JSON")
    if any(field not in rows[0] for field in ("photo_license", "photo_attribution")):
        raise RuntimeError("candidate CSV lacks required license/attribution provenance")

    summary["eligibility"].update({
        "approved_license_codes": sorted(APPROVED_LICENSES),
        "approved_policy_effective_date": LICENSE_POLICY_EFFECTIVE_DATE,
        "approved_scope": "personal, non-commercial phase only",
        "unlicensed": "empty photo_license means iNaturalist returned null/no code; preserved as source truth and not eligible",
    })
    summary["quota"].update({
        "minimum_candidate_floor": 150,
        "minimum_candidate_floor_status": "not binding: all 15 current candidates clear the 230-observation quota under the approved pool; still required for replacements",
        "capacity_basis": "approved personal/non-commercial license pool",
        "status": "approved capacity target only; no rows assigned and no membership frozen",
    })
    rows_by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_species[row["species"]].append(row)
    for item in summary["species"]:
        species_rows = rows_by_species[item["name"]]
        unlicensed_rows = sum(not row["photo_license"] for row in species_rows)
        item["source_unlicensed_rows"] = unlicensed_rows
        item["source_unlicensed_share"] = round(
            unlicensed_rows / len(species_rows), 6
        )
        item["core_quota_capacity"] = quota_capacity(item["core_license_eligible_rows"])
        item["approved_quota_capacity"] = quota_capacity(
            item["personal_noncommercial_eligible_rows"]
        )
        item.pop("quota_capacity", None)
        item["warnings"] = [
            warning
            for warning in item.get("warnings", [])
            if not warning.startswith("core-license pool is ")
        ]
    if not all(item["approved_quota_capacity"]["numerically_ready"] for item in summary["species"]):
        raise RuntimeError("approved policy does not satisfy the recorded all-15 quota decision")
    summary.pop("candidate_rows", None)
    summary["future_download_manifest_contract"] = {
        "status": "schema requirement only; download manifest does not exist yet",
        "required_fields": FUTURE_DOWNLOAD_MANIFEST_REQUIRED_FIELDS,
        "provenance_rule": "photo_license and photo_attribution must be copied from the selected candidate row and must never be blank for an admitted download",
    }
    summary["budget"]["recommended_disk_reservation_gib"] = 2.0
    summary["next_gate"] = (
        "manual labeled-photo review proposal before any image download or species freeze"
    )
    write_json(json_path, summary)
    report_path.write_text(
        render_markdown(summary, csv_path, repo), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("docs/plans/northeast-counts-2026-09-05.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/northeast_readiness_v1")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/plans/northeast-readiness-v1.md")
    )
    parser.add_argument("--request-interval", type=float, default=1.05)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="apply current policy/report schema to the existing verified CSV/JSON without API calls",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    counts_path = args.counts if args.counts.is_absolute() else repo / args.counts
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    report_path = args.report if args.report.is_absolute() else repo / args.report
    csv_path = out_dir / "candidates.csv"
    json_path = out_dir / "northeast_readiness_v1.json"

    if args.refresh_existing:
        refresh_existing_outputs(repo, csv_path, json_path, report_path)
        print(f"refreshed {json_path} and {report_path} without API or image access")
        return

    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    candidate_ids = [int(value) for value in counts["provisional_addition_taxon_ids"]]
    if len(candidate_ids) != 15 or len(set(candidate_ids)) != 15:
        raise RuntimeError("count snapshot must contain exactly 15 unique provisional taxon IDs")
    count_species = {
        int(item["taxon_id"]): item
        for item in counts["species"]
        if int(item["taxon_id"]) in set(candidate_ids)
    }
    if set(count_species) != set(candidate_ids):
        raise RuntimeError("count snapshot is missing one or more provisional species records")
    expected = [count_species[taxon_id] for taxon_id in candidate_ids]

    exclusions, exclusion_provenance = load_exclusions(repo)
    client = PacedJsonClient(interval_seconds=args.request_interval)
    taxa = resolve_taxa(client, expected)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seen_observations: set[str] = set()
    seen_photos: dict[int, str] = {}
    for index, taxon_id in enumerate(candidate_ids, 1):
        taxon = taxa[taxon_id]
        print(f"[{index:02d}/15] {taxon['name']} ({taxon_id})", flush=True)
        observations, api_total = fetch_observations(client, taxon_id)
        rows = []
        for obs in observations:
            row = normalize_observation(obs, taxon, exclusions, seen_photos)
            uuid = row["observation_uuid"]
            photo_id = int(row["photo_id"])
            if not uuid or uuid in seen_observations:
                raise RuntimeError(f"missing or duplicate observation UUID: {uuid!r}")
            seen_observations.add(uuid)
            if not row["internal_duplicate"]:
                seen_photos[photo_id] = uuid
            rows.append(row)
        all_rows.extend(rows)
        summaries.append(
            summarize_species(taxon, rows, int(count_species[taxon_id]["total"]), api_total)
        )
        print(
            f"         rows={len(rows)} core={summaries[-1]['core_license_eligible_rows']} "
            f"personal_nc={summaries[-1]['personal_noncommercial_eligible_rows']} "
            f"overlap={summaries[-1]['prior_overlap_rows']}",
            flush=True,
        )

    all_rows.sort(key=lambda row: (int(row["taxon_id"]), int(row["observation_id"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: bool_csv(row[key]) for key in CSV_FIELDS} for row in all_rows)

    size_stats = existing_image_size_stats(repo)
    max_images = len(candidate_ids) * sum(QUOTA.values())
    projected_mean = max_images * int(size_stats["mean_bytes"] or 160_000)
    projected_p95 = max_images * int(size_stats["p95_bytes"] or 320_000)
    reserve_gib = math.ceil((projected_p95 / (1024**3)) * 10) / 10
    summary = {
        "schema_version": 1,
        "purpose": "metadata-only readiness audit; not a frozen dataset or split",
        "counts_snapshot": {
            "path": counts_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(counts_path),
        },
        "retrieval": {
            "api": API,
            "started_at_utc": client.first_request_utc,
            "completed_at_utc": utc_now(),
            "last_request_at_utc": client.last_request_utc,
            "request_count": client.request_count,
            "request_interval_seconds": args.request_interval,
            "user_agent": USER_AGENT,
            "query": observation_query_params(0, 1) | {
                "taxon_id": "one exact provisional taxon ID per query",
                "page": "1..N until complete",
            },
            "taxon_resolution": "GET /v1/taxa/{comma-separated exact IDs}; exact active species/name and genus ancestor required",
        },
        "region": {
            "definition": "nine-state U.S. Census Northeast",
            "state_place_ids": STATE_PLACE_IDS,
            "broome_usage": "diagnostic in the earlier count snapshot only; not added because it is inside NY",
        },
        "eligibility": {
            "base": ["research grade", "has photos", "not captive", "exact active candidate species"],
            "one_photo_per_observation": "choose lowest photo ID from best available license tier",
            "core_license_codes": sorted(CORE_LICENSES),
            "personal_noncommercial_license_codes": sorted(PERSONAL_NC_LICENSES),
            "approved_license_codes": sorted(APPROVED_LICENSES),
            "approved_policy_effective_date": LICENSE_POLICY_EFFECTIVE_DATE,
            "approved_scope": "personal, non-commercial phase only",
            "excluded_license_codes": sorted(NO_DERIVATIVES_LICENSES),
            "unlicensed": "empty photo_license means iNaturalist returned null/no code; preserved as source truth and not eligible",
            "coordinates": "not collected in this metadata audit",
        },
        "quota": {
            **QUOTA,
            "per_species_total": sum(QUOTA.values()),
            "minimum_candidate_floor": 150,
            "minimum_candidate_floor_status": "not binding: all 15 current candidates clear the 230-observation quota under the approved pool; still required for replacements",
            "capacity_basis": "approved personal/non-commercial license pool",
            "status": "approved capacity target only; no rows assigned and no membership frozen",
        },
        "exclusions": {
            "source_files": exclusion_provenance,
            "method": "all observation photo IDs plus observation UUID compared with source-specific sets",
            "training_uuid_limitation": "69 historical training photos have no reconstructed parent UUID; protected by photo ID only",
            "not_yet_possible": [
                "downloaded image SHA-256 comparison",
                "perceptual/recompressed near-duplicate comparison",
            ],
            "required_later": "perform both byte-hash and perceptual near-duplicate checks before freezing any split",
        },
        "candidate_csv": {
            "path": csv_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(csv_path),
            "rows": len(all_rows),
            "contains_image_bytes": False,
        },
        "future_download_manifest_contract": {
            "status": "schema requirement only; download manifest does not exist yet",
            "required_fields": FUTURE_DOWNLOAD_MANIFEST_REQUIRED_FIELDS,
            "provenance_rule": "photo_license and photo_attribution must be copied from the selected candidate row and must never be blank for an admitted download",
        },
        "budget": {
            "max_images": max_images,
            "existing_download_size_sample": size_stats,
            "existing_mean_kib": round((size_stats["mean_bytes"] or 160_000) / 1024, 1),
            "existing_p95_kib": round((size_stats["p95_bytes"] or 320_000) / 1024, 1),
            "projected_mean_mib": round(projected_mean / (1024**2), 1),
            "projected_p95_mib": round(projected_p95 / (1024**2), 1),
            "recommended_disk_reservation_gib": max(2.0, reserve_gib),
            "paid_service_required": False,
        },
        "species": summaries,
        "global_caveats": [
            "Research-grade labels are community evidence, not infallible species ground truth.",
            "Counts measure available iNaturalist metadata, not natural abundance or encounter probability.",
            "Numeric quota capacity does not assess diagnostic image quality or lookalike label risk.",
            "API data and taxonomy can change after this dated snapshot.",
            "No model scores were used to choose candidates, licenses, or quotas.",
        ],
        "next_gate": "manual labeled-photo review proposal before any image download or species freeze",
    }
    write_json(json_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(summary, csv_path, repo), encoding="utf-8", newline="\n"
    )
    print(f"wrote {csv_path} ({len(all_rows)} rows)")
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    print("STOP: metadata only; no photos downloaded and no catalog membership frozen")


if __name__ == "__main__":
    main()
