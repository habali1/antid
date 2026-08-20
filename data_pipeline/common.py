"""Shared helpers for the AntID data pipeline.

Heavy / optional dependencies (psycopg2, boto3, google-cloud-storage) are
imported lazily inside the functions that need them so that `--help`,
`--dry-run`, and unit-level use stay fast and dependency-free.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


# --------------------------------------------------------------------------- #
# Species list
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetSpecies:
    scientific_name: str            # "Camponotus pennsylvanicus" (canonical key)
    common_name: Optional[str]      # "Black Carpenter Ant" or None

    @property
    def slug(self) -> str:
        return slugify(self.scientific_name)

    @property
    def genus(self) -> str:
        return self.scientific_name.split()[0]

    @property
    def species_epithet(self) -> str:
        parts = self.scientific_name.split()
        return parts[1] if len(parts) > 1 else ""


def slugify(text: str) -> str:
    """'Camponotus pennsylvanicus' -> 'camponotus-pennsylvanicus'."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def load_species_list(path: Path) -> list[TargetSpecies]:
    """Parse the species list file. '#' starts a comment (incl. inline)."""
    species: list[TargetSpecies] = []
    seen: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            name_part, comment = line.split("#", 1)
            scientific = name_part.strip()
            common = comment.strip() or None
        else:
            scientific = line
            common = None
        # Normalize internal whitespace.
        scientific = " ".join(scientific.split())
        if not scientific or scientific.lower() in seen:
            continue
        seen.add(scientific.lower())
        species.append(TargetSpecies(scientific, common))
    return species


# --------------------------------------------------------------------------- #
# Image hashing / dedup helpers
# --------------------------------------------------------------------------- #
def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# --------------------------------------------------------------------------- #
# PostgreSQL
# --------------------------------------------------------------------------- #
def database_url() -> Optional[str]:
    return os.environ.get("DATABASE_URL")


def get_db_connection(url: Optional[str] = None):
    """Open a psycopg2 connection. Returns None if DATABASE_URL is unset."""
    url = url or database_url()
    if not url:
        return None
    import psycopg2  # lazy

    return psycopg2.connect(url)


def upsert_species(conn, sp: "TargetSpecies", taxon_id: Optional[int],
                   class_idx: Optional[int] = None) -> int:
    """Insert or update a species row, returning its primary key id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO species (taxon_id, slug, name, common_name, class_idx)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (slug) DO UPDATE
              SET taxon_id    = COALESCE(EXCLUDED.taxon_id, species.taxon_id),
                  common_name = COALESCE(EXCLUDED.common_name, species.common_name),
                  class_idx   = COALESCE(EXCLUDED.class_idx, species.class_idx)
            RETURNING id
            """,
            (taxon_id, sp.slug, sp.scientific_name, sp.common_name, class_idx),
        )
        species_id = cur.fetchone()[0]
    conn.commit()
    return species_id


def insert_image(conn, species_id: int, source: str, storage_path: str,
                 split: str, width: Optional[int], height: Optional[int],
                 lat: Optional[float] = None, lon: Optional[float] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO images (species_id, source, storage_path, split, width, height, lat, lon)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (storage_path) DO NOTHING
            """,
            (species_id, source, storage_path, split, width, height, lat, lon),
        )
    conn.commit()


# --------------------------------------------------------------------------- #
# Cloud storage (GCS / S3) — used by upload.py and training/data.py
# --------------------------------------------------------------------------- #
def parse_bucket(bucket: str, backend: str = "auto") -> tuple[str, str]:
    """Return (backend, bucket_name). Accepts 'gs://x', 's3://x', or bare 'x'."""
    if bucket.startswith("gs://"):
        return "gcs", bucket[len("gs://"):].strip("/")
    if bucket.startswith("s3://"):
        return "s3", bucket[len("s3://"):].strip("/")
    if backend == "auto":
        # Default to GCS unless AWS creds are clearly present.
        backend = "s3" if os.environ.get("AWS_ACCESS_KEY_ID") else "gcs"
    return backend, bucket.strip("/")


class StorageClient:
    """Thin uniform wrapper over GCS or S3 for put/get of objects."""

    def __init__(self, bucket: str, backend: str = "auto"):
        self.backend, self.bucket_name = parse_bucket(bucket, backend)
        self._client = None  # lazy

    def _gcs(self):
        if self._client is None:
            from google.cloud import storage  # lazy
            self._client = storage.Client().bucket(self.bucket_name)
        return self._client

    def _s3(self):
        if self._client is None:
            import boto3  # lazy
            self._client = boto3.client("s3")
        return self._client

    def upload_bytes(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        if self.backend == "gcs":
            blob = self._gcs().blob(key)
            blob.upload_from_string(data, content_type=content_type)
            return f"gs://{self.bucket_name}/{key}"
        self._s3().put_object(Bucket=self.bucket_name, Key=key, Body=data,
                              ContentType=content_type)
        return f"s3://{self.bucket_name}/{key}"

    def download_bytes(self, key: str) -> bytes:
        if self.backend == "gcs":
            return self._gcs().blob(key).download_as_bytes()
        obj = self._s3().get_object(Bucket=self.bucket_name, Key=key)
        return obj["Body"].read()

    def uri(self, key: str) -> str:
        scheme = "gs" if self.backend == "gcs" else "s3"
        return f"{scheme}://{self.bucket_name}/{key}"


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def iter_image_files(root: Path, exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".webp")) -> Iterator[Path]:
    exts = tuple(e.lower() for e in exts)
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p
