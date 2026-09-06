"""Data loading for AntID training.

Three sources, in priority order:
  1. PostgreSQL `images` table (storage_path + split) — the canonical source.
     Images are streamed from the gs:// / s3:// path, cached under LOCAL_DATA_DIR.
  2. A scraper manifest CSV (MANIFEST_CSV) resolved against LOCAL_DATA_DIR. This
     is the offline equivalent of the DB: it carries taxon_id, lat/lon and an
     explicit train/val split, none of which survive the bare-directory walk.
  3. A local directory laid out as {root}/{species_slug}/{image_id}.jpg — the
     last-resort fallback when there is no DB and no manifest.

Whichever source is used, the class index is contiguous 0..N-1, sorted by species
slug, so the mapping is reproducible and matches taxonomy.json. Every taxonomy
entry from every source also carries "genus" (the species name's first token),
so it survives into a freshly written taxonomy.json regardless of which
loader path produced it.
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Probe order matters when resolving {photo_id} -> a file: the same id must
# always resolve to the same extension. IMG_EXTS stays a set for membership tests.
EXT_PROBE_ORDER = (".jpg", ".jpeg", ".png", ".webp")
IMG_EXTS = set(EXT_PROBE_ORDER)


@dataclass
class Sample:
    storage_path: str          # local path or gs://… / s3://… URI
    label: int
    slug: str


def _genus_from_species_name(species_name: str) -> str:
    """First token of the authoritative binomial species name -- the genus.

    Every taxonomy entry must carry this (see the genus-presentation feature
    in docs/plans/northeast-expansion-v1.md), so it has to survive every
    loader path here, not just a one-off hand-built snapshot: train.py writes
    taxonomy.json straight from whatever load_manifest() returns.
    """
    parts = (species_name or "").split()
    return parts[0] if parts else ""


# ---------------------------------------------------------------- transforms
def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    size = cfg["image_size"]
    norm = cfg["normalize"]
    if train:
        aug = cfg["augmentation"]
        cj = aug["color_jitter"]
        ops = [
            # Squish to a fixed square (resize-to-fill; aspect intentionally
            # distorted) — the SAME geometry as the val transform below and
            # api/inference.py. Augmentations apply on top; no crop, so the
            # framing the model sees matches serving exactly.
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip() if aug["random_horizontal_flip"]
            else transforms.Lambda(lambda im: im),
            transforms.RandomRotation(aug["random_rotation_degrees"]),
            transforms.ColorJitter(
                brightness=cj["brightness"],
                contrast=cj["contrast"],
                saturation=cj["saturation"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ]
    else:
        ops = [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ]
    return transforms.Compose(ops)


# ---------------------------------------------------------------- manifest
def _manifest_from_db(database_url: str) -> tuple[list[Sample], dict[int, dict]]:
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT s.slug, s.name, s.common_name, s.taxon_id, "
                "       i.storage_path, i.split, i.lat, i.lon "
                "FROM images i JOIN species s ON s.id = i.species_id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    slugs = sorted({r[0] for r in rows})
    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    meta_by_slug = {r[0]: (r[1], r[2], r[3]) for r in rows}

    samples = [Sample(r[4], slug_to_idx[r[0]], r[0]) for r in rows]
    splits = {r[4]: r[5] for r in rows}
    coords = {r[4]: (r[6], r[7]) for r in rows}
    taxonomy = {
        slug_to_idx[s]: {
            "species_name": meta_by_slug[s][0],
            "common_name": meta_by_slug[s][1],
            "taxon_id": meta_by_slug[s][2],
            "slug": s,
            "genus": _genus_from_species_name(meta_by_slug[s][0]),
        }
        for s in slugs
    }
    # attach split + coords via parallel dicts so we can partition / geo-index
    for smp in samples:
        smp.__dict__["split"] = splits[smp.storage_path]
        la, lo = coords.get(smp.storage_path, (None, None))
        smp.__dict__["lat"] = la
        smp.__dict__["lon"] = lo
    return samples, taxonomy


def _as_float(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _as_int(v: str | None) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _manifest_from_csv(csv_path: Path, root: Path) -> tuple[list[Sample], dict[int, dict]]:
    """Build the manifest from a scraper CSV.

    Columns used: species, slug, taxon_id, photo_id, lat, lon, split. The CSV has
    no path column — each row resolves to {root}/{slug}/{photo_id}.{ext} over
    IMG_EXTS. Rows whose image did not survive clean.py are skipped, so the
    manifest is allowed to be a superset of what is on disk.
    """
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"Manifest has no rows: {csv_path}")
    for col in ("slug", "photo_id"):
        if col not in rows[0]:
            raise SystemExit(f"Manifest {csv_path} is missing required column {col!r}.")

    slugs = sorted({r["slug"] for r in rows})
    slug_to_idx = {s: i for i, s in enumerate(slugs)}

    def resolve(slug: str, photo_id: str) -> Path | None:
        for ext in EXT_PROBE_ORDER:
            p = root / slug / f"{photo_id}{ext}"
            if p.exists():
                return p
        return None

    samples: list[Sample] = []
    missing = 0
    for r in rows:
        path = resolve(r["slug"], r["photo_id"])
        if path is None:
            missing += 1
            continue
        smp = Sample(str(path), slug_to_idx[r["slug"]], r["slug"])
        split = (r.get("split") or "").strip()
        if split:
            # Only set the key when a real value exists. train.py accepts a
            # completely unsplit manifest (random split) but rejects partial
            # split metadata rather than mixing it with reconstructed membership.
            smp.__dict__["split"] = split
        smp.__dict__["lat"] = _as_float(r.get("lat"))
        smp.__dict__["lon"] = _as_float(r.get("lon"))
        samples.append(smp)

    if not samples:
        raise SystemExit(
            f"No manifest row in {csv_path} resolved to a file under {root}. "
            "Is LOCAL_DATA_DIR pointing at the cleaned {slug}/{photo_id}.jpg tree?"
        )

    meta: dict[str, dict] = {}
    for r in rows:
        species_name = ((r.get("species") or "").strip()
                        or r["slug"].replace("-", " ").capitalize())
        meta.setdefault(r["slug"], {
            "species_name": species_name,
            "common_name": (r.get("common_name") or "").strip() or None,
            "taxon_id": _as_int(r.get("taxon_id")),
            "slug": r["slug"],
            "genus": _genus_from_species_name(species_name),
        })
    taxonomy = {slug_to_idx[s]: meta[s] for s in slugs}

    with_coords = sum(1 for s in samples if s.__dict__.get("lat") is not None)
    print(f"[data] manifest {csv_path.name}: {len(samples)} images / {len(slugs)} classes "
          f"({missing} rows with no file on disk, {with_coords} with coordinates)")
    return samples, taxonomy


def _manifest_from_dir(root: Path) -> tuple[list[Sample], dict[int, dict]]:
    slugs = sorted(p.name for p in root.iterdir() if p.is_dir())
    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    samples: list[Sample] = []
    for slug in slugs:
        for img in sorted((root / slug).iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                samples.append(Sample(str(img), slug_to_idx[slug], slug))
    taxonomy = {
        slug_to_idx[s]: {
            "species_name": s.replace("-", " ").capitalize(),
            "common_name": None,
            "taxon_id": None,
            "slug": s,
            "genus": _genus_from_species_name(s.replace("-", " ").capitalize()),
        }
        for s in slugs
    }
    return samples, taxonomy


def load_manifest(cfg: dict) -> tuple[list[Sample], dict[int, dict]]:
    """Return (samples, taxonomy).

    Precedence: DATABASE_URL, then MANIFEST_CSV (+LOCAL_DATA_DIR), then a bare
    LOCAL_DATA_DIR walk. Prefer the manifest over the bare walk when you have
    one: the walk cannot recover taxon_id, coordinates, or the train/val split,
    which is how taxonomy.json ends up full of nulls and the geo index ends up
    empty.
    """
    db = os.environ.get("DATABASE_URL")
    if db:
        return _manifest_from_db(db)

    manifest = os.environ.get("MANIFEST_CSV")
    local = os.environ.get("LOCAL_DATA_DIR")

    if manifest:
        if not local:
            raise SystemExit(
                "MANIFEST_CSV also needs LOCAL_DATA_DIR — the image root that the "
                "manifest's {slug}/{photo_id} pairs resolve against."
            )
        mpath, root = Path(manifest), Path(local)
        if not mpath.exists():
            raise SystemExit(f"MANIFEST_CSV does not exist: {mpath}")
        if not root.exists():
            raise SystemExit(f"LOCAL_DATA_DIR does not exist: {root}")
        return _manifest_from_csv(mpath, root)

    if not local:
        raise SystemExit(
            "No data source: set DATABASE_URL, or MANIFEST_CSV together with "
            "LOCAL_DATA_DIR, or LOCAL_DATA_DIR alone "
            "(a dir of {species_slug}/{image}.jpg)."
        )
    root = Path(local)
    if not root.exists():
        raise SystemExit(f"LOCAL_DATA_DIR does not exist: {root}")
    return _manifest_from_dir(root)


# ---------------------------------------------------------------- dataset
class AntDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg: dict, train: bool):
        self.samples = samples
        self.tf = build_transforms(cfg, train)
        self._cache = Path(os.environ.get("LOCAL_DATA_DIR", ".cache_imgs"))
        self._storage = None  # lazy cloud client

    def __len__(self) -> int:
        return len(self.samples)

    def _read_bytes(self, path: str) -> bytes:
        if path.startswith(("gs://", "s3://")):
            if self._storage is None:
                # Reuse the pipeline's uniform wrapper.
                import sys

                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
                from common import StorageClient  # type: ignore

                self._storage = StorageClient(path.split("//", 1)[0] + "//" +
                                               path.split("//", 1)[1].split("/", 1)[0])
            key = path.split("//", 1)[1].split("/", 1)[1]
            return self._storage.download_bytes(key)
        return Path(path).read_bytes()

    def __getitem__(self, i: int):
        smp = self.samples[i]
        img = Image.open(io.BytesIO(self._read_bytes(smp.storage_path))).convert("RGB")
        return self.tf(img), smp.label
