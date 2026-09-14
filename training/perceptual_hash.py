#!/usr/bin/env python3
"""perceptual_hash.py — pHash/dHash implementation using ONLY Pillow and
NumPy (no additional third-party dependency), for Phase 5F1's
perceptual-duplicate scan.

Every function here is a pure transform over already-in-memory bytes/arrays
-- nothing in this module reads a manifest, writes a file, or knows about
datasets. `compute_all_hashes` is the one entry point that ties the pieces
together: verify -> EXIF-transpose -> grayscale -> 8 dihedral orientations
-> pHash + dHash per orientation.
"""
from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps

IMPLEMENTATION_SOURCE_REL_PATH = "training/perceptual_hash.py"

PHASH_RESIZE = (32, 32)  # (width, height)
PHASH_KEEP = 8
DHASH_RESIZE = (9, 8)  # (width, height) -> 8x8 = 64 comparisons
N_DIHEDRAL_ORIENTATIONS = 8


class PerceptualHashError(RuntimeError):
    """A source-byte or image-decode problem. Always fails closed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_source_bytes(data: bytes, expected_sha256: str) -> None:
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise PerceptualHashError(
            f"source bytes sha256 {actual} != manifest-recorded {expected_sha256!r}"
        )


def load_grayscale_exif_transposed(data: bytes) -> Image.Image:
    """Decodes `data`, applies EXIF orientation transpose (so a
    phone-rotated JPEG hashes the same as its visually-equivalent upright
    form), then converts to grayscale via Pillow's deterministic ITU-R
    601-2 luma transform ('L' mode)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a controlled error here
        raise PerceptualHashError(f"could not decode image bytes: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    return img.convert("L")


def dihedral_orientations(img: Image.Image) -> list[Image.Image]:
    """All 8 dihedral-group orientations: the 4 rotations (0/90/180/270)
    and their horizontal mirrors, in a fixed, deterministic order."""
    rotations = [
        img,
        img.rotate(90, expand=True),
        img.rotate(180, expand=True),
        img.rotate(270, expand=True),
    ]
    mirrors = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in rotations]
    orientations = rotations + mirrors
    assert len(orientations) == N_DIHEDRAL_ORIENTATIONS
    return orientations


def dct2_matrix(n: int) -> np.ndarray:
    """The explicit orthonormal DCT-II basis matrix C (n x n, float64) such
    that D = C @ A @ C.T is the 2-D DCT-II of A. Built directly from the
    cosine formula -- no FFT-based shortcut -- so the algorithm is exactly
    what the frozen contract documents, not an implementation detail of
    whichever DCT routine happened to be available."""
    k = np.arange(n, dtype=np.float64).reshape(-1, 1)
    x = np.arange(n, dtype=np.float64).reshape(1, -1)
    c = np.cos(np.pi * (2.0 * x + 1.0) * k / (2.0 * n))
    c *= np.sqrt(2.0 / n)
    c[0, :] *= 1.0 / np.sqrt(2.0)
    return c.astype(np.float64)


def dct2(block: np.ndarray) -> np.ndarray:
    n = block.shape[0]
    if block.shape[0] != block.shape[1]:
        raise PerceptualHashError(f"dct2 requires a square block, got shape {block.shape}")
    c = dct2_matrix(n)
    return c @ block.astype(np.float64) @ c.T


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits.flatten().tolist():
        value = (value << 1) | (1 if b else 0)
    return value


def phash64(img: Image.Image) -> int:
    """Resize to 32x32 (Pillow LANCZOS) -> explicit orthonormal 2-D DCT-II
    (float64) -> top-left 8x8 coefficients (row-major) -> median-threshold:
    bit is 1 only when the coefficient is STRICTLY greater than the
    median."""
    resized = img.resize(PHASH_RESIZE, Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float64)
    coeffs_full = dct2(arr)
    coeffs = coeffs_full[:PHASH_KEEP, :PHASH_KEEP]
    median = np.median(coeffs)
    bits = coeffs > median
    return _bits_to_int(bits)


def dhash64(img: Image.Image) -> int:
    """Resize to 9x8 (Pillow LANCZOS) -> bit is 1 only when the left pixel
    is STRICTLY greater than the right neighbor -> 8 rows x 8 comparisons
    = 64 bits."""
    resized = img.resize(DHASH_RESIZE, Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.int16)
    bits = arr[:, :-1] > arr[:, 1:]
    return _bits_to_int(bits)


def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def min_hamming_across_orientations(hashes_a: list[int], hashes_b: list[int]) -> int:
    return min(hamming64(x, y) for x in hashes_a for y in hashes_b)


def compute_all_hashes(data: bytes, expected_sha256: str) -> dict[str, Any]:
    """The one entry point: verify -> EXIF-transpose -> grayscale -> 8
    dihedral orientations -> pHash + dHash per orientation. Returns a dict
    safe to serialize to JSON (no numpy types, no PIL objects)."""
    verify_source_bytes(data, expected_sha256)
    base = load_grayscale_exif_transposed(data)
    orientations = dihedral_orientations(base)
    return {
        "sha256": expected_sha256,
        "phashes": [phash64(o) for o in orientations],
        "dhashes": [dhash64(o) for o in orientations],
        "pillow_version": _pillow_version(),
        "numpy_version": np.__version__,
    }


def _pillow_version() -> str:
    import PIL
    return PIL.__version__


class BKTree:
    """A BK-tree (Burkhard-Keller tree) over the Hamming-distance metric --
    an EXACT radius-search index, not an approximation: `query(value,
    radius)` always returns every indexed value within `radius`, using the
    triangle inequality to prune subtrees that provably cannot contain a
    match, never to skip ones that might. Distance is always `hamming64`.

    Each node stores a value, the list of payloads inserted with that exact
    value (so identical hashes for different images share one node), and a
    dict of child nodes keyed by their distance from this node's value."""

    __slots__ = ("root",)

    def __init__(self) -> None:
        self.root: list | None = None  # [value, [payloads], {distance: child_node}]

    def add(self, value: int, payload) -> None:
        if self.root is None:
            self.root = [value, [payload], {}]
            return
        node = self.root
        while True:
            d = hamming64(value, node[0])
            if d == 0:
                node[1].append(payload)
                return
            child = node[2].get(d)
            if child is None:
                node[2][d] = [value, [payload], {}]
                return
            node = child

    def query(self, value: int, radius: int) -> list:
        """Every payload whose indexed hash is within `radius` of `value`
        (inclusive), in an unspecified but DETERMINISTIC order (insertion
        order within each node, nodes visited in a fixed stack order) --
        callers that need a canonical output order sort it themselves."""
        if self.root is None:
            return []
        results: list = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = hamming64(value, node[0])
            if d <= radius:
                results.extend(node[1])
            lo, hi = d - radius, d + radius
            for child_d in sorted(node[2]):
                if lo <= child_d <= hi:
                    stack.append(node[2][child_d])
        return results


def build_bk_tree(hash_to_payloads: list[tuple[int, object]]) -> BKTree:
    tree = BKTree()
    for value, payload in hash_to_payloads:
        tree.add(value, payload)
    return tree


def is_candidate_pair(phashes_a: list[int], dhashes_a: list[int],
                      phashes_b: list[int], dhashes_b: list[int], *,
                      phash_max: int = 10, dhash_max: int = 8) -> tuple[bool, int, int]:
    """Returns (is_candidate, min_phash_distance, min_dhash_distance).
    Flagged when EITHER minimum distance across all orientation-hash
    combinations is <= its threshold (boundary equality included). This
    generates a CANDIDATE only -- never an automatic duplicate
    declaration."""
    p = min_hamming_across_orientations(phashes_a, phashes_b)
    d = min_hamming_across_orientations(dhashes_a, dhashes_b)
    return (p <= phash_max or d <= dhash_max), p, d
