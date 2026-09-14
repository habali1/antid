#!/usr/bin/env python3
"""test_perceptual_hash.py — offline tests for perceptual_hash.py.

Every image used here is generated synthetically in memory via PIL/NumPy --
never a real dataset photograph.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from PIL import Image

import perceptual_hash as ph  # noqa: E402


def synthetic_image_bytes(seed: int, size=(64, 64), fmt: str = "PNG") -> bytes:
    rng = np.random.RandomState(seed)
    arr = (rng.rand(size[1], size[0], 3) * 255).astype("uint8")
    # add coarse structure (blocks) so resize/DCT behavior isn't pure noise
    arr[: size[1] // 2, : size[0] // 2] = (200, 50, 50)
    arr[size[1] // 2:, size[0] // 2:] = (50, 50, 200)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestVerifySourceBytes(unittest.TestCase):
    def test_matching_hash_passes(self):
        data = synthetic_image_bytes(1)
        ph.verify_source_bytes(data, ph.sha256_bytes(data))

    def test_mismatched_hash_raises(self):
        data = synthetic_image_bytes(1)
        with self.assertRaises(ph.PerceptualHashError):
            ph.verify_source_bytes(data, "0" * 64)


class TestDihedralOrientations(unittest.TestCase):
    def test_produces_exactly_eight(self):
        data = synthetic_image_bytes(2)
        img = ph.load_grayscale_exif_transposed(data)
        orientations = ph.dihedral_orientations(img)
        self.assertEqual(len(orientations), 8)

    def test_all_grayscale(self):
        data = synthetic_image_bytes(2)
        img = ph.load_grayscale_exif_transposed(data)
        for o in ph.dihedral_orientations(img):
            self.assertEqual(o.mode, "L")


class TestDct2(unittest.TestCase):
    def test_matrix_is_orthonormal(self):
        c = ph.dct2_matrix(8)
        product = c @ c.T
        np.testing.assert_allclose(product, np.eye(8), atol=1e-10)

    def test_non_square_block_raises(self):
        with self.assertRaises(ph.PerceptualHashError):
            ph.dct2(np.zeros((4, 8)))


class TestHashDeterminism(unittest.TestCase):
    def test_phash_deterministic_for_same_bytes(self):
        data = synthetic_image_bytes(3)
        img1 = ph.load_grayscale_exif_transposed(data)
        img2 = ph.load_grayscale_exif_transposed(data)
        self.assertEqual(ph.phash64(img1), ph.phash64(img2))

    def test_dhash_deterministic_for_same_bytes(self):
        data = synthetic_image_bytes(3)
        img1 = ph.load_grayscale_exif_transposed(data)
        img2 = ph.load_grayscale_exif_transposed(data)
        self.assertEqual(ph.dhash64(img1), ph.dhash64(img2))

    def test_hashes_are_64_bit(self):
        data = synthetic_image_bytes(3)
        img = ph.load_grayscale_exif_transposed(data)
        self.assertLess(ph.phash64(img), 1 << 64)
        self.assertLess(ph.dhash64(img), 1 << 64)
        self.assertGreaterEqual(ph.phash64(img), 0)
        self.assertGreaterEqual(ph.dhash64(img), 0)


class TestRecompressionRotationMirrorCandidates(unittest.TestCase):
    """Same underlying synthetic source, several derivative transforms --
    each derivative's set of 8 orientation-hashes must include a
    near-zero-distance match against the base image's own 8 orientations
    (since the derivative's un-rotated form IS one of the base's dihedral
    orientations, or a recompression of it), confirming the candidate rule
    actually fires for resize/recompression/rotation/mirror derivatives."""

    def setUp(self):
        self.base_bytes = synthetic_image_bytes(42, size=(80, 60))
        self.base_hashes = ph.compute_all_hashes(self.base_bytes, ph.sha256_bytes(self.base_bytes))

    def _derivative_hashes(self, img: Image.Image, fmt="JPEG", quality=85) -> dict:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality) if fmt == "JPEG" else img.save(buf, format=fmt)
        data = buf.getvalue()
        return ph.compute_all_hashes(data, ph.sha256_bytes(data))

    def test_recompressed_jpeg_is_a_candidate(self):
        base_img = Image.open(io.BytesIO(self.base_bytes))
        derived = self._derivative_hashes(base_img, fmt="JPEG", quality=70)
        is_candidate, p, d = ph.is_candidate_pair(
            self.base_hashes["phashes"], self.base_hashes["dhashes"],
            derived["phashes"], derived["dhashes"])
        self.assertTrue(is_candidate, f"phash_dist={p} dhash_dist={d}")

    def test_resized_image_is_a_candidate(self):
        base_img = Image.open(io.BytesIO(self.base_bytes))
        resized = base_img.resize((40, 30), Image.LANCZOS)
        derived = self._derivative_hashes(resized, fmt="PNG")
        is_candidate, p, d = ph.is_candidate_pair(
            self.base_hashes["phashes"], self.base_hashes["dhashes"],
            derived["phashes"], derived["dhashes"])
        self.assertTrue(is_candidate, f"phash_dist={p} dhash_dist={d}")

    def test_rotated_90_image_is_a_candidate(self):
        base_img = Image.open(io.BytesIO(self.base_bytes))
        rotated = base_img.rotate(90, expand=True)
        derived = self._derivative_hashes(rotated, fmt="PNG")
        is_candidate, p, d = ph.is_candidate_pair(
            self.base_hashes["phashes"], self.base_hashes["dhashes"],
            derived["phashes"], derived["dhashes"])
        self.assertTrue(is_candidate, f"phash_dist={p} dhash_dist={d}")

    def test_mirrored_image_is_a_candidate(self):
        base_img = Image.open(io.BytesIO(self.base_bytes))
        mirrored = base_img.transpose(Image.FLIP_LEFT_RIGHT)
        derived = self._derivative_hashes(mirrored, fmt="PNG")
        is_candidate, p, d = ph.is_candidate_pair(
            self.base_hashes["phashes"], self.base_hashes["dhashes"],
            derived["phashes"], derived["dhashes"])
        self.assertTrue(is_candidate, f"phash_dist={p} dhash_dist={d}")

    def test_unrelated_image_is_not_a_candidate(self):
        other_bytes = synthetic_image_bytes(9999, size=(80, 60))
        other_hashes = ph.compute_all_hashes(other_bytes, ph.sha256_bytes(other_bytes))
        is_candidate, p, d = ph.is_candidate_pair(
            self.base_hashes["phashes"], self.base_hashes["dhashes"],
            other_hashes["phashes"], other_hashes["dhashes"])
        self.assertFalse(is_candidate, f"phash_dist={p} dhash_dist={d}")


FAR_A = 0x0000000000000000
FAR_B = 0xFFFFFFFFFFFFFFFF  # distance 64 from FAR_A -- always safely outside any threshold


class TestBoundaryDistances(unittest.TestCase):
    """Exact pHash/dHash boundary behavior: 10/11 for pHash, 8/9 for dHash,
    using synthetic (hand-constructed) hash integers -- not real images --
    to pin the exact boundary condition of is_candidate_pair itself. The
    OTHER hash type in each test is pinned to FAR_A/FAR_B (distance 64) so
    it can never accidentally satisfy its own threshold and confound the
    result."""

    def test_phash_distance_exactly_10_is_candidate(self):
        a = 0
        b = int("1" * 10 + "0" * 54, 2)  # exactly 10 bits differ
        self.assertEqual(ph.hamming64(a, b), 10)
        is_candidate, p, d = ph.is_candidate_pair([a], [FAR_A], [b], [FAR_B])
        self.assertTrue(is_candidate)
        self.assertEqual(p, 10)

    def test_phash_distance_11_is_not_candidate_via_phash_alone(self):
        a = 0
        b = int("1" * 11 + "0" * 53, 2)  # exactly 11 bits differ
        self.assertEqual(ph.hamming64(a, b), 11)
        is_candidate, p, d = ph.is_candidate_pair([a], [FAR_A], [b], [FAR_B])
        self.assertFalse(is_candidate)
        self.assertEqual(p, 11)

    def test_dhash_distance_exactly_8_is_candidate(self):
        a = 0
        b = int("1" * 8 + "0" * 56, 2)
        self.assertEqual(ph.hamming64(a, b), 8)
        is_candidate, p, d = ph.is_candidate_pair([FAR_A], [a], [FAR_B], [b])
        self.assertTrue(is_candidate)
        self.assertEqual(d, 8)

    def test_dhash_distance_9_is_not_candidate(self):
        a = 0
        b = int("1" * 9 + "0" * 55, 2)
        self.assertEqual(ph.hamming64(a, b), 9)
        is_candidate, p, d = ph.is_candidate_pair([FAR_A], [a], [FAR_B], [b])
        self.assertFalse(is_candidate)
        self.assertEqual(d, 9)

    def test_min_across_orientations_used(self):
        # One close pair (distance 5) buried among decoys verified to be
        # >= 5 away from every OTHER cross-combination, so the true minimum
        # is unambiguous.
        close_a, close_b = 0, int("11111", 2)
        self.assertEqual(ph.hamming64(close_a, close_b), 5)
        decoy_a, decoy_b = 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA
        for x, y in ((decoy_a, close_b), (close_a, decoy_b), (decoy_a, decoy_b)):
            self.assertGreaterEqual(ph.hamming64(x, y), 5)

        a_list, b_list = [close_a, decoy_a], [close_b, decoy_b]
        is_candidate, p, d = ph.is_candidate_pair(a_list, [FAR_A], b_list, [FAR_B])
        self.assertTrue(is_candidate)
        self.assertEqual(p, 5)


class TestBKTree(unittest.TestCase):
    def test_empty_tree_query_returns_empty(self):
        tree = ph.BKTree()
        self.assertEqual(tree.query(0, 10), [])

    def test_single_exact_match(self):
        tree = ph.BKTree()
        tree.add(0, "payload-a")
        self.assertEqual(tree.query(0, 0), ["payload-a"])

    def test_within_radius_returned_outside_radius_excluded(self):
        tree = ph.BKTree()
        tree.add(0, "near")  # distance 0
        tree.add(int("1" * 10, 2), "boundary")  # distance 10
        tree.add(int("1" * 11, 2), "just-outside")  # distance 11
        results = set(tree.query(0, 10))
        self.assertEqual(results, {"near", "boundary"})

    def test_multiple_payloads_at_same_hash(self):
        tree = ph.BKTree()
        tree.add(5, "a")
        tree.add(5, "b")
        self.assertEqual(set(tree.query(5, 0)), {"a", "b"})

    def test_build_bk_tree_helper(self):
        tree = ph.build_bk_tree([(0, "a"), (1, "b"), (int("1" * 20, 2), "far")])
        self.assertEqual(set(tree.query(0, 2)), {"a", "b"})

    def test_random_synthetic_fixture_indexed_matches_brute_force(self):
        import random
        rng = random.Random(12345)
        pool_a = [(rng.getrandbits(64), f"a{i}") for i in range(40)]
        pool_b = [(rng.getrandbits(64), f"b{i}") for i in range(60)]
        # force some genuinely close pairs so radius queries have hits
        for i in range(5):
            base_value, base_payload = pool_a[i]
            flips = rng.sample(range(64), 4)
            close_value = base_value
            for bit in flips:
                close_value ^= (1 << bit)
            pool_b.append((close_value, f"close-to-{base_payload}"))

        tree_b = ph.build_bk_tree(pool_b)
        radius = 10

        indexed_pairs = set()
        for value_a, payload_a in pool_a:
            for payload_b in tree_b.query(value_a, radius):
                indexed_pairs.add((payload_a, payload_b))

        brute_pairs = set()
        for value_a, payload_a in pool_a:
            for value_b, payload_b in pool_b:
                if ph.hamming64(value_a, value_b) <= radius:
                    brute_pairs.add((payload_a, payload_b))

        self.assertEqual(indexed_pairs, brute_pairs)
        self.assertGreater(len(indexed_pairs), 0, "fixture must actually exercise some matches")


if __name__ == "__main__":
    unittest.main(verbosity=2)
