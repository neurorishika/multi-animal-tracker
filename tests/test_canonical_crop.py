"""Tests for multi_tracker.core.tracking.canonical_crop module."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from multi_tracker.core.tracking.canonical_crop import (
    CanonicalCropResult,
    _compose_affine,
    _rotation_matrix,
    apply_headtail_rotation,
    compute_alignment_affine,
    compute_crop_dimensions,
    extract_and_classify_batch,
    extract_canonical_crop,
    invert_keypoints,
)

# ---------------------------------------------------------------------------
# compute_crop_dimensions
# ---------------------------------------------------------------------------


class TestComputeCropDimensions:
    def test_basic(self):
        w, h = compute_crop_dimensions(128, 2.0)
        assert w == 128
        assert h == 64

    def test_aspect_ratio_1(self):
        w, h = compute_crop_dimensions(256, 1.0)
        assert w == 256
        assert h == 256

    def test_aspect_ratio_3(self):
        w, h = compute_crop_dimensions(300, 3.0)
        assert w == 300
        assert h == 100

    def test_minimum_clamping(self):
        # very small long edge
        w, h = compute_crop_dimensions(4, 2.0)
        assert w == 8  # clamped
        assert h == 8  # clamped (4/2=2 → 8 clamped)

    def test_ar_less_than_one_clamped(self):
        # AR < 1 is clamped to 1
        w, h = compute_crop_dimensions(100, 0.5)
        assert w == 100
        assert h == 100  # ar clamped to 1.0


# ---------------------------------------------------------------------------
# compute_alignment_affine
# ---------------------------------------------------------------------------


def _make_obb(cx, cy, w, h, angle_deg=0.0):
    """Create OBB corners for a rectangle centred at (cx, cy)."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    hw, hh = w / 2.0, h / 2.0
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        dx = sx * hw
        dy = sy * hh
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        corners.append([x, y])
    return np.array(corners, dtype=np.float32)


class TestComputeAlignmentAffine:
    def test_axis_aligned(self):
        corners = _make_obb(100, 100, 80, 40, 0)
        M, angle = compute_alignment_affine(corners, 128, 64, 0.0)
        assert M.shape == (2, 3)
        assert abs(angle) < 0.01  # axis-aligned → ~0

    def test_rotated(self):
        corners = _make_obb(200, 200, 80, 40, 45)
        M, angle = compute_alignment_affine(corners, 128, 64, 0.0)
        assert M.shape == (2, 3)
        assert abs(angle - math.radians(45)) < 0.1

    def test_degenerate_raises(self):
        # all four corners at the same point
        corners = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float32)
        with pytest.raises(ValueError, match="Degenerate"):
            compute_alignment_affine(corners, 128, 64, 0.0)

    def test_centroid_maps_to_canvas_centre(self):
        corners = _make_obb(150, 80, 60, 30, 20)
        M, _ = compute_alignment_affine(corners, 128, 64, 0.0)
        # centroid of OBB in frame space
        cx = np.mean(corners[:, 0])
        cy = np.mean(corners[:, 1])
        # map centroid through M
        pt = M @ np.array([cx, cy, 1.0])
        assert abs(pt[0] - 64) < 2.0  # near canvas centre x
        assert abs(pt[1] - 32) < 2.0  # near canvas centre y


# ---------------------------------------------------------------------------
# extract_canonical_crop
# ---------------------------------------------------------------------------


class TestExtractCanonicalCrop:
    def test_basic_crop(self):
        frame = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        corners = _make_obb(200, 150, 80, 40, 0)
        M, _ = compute_alignment_affine(corners, 128, 64, 0.1)
        crop = extract_canonical_crop(frame, M, 128, 64)
        assert crop.shape == (64, 128, 3)
        assert crop.dtype == np.uint8

    def test_crop_with_foreign(self):
        frame = np.full((300, 400, 3), 128, dtype=np.uint8)
        corners = _make_obb(200, 150, 80, 40, 0)
        foreign = [_make_obb(240, 150, 30, 20, 0)]
        M, _ = compute_alignment_affine(corners, 128, 64, 0.1)
        crop = extract_canonical_crop(
            frame, M, 128, 64, bg_color=(0, 0, 0), foreign_corners=foreign
        )
        assert crop.shape == (64, 128, 3)
        # Should have some black pixels from foreign masking
        assert np.any(crop == 0)


# ---------------------------------------------------------------------------
# apply_headtail_rotation
# ---------------------------------------------------------------------------


class TestApplyHeadtailRotation:
    def _make_crop_and_align(self):
        crop = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
        corners = _make_obb(200, 150, 80, 40, 0)
        M_align, _ = compute_alignment_affine(corners, 128, 64, 0.1)
        return crop, M_align

    def test_right_noop(self):
        crop, M_align = self._make_crop_and_align()
        rotated, M_can, M_inv, offset = apply_headtail_rotation(
            crop, M_align, "right", 128, 64
        )
        assert rotated.shape == (64, 128, 3)
        assert abs(offset) < 1e-6
        assert M_can.shape == (2, 3)
        assert M_inv.shape == (2, 3)

    def test_left_flips(self):
        crop, M_align = self._make_crop_and_align()
        rotated, M_can, M_inv, offset = apply_headtail_rotation(
            crop, M_align, "left", 128, 64
        )
        assert rotated.shape == (64, 128, 3)
        assert abs(offset - math.pi) < 0.01

    def test_unknown_noop(self):
        crop, M_align = self._make_crop_and_align()
        rotated, M_can, M_inv, offset = apply_headtail_rotation(
            crop, M_align, "unknown", 128, 64
        )
        assert abs(offset) < 1e-6

    def test_up_treated_as_unknown(self):
        crop, M_align = self._make_crop_and_align()
        rotated, _, _, offset = apply_headtail_rotation(
            crop, M_align, "up", 128, 64, treat_updown_as_unknown=True
        )
        assert abs(offset) < 1e-6
        assert rotated.shape == (64, 128, 3)

    def test_up_rotates_when_allowed(self):
        crop, M_align = self._make_crop_and_align()
        rotated, _, _, offset = apply_headtail_rotation(
            crop, M_align, "up", 128, 64, treat_updown_as_unknown=False
        )
        assert abs(offset + math.pi / 2) < 0.01
        # 90° CW changes shape: (64, 128) → (128, 64)
        assert rotated.shape == (128, 64, 3)

    def test_down_rotates_when_allowed(self):
        crop, M_align = self._make_crop_and_align()
        rotated, _, _, offset = apply_headtail_rotation(
            crop, M_align, "down", 128, 64, treat_updown_as_unknown=False
        )
        assert abs(offset - math.pi / 2) < 0.01
        assert rotated.shape == (128, 64, 3)


# ---------------------------------------------------------------------------
# invert_keypoints
# ---------------------------------------------------------------------------


class TestInvertKeypoints:
    def test_identity(self):
        M_inv = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        kp = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = invert_keypoints(kp, M_inv)
        np.testing.assert_allclose(result, kp, atol=1e-6)

    def test_translation(self):
        M_inv = np.array([[1, 0, 50], [0, 1, 100]], dtype=np.float64)
        kp = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = invert_keypoints(kp, M_inv)
        expected = np.array([[60, 120], [80, 140]], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_with_confidence_column(self):
        M_inv = np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float64)
        kp = np.array([[5, 5, 0.9], [10, 10, 0.8]], dtype=np.float64)
        result = invert_keypoints(kp, M_inv)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[:, :2], [[15, 25], [20, 30]], atol=1e-6)
        np.testing.assert_allclose(result[:, 2], [0.9, 0.8], atol=1e-6)

    def test_round_trip(self):
        corners = _make_obb(200, 150, 80, 40, 30)
        M_align, _ = compute_alignment_affine(corners, 128, 64, 0.1)
        M_inv = cv2.invertAffineTransform(M_align)

        # A known point in frame space
        frame_pts = np.array([[200, 150], [180, 140]], dtype=np.float64)

        # Frame → canonical
        ones = np.ones((2, 1), dtype=np.float64)
        pts_h = np.hstack([frame_pts, ones])
        canonical = (M_align @ pts_h.T).T

        # Canonical → frame via invert_keypoints
        back = invert_keypoints(canonical, M_inv)
        np.testing.assert_allclose(back, frame_pts, atol=0.5)

    def test_empty(self):
        M_inv = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        kp = np.zeros((0, 2), dtype=np.float64)
        result = invert_keypoints(kp, M_inv)
        assert result.shape == (0, 2)


# ---------------------------------------------------------------------------
# _compose_affine
# ---------------------------------------------------------------------------


class TestComposeAffine:
    def test_identity_composition(self):
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        M = np.array([[2, 0, 10], [0, 3, 20]], dtype=np.float64)
        result = _compose_affine(I, M)
        np.testing.assert_allclose(result, M, atol=1e-10)

        result2 = _compose_affine(M, I)
        np.testing.assert_allclose(result2, M, atol=1e-10)

    def test_translation_composes(self):
        T1 = np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float64)
        T2 = np.array([[1, 0, 5], [0, 1, 7]], dtype=np.float64)
        result = _compose_affine(T2, T1)
        expected = np.array([[1, 0, 15], [0, 1, 27]], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# _rotation_matrix
# ---------------------------------------------------------------------------


class TestRotationMatrix:
    def test_zero_rotation(self):
        M = _rotation_matrix(0.0, 128, 64, 128, 64)
        np.testing.assert_allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-10)

    def test_180_rotation(self):
        M = _rotation_matrix(math.pi, 128, 64, 128, 64)
        # 180° about centre should map (0,0) → (128, 64)
        pt = M @ np.array([0, 0, 1])
        np.testing.assert_allclose(pt, [128, 64], atol=0.5)


# ---------------------------------------------------------------------------
# extract_and_classify_batch
# ---------------------------------------------------------------------------


class TestExtractAndClassifyBatch:
    def test_single_frame_single_det(self):
        frame = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        corners = _make_obb(200, 150, 80, 40, 15)
        results = extract_and_classify_batch(
            [frame], [[corners]], 128, 64, padding_fraction=0.1
        )
        assert len(results) == 1
        assert len(results[0]) == 1
        r = results[0][0]
        assert isinstance(r, CanonicalCropResult)
        assert r.crop.shape == (64, 128, 3)
        assert r.M_canonical.shape == (2, 3)
        assert r.M_inverse.shape == (2, 3)
        assert not r.directed

    def test_empty_detection(self):
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        results = extract_and_classify_batch([frame], [[]], 128, 64)
        assert len(results) == 1
        assert len(results[0]) == 0

    def test_degenerate_obb_returns_none(self):
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        degen = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float32)
        results = extract_and_classify_batch([frame], [[degen]], 128, 64)
        assert results[0][0] is None

    def test_multi_frame(self):
        frames = [
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8) for _ in range(3)
        ]
        corners_per_frame = [[_make_obb(100, 100, 60, 30, i * 10)] for i in range(3)]
        results = extract_and_classify_batch(frames, corners_per_frame, 128, 64)
        assert len(results) == 3
        for fr in results:
            assert len(fr) == 1
            assert fr[0] is not None
