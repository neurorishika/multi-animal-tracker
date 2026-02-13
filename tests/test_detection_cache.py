from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.helpers.module_loader import load_src_module

detection_cache_mod = load_src_module(
    "multi_tracker/data/detection_cache.py",
    "detection_cache_under_test",
)
DetectionCache = detection_cache_mod.DetectionCache


def test_detection_cache_roundtrip_and_range_checks(tmp_path: Path) -> None:
    cache_path = tmp_path / "detections.npz"

    with DetectionCache(cache_path, mode="w", start_frame=10, end_frame=12) as cache:
        cache.add_frame(
            10,
            meas=[np.array([1.0, 2.0, 0.1], dtype=np.float32)],
            sizes=[12.0],
            shapes=[(10.0, 1.2)],
            confidences=[0.95],
            obb_corners=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)],
            detection_ids=[100001],
        )
        cache.add_frame(
            11,
            meas=[],
            sizes=[],
            shapes=[],
            confidences=[],
            obb_corners=[],
            detection_ids=[],
        )
        cache.add_frame(
            12,
            meas=[np.array([3.0, 4.0, 0.2], dtype=np.float32)],
            sizes=[15.0],
            shapes=[(12.0, 1.5)],
            confidences=[0.8],
            obb_corners=[],
            detection_ids=[120000],
        )
        cache.save()

    assert cache_path.exists()

    with DetectionCache(cache_path, mode="r") as cache:
        assert cache.get_total_frames() >= 13
        assert cache.get_frame_range() == (10, 12)
        assert cache.matches_frame_range(10, 12)
        assert cache.covers_frame_range(10, 12)
        assert cache.get_missing_frames(10, 12) == []

        meas, sizes, shapes, confidences, obb, det_ids = cache.get_frame(10)
        assert len(meas) == 1 and meas[0].shape == (3,)
        assert sizes == [12.0]
        assert shapes == [(10.0, 1.2)]
        assert confidences == [0.949999988079071] or confidences == [0.95]
        assert len(obb) == 1
        assert det_ids == [100001.0]

        meas, sizes, shapes, confidences, obb, det_ids = cache.get_frame(11)
        assert meas == []
        assert sizes == []
        assert shapes == []
        assert confidences == []
        assert obb == []
        assert det_ids == []


def test_detection_cache_missing_frame_reporting(tmp_path: Path) -> None:
    cache_path = tmp_path / "partial.npz"
    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=4) as cache:
        for frame in (0, 2, 4):
            cache.add_frame(frame, [], [], [], [])
        cache.save()

    with DetectionCache(cache_path, mode="r") as cache:
        assert not cache.covers_frame_range(0, 4)
        missing = cache.get_missing_frames(0, 4, max_report=10)
        assert missing == [1, 3]
