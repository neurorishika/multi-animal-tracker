"""Tests for TagObservationCache (NPZ-backed per-frame tag storage)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "multi_tracker/data/tag_observation_cache.py",
    "tag_obs_cache_under_test",
)
TagObservationCache = mod.TagObservationCache


def test_roundtrip_write_read(tmp_path: Path) -> None:
    """Write a few frames, save, reopen read-only, and verify contents."""
    cache_path = tmp_path / "tags.npz"

    cache = TagObservationCache(str(cache_path), mode="w")
    cache.add_frame(
        frame_idx=0,
        tag_ids=np.array([1, 5], dtype=np.int32),
        centers_xy=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
        det_indices=np.array([0, 2], dtype=np.int32),
        hammings=np.array([0, 1], dtype=np.int32),
    )
    cache.add_frame(
        frame_idx=1,
        tag_ids=np.array([], dtype=np.int32),
        centers_xy=np.zeros((0, 2), dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
        det_indices=np.array([], dtype=np.int32),
        hammings=np.array([], dtype=np.int32),
    )
    cache.add_frame(
        frame_idx=2,
        tag_ids=np.array([3], dtype=np.int32),
        centers_xy=np.array([[50.0, 60.0]], dtype=np.float32),
        corners=np.zeros((1, 4, 2), dtype=np.float32),
        det_indices=np.array([1], dtype=np.int32),
        hammings=np.array([0], dtype=np.int32),
    )
    cache.save()
    cache.close()

    # Reopen in read mode
    cache2 = TagObservationCache(str(cache_path), mode="r")

    obs0 = cache2.get_frame(0)
    assert len(obs0["tag_ids"]) == 2
    np.testing.assert_array_equal(obs0["tag_ids"], [1, 5])
    np.testing.assert_array_equal(obs0["det_indices"], [0, 2])

    obs1 = cache2.get_frame(1)
    assert len(obs1["tag_ids"]) == 0

    obs2 = cache2.get_frame(2)
    assert len(obs2["tag_ids"]) == 1
    assert obs2["tag_ids"][0] == 3

    # Non-existent frame returns empty
    obs_missing = cache2.get_frame(999)
    assert len(obs_missing["tag_ids"]) == 0

    cache2.close()


def test_is_compatible(tmp_path: Path) -> None:
    """Cache with matching version should be compatible."""
    cache_path = tmp_path / "tags.npz"
    cache = TagObservationCache(str(cache_path), mode="w")
    cache.add_frame(
        frame_idx=0,
        tag_ids=np.array([1], dtype=np.int32),
        centers_xy=np.array([[0.0, 0.0]], dtype=np.float32),
        corners=np.zeros((1, 4, 2), dtype=np.float32),
        det_indices=np.array([0], dtype=np.int32),
        hammings=np.array([0], dtype=np.int32),
    )
    cache.save()
    cache.close()

    cache2 = TagObservationCache(str(cache_path), mode="r")
    assert cache2.is_compatible()
    cache2.close()
