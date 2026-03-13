"""Tests for per-frame tag feature helpers (tag_features.py)."""

from __future__ import annotations

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "multi_tracker/core/tracking/tag_features.py",
    "tag_features_under_test",
)
NO_TAG = mod.NO_TAG
build_tag_detection_map = mod.build_tag_detection_map
build_detection_tag_id_list = mod.build_detection_tag_id_list
TrackTagHistory = mod.TrackTagHistory


# --- build_tag_detection_map / build_detection_tag_id_list ---


class FakeTagCache:
    """Minimal stand-in for TagObservationCache in read mode."""

    def __init__(self, data):
        self._data = data

    def get_frame(self, frame_idx):
        import numpy as np

        empty = {
            "tag_ids": np.array([], dtype=np.int32),
            "det_indices": np.array([], dtype=np.int32),
        }
        return self._data.get(frame_idx, empty)


def test_build_tag_detection_map_basic():
    import numpy as np

    cache = FakeTagCache(
        {
            5: {
                "tag_ids": np.array([10, 20], dtype=np.int32),
                "det_indices": np.array([0, 3], dtype=np.int32),
            }
        }
    )
    result = build_tag_detection_map(cache, 5)
    assert result == {0: 10, 3: 20}


def test_build_tag_detection_map_none_cache():
    result = build_tag_detection_map(None, 0)
    assert result == {}


def test_build_detection_tag_id_list_basic():
    tag_map = {0: 10, 3: 20}
    result = build_detection_tag_id_list(tag_map, 5)
    assert result == [10, NO_TAG, NO_TAG, 20, NO_TAG]


def test_build_detection_tag_id_list_empty():
    result = build_detection_tag_id_list({}, 3)
    assert result == [NO_TAG, NO_TAG, NO_TAG]


# --- TrackTagHistory ---


def test_track_tag_history_majority_vote():
    h = TrackTagHistory(n_tracks=4, window=10)
    # Track 0 sees tag 5 three times, tag 7 once
    h.record(0, 1, 5)
    h.record(0, 2, 5)
    h.record(0, 3, 7)
    h.record(0, 4, 5)
    assert h.majority_tag(0) == 5


def test_track_tag_history_no_observations():
    h = TrackTagHistory(n_tracks=2, window=5)
    assert h.majority_tag(0) == NO_TAG
    assert h.majority_tag(1) == NO_TAG


def test_track_tag_history_window_eviction():
    h = TrackTagHistory(n_tracks=1, window=3)
    h.record(0, 1, 10)
    h.record(0, 2, 10)
    # Frame 5 is beyond window from frame 1 (5 - 3 = 2, so frame 1 is evicted)
    h.record(0, 5, 20)
    h.record(0, 6, 20)
    # Tag 10 at frame 1 was evicted (cutoff=6-3=3), frame 2 also evicted
    assert h.majority_tag(0) == 20


def test_track_tag_history_clear():
    h = TrackTagHistory(n_tracks=2, window=10)
    h.record(0, 1, 5)
    h.clear_track(0)
    assert h.majority_tag(0) == NO_TAG


def test_track_tag_history_resize():
    h = TrackTagHistory(n_tracks=2, window=10)
    h.resize(5)
    assert h.n_tracks == 5
    h.record(4, 1, 99)
    assert h.majority_tag(4) == 99


def test_track_tag_history_build_list():
    h = TrackTagHistory(n_tracks=3, window=10)
    h.record(0, 1, 5)
    h.record(2, 1, 7)
    result = h.build_track_tag_id_list(3)
    assert result == [5, NO_TAG, 7]


def test_no_tag_skipped_on_record():
    """Recording NO_TAG should be a no-op."""
    h = TrackTagHistory(n_tracks=1, window=10)
    h.record(0, 1, NO_TAG)
    assert h.majority_tag(0) == NO_TAG
