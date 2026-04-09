"""Tests for FilterKit core deduplication behavior."""

from __future__ import annotations

import numpy as np

from hydra_suite.filterkit.core import FilterKitCore


def _make_item(path: str, det_id: int, signature, color_signature=None):
    item = {
        "path": path,
        "filename": path.rsplit("/", 1)[-1],
        "det_id": det_id,
        "frame_idx": det_id // 10000,
        "det_idx": det_id % 10000,
        "dedup_signature": signature,
    }
    if color_signature is not None:
        item["color_signature"] = color_signature
    return item


def _one_hot(size: int, index: int) -> np.ndarray:
    signature = np.zeros((size,), dtype=np.float32)
    signature[index] = 1.0
    return signature


def test_filterkit_hash_dedup_keeps_first_nonduplicate_match() -> None:
    core = FilterKitCore()
    dataset = [
        _make_item("/tmp/a.png", 1, 0b0000),
        _make_item("/tmp/b.png", 2, 0b0001),
        _make_item("/tmp/c.png", 3, 0b0011),
        _make_item("/tmp/d.png", 4, 0b1000),
    ]

    kept, groups = core.deduplicate_by_hash(
        dataset,
        threshold=1,
        method="phash",
        return_groups=True,
    )

    assert [item["path"] for item in kept] == ["/tmp/a.png", "/tmp/c.png"]
    assert groups == [
        {
            "hash": "0",
            "count": 3,
            "paths": ["/tmp/a.png", "/tmp/b.png", "/tmp/d.png"],
            "method": "phash",
        }
    ]


def test_filterkit_hash_dedup_preserves_distinct_colors() -> None:
    core = FilterKitCore()
    color_a = _one_hot(8 * 8 * 8, 0)
    color_b = _one_hot(8 * 8 * 8, 1)
    dataset = [
        _make_item("/tmp/a.png", 1, 123, color_signature=color_a),
        _make_item("/tmp/b.png", 2, 123, color_signature=color_b),
        _make_item("/tmp/c.png", 3, 123, color_signature=color_a),
    ]

    kept, groups = core.deduplicate_by_hash(
        dataset,
        threshold=0,
        method="phash",
        return_groups=True,
        color_threshold=0.2,
    )

    assert [item["path"] for item in kept] == ["/tmp/a.png", "/tmp/b.png"]
    assert groups == [
        {
            "hash": "123",
            "count": 2,
            "paths": ["/tmp/a.png", "/tmp/c.png"],
            "method": "phash",
        }
    ]


def test_filterkit_histogram_dedup_matches_existing_behavior() -> None:
    core = FilterKitCore()
    dataset = [
        _make_item("/tmp/a.png", 1, _one_hot(32, 0)),
        _make_item("/tmp/b.png", 2, _one_hot(32, 0)),
        _make_item("/tmp/c.png", 3, _one_hot(32, 5)),
    ]

    kept, groups = core.deduplicate_by_hash(
        dataset,
        threshold=0.1,
        method="histogram",
        return_groups=True,
    )

    assert [item["path"] for item in kept] == ["/tmp/a.png", "/tmp/c.png"]
    assert groups == [
        {
            "hash": str(_one_hot(32, 0)),
            "count": 2,
            "paths": ["/tmp/a.png", "/tmp/b.png"],
            "method": "histogram",
        }
    ]
