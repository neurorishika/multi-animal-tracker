from __future__ import annotations

from pathlib import Path

import numpy as np

from hydra_suite.classkit.core.store.db import ClassKitDB


def test_cluster_cache_requires_matching_embedding_cache_id(tmp_path: Path) -> None:
    db_path = tmp_path / "classkit.db"
    image_paths = [tmp_path / "img_0.jpg", tmp_path / "img_1.jpg"]
    db = ClassKitDB(db_path)
    db.add_images(image_paths)

    first_embedding_id = db.save_embeddings(
        np.ones((2, 4), dtype=np.float32),
        "resnet50.a1_in1k",
        "cpu",
        image_paths=image_paths,
    )
    db.save_cluster_cache(
        np.array([0, 1], dtype=np.int32),
        centers=None,
        n_clusters=2,
        method="minibatch",
        meta={"embedding_cache_id": first_embedding_id},
    )

    second_embedding_id = db.save_embeddings(
        np.full((2, 4), 2.0, dtype=np.float32),
        "resnet50.a1_in1k",
        "cpu",
        image_paths=image_paths,
    )

    assert (
        db.get_most_recent_cluster_cache(embedding_cache_id=second_embedding_id) is None
    )

    cached = db.get_most_recent_cluster_cache(embedding_cache_id=first_embedding_id)
    assert cached is not None
    assert cached["embedding_cache_id"] == first_embedding_id
