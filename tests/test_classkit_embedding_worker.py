from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from hydra_suite.classkit.core.embed import embedder as embedder_module
from hydra_suite.classkit.core.store.db import ClassKitDB
from hydra_suite.classkit.jobs.task_workers import EmbeddingWorker


def test_embedding_worker_force_recompute_still_saves_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeEmbedder:
        def __init__(self, model_name: str, device: str):
            self.model_name = model_name
            self.device = device
            self._dim = 4

        def load_model(self) -> None:
            return None

        @property
        def dimension(self) -> int:
            return self._dim

        def embed(self, image_paths, batch_size: int = 32, preprocess_fn=None):
            assert batch_size == 8
            return np.ones((len(image_paths), self._dim), dtype=np.float32)

    monkeypatch.setattr(embedder_module, "TimmEmbedder", FakeEmbedder)

    db_path = tmp_path / "classkit.db"
    image_paths = [tmp_path / "img_0.jpg", tmp_path / "img_1.jpg"]
    ClassKitDB(db_path).add_images(image_paths)
    worker = EmbeddingWorker(
        image_paths=image_paths,
        model_name="resnet50.a1_in1k",
        device="cpu",
        batch_size=8,
        db_path=db_path,
        force_recompute=True,
    )

    successes: list[dict] = []
    errors: list[str] = []
    worker.signals.success.connect(successes.append)
    worker.signals.error.connect(errors.append)

    worker.run()

    assert not errors
    assert len(successes) == 1

    cached = ClassKitDB(db_path).get_embeddings("resnet50.a1_in1k", "cpu")
    assert cached is not None

    embeddings, metadata = cached
    assert embeddings.shape == (2, 4)
    assert metadata["model_name"] == "resnet50.a1_in1k"


def test_embedding_worker_reuses_cached_prefix_for_new_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embed_calls: list[list[str]] = []

    class FakeEmbedder:
        def __init__(self, model_name: str, device: str):
            self.model_name = model_name
            self.device = device
            self._dim = 4

        def load_model(self) -> None:
            return None

        @property
        def dimension(self) -> int:
            return self._dim

        def embed(self, image_paths, batch_size: int = 32, preprocess_fn=None):
            embed_calls.append([Path(path).name for path in image_paths])
            fill_value = float(len(embed_calls))
            return np.full((len(image_paths), self._dim), fill_value, dtype=np.float32)

    monkeypatch.setattr(embedder_module, "TimmEmbedder", FakeEmbedder)

    db_path = tmp_path / "classkit.db"
    db = ClassKitDB(db_path)

    initial_paths = [tmp_path / f"img_{index}.jpg" for index in range(3)]
    db.add_images(initial_paths)

    first_worker = EmbeddingWorker(
        image_paths=initial_paths,
        model_name="resnet50.a1_in1k",
        device="cpu",
        batch_size=8,
        db_path=db_path,
        force_recompute=True,
    )
    first_worker.run()

    extended_paths = initial_paths + [tmp_path / "img_3.jpg", tmp_path / "img_4.jpg"]
    db.add_images(extended_paths[3:])

    successes: list[dict] = []
    errors: list[str] = []
    second_worker = EmbeddingWorker(
        image_paths=extended_paths,
        model_name="resnet50.a1_in1k",
        device="cpu",
        batch_size=8,
        db_path=db_path,
        force_recompute=False,
    )
    second_worker.signals.success.connect(successes.append)
    second_worker.signals.error.connect(errors.append)

    second_worker.run()

    assert not errors
    assert embed_calls == [
        ["img_0.jpg", "img_1.jpg", "img_2.jpg"],
        ["img_3.jpg", "img_4.jpg"],
    ]
    assert len(successes) == 1
    result = successes[0]
    assert result["embeddings"].shape == (5, 4)
    assert np.all(result["embeddings"][:3] == 1.0)
    assert np.all(result["embeddings"][3:] == 2.0)
    assert result["metadata"]["reused_prefix_count"] == 3
