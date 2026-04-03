from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from hydra_suite.classkit.embed import embedder as embedder_module
from hydra_suite.classkit.jobs.task_workers import EmbeddingWorker
from hydra_suite.classkit.store.db import ClassKitDB


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
