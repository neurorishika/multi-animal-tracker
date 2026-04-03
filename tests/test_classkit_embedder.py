from __future__ import annotations

import pytest

from hydra_suite.classkit.embed import embedder as embedder_module


def test_timm_embedder_retries_with_local_cache_after_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedder_module, "torch", object())
    monkeypatch.setattr(embedder_module, "timm", object())

    calls: list[bool] = []

    def fake_create(self, *, local_files_only: bool):
        calls.append(local_files_only)
        if not local_files_only:
            raise RuntimeError("Network is unreachable")
        return object()

    def fake_configure(self) -> None:
        self._dim = 4

    monkeypatch.setattr(embedder_module.TimmEmbedder, "_create_model", fake_create)
    monkeypatch.setattr(
        embedder_module.TimmEmbedder, "_configure_loaded_model", fake_configure
    )

    embedder = embedder_module.TimmEmbedder(
        model_name="efficientnet_b3.ra2_in1k", device="cpu"
    )
    embedder.load_model()

    assert calls == [False, True]
    assert embedder.model is not None
    assert embedder.dimension == 4


def test_timm_embedder_raises_clear_error_when_weights_missing_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedder_module, "torch", object())
    monkeypatch.setattr(embedder_module, "timm", object())

    calls: list[bool] = []

    def fake_create(self, *, local_files_only: bool):
        calls.append(local_files_only)
        if not local_files_only:
            raise RuntimeError("Network is unreachable")
        raise RuntimeError("Cannot find the requested files in the local cache")

    def fail_if_configured(self) -> None:
        raise AssertionError("load_model should not configure a model after failure")

    monkeypatch.setattr(embedder_module.TimmEmbedder, "_create_model", fake_create)
    monkeypatch.setattr(
        embedder_module.TimmEmbedder,
        "_configure_loaded_model",
        fail_if_configured,
    )

    embedder = embedder_module.TimmEmbedder(
        model_name="efficientnet_b3.ra2_in1k", device="cpu"
    )

    with pytest.raises(embedder_module.ModelLoadError) as exc_info:
        embedder.load_model()

    assert calls == [False, True]
    assert "Could not load pretrained weights for 'efficientnet_b3.ra2_in1k'" in str(
        exc_info.value
    )
    assert "cached locally" in str(exc_info.value)
    assert "Connect to the internet once" in str(exc_info.value)
