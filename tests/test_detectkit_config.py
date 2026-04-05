"""Tests for DetectKitConfig schema."""


def test_detectkit_config_defaults():
    from hydra_suite.detectkit.config.schemas import DetectKitConfig

    cfg = DetectKitConfig()
    assert cfg.last_project_path == ""
    assert cfg.compute_runtime == "cpu"


def test_detectkit_config_round_trip():
    from hydra_suite.detectkit.config.schemas import DetectKitConfig

    cfg = DetectKitConfig(last_project_path="/tmp/proj", compute_runtime="mps")
    restored = DetectKitConfig.from_dict(cfg.to_dict())
    assert restored.last_project_path == "/tmp/proj"
    assert restored.compute_runtime == "mps"
