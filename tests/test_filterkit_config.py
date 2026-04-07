"""Tests for FilterKit configuration schema."""


def test_filterkit_config_defaults():
    from hydra_suite.filterkit.config.schemas import FilterKitConfig

    cfg = FilterKitConfig()
    assert cfg.dataset_path == ""
    assert cfg.images_per_page == 200
    assert cfg.removed_per_page == 150


def test_filterkit_config_round_trip():
    from hydra_suite.filterkit.config.schemas import FilterKitConfig

    cfg = FilterKitConfig(dataset_path="/tmp/data", images_per_page=100)
    restored = FilterKitConfig.from_dict(cfg.to_dict())
    assert restored.dataset_path == "/tmp/data"
    assert restored.images_per_page == 100
