"""Test RefineKitConfig schema."""


def test_refinekit_config_defaults():
    from hydra_suite.refinekit.config.schemas import RefineKitConfig

    cfg = RefineKitConfig()
    assert cfg.last_video_path == ""
    assert cfg.sessions == []


def test_refinekit_config_round_trip():
    from hydra_suite.refinekit.config.schemas import RefineKitConfig

    cfg = RefineKitConfig(
        last_video_path="/tmp/video.mp4", sessions=["/tmp/a.mp4", "/tmp/b.mp4"]
    )
    restored = RefineKitConfig.from_dict(cfg.to_dict())
    assert restored.last_video_path == "/tmp/video.mp4"
    assert restored.sessions == ["/tmp/a.mp4", "/tmp/b.mp4"]
