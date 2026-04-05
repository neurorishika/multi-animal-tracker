"""Tests for TrackerConfig schema."""


def test_tracker_config_defaults():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig

    cfg = TrackerConfig()
    assert cfg.current_video_path == ""
    assert cfg.roi_current_mode == "circle"
    assert cfg.roi_current_zone_type == "include"
    assert cfg.roi_shapes == []
    assert cfg.batch_videos == []


def test_tracker_config_round_trip():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig

    cfg = TrackerConfig(
        current_video_path="/tmp/test.mp4",
        roi_current_mode="polygon",
        batch_videos=["/tmp/a.mp4", "/tmp/b.mp4"],
    )
    d = cfg.to_dict()
    restored = TrackerConfig.from_dict(d)
    assert restored.current_video_path == "/tmp/test.mp4"
    assert restored.roi_current_mode == "polygon"
    assert restored.batch_videos == ["/tmp/a.mp4", "/tmp/b.mp4"]


def test_tracker_config_path_fields_are_str():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig

    cfg = TrackerConfig(current_video_path="/some/path.mp4")
    d = cfg.to_dict()
    assert isinstance(d["current_video_path"], str)
