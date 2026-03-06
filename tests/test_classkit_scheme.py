from pathlib import Path

from multi_tracker.classkit.config.schemas import Factor, LabelingScheme, ProjectConfig


def test_factor_has_name_and_labels():
    f = Factor(name="tag_1", labels=["red", "blue", "green"])
    assert f.name == "tag_1"
    assert f.labels == ["red", "blue", "green"]
    assert f.shortcut_keys == []


def test_scheme_single_factor():
    scheme = LabelingScheme(
        name="age",
        factors=[Factor(name="age", labels=["young", "old"])],
        training_modes=["flat_tiny", "flat_yolo"],
    )
    assert len(scheme.factors) == 1
    assert scheme.total_classes == 2


def test_scheme_two_factor_cartesian():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red", "blue", "green"]),
            Factor(name="tag_2", labels=["red", "blue", "green"]),
        ],
        training_modes=["flat_yolo", "multihead_yolo"],
    )
    assert scheme.total_classes == 9


def test_scheme_composite_label_round_trip():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red", "blue"]),
            Factor(name="tag_2", labels=["green", "yellow"]),
        ],
        training_modes=["flat_yolo"],
    )
    composite = scheme.encode_label(["red", "green"])
    assert composite == "red|green"
    decoded = scheme.decode_label(composite)
    assert decoded == ["red", "green"]


def test_project_config_accepts_scheme():
    scheme = LabelingScheme(
        name="test",
        factors=[Factor(name="f", labels=["a", "b"])],
        training_modes=["flat_tiny"],
    )
    cfg = ProjectConfig(name="proj", classes=[], root_dir=Path("/tmp"), scheme=scheme)
    assert cfg.scheme is not None


def test_project_config_scheme_defaults_none():
    cfg = ProjectConfig(name="proj", classes=[], root_dir=Path("/tmp"))
    assert cfg.scheme is None


def test_encode_label_wrong_length_raises():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red"]),
            Factor(name="tag_2", labels=["blue"]),
        ],
        training_modes=["flat_yolo"],
    )
    import pytest

    with pytest.raises(ValueError, match="Expected 2 factor values"):
        scheme.encode_label(["red"])  # missing second factor


def test_decode_label_wrong_parts_raises():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red"]),
            Factor(name="tag_2", labels=["blue"]),
        ],
        training_modes=["flat_yolo"],
    )
    import pytest

    with pytest.raises(ValueError, match="Expected 2 parts"):
        scheme.decode_label("red|blue|green")  # too many parts


from multi_tracker.classkit.presets import (
    age_preset,
    color_tag_preset,
    head_tail_preset,
)


def test_head_tail_preset():
    scheme = head_tail_preset()
    assert scheme.name == "head_tail"
    assert len(scheme.factors) == 1
    assert set(scheme.factors[0].labels) == {"left", "right", "up", "down"}
    assert scheme.total_classes == 4
    assert scheme.training_modes == ["flat_tiny", "flat_yolo"]


def test_color_tag_preset_1factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=1, colors=colors)
    assert scheme.total_classes == 5
    assert len(scheme.factors) == 1
    assert scheme.training_modes == ["flat_tiny", "flat_yolo"]


def test_color_tag_preset_2factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=2, colors=colors)
    assert scheme.total_classes == 25
    assert len(scheme.factors) == 2
    assert scheme.factors[0].name == "tag_1"
    assert scheme.factors[1].name == "tag_2"
    assert scheme.training_modes == [
        "flat_tiny",
        "flat_yolo",
        "multihead_tiny",
        "multihead_yolo",
    ]


def test_color_tag_preset_3factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=3, colors=colors)
    assert scheme.total_classes == 125
    assert scheme.training_modes == [
        "flat_tiny",
        "flat_yolo",
        "multihead_tiny",
        "multihead_yolo",
    ]


def test_age_preset_default():
    scheme = age_preset()
    assert scheme.total_classes == 2
    assert "young" in scheme.factors[0].labels
    assert "old" in scheme.factors[0].labels
    assert scheme.training_modes == ["flat_tiny", "flat_yolo"]


def test_age_preset_extra_classes():
    scheme = age_preset(extra_classes=["juvenile"])
    assert scheme.total_classes == 3
    assert "juvenile" in scheme.factors[0].labels


def test_color_tag_preset_custom_colors():
    scheme = color_tag_preset(n_factors=2, colors=["a", "b", "c"])
    assert scheme.total_classes == 9


def test_color_tag_preset_invalid_n_factors_raises():
    import pytest

    with pytest.raises(ValueError):
        color_tag_preset(n_factors=0, colors=["red"])


def test_color_tag_preset_empty_colors_raises():
    import pytest

    with pytest.raises(ValueError):
        color_tag_preset(n_factors=1, colors=[])
