"""Tests that packaging metadata and bundled data are correct."""


def test_version_matches_metadata():
    """Package version from importlib.metadata matches __init__.__version__."""
    from importlib.metadata import version

    from multi_tracker import __version__

    assert __version__ == version("multi-animal-tracker")


def test_brand_svgs_exist_in_package():
    """Brand SVG files are accessible via importlib.resources."""
    from importlib.resources import files

    brand = files("multi_tracker.resources.brand")
    names = [r.name for r in brand.iterdir()]
    assert "multianimaltracker.svg" in names
    assert "filterkit.svg" in names
    assert "classkit.svg" in names
    assert "posekit.svg" in names
    assert "refinekit.svg" in names


def test_default_configs_exist_in_package():
    """Default config JSON files are accessible via importlib.resources."""
    from importlib.resources import files

    configs = files("multi_tracker.resources.configs")
    names = [r.name for r in configs.iterdir()]
    assert "default.json" in names


def test_skeleton_configs_exist_in_package():
    """Skeleton JSON files are accessible via importlib.resources."""
    from importlib.resources import files

    skeletons = files("multi_tracker.resources.configs.skeletons")
    names = [r.name for r in skeletons.iterdir()]
    assert "ooceraea_biroi.json" in names


def test_platformdirs_importable():
    """platformdirs is available as a dependency."""
    import platformdirs

    assert hasattr(platformdirs, "user_config_dir")
