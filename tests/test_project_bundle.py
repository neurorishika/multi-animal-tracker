"""Tests for shared project-bundle manifest loading behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra_suite.data.project_bundle import (
    ProjectBundleManifest,
    load_project_bundle_manifest,
    save_project_bundle_manifest,
)


def test_load_project_bundle_manifest_returns_none_for_malformed_json(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text("{not-json", encoding="utf-8")

    assert load_project_bundle_manifest(tmp_path) is None


def test_load_project_bundle_manifest_returns_none_for_unsupported_version(
    tmp_path: Path,
) -> None:
    save_project_bundle_manifest(
        tmp_path,
        ProjectBundleManifest(
            kit="detectkit",
            state_path="state/detectkit_project.json",
            bundle_version=2,
        ),
    )

    assert load_project_bundle_manifest(tmp_path) is None


def test_load_project_bundle_manifest_strict_raises_for_invalid_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text(json.dumps(["bad"]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_project_bundle_manifest(tmp_path, strict=True)
