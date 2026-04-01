"""Tests for the one-time data migration helper."""


def test_migrate_copies_models(tmp_path, monkeypatch):
    """Migration copies models from old repo location to new user data dir."""
    old_models = tmp_path / "old_repo" / "models"
    old_models.mkdir(parents=True)
    (old_models / "test_model.pt").write_bytes(b"fake model data")

    new_data = tmp_path / "new_data"
    monkeypatch.setattr("multi_tracker.paths._user_data_dir", lambda: new_data)

    from multi_tracker.paths_migrate import migrate_repo_data

    result = migrate_repo_data(repo_root=tmp_path / "old_repo", dry_run=False)

    assert result["models_copied"] >= 1
    from multi_tracker.paths import get_models_dir

    assert (get_models_dir() / "test_model.pt").exists()


def test_migrate_dry_run_copies_nothing(tmp_path, monkeypatch):
    """Dry run reports what would happen without copying."""
    old_models = tmp_path / "old_repo" / "models"
    old_models.mkdir(parents=True)
    (old_models / "test_model.pt").write_bytes(b"fake model data")

    new_data = tmp_path / "new_data"
    monkeypatch.setattr("multi_tracker.paths._user_data_dir", lambda: new_data)

    from multi_tracker.paths_migrate import migrate_repo_data

    result = migrate_repo_data(repo_root=tmp_path / "old_repo", dry_run=True)

    assert result["models_copied"] >= 1
    from multi_tracker.paths import get_models_dir

    assert not (get_models_dir() / "test_model.pt").exists()
