"""One-time migration helper: copy models/training data from repo to user dirs.

Usage:
    python -m hydra_suite.paths_migrate /path/to/hydra-suite
    python -m hydra_suite.paths_migrate /path/to/hydra-suite --dry-run
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .paths import get_models_dir, get_presets_dir, get_training_runs_dir

logger = logging.getLogger(__name__)


def migrate_repo_data(
    repo_root: str | Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Migrate data from old repo-relative locations to user dirs."""
    repo_root = Path(repo_root).resolve()
    summary: dict[str, Any] = {
        "models_copied": 0,
        "training_copied": 0,
        "configs_copied": 0,
    }

    old_models = repo_root / "models"
    if old_models.exists():
        new_models = get_models_dir()
        for src in old_models.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(old_models)
            dst = new_models / rel
            summary["models_copied"] += 1
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                    logger.info("Copied model: %s -> %s", rel, dst)
                else:
                    logger.info("Skipped (exists): %s", rel)

    old_runs = repo_root / "training" / "runs"
    if old_runs.exists():
        new_runs = get_training_runs_dir()
        for src in old_runs.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(old_runs)
            dst = new_runs / rel
            summary["training_copied"] += 1
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)

    old_configs = repo_root / "configs"
    if old_configs.exists():
        new_presets = get_presets_dir()
        for src in old_configs.glob("*.json"):
            dst = new_presets / src.name
            summary["configs_copied"] += 1
            if not dry_run:
                if not dst.exists():
                    shutil.copy2(src, dst)

    action = "Would copy" if dry_run else "Copied"
    logger.info(
        "%s: %d models, %d training items, %d configs",
        action,
        summary["models_copied"],
        summary["training_copied"],
        summary["configs_copied"],
    )
    return summary


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Migrate data from repo to user directories"
    )
    parser.add_argument("repo_root", help="Path to hydra-suite repo root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying",
    )
    args = parser.parse_args()

    if not Path(args.repo_root).is_dir():
        print(f"Error: {args.repo_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = migrate_repo_data(args.repo_root, dry_run=args.dry_run)
    print(f"Models: {result['models_copied']}")
    print(f"Training: {result['training_copied']}")
    print(f"Configs: {result['configs_copied']}")
