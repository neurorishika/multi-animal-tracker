from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from .constants import DEFAULT_DATASET_IMAGES_DIR
from .main_window import MainWindow
from .models import Project
from .project import (
    create_empty_startup_project,
    create_project_via_wizard,
    find_project,
    load_project_with_repairs,
    resolve_dataset_paths,
)
from .utils import _resolve_project_path, list_images


def parse_args() -> argparse.Namespace:
    """parse_args function documentation."""
    ap = argparse.ArgumentParser(description="PoseKit labeler")
    ap.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset root folder containing images/ subdirectory",
    )
    ap.add_argument("--project", default=None, help="Explicit project json path")
    ap.add_argument(
        "--new", action="store_true", help="Force setup wizard even if project exists"
    )
    return ap.parse_args()


def main() -> None:
    """main function documentation."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("PoseKitLabeler")
    app.setApplicationDisplayName("PoseKit Labeler")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("posekit-labeler")
    try:
        project_root = Path(__file__).resolve().parents[3]
        icon_path = project_root / "brand" / "posekit.svg"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass

    # Startup-without-input mode: open full UI first, then show chooser overlay.
    if not args.dataset and not args.project:
        proj = create_empty_startup_project()
        win = MainWindow(proj, [])
        try:
            win.setWindowIcon(app.windowIcon())
        except Exception:
            pass
        win.resize(1500, 860)
        win.showMaximized()
        QTimer.singleShot(0, win.show_startup_open_overlay)
        sys.exit(app.exec())

    dataset_dir: Optional[Path] = None
    if args.dataset:
        dataset_dir = Path(args.dataset).expanduser().resolve()
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
            sys.exit(2)
        # Backward compatibility: allow passing .../images directly.
        if dataset_dir.name == DEFAULT_DATASET_IMAGES_DIR:
            dataset_dir = dataset_dir.parent

        _, images_dir, _out_root, _labels_dir, _project_path = resolve_dataset_paths(
            dataset_dir
        )
        if not images_dir.exists() or not images_dir.is_dir():
            print(
                f"Invalid dataset root (missing `{DEFAULT_DATASET_IMAGES_DIR}/`): {dataset_dir}",
                file=sys.stderr,
            )
            sys.exit(2)

    proj: Optional[Project] = None

    if args.project:
        project_path = Path(args.project).expanduser().resolve()
        if project_path.exists():
            try:
                data = json.loads(project_path.read_text(encoding="utf-8"))
                base = project_path.parent
                project_images_dir = _resolve_project_path(
                    Path(data["images_dir"]), base, data
                )
                project_dataset_dir = project_images_dir.parent.resolve()
            except Exception as e:
                print(f"Failed to read project images_dir: {e}", file=sys.stderr)
                sys.exit(2)

            if dataset_dir is None:
                dataset_dir = project_dataset_dir
            if not args.new:
                proj = load_project_with_repairs(project_path, dataset_dir)
        else:
            print(f"Project file not found: {project_path}", file=sys.stderr)
            sys.exit(2)

    if proj is None and not args.new:
        found = find_project(dataset_dir) if dataset_dir is not None else None
        if found:
            proj = load_project_with_repairs(found, dataset_dir)

    if proj is None:
        if dataset_dir is None:
            print("No dataset directory provided.", file=sys.stderr)
            sys.exit(2)
        proj = create_project_via_wizard(dataset_dir)
        if proj is None:
            sys.exit(0)

    imgs = list_images(proj.images_dir)
    if not imgs:
        QMessageBox.critical(
            None, "No images", f"No images found under: {proj.images_dir}"
        )
        sys.exit(2)

    win = MainWindow(proj, imgs)
    try:
        win.setWindowIcon(app.windowIcon())
    except Exception:
        pass
    win.resize(1500, 860)
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
