from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..constants import DEFAULT_DATASET_IMAGES_DIR, DEFAULT_POSEKIT_PROJECT_DIR
from ..models import Project
from ..utils import get_default_skeleton_dir
from .skeleton import SkeletonEditorDialog


class ProjectWizard(QDialog):
    """
    Used on first run (no project file) OR to edit project settings.
    """

    def __init__(
        self, dataset_dir: Path, existing: Optional[Project] = None, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("PoseKit Setup" if existing is None else "Project Settings")
        self.setMinimumSize(QSize(820, 560))

        self.dataset_dir = dataset_dir.resolve()
        self.images_dir = self.dataset_dir / DEFAULT_DATASET_IMAGES_DIR
        self.existing = existing

        # Defaults
        default_root = self.dataset_dir / DEFAULT_POSEKIT_PROJECT_DIR
        default_labels = default_root / "labels"
        if existing is None:
            default_classes = ["object"]
            default_kpts = ["kp1", "kp2"]
            default_pad = 0.03
            default_autosave = True
            default_edges: List[Tuple[int, int]] = []
        else:
            default_root = existing.out_root
            default_labels = existing.labels_dir
            default_classes = existing.class_names
            default_kpts = existing.keypoint_names
            default_pad = existing.bbox_pad_frac
            default_autosave = existing.autosave
            default_edges = existing.skeleton_edges

        self._edges = list(default_edges)
        self._kpt_names = list(default_kpts)

        layout = QVBoxLayout(self)

        # Paths
        form = QFormLayout()
        self.dataset_dir_line = QLineEdit(str(self.dataset_dir))
        self.dataset_dir_line.setReadOnly(True)
        form.addRow("Dataset folder:", self.dataset_dir_line)

        self.images_dir_line = QLineEdit(str(self.images_dir))
        self.images_dir_line.setReadOnly(True)
        form.addRow("Images folder:", self.images_dir_line)

        self.out_root = QLineEdit(str(default_root))
        self.out_root.setReadOnly(True)
        form.addRow("PoseKit project folder:", self.out_root)

        self.labels_dir = QLineEdit(str(default_labels))
        self.labels_dir.setReadOnly(True)
        form.addRow("Annotation labels folder:", self.labels_dir)

        self.autosave_cb = QCheckBox("Autosave when changing frames")
        self.autosave_cb.setChecked(default_autosave)
        form.addRow("", self.autosave_cb)

        self.pad_spin = QDoubleSpinBox()
        self.pad_spin.setRange(0.0, 0.25)
        self.pad_spin.setSingleStep(0.01)
        self.pad_spin.setValue(default_pad)
        form.addRow(
            "How much padding should be added around keypoints for bbox export?",
            self.pad_spin,
        )

        layout.addLayout(form)

        # Classes
        cls_box = QWidget()
        cls_layout = QVBoxLayout(cls_box)
        cls_layout.addWidget(QLabel("Classes (one per line)"))
        self.classes_edit = QPlainTextEdit("\n".join(default_classes))
        cls_layout.addWidget(self.classes_edit, 1)
        layout.addWidget(cls_box, 1)

        # Skeleton Editor button
        skel_box = QWidget()
        skel_layout = QVBoxLayout(skel_box)
        skel_layout.addWidget(QLabel("Keypoints and skeleton"))
        self.btn_skel = QPushButton("Edit Keypoints & Skeleton…")
        skel_layout.addWidget(self.btn_skel)

        # Summary label
        self.skel_summary = QLabel(self._get_skeleton_summary())
        self.skel_summary.setWordWrap(True)
        skel_layout.addWidget(self.skel_summary)
        skel_layout.addStretch(1)

        # Migration (only if editing existing project)
        self.mig_box = QWidget()
        mig_layout = QVBoxLayout(self.mig_box)
        self.migrate_cb = QCheckBox(
            "Migrate existing label files to new keypoint layout"
        )
        self.migrate_cb.setChecked(True)
        mig_layout.addWidget(self.migrate_cb)

        self.mig_mode_group = QButtonGroup(self)
        self.rb_by_name = QRadioButton(
            "Map by keypoint NAME (recommended if names stable)"
        )
        self.rb_by_index = QRadioButton(
            "Map by INDEX (recommended if you append/remove at end)"
        )
        self.rb_by_name.setChecked(True)
        self.mig_mode_group.addButton(self.rb_by_name)
        self.mig_mode_group.addButton(self.rb_by_index)
        mig_layout.addWidget(self.rb_by_name)
        mig_layout.addWidget(self.rb_by_index)

        if existing is None:
            self.mig_box.setVisible(False)

        skel_layout.addWidget(self.mig_box)
        layout.addWidget(skel_box)

        # OK / Cancel
        bottom = QHBoxLayout()
        self.ok_btn = QPushButton("Create Project" if existing is None else "Apply")
        self.cancel_btn = QPushButton("Cancel")
        bottom.addStretch(1)
        bottom.addWidget(self.ok_btn)
        bottom.addWidget(self.cancel_btn)
        layout.addLayout(bottom)

        # wiring
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.btn_skel.clicked.connect(self._edit_skeleton)

    def _get_skeleton_summary(self) -> str:
        k = len(self._kpt_names)
        e = len(self._edges)
        return f"{k} keypoint{'s' if k != 1 else ''}, {e} edge{'s' if e != 1 else ''}"

    def _edit_skeleton(self):
        dlg = SkeletonEditorDialog(
            self._kpt_names, self._edges, self, default_dir=get_default_skeleton_dir()
        )
        if dlg.exec() == QDialog.Accepted:
            names, edges = dlg.get_result()
            self._kpt_names = list(names) if names else ["kp1", "kp2"]
            k = len(self._kpt_names)
            self._edges = [
                (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
            ]
            self.skel_summary.setText(self._get_skeleton_summary())

    def get_classes(self) -> List[str]:
        """get_classes method documentation."""
        lines = [s.strip() for s in self.classes_edit.toPlainText().splitlines()]
        out = [s for s in lines if s]
        return out if out else ["object"]

    def get_keypoints(self) -> List[str]:
        """get_keypoints method documentation."""
        return list(self._kpt_names) if self._kpt_names else ["kp1", "kp2"]

    def get_edges(self) -> List[Tuple[int, int]]:
        """get_edges method documentation."""
        return list(self._edges)

    def get_paths(self) -> Tuple[Path, Path]:
        """get_paths method documentation."""
        root = (self.dataset_dir / DEFAULT_POSEKIT_PROJECT_DIR).resolve()
        labels = (root / "labels").resolve()
        return root, labels

    def get_options(self) -> Tuple[bool, float]:
        """get_options method documentation."""
        return bool(self.autosave_cb.isChecked()), float(self.pad_spin.value())

    def get_migration(self) -> Tuple[bool, str]:
        """get_migration method documentation."""
        if self.existing is None:
            return False, "name"
        do = bool(self.migrate_cb.isChecked())
        mode = "name" if self.rb_by_name.isChecked() else "index"
        return do, mode
