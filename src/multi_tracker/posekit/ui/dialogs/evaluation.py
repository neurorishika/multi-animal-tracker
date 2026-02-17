#!/usr/bin/env python3
"""
Evaluation dialog and worker.
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QSize, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.extensions import list_labeled_indices, load_yolo_pose_label
from .utils import (
    _load_dialog_settings,
    _save_dialog_settings,
    get_available_devices,
    list_sleap_envs,
    make_histogram_image,
    make_pose_infer,
)

logger = logging.getLogger("pose_label.dialogs.evaluation")


def _read_image_size(img_path: Path) -> Optional[Tuple[int, int]]:
    try:
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


def _load_kpts_px(
    label_path: Path, k: int, img_path: Path
) -> Optional[List[Tuple[float, float, float]]]:
    size = _read_image_size(img_path)
    if size is None:
        return None
    w, h = size
    parsed = load_yolo_pose_label(label_path, k)
    if parsed is None:
        return None
    _, kpts, _ = parsed
    out = []
    for kp in kpts:
        out.append((float(kp.x) * w, float(kp.y) * h, float(kp.v)))
    return out


def evaluate_pose_predictions(
    image_paths: List[Path],
    gt_labels_dir: Path,
    pred_labels_dir: Path,
    k: int,
    pck_thresh_frac: float,
    oks_sigma: float,
) -> Dict[str, object]:
    """Evaluate predictions against ground-truth labels."""
    per_kpt_errors: List[List[float]] = [[] for _ in range(k)]
    per_kpt_oks: List[List[float]] = [[] for _ in range(k)]
    per_kpt_pck: List[List[int]] = [[] for _ in range(k)]

    frame_errors: List[Tuple[int, float]] = []
    all_errors: List[float] = []

    for idx, img_path in enumerate(image_paths):
        gt_path = gt_labels_dir / f"{img_path.stem}.txt"
        pred_path = pred_labels_dir / f"{img_path.stem}.txt"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt = _load_kpts_px(gt_path, k, img_path)
        pred = _load_kpts_px(pred_path, k, img_path)
        if gt is None or pred is None:
            continue

        size = _read_image_size(img_path)
        if size is None:
            continue
        w, h = size
        thresh = pck_thresh_frac * max(w, h)
        sigma = oks_sigma * max(w, h)
        sigma = max(1e-6, sigma)

        frame_errs = []
        for j in range(k):
            gx, gy, gv = gt[j]
            px, py, pv = pred[j]
            if gv <= 0 or pv <= 0:
                continue
            d = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            all_errors.append(d)
            per_kpt_errors[j].append(d)
            frame_errs.append(d)

            per_kpt_pck[j].append(1 if d <= thresh else 0)
            oks = math.exp(-((d * d) / (2.0 * sigma * sigma)))
            per_kpt_oks[j].append(oks)

        if frame_errs:
            frame_errors.append((idx, float(np.mean(frame_errs))))

    def _safe_mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    per_kpt_stats = []
    for j in range(k):
        per_kpt_stats.append(
            {
                "mean_error": _safe_mean(per_kpt_errors[j]),
                "pck": _safe_mean(per_kpt_pck[j]),
                "oks": _safe_mean(per_kpt_oks[j]),
                "n": len(per_kpt_errors[j]),
            }
        )

    overall = {
        "mean_error": _safe_mean(all_errors),
        "pck": _safe_mean([x for sub in per_kpt_pck for x in sub]),
        "oks": _safe_mean([x for sub in per_kpt_oks for x in sub]),
        "n": len(all_errors),
    }

    return {
        "overall": overall,
        "per_kpt": per_kpt_stats,
        "frame_errors": frame_errors,
        "all_errors": all_errors,
    }


class EvaluationWorker(QObject):
    """Worker for running evaluation in background."""

    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        weights_path: Path,
        eval_paths: List[Path],
        labels_dir: Path,
        keypoint_names: List[str],
        run_dir: Path,
        out_root: Path,
        device: str,
        imgsz: int,
        conf: float,
        pck_thr: float,
        oks_sigma: float,
        batch: int,
        backend: str = "yolo",
        sleap_env: Optional[str] = None,
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        pred_cache: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    ):
        super().__init__()
        self.weights_path = Path(weights_path)
        self.eval_paths = list(eval_paths)
        self.labels_dir = Path(labels_dir)
        self.keypoint_names = list(keypoint_names)
        self.run_dir = Path(run_dir)
        self.out_root = Path(out_root)
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.pck_thr = float(pck_thr)
        self.oks_sigma = float(oks_sigma)
        self.batch = int(batch)
        self.backend = (backend or "yolo").lower()
        self.sleap_env = sleap_env
        self.sleap_device = sleap_device
        self.sleap_batch = int(sleap_batch)
        self.pred_cache = pred_cache
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self.backend != "sleap":
                try:
                    import ultralytics  # noqa: F401
                except ImportError as e:
                    self.failed.emit(
                        f"Ultralytics not available. Install with: pip install ultralytics\n{e}"
                    )
                    return

            if self.backend == "sleap":
                if not self.weights_path.exists() or not self.weights_path.is_dir():
                    self.failed.emit(f"SLEAP model dir not found: {self.weights_path}")
                    return
                if not self.sleap_env:
                    self.failed.emit("Select a SLEAP conda env.")
                    return
            else:
                if not self.weights_path.exists() and not self.pred_cache:
                    self.failed.emit(f"Weights not found: {self.weights_path}")
                    return

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Eval run dir: {self.run_dir}")
            self.log.emit(f"Evaluating {len(self.eval_paths)} frames...")

            total = len(self.eval_paths)
            num_kpts = len(self.keypoint_names)

            per_frame = []
            per_kpt_errors = [[] for _ in range(num_kpts)]
            kpt_counts = [0] * num_kpts
            kpt_pck = [0] * num_kpts
            kpt_oks = [0.0] * num_kpts
            kpt_err_sum = [0.0] * num_kpts
            kpt_conf_sum = [0.0] * num_kpts

            total_kpts = 0
            total_pck = 0
            total_oks = 0.0
            total_err = 0.0
            total_conf = 0.0

            if self.pred_cache is None:
                infer = make_pose_infer(self.out_root, self.keypoint_names)
                preds, err = infer.predict(
                    self.weights_path,
                    self.eval_paths,
                    device=self.device,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    batch=self.batch,
                    progress_cb=self.progress.emit,
                    backend=self.backend,
                    sleap_env=self.sleap_env,
                    sleap_device=self.sleap_device,
                    sleap_batch=self.sleap_batch,
                    sleap_max_instances=1,
                )
                if preds is None:
                    self.failed.emit(err or "Prediction failed.")
                    return
                self.pred_cache = preds

            if self.pred_cache:
                try:
                    from PIL import Image
                except Exception as e:
                    self.failed.emit(f"PIL not available: {e}")
                    return

                for idx, img_path in enumerate(self.eval_paths):
                    if self._cancel:
                        self.log.emit("Canceled.")
                        return

                    label_path = self.labels_dir / f"{img_path.stem}.txt"
                    gt = load_yolo_pose_label(label_path, num_kpts)
                    if not gt:
                        self.progress.emit(idx + 1, total)
                        continue

                    _, gt_kpts, bbox = gt
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                    except Exception:
                        w, h = (1, 1)
                    if bbox is not None:
                        _, _, bw, bh = bbox
                        scale = max(bw * w, bh * h)
                    else:
                        scale = max(w, h)
                    if scale <= 0:
                        scale = max(w, h, 1)

                    pred_list = self.pred_cache.get(str(img_path))
                    if pred_list is None:
                        pred_list = self.pred_cache.get(str(img_path.resolve()))
                    if not pred_list:
                        self.progress.emit(idx + 1, total)
                        continue
                    if len(pred_list) != num_kpts:
                        self.failed.emit(
                            "Prediction keypoint count mismatch. "
                            f"Model has {len(pred_list)} keypoints, project expects {num_kpts}. "
                            "Please select a matching model."
                        )
                        return

                    pred_xy = np.array(
                        [[p[0], p[1]] for p in pred_list], dtype=np.float32
                    )
                    pred_conf = np.array([p[2] for p in pred_list], dtype=np.float32)

                    frame_errs = [None] * num_kpts
                    frame_confs = [None] * num_kpts
                    for k in range(num_kpts):
                        if gt_kpts[k].v <= 0:
                            continue
                        if pred_conf is not None and pred_conf[k] <= 0.0:
                            continue

                        px, py = pred_xy[k]
                        err = float(
                            math.hypot(px - (gt_kpts[k].x * w), py - (gt_kpts[k].y * h))
                        )
                        conf = float(pred_conf[k]) if pred_conf is not None else 0.0

                        ok = err <= (self.pck_thr * scale)
                        oks = math.exp(
                            -((err**2) / (2 * (self.oks_sigma * scale) ** 2))
                        )

                        total_kpts += 1
                        total_pck += 1 if ok else 0
                        total_oks += oks
                        total_err += err
                        total_conf += conf

                        kpt_counts[k] += 1
                        kpt_pck[k] += 1 if ok else 0
                        kpt_oks[k] += oks
                        kpt_err_sum[k] += err
                        kpt_conf_sum[k] += conf

                        per_kpt_errors[k].append(err)
                        frame_errs[k] = err
                        frame_confs[k] = conf

                    valid_errs = [e for e in frame_errs if e is not None]
                    valid_confs = [c for c in frame_confs if c is not None]
                    if valid_errs:
                        mean_err = float(np.mean(valid_errs))
                        mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
                        mean_err_norm = mean_err / (scale + 1e-6)
                        per_frame.append(
                            {
                                "image_path": str(img_path),
                                "mean_error_px": mean_err,
                                "mean_error_norm": mean_err_norm,
                                "mean_conf": mean_conf,
                                "kpt_errors": frame_errs,
                                "kpt_confs": frame_confs,
                            }
                        )

                    self.progress.emit(idx + 1, total)

            # finalize
            overall = {
                "frames": len(per_frame),
                "total_kpts": total_kpts,
                "pck": (total_pck / total_kpts) if total_kpts else 0.0,
                "oks": (total_oks / total_kpts) if total_kpts else 0.0,
                "mean_error_px": (total_err / total_kpts) if total_kpts else 0.0,
                "mean_conf": (total_conf / total_kpts) if total_kpts else 0.0,
            }

            per_kpt = []
            for i, name in enumerate(self.keypoint_names):
                count = kpt_counts[i]
                per_kpt.append(
                    {
                        "name": name,
                        "count": count,
                        "pck": (kpt_pck[i] / count) if count else 0.0,
                        "oks": (kpt_oks[i] / count) if count else 0.0,
                        "mean_error_px": (kpt_err_sum[i] / count) if count else 0.0,
                        "mean_conf": (kpt_conf_sum[i] / count) if count else 0.0,
                    }
                )

            per_frame_sorted = sorted(
                per_frame, key=lambda x: x.get("mean_error_norm", 0.0), reverse=True
            )
            worst = per_frame_sorted[:50]

            self.log.emit("Evaluation finished.")
            self.finished.emit(
                {
                    "overall": overall,
                    "per_keypoint": per_kpt,
                    "per_frame": per_frame,
                    "worst": worst,
                    "per_kpt_errors": per_kpt_errors,
                }
            )

        except Exception as e:
            self.failed.emit(str(e))
            logger.error(f"Eval worker failed: {e}", exc_info=True)


class EvaluationDashboardDialog(QDialog):
    """Dashboard for evaluating models."""

    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        weights_path: Optional[str] = None,
        add_frames_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Evaluation Dashboard")
        self.setMinimumSize(QSize(900, 700))

        self.project = project
        self.image_paths = image_paths
        self.add_frames_callback = add_frames_callback
        self._lock_model = False
        self._thread = None
        self._worker = None
        self._path_to_index = {}
        for i, p in enumerate(image_paths):
            self._path_to_index[str(p)] = i
            try:
                self._path_to_index[str(p.resolve())] = i
            except Exception:
                pass
        self.infer = make_pose_infer(self.project.out_root, self.project.keypoint_names)

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        # Config
        cfg_group = QGroupBox("Config")
        cfg_layout = QFormLayout(cfg_group)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["YOLO", "SLEAP"])
        cfg_layout.addRow("Backend", self.backend_combo)

        self.weights_label = QLabel("Weights")
        self.weights_edit = QLineEdit(weights_path or "")
        self.btn_weights_browse = QPushButton("Browse…")
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.weights_edit, 1)
        weights_row.addWidget(self.btn_weights_browse)
        cfg_layout.addRow(self.weights_label, weights_row)

        self.sleap_env_row = QWidget()
        sleap_env_layout = QHBoxLayout(self.sleap_env_row)
        sleap_env_layout.setContentsMargins(0, 0, 0, 0)
        self.sleap_env_combo = QComboBox()
        self.btn_sleap_refresh = QPushButton("↻")
        self.btn_sleap_refresh.setMaximumWidth(40)
        sleap_env_layout.addWidget(self.sleap_env_combo, 1)
        sleap_env_layout.addWidget(self.btn_sleap_refresh)
        cfg_layout.addRow("SLEAP environment", self.sleap_env_row)
        self.lbl_sleap_env_status = QLabel("")
        self.lbl_sleap_env_status.setStyleSheet("QLabel { color: #b00; }")
        cfg_layout.addRow("", self.lbl_sleap_env_status)

        self.device_combo = QComboBox()
        self.device_combo.addItems(get_available_devices())
        cfg_layout.addRow("Device", self.device_combo)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        cfg_layout.addRow("Image size", self.imgsz_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        cfg_layout.addRow("Batch size", self.batch_spin)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        cfg_layout.addRow("Confidence threshold", self.conf_spin)

        self.pck_spin = QDoubleSpinBox()
        self.pck_spin.setRange(0.01, 0.5)
        self.pck_spin.setSingleStep(0.01)
        self.pck_spin.setValue(0.05)
        cfg_layout.addRow("PCK threshold", self.pck_spin)

        self.oks_spin = QDoubleSpinBox()
        self.oks_spin.setRange(0.01, 1.0)
        self.oks_spin.setSingleStep(0.05)
        self.oks_spin.setValue(0.1)
        cfg_layout.addRow("OKS sigma", self.oks_spin)

        self.cb_use_cache = QCheckBox("Use cached predictions when available")
        self.cb_use_cache.setChecked(True)
        cfg_layout.addRow("", self.cb_use_cache)

        self.out_dir_edit = QLineEdit("")
        self.btn_out_browse = QPushButton("Browse…")
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(self.btn_out_browse)
        cfg_layout.addRow("Output directory", out_row)

        content_layout.addWidget(cfg_group)

        # Progress + logs
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        content_layout.addWidget(self.progress)

        self.lbl_loss_plot = QLabel()
        self.lbl_loss_plot.setMinimumHeight(220)
        content_layout.addWidget(self.lbl_loss_plot)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        content_layout.addWidget(self.log_view, 1)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.lbl_overall = QLabel("Overall: --")
        results_layout.addWidget(self.lbl_overall)

        self.kpt_table = QTableWidget(0, 5)
        self.kpt_table.setHorizontalHeaderLabels(
            ["Keypoint", "Mean Err (px)", "PCK", "OKS", "Mean Conf"]
        )
        results_layout.addWidget(self.kpt_table)

        charts_row = QHBoxLayout()
        self.lbl_hist = QLabel()
        self.lbl_heat = QLabel()
        charts_row.addWidget(self.lbl_hist, 1)
        charts_row.addWidget(self.lbl_heat, 1)
        results_layout.addLayout(charts_row)

        self.worst_list = QListWidget()
        results_layout.addWidget(QLabel("Worst frames"))
        results_layout.addWidget(self.worst_list, 1)

        content_layout.addWidget(results_group, 2)

        # Buttons
        btns = QHBoxLayout()
        self.btn_run = QPushButton("Run Evaluation")
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_run)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        content_layout.addLayout(btns)

        # Wiring
        self.btn_weights_browse.clicked.connect(self._browse_weights)
        self.btn_out_browse.clicked.connect(self._browse_out)
        self.btn_run.clicked.connect(self._run_eval)
        self.btn_close.clicked.connect(self.reject)
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self.btn_sleap_refresh.clicked.connect(self._refresh_sleap_envs)

        self._update_default_out_dir()
        self._apply_settings()
        self._apply_latest_weights_default()
        self._refresh_sleap_envs()
        self._apply_backend_ui()

    def lock_model_path(self: object, path: str) -> object:
        if path:
            self.weights_edit.setText(path)
        self._lock_model = True
        self.weights_edit.setReadOnly(True)
        self.btn_weights_browse.setEnabled(False)
        self._apply_backend_ui()

    def _apply_settings(self):
        settings = _load_dialog_settings("evaluation_dashboard")
        if not settings:
            return
        if not self._lock_model:
            self.weights_edit.setText(settings.get("weights", ""))
        self.device_combo.setCurrentText(settings.get("device", "auto"))
        self.imgsz_spin.setValue(int(settings.get("imgsz", 640)))
        self.conf_spin.setValue(float(settings.get("conf", 0.25)))
        self.pck_spin.setValue(float(settings.get("pck", 0.05)))
        self.oks_spin.setValue(float(settings.get("oks", 0.1)))
        self.batch_spin.setValue(int(settings.get("batch", 16)))
        self.cb_use_cache.setChecked(bool(settings.get("use_cache", True)))
        self.backend_combo.setCurrentText(settings.get("backend", "YOLO"))
        self.sleap_env_combo.setCurrentText(settings.get("sleap_env", ""))

    def _save_settings(self):
        _save_dialog_settings(
            "evaluation_dashboard",
            {
                "weights": self.weights_edit.text() if not self._lock_model else "",
                "device": self.device_combo.currentText(),
                "imgsz": self.imgsz_spin.value(),
                "conf": self.conf_spin.value(),
                "pck": self.pck_spin.value(),
                "oks": self.oks_spin.value(),
                "batch": self.batch_spin.value(),
                "use_cache": self.cb_use_cache.isChecked(),
                "backend": self.backend_combo.currentText(),
                "sleap_env": self.sleap_env_combo.currentText(),
            },
        )

    def closeEvent(self, event):
        self._save_settings()
        if self._thread:
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)

    def _apply_latest_weights_default(self):
        if self._lock_model or self.weights_edit.text():
            return
        # Try to find recent run
        runs_dir = self.project.out_root / "runs" / "train"
        if not runs_dir.exists():
            return
        runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs:
            best = r / "weights" / "best.pt"
            if best.exists():
                self.weights_edit.setText(str(best))
                return

    def _on_backend_changed(self, _text: str):
        self._apply_backend_ui()
        if self.backend_combo.currentText().strip().lower() == "sleap":
            self._refresh_sleap_envs()

    def _apply_backend_ui(self):
        is_sleap = self.backend_combo.currentText().strip().lower() == "sleap"
        self.weights_label.setText("Model dir:" if is_sleap else "Weights:")
        self.sleap_env_row.setVisible(is_sleap)
        self.lbl_sleap_env_status.setVisible(is_sleap)
        self.imgsz_spin.setEnabled(not is_sleap)
        self.conf_spin.setEnabled(not is_sleap)
        if self._lock_model:
            self.weights_label.setVisible(False)
            self.weights_edit.setVisible(False)
            self.btn_weights_browse.setVisible(False)

    def _refresh_sleap_envs(self):
        current = self.sleap_env_combo.currentText().strip()
        self.sleap_env_combo.clear()
        self.sleap_env_combo.setEnabled(True)
        envs, err = list_sleap_envs()
        if not envs:
            self.sleap_env_combo.addItem("No sleap envs found")
            self.sleap_env_combo.setEnabled(False)
            self.lbl_sleap_env_status.setText(
                err or "No conda envs starting with 'sleap' found."
            )
        else:
            self.sleap_env_combo.addItems(envs)
            if current and current in envs:
                self.sleap_env_combo.setCurrentText(current)
            self.lbl_sleap_env_status.setText("")

    def _get_sleap_env(self) -> Optional[str]:
        env = self.sleap_env_combo.currentText().strip()
        if not env or env.lower().startswith("no sleap"):
            return None
        return env

    def _browse_weights(self):
        if self._lock_model:
            return
        if self.backend_combo.currentText().strip().lower() == "sleap":
            path = QFileDialog.getExistingDirectory(
                self, "Select SLEAP model directory"
            )
            if path:
                self.weights_edit.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select weights", "", "*.pt")
            if path:
                self.weights_edit.setText(path)
                self._update_default_out_dir()

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select output dir")
        if path:
            self.out_dir_edit.setText(path)

    def _update_default_out_dir(self):
        weights = Path(self.weights_edit.text().strip())
        if weights.exists() and weights.parent.name == "weights":
            out_dir = weights.parent.parent
        else:
            out_dir = (
                self.project.out_root
                / "runs"
                / "eval"
                / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        self.out_dir_edit.setText(str(out_dir))

    def _collect_eval_paths(self) -> List[Path]:
        labeled = list_labeled_indices(self.image_paths, self.project.labels_dir)
        return [self.image_paths[i] for i in labeled]

    def _run_eval(self):
        backend = self.backend_combo.currentText().strip().lower()
        weights_txt = self.weights_edit.text().strip()
        if not weights_txt:
            QMessageBox.warning(self, "Missing model", "Select model first.")
            return
        weights = Path(weights_txt)
        if backend == "sleap":
            if not weights.exists() or not weights.is_dir():
                QMessageBox.warning(self, "Missing model", "Select SLEAP model dir.")
                return
            sleap_env = self._get_sleap_env()
        else:
            if not weights.exists():
                QMessageBox.warning(self, "Missing weights", "Weights file missing.")
                return
            sleap_env = None

        out_dir = Path(self.out_dir_edit.text().strip())
        eval_paths = self._collect_eval_paths()
        if not eval_paths:
            QMessageBox.warning(self, "No labels", "No labeled frames to evaluate.")
            return

        self.btn_run.setEnabled(False)
        self.log_view.setPlainText("")
        self.progress.setValue(0)
        self.lbl_loss_plot.clear()

        # Check existing predictions cache
        pred_cache = None
        if self.cb_use_cache.isChecked() and self._worker and self._worker.pred_cache:
            # Reuse previous worker's cache if available (simple optimization)
            pass

        self._thread = QThread(self)
        self._worker = EvaluationWorker(
            weights_path=weights,
            eval_paths=eval_paths,
            labels_dir=self.project.labels_dir,
            keypoint_names=self.project.keypoint_names,
            run_dir=out_dir,
            out_root=self.project.out_root,
            device=self.device_combo.currentText(),
            imgsz=self.imgsz_spin.value(),
            conf=self.conf_spin.value(),
            pck_thr=self.pck_spin.value(),
            oks_sigma=self.oks_spin.value(),
            batch=self.batch_spin.value(),
            backend=backend,
            sleap_env=sleap_env,
            pred_cache=pred_cache,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_view.appendPlainText)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, done, total):
        self.progress.setValue(int(done / total * 100))

    def _on_failed(self, msg):
        self._cleanup()
        QMessageBox.critical(self, "Evaluation Failed", msg)

    def _on_finished(self, results):
        self._cleanup()
        overall = results["overall"]
        self.lbl_overall.setText(
            f"Overall PCK: {overall['pck']:.3f} | OKS: {overall['oks']:.3f} | "
            f"Mean Err: {overall['mean_error_px']:.1f}px"
        )

        self.kpt_table.setRowCount(len(results["per_keypoint"]))
        for i, row in enumerate(results["per_keypoint"]):
            self.kpt_table.setItem(i, 0, QTableWidgetItem(row["name"]))
            self.kpt_table.setItem(
                i, 1, QTableWidgetItem(f"{row['mean_error_px']:.1f}")
            )
            self.kpt_table.setItem(i, 2, QTableWidgetItem(f"{row['pck']:.3f}"))
            self.kpt_table.setItem(i, 3, QTableWidgetItem(f"{row['oks']:.3f}"))
            self.kpt_table.setItem(i, 4, QTableWidgetItem(f"{row['mean_conf']:.2f}"))

        self.worst_list.clear()
        for row in results["worst"]:
            item_txt = (
                f"{Path(row['image_path']).name}  "
                f"err={row['mean_error_px']:.1f}px  conf={row['mean_conf']:.2f}"
            )
            self.worst_list.addItem(item_txt)

        # Hist + Heatmap
        all_errs = [
            e for sub in results["per_kpt_errors"] for e in sub if isinstance(e, float)
        ]
        if all_errs:
            pix = make_histogram_image(
                all_errs, width=self.lbl_hist.width(), height=120
            )
            self.lbl_hist.setPixmap(pix)

    def _cleanup(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self.btn_run.setEnabled(True)
