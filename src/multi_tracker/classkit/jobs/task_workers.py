"""
Specialized worker classes for ClassKit background tasks.
"""

import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class TaskSignals(QObject):
    """
    Signals for task progress and completion.
    """

    started = Signal()
    progress = Signal(int, str)  # (percentage, message)
    finished = Signal()
    success = Signal(object)  # result data
    error = Signal(str)  # error message


class IngestWorker(QRunnable):
    """Worker for ingesting images from folders/videos."""

    def __init__(self, source_path: Path, db_path: Path):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.source_path = source_path
        self.db_path = db_path
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Initializing ingestion...")

            from ..data.ingest import IngestWorker as Ingester
            from ..store.db import ClassKitDB

            # Initialize
            self.signals.progress.emit(5, "Connecting to database...")
            db = ClassKitDB(self.db_path)
            ingester = Ingester(db)

            # Scan and ingest
            self.signals.progress.emit(10, f"Scanning folder: {self.source_path}...")
            # Prefer images/ subdirectory when present (PoseKit convention).
            from ...classkit.data.ingest import scan_images as _scan

            image_paths = list(_scan(self.source_path))
            self.signals.progress.emit(40, f"Found {len(image_paths):,} images")

            self.signals.progress.emit(50, "Computing image hashes...")
            ingester.ingest(image_paths)
            self.signals.progress.emit(90, "Writing to database...")

            self.signals.progress.emit(
                100, f"Complete! Ingested {len(image_paths):,} images"
            )
            self.signals.success.emit({"num_images": len(image_paths)})

        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class EmbeddingWorker(QRunnable):
    """Worker for computing embeddings."""

    def __init__(
        self,
        image_paths: List[Path],
        model_name: str,
        device: str,
        batch_size: int = 32,
        db_path: Optional[Path] = None,
        force_recompute: bool = False,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.image_paths = image_paths
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.db_path = db_path
        self.force_recompute = force_recompute
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        db_cls = None
        if self.db_path:
            from ..store.db import ClassKitDB as _ClassKitDB

            db_cls = _ClassKitDB

        from ..embed.embedder import (
            ModelLoadError,
            TimmEmbedder,
            resolve_embedder_device,
        )

        resolved_device = resolve_embedder_device(self.device)

        try:
            self.signals.started.emit()
            cache_model_name = self.model_name

            # Check for cached embeddings
            if db_cls is not None and not self.force_recompute:
                self.signals.progress.emit(0, "Checking for cached embeddings...")

                db = db_cls(self.db_path)

                cached = db.get_embeddings(cache_model_name, resolved_device)
                if cached is not None:
                    embeddings, metadata = cached
                    self.signals.progress.emit(
                        50, f"Found cached embeddings from {metadata['timestamp']}"
                    )
                    self.signals.progress.emit(
                        70,
                        f"Loaded {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})",
                    )
                    self.signals.progress.emit(100, "Using cached embeddings!")
                    self.signals.success.emit(
                        {
                            "embeddings": embeddings,
                            "dimension": embeddings.shape[1],
                            "cached": True,
                            "metadata": metadata,
                        }
                    )
                    return

                self.signals.progress.emit(0, "No cache found, computing embeddings...")

            self.signals.progress.emit(0, f"Loading model: {self.model_name}...")

            # Create embedder
            self.signals.progress.emit(5, f"Creating embedder on {resolved_device}...")
            embedder = TimmEmbedder(model_name=self.model_name, device=resolved_device)

            self.signals.progress.emit(8, "Loading model weights...")
            embedder.load_model()

            self.signals.progress.emit(
                10, f"Computing embeddings for {len(self.image_paths):,} images..."
            )
            self.signals.progress.emit(
                12, f"Batch size: {self.batch_size}, Device: {embedder.device}"
            )

            # Compute embeddings
            embeddings = embedder.embed(
                self.image_paths,
                batch_size=self.batch_size,
            )

            self.signals.progress.emit(
                90, f"Computed {embeddings.shape[0]:,} embeddings"
            )

            # Save to cache
            if db_cls is not None:
                self.signals.progress.emit(92, "Saving embeddings to cache...")
                db = db_cls(self.db_path)
                db.save_embeddings(
                    embeddings,
                    cache_model_name,
                    embedder.device,
                    self.batch_size,
                    meta={
                        "model_name": self.model_name,
                    },
                )
                self.signals.progress.emit(95, "Cached for future use")

            self.signals.progress.emit(100, f"Complete! Shape: {embeddings.shape}")
            self.signals.success.emit(
                {
                    "embeddings": embeddings,
                    "dimension": embedder.dimension,
                    "cached": False,
                    "metadata": {
                        "model_name": self.model_name,
                    },
                }
            )

        except (ImportError, ModelLoadError) as e:
            self.signals.error.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ClusteringWorker(QRunnable):
    """Worker for clustering embeddings."""

    def __init__(self, embeddings, n_clusters: int = 500, method: str = "minibatch"):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        self.method = method
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Starting clustering pipeline...")
            self.signals.progress.emit(
                2,
                f"Input: {self.embeddings.shape[0]:,} vectors, dim={self.embeddings.shape[1]}",
            )
            self.signals.progress.emit(
                4, f"Target: {self.n_clusters} clusters  method={self.method}"
            )

            import numpy as np

            from ..cluster.clusterer import SKLearnClusterer

            self.signals.progress.emit(8, f"Initializing {self.method} clusterer...")
            clusterer = SKLearnClusterer(
                n_clusters=self.n_clusters,
                method=self.method,
                verbose=False,
            )

            self.signals.progress.emit(
                20,
                f"Fitting {self.embeddings.shape[0]:,} samples...",
            )
            assignments = clusterer.fit(self.embeddings)

            self.signals.progress.emit(75, "Computing cluster statistics...")
            stats = clusterer.compute_cluster_stats(self.embeddings, assignments)

            n_actual = len(np.unique(assignments))
            self.signals.progress.emit(
                90, f"Created {n_actual} clusters (requested {self.n_clusters})"
            )
            self.signals.progress.emit(100, "Clustering complete!")

            self.signals.success.emit(
                {
                    "assignments": assignments,
                    "centers": clusterer.cluster_centers,
                    "stats": stats,
                    "method": self.method,
                }
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class UMAPWorker(QRunnable):
    """Worker for computing UMAP projection."""

    def __init__(self, embeddings, n_neighbors: int = 15, min_dist: float = 0.1):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Initializing UMAP...")
            self.signals.progress.emit(
                5,
                f"Input: {self.embeddings.shape[0]:,} points, {self.embeddings.shape[1]} dims",
            )

            from ..viz.umap_reduce import UMAPReducer

            self.signals.progress.emit(
                10,
                f"Parameters: n_neighbors={self.n_neighbors}, min_dist={self.min_dist}",
            )
            reducer = UMAPReducer(n_neighbors=self.n_neighbors, min_dist=self.min_dist)

            self.signals.progress.emit(
                15, "Computing UMAP projection (this may take a while)..."
            )
            coords = reducer.fit_transform(self.embeddings)

            self.signals.progress.emit(95, f"Generated 2D coordinates: {coords.shape}")
            self.signals.progress.emit(100, "Complete! UMAP projection ready")
            self.signals.success.emit({"coords": coords})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class TrainingWorker(QRunnable):
    """Worker for training classifier."""

    def __init__(
        self,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        num_classes: int,
        model_type: str = "linear",
        device: str = "cpu",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        batch_size: int = 256,
        epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        early_stop_patience: int = 10,
        calibrate: bool = True,
    ):
        super().__init__()
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_labels = val_labels
        self.num_classes = num_classes
        self.model_type = model_type
        self.device = device
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.calibrate = calibrate
        self.signals = TaskSignals()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Initializing trainer...")

            from ..train.trainer import EmbeddingHeadTrainer

            input_dim = self.train_embeddings.shape[1]
            trainer = EmbeddingHeadTrainer(
                model_type=self.model_type,
                input_dim=input_dim,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                device=self.device,
            )

            self.signals.progress.emit(10, "Training...")

            history = trainer.fit(
                self.train_embeddings,
                self.train_labels,
                self.val_embeddings,
                self.val_labels,
                batch_size=self.batch_size,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                early_stop_patience=self.early_stop_patience,
                verbose=False,
            )

            self.signals.progress.emit(80, "Calibrating...")

            metrics = None
            if self.val_embeddings is not None:
                if self.calibrate:
                    trainer.calibrate(self.val_embeddings, self.val_labels)

                # Compute metrics on validation set
                metrics = trainer.evaluate(
                    self.val_embeddings, self.val_labels, calibrated=self.calibrate
                )

            self.signals.progress.emit(100, "Complete!")
            self.signals.success.emit(
                {"trainer": trainer, "history": history, "metrics": metrics}
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ClassKitTrainingWorker(QRunnable):
    """Worker for ClassKit classification training (flat or multi-head)."""

    def __init__(
        self,
        *,
        role,
        specs,
        run_dir: str,
        multi_head: bool = False,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.role = role
        self.specs = (
            specs  # list of TrainingRunSpec (one per factor if multi_head, else one)
        )
        self.run_dir = run_dir
        self.multi_head = multi_head
        self.signals = TaskSignals()
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def run(self):
        from pathlib import Path

        from ...training.runner import run_training

        try:
            self.signals.started.emit()
            results = []

            for i, spec in enumerate(self.specs):
                if self._canceled:
                    self.signals.error.emit("Canceled by user")
                    return

                factor_run_dir = (
                    Path(self.run_dir) / f"factor_{i}"
                    if self.multi_head
                    else Path(self.run_dir) / "run"
                )
                n_specs = len(self.specs)

                def log_cb(msg: str, _i: int = i) -> None:
                    prefix = f"[factor {_i}] " if self.multi_head else ""
                    self.signals.progress.emit(0, f"{prefix}{msg}")

                def progress_cb(
                    current: int, total: int, _i: int = i, _n: int = n_specs
                ) -> None:
                    if self.multi_head:
                        overall = int((_i * 100 + current * 100 // max(1, total)) / _n)
                    else:
                        overall = current * 100 // max(1, total)
                    self.signals.progress.emit(overall, "")

                result = run_training(
                    spec,
                    factor_run_dir,
                    log_cb=log_cb,
                    progress_cb=progress_cb,
                    should_cancel=lambda: self._canceled,
                )
                results.append(result)

                if not result.get("success"):
                    self.signals.error.emit(f"Training failed for spec {i}")
                    return

            self.signals.success.emit(results)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class ExportWorker(QRunnable):
    """Worker for exporting datasets."""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        output_path: Path,
        format: str = "imagefolder",
        class_names: Optional[Dict[int, str]] = None,
        val_fraction: float = 0.2,
        test_fraction: float = 0.0,
        copy_files: bool = True,
        temp_dir: Optional[Path] = None,
        label_expansion: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.image_paths = image_paths
        self.labels = labels
        self.output_path = output_path
        self.format = format
        self.class_names = class_names or {}
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.copy_files = copy_files
        self.temp_dir = temp_dir
        # label_expansion: {"fliplr": {"left": "right", "right": "left"}, ...}
        self.label_expansion: Dict[str, Dict[str, str]] = label_expansion or {}
        self.signals = TaskSignals()

    def _class_name(self, label: int) -> str:
        """Resolve class display name for a label ID."""
        return self.class_names.get(label, f"class_{label}")

    def _build_splits(self):
        """Build deterministic train/val/test split assignments."""
        import numpy as np

        n_items = len(self.image_paths)
        if n_items == 0:
            return []

        indices = np.arange(n_items)
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        n_test = int(round(n_items * self.test_fraction))
        n_val = int(round(n_items * self.val_fraction))

        if n_test + n_val >= n_items and n_items > 1:
            overflow = (n_test + n_val) - (n_items - 1)
            if n_test >= overflow:
                n_test -= overflow
            else:
                n_val = max(0, n_val - (overflow - n_test))
                n_test = 0

        split_by_index = ["train"] * n_items
        for idx in indices[:n_test]:
            split_by_index[int(idx)] = "test"
        for idx in indices[n_test : n_test + n_val]:
            split_by_index[int(idx)] = "val"
        return split_by_index

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, f"Exporting to {self.format}...")

            # Ensure temp_dir exists when label expansion is requested
            if self.label_expansion and self.temp_dir is None:
                import tempfile

                self._expansion_tmpdir = tempfile.mkdtemp(prefix="classkit_exp_")
                self.temp_dir = Path(self._expansion_tmpdir)
            else:
                self._expansion_tmpdir = None

            valid = [
                (img_path, int(label))
                for img_path, label in zip(self.image_paths, self.labels)
                if int(label) >= 0
            ]

            if not valid:
                raise RuntimeError("No labeled samples found to export.")

            image_paths = [item[0] for item in valid]
            labels = [item[1] for item in valid]

            splits = self._build_splits()
            splits = [
                split
                for split, (_, lbl) in zip(splits, zip(self.image_paths, self.labels))
                if int(lbl) >= 0
            ]

            class_names = {label: self._class_name(label) for label in set(labels)}

            # ----------------------------------------------------------------
            # Label-switching expansion: physically write flipped copies with
            # the remapped label so the training split contains both the
            # original and its mirror with correct labels.
            # Only added to the train split to avoid leaking flipped versions
            # into the evaluation set.
            # ----------------------------------------------------------------
            if self.label_expansion and self.temp_dir:
                import cv2

                # Build reverse name→int map so we can look up the remapped int label.
                name_to_int = {v: k for k, v in class_names.items()}
                name_to_int_ci = {
                    str(v).strip().lower(): k for k, v in class_names.items()
                }

                # CV2 flip codes: 0=vertical (flipud), 1=horizontal (fliplr)
                flip_code = {"fliplr": 1, "flipud": 0}

                exp_dir = Path(self.temp_dir) / "label_expansion"
                exp_dir.mkdir(parents=True, exist_ok=True)

                extra_paths: List[Path] = []
                extra_labels: List[int] = []
                extra_splits: List[str] = []
                n_exp = 0
                n_skipped_unknown_dst = 0

                for axis, mapping in self.label_expansion.items():
                    if axis not in flip_code:
                        continue
                    code = flip_code[axis]

                    for img_path, label_int, split in zip(image_paths, labels, splits):
                        if split != "train":
                            continue  # only expand training data
                        src_name = class_names.get(label_int, "")
                        dst_name = mapping.get(src_name)
                        if dst_name is None:
                            # Backward compatibility for legacy configs with case drift.
                            src_name_ci = str(src_name).strip().lower()
                            for key, value in mapping.items():
                                if str(key).strip().lower() == src_name_ci:
                                    dst_name = value
                                    break
                        if dst_name is None:
                            continue  # label not in this expansion rule

                        dst_name_clean = str(dst_name).strip()
                        dst_int = name_to_int.get(dst_name_clean)
                        if dst_int is None:
                            dst_int = name_to_int_ci.get(dst_name_clean.lower())
                        if dst_int is None:
                            # Unknown destinations usually come from stale free-text
                            # mappings. Skip rather than inventing a synthetic class.
                            n_skipped_unknown_dst += 1
                            continue

                        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        flipped = cv2.flip(img, code)
                        stem = f"expn_{axis}_{n_exp}_{img_path.stem}"
                        out_p = exp_dir / (stem + img_path.suffix)
                        cv2.imwrite(str(out_p), flipped)
                        extra_paths.append(out_p)
                        extra_labels.append(dst_int)
                        extra_splits.append("train")
                        n_exp += 1

                if n_exp:
                    self.signals.progress.emit(
                        55, f"Added {n_exp} label-expansion copies"
                    )
                    image_paths = image_paths + extra_paths
                    labels = labels + extra_labels
                    splits = splits + extra_splits
                    self.copy_files = True  # expanded images live in temp_dir
                if n_skipped_unknown_dst:
                    self.signals.progress.emit(
                        58,
                        f"Skipped {n_skipped_unknown_dst} expansion rows with unknown destination class",
                    )

            if self.format == "imagefolder":
                from ..export.imagefolder import export_to_imagefolder

                records = [
                    (path, class_names[label], split)
                    for path, label, split in zip(image_paths, labels, splits)
                ]
                export_to_imagefolder(
                    dataset_root=self.output_path,
                    images=records,
                    copy=self.copy_files,
                )
            elif self.format == "csv":
                from ..export.parquet_csv import export_to_csv

                csv_path = self.output_path
                if csv_path.suffix.lower() != ".csv":
                    csv_path = csv_path / "labels.csv"

                export_to_csv(
                    output_path=csv_path,
                    image_paths=image_paths,
                    labels=labels,
                    class_names=class_names,
                    splits=splits,
                    include_header=True,
                )
                self.output_path = csv_path
            elif self.format == "parquet":
                from ..export.parquet_csv import export_to_parquet

                parquet_path = self.output_path
                if parquet_path.suffix.lower() != ".parquet":
                    parquet_path = parquet_path / "labels.parquet"

                export_to_parquet(
                    output_path=parquet_path,
                    image_paths=image_paths,
                    labels=labels,
                    class_names=class_names,
                    splits=splits,
                )
                self.output_path = parquet_path
            elif self.format == "ultralytics":
                from ..export.ultralytics_classify import export_ultralytics_classify

                train_images, train_labels = [], []
                val_images, val_labels = [], []
                test_images, test_labels = [], []

                for path, label, split in zip(image_paths, labels, splits):
                    if split == "val":
                        val_images.append(path)
                        val_labels.append(label)
                    elif split == "test":
                        test_images.append(path)
                        test_labels.append(label)
                    else:
                        train_images.append(path)
                        train_labels.append(label)

                if not val_images and train_images:
                    val_images.append(train_images[-1])
                    val_labels.append(train_labels[-1])
                    train_images = train_images[:-1]
                    train_labels = train_labels[:-1]

                export_ultralytics_classify(
                    output_path=self.output_path,
                    train_images=train_images,
                    train_labels=train_labels,
                    val_images=val_images,
                    val_labels=val_labels,
                    test_images=test_images if test_images else None,
                    test_labels=test_labels if test_labels else None,
                    class_names=class_names,
                    copy=self.copy_files,
                )
            else:
                raise ValueError(f"Unsupported export format: {self.format}")

            self.signals.progress.emit(100, "Complete!")
            self.signals.success.emit(
                {
                    "output_path": self.output_path,
                    "format": self.format,
                    "num_exported": len(image_paths),
                    "num_classes": len(set(labels)),
                }
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
            # Clean up auto-created expansion temp dir (if we made it)
            if getattr(self, "_expansion_tmpdir", None):
                import shutil

                try:
                    shutil.rmtree(self._expansion_tmpdir, ignore_errors=True)
                except Exception:
                    pass
                self._expansion_tmpdir = None


class YoloInferenceWorker(QRunnable):
    """Run YOLO classification inference on all images, return per-image probabilities."""

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        compute_runtime: str = "cpu",
        batch_size: int = 64,
        # Deprecated — use compute_runtime instead.
        device: str = "",
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        if not compute_runtime or compute_runtime == "cpu":
            if device and device not in ("", "cpu"):
                compute_runtime = device
        self.compute_runtime = str(compute_runtime or "cpu")
        self.batch_size = batch_size
        self.signals = TaskSignals()

    @staticmethod
    def _torch_device(rt: str) -> str:
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt == "mps":
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"  # PyTorch uses "cuda" namespace for ROCm.
        return "cpu"

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            import numpy as np
            from ultralytics import YOLO

            from ...runtime.compute_runtime import _normalize_runtime

            rt = _normalize_runtime(self.compute_runtime)
            use_onnx = rt.startswith("onnx_")
            use_tensorrt = rt == "tensorrt"

            model_path = self.model_path
            model_suffix = model_path.suffix.lower()

            if model_suffix == ".pth":
                raise ValueError(
                    f"YoloInferenceWorker requires a YOLO classify artifact (.pt/.onnx/.engine/.trt); got {model_path.name}. "
                    "Tiny CNN (.pth) models are not supported here."
                )

            if model_suffix == ".pt" and (use_onnx or use_tensorrt):
                artifact_suffix = ".onnx" if use_onnx else ".engine"
                artifact_path = model_path.with_name(
                    f"{model_path.stem}_classify_b1{artifact_suffix}"
                )
                needs_export = (not artifact_path.exists()) or (
                    artifact_path.stat().st_mtime_ns < model_path.stat().st_mtime_ns
                )
                if needs_export:
                    self.signals.progress.emit(
                        1,
                        f"Preparing YOLO classify {artifact_suffix} runtime artifact...",
                    )
                    export_model = YOLO(str(model_path), task="classify")
                    if use_onnx:
                        export_path = export_model.export(
                            format="onnx",
                            dynamic=False,
                            simplify=False,
                            opset=17,
                            batch=1,
                            verbose=False,
                        )
                    else:
                        export_device = self._torch_device(rt)
                        export_model.to(export_device)
                        export_path = export_model.export(
                            format="engine",
                            device=export_device,
                            half=True,
                            workspace=4,
                            dynamic=False,
                            batch=1,
                            verbose=False,
                        )
                    exported = Path(export_path).expanduser().resolve()
                    if not exported.exists():
                        raise RuntimeError(
                            f"YOLO runtime export output missing: {exported}"
                        )
                    if exported != artifact_path:
                        shutil.copy2(str(exported), str(artifact_path))
                model_path = artifact_path

            if model_path.suffix.lower() not in {".pt", ".onnx", ".engine", ".trt"}:
                raise ValueError(
                    f"Unsupported YOLO model artifact for inference: {model_path.name}"
                )

            self.signals.progress.emit(
                0, f"Loading YOLO model ({rt}): {model_path.name}..."
            )
            model = YOLO(str(model_path), task="classify")
            class_names = (
                [model.names[i] for i in sorted(model.names.keys())]
                if hasattr(model, "names")
                else []
            )

            predict_device = None
            if model_path.suffix.lower() == ".pt":
                # Place native PyTorch model on requested torch device.
                # For exported runtimes, device routing is backend/provider-driven.
                predict_device = self._torch_device(rt)
                try:
                    model.to(predict_device)
                    predict_device = None
                except Exception:
                    pass

            num_images = len(self.image_paths)
            self.signals.progress.emit(
                5, f"Running inference on {num_images:,} images ({rt})..."
            )
            all_probs = []

            for batch_start in range(0, num_images, self.batch_size):
                batch_paths = self.image_paths[
                    batch_start : batch_start + self.batch_size
                ]
                batch_input = [str(p) for p in batch_paths]
                kwargs = {"verbose": False}
                if predict_device is not None:
                    kwargs["device"] = predict_device
                results = model(batch_input, **kwargs)
                for r in results:
                    if r.probs is not None:
                        all_probs.append(r.probs.data.cpu().numpy())
                    else:
                        n_cls = max(len(class_names), 1)
                        all_probs.append(np.ones(n_cls) / n_cls)

                done = batch_start + len(batch_paths)
                pct = min(95, 5 + int(90 * done / num_images))
                self.signals.progress.emit(pct, f"Processed {done:,}/{num_images:,}")

            probs = np.array(all_probs)  # (N, num_classes)
            self.signals.progress.emit(100, "Inference complete!")
            self.signals.success.emit({"probs": probs, "class_names": class_names})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class LogitsUMAPWorker(QRunnable):
    """Compute UMAP projection from a probability / logit matrix (N × C)."""

    def __init__(self, probs, n_neighbors: int = 15, min_dist: float = 0.1):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.probs = probs
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            n, c = self.probs.shape
            self.signals.progress.emit(0, f"Input: {n:,} images × {c} classes")

            from ..viz.umap_reduce import UMAPReducer

            self.signals.progress.emit(
                10,
                f"Computing UMAP in model logits space (n_neighbors={self.n_neighbors})...",
            )
            reducer = UMAPReducer(
                n_neighbors=min(self.n_neighbors, max(2, n - 1)),
                min_dist=self.min_dist,
                metric="euclidean",
            )
            coords = reducer.fit_transform(self.probs)
            self.signals.progress.emit(100, "Model-space UMAP complete!")
            self.signals.success.emit({"coords": coords})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class LogitsPCAWorker(QRunnable):
    """Compute a 2D PCA projection from a probability / logit matrix (N × C)."""

    def __init__(self, probs):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.probs = probs
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            import numpy as np

            mat = np.asarray(self.probs, dtype=np.float64)
            if mat.ndim != 2:
                raise ValueError("Expected 2D probability/logit matrix for PCA")
            n, c = mat.shape
            self.signals.progress.emit(0, f"Input: {n:,} images × {c} classes")

            if n < 2:
                raise ValueError("Need at least 2 samples to compute PCA")

            self.signals.progress.emit(20, "Centering logits/probabilities...")
            centered = mat - mat.mean(axis=0, keepdims=True)

            self.signals.progress.emit(50, "Running SVD for PCA projection...")
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)

            components = vt[:2].T if vt.shape[0] >= 2 else vt[:1].T
            coords = centered @ components
            if coords.shape[1] == 1:
                coords = np.concatenate(
                    [coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)],
                    axis=1,
                )

            self.signals.progress.emit(100, "Model-space PCA complete!")
            self.signals.success.emit({"coords": coords.astype(np.float32)})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ALBatchWorker(QRunnable):
    """Select the next active learning labeling batch using acquisition strategies."""

    def __init__(
        self,
        embeddings,
        probs,
        labeled_mask,
        cluster_assignments=None,
        batch_size: int = 50,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.embeddings = embeddings
        self.probs = probs
        self.labeled_mask = labeled_mask
        self.cluster_assignments = cluster_assignments
        self.batch_size = batch_size
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            import numpy as np

            from ..al.acquisition import BatchAcquisition, BatchConfig

            self.signals.progress.emit(0, "Computing active learning batch...")
            unlabeled_mask = ~self.labeled_mask

            if unlabeled_mask.sum() == 0:
                self.signals.success.emit(
                    {"selected_indices": np.array([], dtype=int), "breakdown": {}}
                )
                return

            cluster_densities = None
            label_coverage = {}

            if self.cluster_assignments is not None:
                self.signals.progress.emit(20, "Computing cluster densities...")
                try:
                    from ..al.density import compute_cluster_densities

                    cluster_densities = compute_cluster_densities(
                        self.embeddings, self.cluster_assignments
                    )
                except Exception:
                    pass

                try:
                    from ..train.metrics import compute_label_coverage

                    labeled_int = np.where(self.labeled_mask, 0, -1)
                    label_coverage = compute_label_coverage(
                        labeled_int, self.cluster_assignments
                    )
                except Exception:
                    pass

            self.signals.progress.emit(50, "Running batch acquisition selection...")
            cfg = BatchConfig(
                batch_size=min(self.batch_size, int(unlabeled_mask.sum()))
            )
            acq = BatchAcquisition(cfg)
            selected, breakdown = acq.select_batch(
                embeddings=self.embeddings,
                probs=self.probs,
                unlabeled_mask=unlabeled_mask,
                cluster_assignments=self.cluster_assignments,
                label_coverage=label_coverage,
                cluster_densities=cluster_densities,
            )

            self.signals.progress.emit(100, f"Selected {len(selected)} candidates!")
            self.signals.success.emit(
                {"selected_indices": selected, "breakdown": breakdown}
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class TinyCNNInferenceWorker(QRunnable):
    """Run tiny CNN classification inference on all project images.

    Supports PyTorch (cpu/mps/cuda/rocm) and ONNX/TensorRT runtimes via the
    canonical ``compute_runtime`` parameter (see ``runtime.compute_runtime``).
    ONNX inference requires a ``<stem>.onnx`` sibling of the ``.pth`` model file,
    which is auto-exported by ``runner.py`` after training.

    Returns the same ``{"probs": ndarray, "class_names": list}`` contract as
    ``YoloInferenceWorker`` so callers share a unified post-inference path.
    """

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        class_names: List[str],
        compute_runtime: str = "cpu",
        batch_size: int = 64,
        # Deprecated — use compute_runtime instead.
        device: str = "",
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        self.class_names = list(class_names)
        # Backward-compat: if caller only passed the old `device` kwarg, honour it.
        if not compute_runtime or compute_runtime == "cpu":
            if device and device not in ("", "cpu"):
                compute_runtime = device
        self.compute_runtime = str(compute_runtime or "cpu")
        self.batch_size = batch_size
        self.signals = TaskSignals()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _torch_device(rt: str) -> str:
        """Map a canonical runtime string to a plain PyTorch device string."""
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt == "mps":
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"  # PyTorch uses "cuda" for ROCm
        return "cpu"

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            import cv2
            import numpy as np

            from ...runtime.compute_runtime import _normalize_runtime

            rt = _normalize_runtime(self.compute_runtime)
            use_onnx = rt.startswith("onnx_") or rt == "tensorrt"
            onnx_path = self.model_path.with_suffix(".onnx")
            if use_onnx and not onnx_path.exists():
                # ONNX file not available — fall back to PyTorch silently.
                use_onnx = False

            self.signals.progress.emit(
                0, f"Loading tiny CNN ({rt}): {self.model_path.name}..."
            )

            num_images = len(self.image_paths)

            # ── Image-loading helper (shared between both inference paths) ────
            def _load_batch_images(batch_paths, input_w, input_h):
                tensors = []
                for p in batch_paths:
                    try:
                        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        if img is None:
                            tensors.append(
                                np.zeros((3, input_h, input_w), dtype=np.float32)
                            )
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(
                            img, (input_w, input_h), interpolation=cv2.INTER_LINEAR
                        )
                        tensors.append(
                            img.transpose(2, 0, 1).astype(np.float32) / 255.0
                        )
                    except Exception:
                        tensors.append(
                            np.zeros((3, input_h, input_w), dtype=np.float32)
                        )
                return (
                    np.stack(tensors, axis=0)
                    if tensors
                    else np.zeros((0, 3, input_h, input_w), dtype=np.float32)
                )

            all_probs: list = []
            resolved_names: list = list(self.class_names)

            # ── ONNX / TensorRT inference path ────────────────────────────────
            if use_onnx:
                import torch  # for reading ckpt metadata only

                from ...training.tiny_model import load_tiny_onnx, run_tiny_onnx

                # Read input size from .pth checkpoint (no GPU needed)
                try:
                    _ckpt = torch.load(
                        str(self.model_path), map_location="cpu", weights_only=False
                    )
                    input_w, input_h = _ckpt.get("input_size", [128, 64])
                    ckpt_names = _ckpt.get("class_names")
                    if ckpt_names:
                        resolved_names = list(ckpt_names)
                except Exception:
                    input_w, input_h = 128, 64

                session = load_tiny_onnx(str(onnx_path), compute_runtime=rt)
                self.signals.progress.emit(
                    5, f"Tiny CNN ONNX inference on {num_images:,} images..."
                )

                for batch_start in range(0, num_images, self.batch_size):
                    batch_paths = self.image_paths[
                        batch_start : batch_start + self.batch_size
                    ]
                    batch_np = _load_batch_images(batch_paths, input_w, input_h)
                    if batch_np.shape[0] == 0:
                        continue
                    probs = run_tiny_onnx(session, batch_np)
                    all_probs.append(probs)
                    done = batch_start + len(batch_paths)
                    pct = min(95, 5 + int(90 * done / num_images))
                    self.signals.progress.emit(
                        pct, f"Processed {done:,}/{num_images:,}"
                    )

            # ── PyTorch inference path ─────────────────────────────────────────
            else:
                import torch
                import torch.nn.functional as F

                from ...training.tiny_model import load_tiny_classifier

                torch_device = self._torch_device(rt)
                model, ckpt = load_tiny_classifier(
                    str(self.model_path), device=torch_device
                )
                input_w, input_h = ckpt.get("input_size", [128, 64])
                ckpt_names = ckpt.get("class_names")
                if ckpt_names:
                    resolved_names = list(ckpt_names)

                self.signals.progress.emit(
                    5, f"Tiny CNN inference on {num_images:,} images..."
                )

                for batch_start in range(0, num_images, self.batch_size):
                    batch_paths = self.image_paths[
                        batch_start : batch_start + self.batch_size
                    ]
                    batch_np = _load_batch_images(batch_paths, input_w, input_h)
                    if batch_np.shape[0] == 0:
                        continue
                    x = torch.from_numpy(batch_np).to(torch_device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
                    done = batch_start + len(batch_paths)
                    pct = min(95, 5 + int(90 * done / num_images))
                    self.signals.progress.emit(
                        pct, f"Processed {done:,}/{num_images:,}"
                    )

            if all_probs:
                result_probs = np.concatenate(all_probs, axis=0)
            else:
                n_cls = max(len(resolved_names), 1)
                result_probs = np.full((num_images, n_cls), 1.0 / n_cls)

            self.signals.progress.emit(100, "Tiny CNN inference complete!")
            self.signals.success.emit(
                {"probs": result_probs, "class_names": resolved_names}
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class TorchvisionInferenceWorker(QRunnable):
    """Run torchvision Custom CNN classification inference on all project images.

    Supports PyTorch (cpu/mps/cuda/rocm) and ONNX runtimes via
    ``compute_runtime``. Output contract: same as TinyCNNInferenceWorker —
    emits ``{"probs": ndarray(N, C), "class_names": list}`` via success signal.
    """

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        class_names: List[str],
        input_size: int = 224,
        compute_runtime: str = "cpu",
        batch_size: int = 64,
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        self.class_names = list(class_names)
        self.input_size = input_size
        self.compute_runtime = str(compute_runtime or "cpu")
        self.batch_size = batch_size
        self.signals = TaskSignals()

    @staticmethod
    def _torch_device(rt: str) -> str:
        """Map canonical runtime to PyTorch device string."""
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt == "mps":
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"
        return "cpu"

    @Slot()
    def run(self) -> None:
        import numpy as _np
        import torch
        from torchvision import transforms

        try:
            self.signals.started.emit()
            from ...runtime.compute_runtime import _normalize_runtime

            rt = _normalize_runtime(self.compute_runtime)
            sz = self.input_size
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            tf = transforms.Compose(
                [
                    transforms.Resize((sz, sz)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            use_onnx = rt in ("onnx_cpu", "onnx_cuda", "onnx_rocm", "tensorrt")
            if use_onnx:
                onnx_path = self.model_path.with_suffix(".onnx")
                if not onnx_path.exists():
                    use_onnx = False

            if use_onnx:
                import onnxruntime as ort

                onnx_path = self.model_path.with_suffix(".onnx")
                # ROCm: onnxruntime has no ROCMExecutionProvider; falls through to CPU
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "cuda" in rt
                    else ["CPUExecutionProvider"]
                )
                sess = ort.InferenceSession(str(onnx_path), providers=providers)
                input_name = sess.get_inputs()[0].name

                def _infer(batch_np):
                    return sess.run(None, {input_name: batch_np})[0]

            else:
                from ...training.torchvision_model import load_torchvision_classifier

                device = self._torch_device(rt)
                model, _ = load_torchvision_classifier(
                    str(self.model_path), device=device
                )

                def _infer(batch_np):
                    t = torch.from_numpy(batch_np).to(device)
                    with torch.no_grad():
                        return model(t).cpu().numpy()

            from PIL import Image

            all_probs = []
            total = len(self.image_paths)
            for i in range(0, total, self.batch_size):
                batch_paths = self.image_paths[i : i + self.batch_size]
                batch_tensors = []
                for p in batch_paths:
                    try:
                        img = Image.open(str(p)).convert("RGB")
                        batch_tensors.append(tf(img).numpy())
                    except Exception:
                        batch_tensors.append(_np.zeros((3, sz, sz), dtype=_np.float32))
                batch_np = _np.stack(batch_tensors).astype(_np.float32)
                logits = _infer(batch_np)
                # Softmax
                exp = _np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                all_probs.append(probs)
                pct = int(min(i + self.batch_size, total) * 100 / total)
                self.signals.progress.emit(
                    pct, f"Inferring {min(i + self.batch_size, total)}/{total}"
                )

            all_probs_np = (
                _np.concatenate(all_probs, axis=0)
                if all_probs
                else _np.zeros((0, len(self.class_names)))
            )
            self.signals.success.emit(
                {"probs": all_probs_np, "class_names": self.class_names}
            )

        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class AprilTagAutoLabelWorker(QRunnable):
    """Background worker that runs AprilTag auto-labeling on a list of images.

    The caller (mainwindow) is responsible for:
    - pre-filtering image_paths to unlabeled images only
    - writing the labeling scheme before starting the worker
    - connecting signals to update the UI

    This worker uses ``db`` only for writing labels — it never reads from it.
    """

    def __init__(
        self,
        image_paths: List[Path],
        config,  # AprilTagConfig — imported lazily to avoid hard dep at module load
        threshold: float,
        db,  # ClassKitDB — imported lazily
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.image_paths = image_paths
        self.config = config
        self.threshold = threshold
        self.db = db
        self.signals = TaskSignals()
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    @Slot()
    def run(self) -> None:
        from ..autolabel.apriltag import autolabel_images

        try:
            self.signals.started.emit()
            total = len(self.image_paths)
            if total == 0:
                self.signals.success.emit(
                    {"n_labeled": 0, "n_skipped": 0, "n_no_tag": 0}
                )
                return

            self.signals.progress.emit(0, f"Auto-labeling {total:,} images...")

            n_labeled = 0
            n_skipped = 0
            n_no_tag = 0
            batch_size = 100

            for batch_start in range(0, total, batch_size):
                if self._canceled:
                    self.signals.error.emit("Canceled by user")
                    return

                batch = self.image_paths[batch_start : batch_start + batch_size]
                results = autolabel_images(batch, self.config, self.threshold)

                updates: Dict[str, tuple] = {}
                for path, result in zip(batch, results):
                    if result.label is not None:
                        updates[str(path)] = (result.label, result.confidence)
                        if result.label == "no_tag":
                            n_no_tag += 1
                        else:
                            n_labeled += 1
                    else:
                        n_skipped += 1

                if updates:
                    self.db.update_labels_with_confidence_batch(updates)

                done = min(batch_start + batch_size, total)
                pct = int(done * 100 / total)
                self.signals.progress.emit(
                    pct,
                    f"Labeled {n_labeled + n_no_tag}, skipped {n_skipped} of {total}",
                )

            self.signals.success.emit(
                {"n_labeled": n_labeled, "n_skipped": n_skipped, "n_no_tag": n_no_tag}
            )

        except Exception as exc:
            traceback.print_exc()
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
