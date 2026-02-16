"""
Specialized worker classes for ClassKit background tasks.
"""

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
            image_paths = ingester.scan_folder(self.source_path)
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
        self.image_paths = image_paths
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.db_path = db_path
        self.force_recompute = force_recompute
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()

            # Check for cached embeddings
            if self.db_path and not self.force_recompute:
                self.signals.progress.emit(0, "Checking for cached embeddings...")

                from ..store.db import ClassKitDB

                db = ClassKitDB(self.db_path)

                cached = db.get_embeddings(self.model_name, self.device)
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

            from ..embed.embedder import TimmEmbedder

            # Create embedder
            self.signals.progress.emit(5, f"Creating embedder on {self.device}...")
            embedder = TimmEmbedder(model_name=self.model_name, device=self.device)

            self.signals.progress.emit(8, "Loading model weights...")
            embedder.load_model()

            self.signals.progress.emit(
                10, f"Computing embeddings for {len(self.image_paths):,} images..."
            )
            self.signals.progress.emit(
                12, f"Batch size: {self.batch_size}, Device: {self.device}"
            )

            # Compute embeddings
            embeddings = embedder.embed(self.image_paths, batch_size=self.batch_size)

            self.signals.progress.emit(
                90, f"Computed {embeddings.shape[0]:,} embeddings"
            )

            # Save to cache
            if self.db_path:
                self.signals.progress.emit(92, "Saving embeddings to cache...")
                db = ClassKitDB(self.db_path)
                db.save_embeddings(
                    embeddings, self.model_name, self.device, self.batch_size
                )
                self.signals.progress.emit(95, "Cached for future use")

            self.signals.progress.emit(100, f"Complete! Shape: {embeddings.shape}")
            self.signals.success.emit(
                {
                    "embeddings": embeddings,
                    "dimension": embedder.dimension,
                    "cached": False,
                }
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ClusteringWorker(QRunnable):
    """Worker for clustering embeddings."""

    def __init__(self, embeddings, n_clusters: int = 500, gpu: bool = False):
        super().__init__()
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        self.gpu = gpu
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
            self.signals.progress.emit(4, f"Target: {self.n_clusters} clusters")

            import platform

            import numpy as np

            is_macos = platform.system() == "Darwin"

            # On macOS, try metalfaiss first, then sklearn (FAISS causes segfaults)
            if is_macos:
                self.signals.progress.emit(
                    6, "Detected macOS - using safe clustering backend..."
                )

                # Try metalfaiss (Metal-accelerated FAISS for Apple Silicon)
                try:
                    from ..cluster.metalfaiss_backend import load_metalfaiss_backend

                    mx, AnyClustering, ClusteringParameters, backend_info = (
                        load_metalfaiss_backend()
                    )

                    self.signals.progress.emit(
                        8, "Using Metal-accelerated FAISS (metalfaiss)..."
                    )
                    if backend_info.get("shadow_path_removed"):
                        self.signals.progress.emit(
                            9,
                            "Detected and bypassed local Faiss-mlx shadow path; using active environment backend.",
                        )
                    self.signals.progress.emit(
                        10, "Converting embeddings to MLX format..."
                    )

                    # Convert to list format (metalfaiss expects List[List[float]])
                    embeddings_list = self.embeddings.astype(np.float32).tolist()
                    d = self.embeddings.shape[1]

                    self.signals.progress.emit(
                        15,
                        f"Initializing Metal k-means (d={d}, k={self.n_clusters})...",
                    )
                    params = ClusteringParameters(max_iterations=20)
                    kmeans = AnyClustering.new(
                        d=d, k=self.n_clusters, parameters=params
                    )

                    self.signals.progress.emit(
                        20, f"Training on {len(embeddings_list):,} vectors..."
                    )
                    kmeans.train(embeddings_list)

                    self.signals.progress.emit(70, "Computing cluster assignments...")
                    centroids_mlx = kmeans.centroids()
                    centers = np.array(centroids_mlx)

                    # Compute assignments: find nearest centroid for each point
                    embeddings_mlx = mx.array(self.embeddings.astype(np.float32))
                    dists = mx.sum(
                        mx.square(
                            mx.subtract(embeddings_mlx[:, None], centroids_mlx[None])
                        ),
                        axis=2,
                    )
                    assignments = np.array(mx.argmin(dists, axis=1))

                    self.signals.progress.emit(
                        90, f"Clustered into {len(np.unique(assignments))} groups"
                    )
                    self.signals.progress.emit(100, "Metal clustering complete!")

                    self.signals.success.emit(
                        {
                            "assignments": assignments,
                            "centers": centers,
                            "stats": {},
                            "method": "metalfaiss",
                        }
                    )
                    return

                except ImportError as e:
                    self.signals.progress.emit(
                        8, f"Metal FAISS unavailable ({str(e)}), using sklearn..."
                    )
                except Exception as e:
                    self.signals.progress.emit(
                        8, f"Metal FAISS error: {str(e)}, falling back to sklearn..."
                    )

                # Fallback to sklearn on macOS
                self.signals.progress.emit(
                    10, "Using scikit-learn MiniBatchKMeans (CPU-based, stable)..."
                )
                from sklearn.cluster import MiniBatchKMeans

                self.signals.progress.emit(
                    15,
                    f"Initializing MiniBatchKMeans with {self.n_clusters} clusters...",
                )
                kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    batch_size=min(1024, self.embeddings.shape[0]),
                    verbose=0,
                    random_state=42,
                    n_init=5,
                )

                self.signals.progress.emit(
                    20,
                    f"Fitting {self.embeddings.shape[0]:,} samples (batch_size={kmeans.batch_size})...",
                )
                self.signals.progress.emit(
                    25, "This may take a few minutes for large datasets..."
                )

                assignments = kmeans.fit_predict(self.embeddings)
                centers = kmeans.cluster_centers_

                n_actual = len(np.unique(assignments))
                self.signals.progress.emit(
                    90, f"Created {n_actual} clusters (requested {self.n_clusters})"
                )
                self.signals.progress.emit(
                    100,
                    f"sklearn clustering complete! {len(assignments):,} points assigned",
                )

                self.signals.success.emit(
                    {
                        "assignments": assignments,
                        "centers": centers,
                        "stats": {},
                        "method": "sklearn",
                    }
                )

            else:
                # On Linux/Windows, try FAISS
                self.signals.progress.emit(
                    8, "Using FAISS k-means (optimized for CPU/CUDA)..."
                )

                try:
                    from ..cluster.clusterer import FAISSClusterer

                    self.signals.progress.emit(10, "Initializing FAISS clusterer...")
                    clusterer = FAISSClusterer(
                        n_clusters=self.n_clusters, verbose=False
                    )

                    use_gpu = self.gpu
                    if use_gpu:
                        self.signals.progress.emit(15, "GPU acceleration enabled")

                    self.signals.progress.emit(
                        20,
                        f"Running k-means on {self.embeddings.shape[0]:,} vectors...",
                    )
                    assignments = clusterer.fit(self.embeddings, gpu=use_gpu)

                    self.signals.progress.emit(75, "Computing cluster statistics...")
                    stats = clusterer.compute_cluster_stats(
                        self.embeddings, assignments
                    )

                    self.signals.progress.emit(
                        90, f"Computed stats for {len(set(assignments))} clusters"
                    )
                    self.signals.progress.emit(100, "FAISS clustering complete!")

                    self.signals.success.emit(
                        {
                            "assignments": assignments,
                            "centers": clusterer.cluster_centers,
                            "stats": stats,
                            "method": "faiss",
                        }
                    )

                except Exception as e:
                    self.signals.progress.emit(
                        15, f"FAISS failed ({str(e)}), using sklearn..."
                    )
                    raise

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class UMAPWorker(QRunnable):
    """Worker for computing UMAP projection."""

    def __init__(self, embeddings, n_neighbors: int = 15, min_dist: float = 0.1):
        super().__init__()
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

            if self.calibrate and self.val_embeddings is not None:
                trainer.calibrate(self.val_embeddings, self.val_labels)

            self.signals.progress.emit(100, "Complete!")
            self.signals.success.emit({"trainer": trainer, "history": history})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
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
    ):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.output_path = output_path
        self.format = format
        self.class_names = class_names or {}
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.copy_files = copy_files
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
