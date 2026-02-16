from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..data.ingest import compute_image_hash, scan_images
from ..embed.embedder import TimmEmbedder
from ..index.faiss_index import KnnIndex
from ..store.db import ClassKitDB
from ..viz.umap_reduce import compute_umap_viz


class IngestWorker(QRunnable):
    """
    Background worker to ingest images, compute embeddings, build index, and run UMAP.
    """

    class Signals(QObject):
        progress = Signal(str, int)  # status message, percentage
        finished = Signal()
        error = Signal(str)
        data_ready = Signal(object, object, object)  # embeddings, index, umap_coords

    def __init__(
        self,
        image_dir: Path,
        db_path: Path,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        device: str = "cuda",
    ):
        super().__init__()
        self.image_dir = image_dir
        self.db_path = db_path
        self.model_name = model_name
        self.device = device
        self.signals = self.Signals()

    @Slot()
    def run(self):
        try:
            # 1. Scan Images
            self.signals.progress.emit("Scanning directory...", 0)
            image_paths = list(scan_images(self.image_dir))
            if not image_paths:
                self.signals.error.emit(f"No images found in {self.image_dir}")
                return

            self.signals.progress.emit(
                f"Found {len(image_paths)} images. Hashing...", 5
            )

            # 2. Hash & DB Ingest
            # For large datasets, we might want to parallelize hashing or do it lazily
            # MVP: just do it linearly for now, it's IO bound
            hashes = []
            for i, p in enumerate(image_paths):
                if i % 100 == 0:
                    self.signals.progress.emit(
                        f"Hashing {i}/{len(image_paths)}...",
                        int(5 + 10 * (i / len(image_paths))),
                    )
                hashes.append(compute_image_hash(p))

            self.signals.progress.emit("Saving to database...", 15)
            db = ClassKitDB(self.db_path)
            db.add_images(image_paths, hashes)

            # 3. Embeddings
            self.signals.progress.emit("Loading Embedder...", 20)
            try:
                embedder = TimmEmbedder(model_name=self.model_name, device=self.device)
                embedder.load_model()
            except ImportError as e:
                self.signals.error.emit(f"Failed to load embedder: {e}")
                return

            self.signals.progress.emit(
                "Computing embeddings (this may take a while)...", 25
            )
            # Todo: check DB for existing embeddings to skip?
            # For now, simplistic re-compute

            # We need to process in chunks to update progress
            BATCH_SIZE = 32
            total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
            all_embeddings = []

            for i in range(0, len(image_paths), BATCH_SIZE):
                batch_paths = image_paths[i : i + BATCH_SIZE]
                batch_emb = embedder.embed(batch_paths, batch_size=BATCH_SIZE)
                all_embeddings.append(batch_emb)

                percent = 25 + int(50 * ((i // BATCH_SIZE) / total_batches))
                self.signals.progress.emit(
                    f"Embedding batch {i // BATCH_SIZE}/{total_batches}", percent
                )

            embeddings = np.concatenate(all_embeddings, axis=0)

            # 4. Save Embeddings to DB (blob) - simplified for now, usually use numpy/disk
            # db.save_embeddings(embeddings) # Helper needed in DB

            # 5. Build FAISS Index
            self.signals.progress.emit("Building FAISS Index...", 80)
            d = embeddings.shape[1]
            index = KnnIndex(d, metric="l2")
            index.add(embeddings)

            # 6. Compute UMAP
            self.signals.progress.emit("Computing UMAP visualization...", 90)
            # Subsample for UMAP if too large?
            umap_coords = compute_umap_viz(embeddings)

            self.signals.progress.emit("Done!", 100)
            self.signals.data_ready.emit(embeddings, index, umap_coords)
            self.signals.finished.emit()

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.signals.error.emit(str(e))
