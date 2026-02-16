import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ClassKitDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if not exist."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()

            # Images table
            c.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT,
                    label TEXT,
                    confidence REAL,
                    embedding_idx INTEGER,
                    split TEXT DEFAULT 'train', -- train, val, test, unassigned
                    meta_json TEXT
                )
            """)

            # Clusters table
            c.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY,
                    cluster_idx INTEGER,
                    label_consensus TEXT,
                    purity REAL
                )
            """)

            # Runs/Versions table
            c.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT,
                    metrics_json TEXT
                )
            """)

            # Embeddings cache table
            c.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    device TEXT,
                    batch_size INTEGER,
                    num_images INTEGER,
                    dimension INTEGER,
                    file_path TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    meta_json TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS cluster_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    num_images INTEGER,
                    num_embeddings INTEGER,
                    n_clusters INTEGER,
                    method TEXT,
                    assignments_path TEXT NOT NULL,
                    centers_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    meta_json TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS umap_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    num_images INTEGER,
                    n_neighbors INTEGER,
                    min_dist REAL,
                    coords_path TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    meta_json TEXT
                )
            """)
            conn.commit()

    def add_images(self, paths: List[Path], hashes: List[str] = None):
        """Batch insert images."""
        if hashes is None:
            hashes = [None] * len(paths)

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Use executemany for speed
            data = [(str(p), h) for p, h in zip(paths, hashes)]
            c.executemany(
                "INSERT OR IGNORE INTO images (file_path, file_hash) VALUES (?, ?)",
                data,
            )
            conn.commit()

    def get_unlabeled(self, limit: int = 100) -> List[Tuple[Any]]:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM images WHERE label IS NULL LIMIT ?", (limit,))
            return c.fetchall()

    def update_label(self, path: str, label: str):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("UPDATE images SET label = ? WHERE file_path = ?", (label, path))
            conn.commit()

    def get_image_path_by_id(self, idx: int) -> Optional[str]:
        """Get path for image at index (0-based ID or offset-based in list).
        Actually let's use ROWID/Order for now to match embedding index.
        """
        # CAUTION: This assumes rows are never deleted so ID matches index + 1
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Assuming id corresponds to embedding index + 1
            c.execute("SELECT file_path FROM images WHERE id = ?", (idx + 1,))
            res = c.fetchone()
            return res[0] if res else None

    def get_all_labels(self) -> List[Optional[str]]:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT label FROM images ORDER BY id"
            )  # Order must match embeddings
            return [r[0] for r in c.fetchall()]

    def count_images(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM images")
            res = c.fetchone()
            return res[0] if res else 0

    def get_all_image_paths(self) -> List[str]:
        """Get all image paths ordered by ID."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT file_path FROM images ORDER BY id")
            return [r[0] for r in c.fetchall()]

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save embeddings to disk and metadata to database.

        Args:
            embeddings: Embeddings array (N, D)
            model_name: Name of the model used
            device: Device used for computation
            batch_size: Batch size used
            meta: Optional metadata dictionary

        Returns:
            ID of the saved embedding record
        """
        # Create embeddings directory
        embeddings_dir = self.db_path.parent / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        # Generate filename based on model and timestamp
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace('/', '_')}_{timestamp}.npy"
        file_path = embeddings_dir / filename

        # Save embeddings to disk
        np.save(file_path, embeddings)

        # Save metadata to database
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO embeddings
                (model_name, device, batch_size, num_images, dimension, file_path, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    device,
                    batch_size,
                    embeddings.shape[0],
                    embeddings.shape[1],
                    str(file_path),
                    json.dumps(meta) if meta else None,
                ),
            )
            conn.commit()
            return c.lastrowid

    def get_embeddings(
        self, model_name: str, device: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Load cached embeddings if they exist for the given model.

        Args:
            model_name: Name of the model
            device: Optional device filter (if None, returns any device)

        Returns:
            Tuple of (embeddings, metadata) or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()

            # Get the most recent embeddings for this model
            if device:
                c.execute(
                    """
                    SELECT file_path, num_images, dimension, device, batch_size,
                           timestamp, meta_json
                    FROM embeddings
                    WHERE model_name = ? AND device = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (model_name, device),
                )
            else:
                c.execute(
                    """
                    SELECT file_path, num_images, dimension, device, batch_size,
                           timestamp, meta_json
                    FROM embeddings
                    WHERE model_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (model_name,),
                )

            result = c.fetchone()

            if not result:
                return None

            file_path, num_images, dimension, dev, batch_size, timestamp, meta_json = (
                result
            )

            # Check if file exists
            if not Path(file_path).exists():
                return None

            # Load embeddings
            embeddings = np.load(file_path)

            # Verify shape matches expected (image count should match DB)
            current_image_count = self.count_images()
            if embeddings.shape[0] != current_image_count:
                # Image count changed, invalidate cache
                return None

            # Build metadata
            metadata = {
                "model_name": model_name,
                "device": dev,
                "batch_size": batch_size,
                "num_images": num_images,
                "dimension": dimension,
                "timestamp": timestamp,
                "file_path": file_path,
            }

            if meta_json:
                metadata.update(json.loads(meta_json))

            return embeddings, metadata

    def list_cached_embeddings(self) -> List[Dict[str, Any]]:
        """
        List all cached embeddings.

        Returns:
            List of embedding metadata dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT model_name, device, num_images, dimension, timestamp, file_path
                FROM embeddings
                ORDER BY timestamp DESC
                """)

            results = []
            for row in c.fetchall():
                model_name, device, num_images, dimension, timestamp, file_path = row
                results.append(
                    {
                        "model_name": model_name,
                        "device": device,
                        "num_images": num_images,
                        "dimension": dimension,
                        "timestamp": timestamp,
                        "file_path": file_path,
                        "exists": Path(file_path).exists(),
                    }
                )

            return results

    def get_most_recent_embeddings(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load most recently cached embeddings regardless of model/device."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT model_name, device, batch_size, num_images, dimension,
                       file_path, timestamp, meta_json
                FROM embeddings
                ORDER BY timestamp DESC
                LIMIT 1
                """)
            row = c.fetchone()
            if not row:
                return None

            (
                model_name,
                device,
                batch_size,
                num_images,
                dimension,
                file_path,
                timestamp,
                meta_json,
            ) = row
            cache_path = Path(file_path)
            if not cache_path.exists():
                return None

            embeddings = np.load(cache_path)
            if embeddings.shape[0] != self.count_images():
                return None

            metadata = {
                "model_name": model_name,
                "device": device,
                "batch_size": batch_size,
                "num_images": num_images,
                "dimension": dimension,
                "timestamp": timestamp,
                "file_path": file_path,
            }
            if meta_json:
                metadata.update(json.loads(meta_json))
            return embeddings, metadata

    def save_cluster_cache(
        self,
        assignments: np.ndarray,
        centers: Optional[np.ndarray],
        n_clusters: int,
        method: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist clustering outputs for reuse."""
        cache_dir = self.db_path.parent / "clusters"
        cache_dir.mkdir(exist_ok=True)

        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        assignments_path = cache_dir / f"assignments_{timestamp}.npy"
        np.save(assignments_path, assignments)

        centers_path = None
        if centers is not None:
            centers_path = cache_dir / f"centers_{timestamp}.npy"
            np.save(centers_path, centers)

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO cluster_cache
                (num_images, num_embeddings, n_clusters, method, assignments_path, centers_path, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.count_images(),
                    int(len(assignments)),
                    int(n_clusters),
                    method,
                    str(assignments_path),
                    str(centers_path) if centers_path else None,
                    json.dumps(meta) if meta else None,
                ),
            )
            conn.commit()
            return c.lastrowid

    def get_most_recent_cluster_cache(self) -> Optional[Dict[str, Any]]:
        """Load most recently cached cluster assignments and optional centers."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT num_images, num_embeddings, n_clusters, method,
                       assignments_path, centers_path, timestamp, meta_json
                FROM cluster_cache
                ORDER BY timestamp DESC
                LIMIT 1
                """)
            row = c.fetchone()
            if not row:
                return None

            (
                num_images,
                num_embeddings,
                n_clusters,
                method,
                assignments_path,
                centers_path,
                timestamp,
                meta_json,
            ) = row
            if num_images != self.count_images():
                return None

            assign_path = Path(assignments_path)
            if not assign_path.exists():
                return None

            assignments = np.load(assign_path)
            centers = None
            if centers_path:
                centers_path_obj = Path(centers_path)
                if centers_path_obj.exists():
                    centers = np.load(centers_path_obj)

            payload = {
                "assignments": assignments,
                "centers": centers,
                "n_clusters": n_clusters,
                "method": method,
                "timestamp": timestamp,
                "num_embeddings": num_embeddings,
            }
            if meta_json:
                payload.update(json.loads(meta_json))
            return payload

    def save_umap_cache(
        self,
        coords: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist UMAP coordinates for reuse."""
        cache_dir = self.db_path.parent / "umap"
        cache_dir.mkdir(exist_ok=True)

        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        coords_path = cache_dir / f"coords_{timestamp}.npy"
        np.save(coords_path, coords)

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO umap_cache
                (num_images, n_neighbors, min_dist, coords_path, meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.count_images(),
                    int(n_neighbors),
                    float(min_dist),
                    str(coords_path),
                    json.dumps(meta) if meta else None,
                ),
            )
            conn.commit()
            return c.lastrowid

    def get_most_recent_umap_cache(self) -> Optional[Dict[str, Any]]:
        """Load most recent cached UMAP projection if still valid."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT num_images, n_neighbors, min_dist, coords_path, timestamp, meta_json
                FROM umap_cache
                ORDER BY timestamp DESC
                LIMIT 1
                """)
            row = c.fetchone()
            if not row:
                return None

            num_images, n_neighbors, min_dist, coords_path, timestamp, meta_json = row
            if num_images != self.count_images():
                return None

            path = Path(coords_path)
            if not path.exists():
                return None

            coords = np.load(path)
            payload = {
                "coords": coords,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "timestamp": timestamp,
            }
            if meta_json:
                payload.update(json.loads(meta_json))
            return payload

    def save_predictions(
        self,
        paths: List[str],
        predicted_labels: List[str],
        predicted_indices: List[int],
        confidences: List[float],
    ):
        """Persist prediction outputs for images.

        Writes confidence to the dedicated `confidence` column and stores
        prediction metadata inside `meta_json`.
        """
        rows = zip(paths, predicted_labels, predicted_indices, confidences)

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            for path, pred_label, pred_idx, confidence in rows:
                c.execute("SELECT meta_json FROM images WHERE file_path = ?", (path,))
                existing = c.fetchone()
                meta = {}
                if existing and existing[0]:
                    try:
                        meta = json.loads(existing[0])
                    except Exception:
                        meta = {}

                meta["predicted_label"] = pred_label
                meta["predicted_index"] = int(pred_idx)
                meta["prediction_confidence"] = float(confidence)

                c.execute(
                    """
                    UPDATE images
                    SET confidence = ?, meta_json = ?
                    WHERE file_path = ?
                    """,
                    (float(confidence), json.dumps(meta), path),
                )
            conn.commit()

    def close(self):
        pass
