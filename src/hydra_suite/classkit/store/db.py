import csv
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
                    kind TEXT DEFAULT 'embedding',
                    coords_path TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    meta_json TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS candidate_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    num_images INTEGER,
                    candidate_indices_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    meta_json TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    num_images INTEGER,
                    num_classes INTEGER,
                    class_names_json TEXT,
                    active_model_mode TEXT,
                    canonicalize_mat INTEGER DEFAULT 0,
                    model_cache_id INTEGER,
                    probs_path TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    meta_json TEXT
                )
            """)

            # Migrate umap_cache: add 'kind' column if absent (live DB upgrade)
            try:
                c.execute(
                    "ALTER TABLE umap_cache ADD COLUMN kind TEXT DEFAULT 'embedding'"
                )
                conn.commit()
            except Exception:
                pass  # column already exists

            c.execute("""
                CREATE TABLE IF NOT EXISTS model_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    artifact_paths_json TEXT NOT NULL,
                    class_names_json TEXT,
                    canonicalize_mat INTEGER DEFAULT 0,
                    best_val_acc REAL,
                    num_classes INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    meta_json TEXT
                )
            """)
            conn.commit()

    def add_images(self, paths: List[Path], hashes: List[str] = None):
        """Batch insert images, storing resolved absolute paths."""
        if hashes is None:
            hashes = [None] * len(paths)

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Resolve paths to canonical absolute form to prevent duplicates
            # caused by symlinks or relative path representations.
            data = [(str(Path(p).resolve()), h) for p, h in zip(paths, hashes)]
            c.executemany(
                "INSERT OR IGNORE INTO images (file_path, file_hash) VALUES (?, ?)",
                data,
            )
            conn.commit()

    def migrate_paths_to_resolved(self) -> int:
        """One-time migration: ensure all stored file_path values are resolved absolute paths.

        Returns the number of rows updated.
        """
        from pathlib import Path

        updated = 0
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT id, file_path FROM images")
            rows = c.fetchall()
            for row_id, stored_path in rows:
                try:
                    resolved = str(Path(stored_path).resolve())
                except Exception:
                    continue
                if resolved != stored_path:
                    c.execute(
                        "UPDATE images SET file_path = ? WHERE id = ?",
                        (resolved, row_id),
                    )
                    updated += 1
            conn.commit()
        return updated

    def update_labels_batch(self, updates: Dict[str, Optional[str]]) -> int:
        """Batch update labels for multiple images in a single transaction.

        Returns the total number of rows affected.
        Uses an explicit per-row count to work around sqlite3 executemany rowcount
        inconsistencies across Python versions.
        """
        if not updates:
            return 0

        updated_count = 0
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            for path, label in updates.items():
                c.execute(
                    "UPDATE images SET label = ? WHERE file_path = ?", (label, path)
                )
                updated_count += c.rowcount
            conn.commit()
        return updated_count

    def update_labels_with_confidence_batch(
        self, updates: Dict[str, Tuple[str, float]]
    ) -> None:
        """Write label and confidence for multiple images in one transaction.

        Args:
            updates: mapping of {file_path: (label, confidence)}
        """
        if not updates:
            return
        rows = [
            (label, confidence, path) for path, (label, confidence) in updates.items()
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "UPDATE images SET label = ?, confidence = ? WHERE file_path = ?",
                rows,
            )
            conn.commit()

    def clear_all_labels(self) -> None:
        """Set label=NULL and confidence=NULL for all images."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE images SET label = NULL, confidence = NULL")
            conn.commit()

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
        kind: str = "embedding",
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist UMAP coordinates for reuse. kind='embedding' or 'model'."""
        cache_dir = self.db_path.parent / "umap"
        cache_dir.mkdir(exist_ok=True)

        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        coords_path = cache_dir / f"coords_{kind}_{timestamp}.npy"
        np.save(coords_path, coords)

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO umap_cache
                (num_images, n_neighbors, min_dist, kind, coords_path, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.count_images(),
                    int(n_neighbors),
                    float(min_dist),
                    kind,
                    str(coords_path),
                    json.dumps(meta) if meta else None,
                ),
            )
            conn.commit()
            return c.lastrowid

    def get_most_recent_umap_cache(
        self, kind: str = "embedding"
    ) -> Optional[Dict[str, Any]]:
        """Load most recent cached UMAP projection of the given kind if still valid."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT num_images, n_neighbors, min_dist, coords_path, timestamp, meta_json
                FROM umap_cache
                WHERE kind = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (kind,),
            )
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

    def save_candidate_cache(
        self,
        candidate_indices: List[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist current labeling candidate indices for reuse."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO candidate_cache
                (num_images, candidate_indices_json, meta_json)
                VALUES (?, ?, ?)
                """,
                (
                    self.count_images(),
                    json.dumps(candidate_indices),
                    json.dumps(meta) if meta else None,
                ),
            )
            conn.commit()
            return c.lastrowid

    def get_most_recent_candidate_cache(self) -> Optional[Dict[str, Any]]:
        """Load most recent cached candidate set if still valid."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT num_images, candidate_indices_json, timestamp, meta_json
                FROM candidate_cache
                ORDER BY timestamp DESC
                LIMIT 1
                """)
            row = c.fetchone()
            if not row:
                return None

            num_images, indices_json, timestamp, meta_json = row
            if num_images != self.count_images():
                return None

            try:
                indices = json.loads(indices_json)
            except Exception:
                return None

            payload = {
                "candidate_indices": indices,
                "timestamp": timestamp,
            }
            if meta_json:
                payload.update(json.loads(meta_json))
            return payload

    def close(self):
        pass

    # ── Prediction cache ────────────────────────────────────────────────

    def save_prediction_cache(
        self,
        probs: np.ndarray,
        class_names: List[str],
        active_model_mode: str = "",
        model_cache_id: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist per-image probability matrix (N, C) to disk and record metadata."""
        cache_dir = self.db_path.parent / "predictions"
        cache_dir.mkdir(exist_ok=True)
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        probs_path = cache_dir / f"probs_{active_model_mode}_{timestamp}.npy"
        np.save(probs_path, probs)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO prediction_cache
                (num_images, num_classes, class_names_json, active_model_mode,
                 canonicalize_mat, model_cache_id, probs_path, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(probs.shape[0]),
                    int(probs.shape[1]),
                    json.dumps(list(class_names)),
                    str(active_model_mode),
                    0,
                    model_cache_id,
                    str(probs_path),
                    json.dumps(meta) if meta else None,
                ),
            )
            conn.commit()
            return c.lastrowid

    def get_most_recent_prediction_cache(self) -> Optional[Dict[str, Any]]:
        """Load the most recent prediction cache if num_images still matches."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT num_images, num_classes, class_names_json, active_model_mode,
                       canonicalize_mat, model_cache_id, probs_path, timestamp, meta_json
                FROM prediction_cache
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = c.fetchone()
        if not row:
            return None
        (
            num_images,
            num_classes,
            names_json,
            mode,
            canon,
            model_id,
            probs_path,
            timestamp,
            meta_json,
        ) = row
        if num_images != self.count_images():
            return None
        path = Path(probs_path)
        if not path.exists():
            return None
        probs = np.load(path)
        try:
            class_names = json.loads(names_json) if names_json else []
        except Exception:
            class_names = []
        payload = {
            "probs": probs,
            "class_names": class_names,
            "active_model_mode": mode,
            "model_cache_id": model_id,
            "num_classes": num_classes,
            "timestamp": timestamp,
        }
        if meta_json:
            from contextlib import suppress

            with suppress(Exception):
                payload.update(json.loads(meta_json))
        return payload

    # ── Model cache ──────────────────────────────────────────────────────────

    def save_model_cache(
        self,
        mode: str,
        artifact_paths: List[str],
        class_names: List[str],
        best_val_acc: Optional[float] = None,
        num_classes: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a trained model record with its artifact paths and metadata."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO model_cache
                (mode, artifact_paths_json, class_names_json,
                 canonicalize_mat, best_val_acc, num_classes, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(mode),
                    json.dumps([str(p) for p in artifact_paths]),
                    json.dumps(list(class_names)),
                    0,
                    (float(best_val_acc) if best_val_acc is not None else None),
                    int(num_classes),
                    json.dumps(meta) if meta else None,
                ),
            )
            conn.commit()
            return c.lastrowid

    def set_model_cache_display_name(
        self, model_cache_id: int, display_name: str
    ) -> int:
        """Set or clear a user-facing display name in model cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT meta_json FROM model_cache WHERE id = ?", (int(model_cache_id),)
            )
            row = c.fetchone()
            if not row:
                return 0

            meta: Dict[str, Any] = {}
            meta_json = row[0]
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                except Exception:
                    meta = {}

            name_clean = str(display_name).strip()
            if name_clean:
                meta["display_name"] = name_clean
            else:
                meta.pop("display_name", None)

            c.execute(
                "UPDATE model_cache SET meta_json = ? WHERE id = ?",
                (json.dumps(meta) if meta else None, int(model_cache_id)),
            )
            conn.commit()
            return int(c.rowcount)

    def delete_model_cache_entry(self, model_cache_id: int) -> int:
        """Delete one model cache row by id and return affected row count."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM model_cache WHERE id = ?", (int(model_cache_id),))
            conn.commit()
            return int(c.rowcount)

    @staticmethod
    def _accuracy_columns(field_names: List[str]) -> List[str]:
        """Return candidate accuracy columns sorted by preference."""
        cols = [
            c for c in field_names if ("acc" in c.lower()) or ("accuracy" in c.lower())
        ]
        return sorted(
            cols,
            key=lambda c: (
                0 if ("top1" in c.lower() and "metrics" in c.lower()) else 1,
                c.lower(),
            ),
        )

    @staticmethod
    def _max_float_from_rows(
        rows: List[Dict[str, Any]], column: str
    ) -> Optional[float]:
        """Return the maximum parseable float from one CSV column."""
        values: List[float] = []
        for row in rows:
            raw = row.get(column)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                values.append(float(raw))
            except Exception:
                continue
        return max(values) if values else None

    @staticmethod
    def _extract_best_acc_from_results_csv(path: Path) -> Optional[float]:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                cols = ClassKitDB._accuracy_columns(fields)
                if not cols:
                    return None
                rows = list(reader)
                for col in cols:
                    value = ClassKitDB._max_float_from_rows(rows, col)
                    if value is not None:
                        return value
                return None
        except Exception:
            return None

    @staticmethod
    def _infer_acc_from_tiny_metrics(path: Path) -> Optional[float]:
        """Read best validation accuracy from Tiny CNN metrics JSON."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        direct = data.get("best_val_acc")
        if direct is not None:
            return float(direct)
        history = data.get("history")
        if not isinstance(history, list):
            return None
        vals = [
            float(item.get("val_acc"))
            for item in history
            if isinstance(item, dict) and item.get("val_acc") is not None
        ]
        return max(vals) if vals else None

    @classmethod
    def _infer_acc_from_artifact(cls, artifact: Path) -> Optional[float]:
        """Infer validation accuracy from metrics artifacts beside a model file."""
        if not artifact.exists():
            return None
        run_dir = (
            artifact.parent.parent
            if artifact.parent.name == "weights"
            else artifact.parent
        )
        value = cls._infer_acc_from_tiny_metrics(run_dir / "tiny_metrics.json")
        if value is not None:
            return value
        return cls._extract_best_acc_from_results_csv(run_dir / "results.csv")

    @classmethod
    def _infer_best_val_acc_from_artifacts(
        cls, artifact_paths: List[str]
    ) -> Optional[float]:
        inferred_values: List[float] = []
        for src in artifact_paths or []:
            artifact = Path(str(src))
            value = cls._infer_acc_from_artifact(artifact)
            if value is not None:
                inferred_values.append(float(value))

        return max(inferred_values) if inferred_values else None

    @staticmethod
    def _deserialize_model_cache_lists(
        paths_json: str, names_json: Optional[str]
    ) -> tuple[List[str], List[str]]:
        """Deserialize artifact/class-name JSON columns safely."""
        try:
            artifact_paths = json.loads(paths_json)
        except Exception:
            artifact_paths = []
        try:
            class_names = json.loads(names_json) if names_json else []
        except Exception:
            class_names = []
        return artifact_paths, class_names

    def _build_model_cache_entry(self, row) -> Optional[Dict[str, Any]]:
        """Convert one model_cache SQL row into an API entry dict."""
        id_, mode, paths_json, names_json, _canon, acc, n_cls, ts, meta_json = row
        artifact_paths, class_names = self._deserialize_model_cache_lists(
            paths_json, names_json
        )
        best_val_acc = float(acc) if acc is not None else None
        if (best_val_acc is None or best_val_acc <= 0.0) and artifact_paths:
            inferred = self._infer_best_val_acc_from_artifacts(artifact_paths)
            if inferred is not None:
                best_val_acc = float(inferred)

        entry: Dict[str, Any] = {
            "id": id_,
            "mode": mode,
            "artifact_paths": artifact_paths,
            "class_names": class_names,
            "best_val_acc": best_val_acc,
            "num_classes": int(n_cls) if n_cls else 0,
            "timestamp": ts,
        }
        if meta_json:
            try:
                meta = json.loads(meta_json)
                entry["meta"] = meta
                if isinstance(meta, dict) and str(meta.get("display_name", "")).strip():
                    entry["display_name"] = str(meta.get("display_name")).strip()
            except Exception:
                pass
        if artifact_paths and Path(artifact_paths[0]).exists():
            return entry
        return None

    def list_model_caches(self) -> List[Dict[str, Any]]:
        """Return all model cache entries ordered newest-first."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, mode, artifact_paths_json, class_names_json,
                       canonicalize_mat, best_val_acc, num_classes, timestamp, meta_json
                FROM model_cache
                ORDER BY timestamp DESC
                """)
            rows = c.fetchall()

        results = []
        for row in rows:
            entry = self._build_model_cache_entry(row)
            if entry is not None:
                results.append(entry)
        return results

    def get_most_recent_model_cache(self) -> Optional[Dict[str, Any]]:
        """Return the most recent model cache entry whose artifact still exists."""
        entries = self.list_model_caches()
        return entries[0] if entries else None

    # ── Source folder helpers ────────────────────────────────────────────────

    def get_source_folders(self) -> List[Dict[str, Any]]:
        """Derive distinct source folders from stored image paths.

        Returns a list of dicts with keys:
            folder (str): parent directory path
            count  (int): number of images from that folder
        Sorted by folder name.
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT file_path FROM images ORDER BY id")
            rows = c.fetchall()

        folder_counts: Dict[str, int] = {}
        for (fp,) in rows:
            parent = str(Path(fp).parent)
            folder_counts[parent] = folder_counts.get(parent, 0) + 1

        return sorted(
            [{"folder": f, "count": n} for f, n in folder_counts.items()],
            key=lambda d: d["folder"],
        )

    def remove_images_by_folder(self, folder: str) -> int:
        """Delete all images whose file_path lives under *folder*.

        Uses a prefix match on the resolved folder string so that
        ``/data/set1/images/img001.png`` is matched by ``/data/set1/images``.

        Returns the number of rows deleted.
        """
        folder_prefix = str(Path(folder).resolve())
        if not folder_prefix.endswith("/"):
            folder_prefix += "/"

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Match images whose path starts with the folder prefix, or whose
            # parent *is* the folder exactly.
            c.execute(
                "DELETE FROM images WHERE file_path LIKE ? OR "
                "REPLACE(file_path, RTRIM(file_path, REPLACE(file_path, '/', '')), '') = ?",
                (folder_prefix + "%", folder_prefix.rstrip("/")),
            )
            # Simpler: just match on prefix
            deleted = c.rowcount
            conn.commit()
        return deleted

    def remove_images_by_folder_exact(self, folder: str) -> int:
        """Delete images whose parent directory is exactly *folder*.

        More precise than prefix matching — only removes images that live
        directly inside the given directory.

        Returns the number of rows deleted.
        """
        folder_resolved = str(Path(folder).resolve())

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Fetch ids to delete (SQLite doesn't have dirname())
            c.execute("SELECT id, file_path FROM images")
            ids_to_delete = []
            for row_id, fp in c.fetchall():
                if str(Path(fp).parent) == folder_resolved:
                    ids_to_delete.append(row_id)

            if ids_to_delete:
                placeholders = ",".join("?" for _ in ids_to_delete)
                c.execute(
                    f"DELETE FROM images WHERE id IN ({placeholders})",  # noqa: S608
                    ids_to_delete,
                )
            conn.commit()
        return len(ids_to_delete)
