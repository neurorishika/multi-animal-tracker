#!/usr/bin/env python3
"""
Extensions for PoseKit Labeler:
- Crash-safe recovery with temp files
- Versioned label backups
- Per-frame metadata/tags
- Cluster-stratified dataset splitting
- Embedding/clustering utilities
- YOLO pose I/O
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
import logging

import cv2
import numpy as np

logger = logging.getLogger("pose_label.extensions")


# -----------------------------
# Frame Metadata
# -----------------------------
@dataclass
class FrameMetadata:
    """Metadata for a single frame."""

    image_path: str
    tags: Set[str] = field(
        default_factory=set
    )  # e.g., "occluded", "weird_posture", "motion_blur"
    notes: str = ""
    cluster_id: Optional[int] = None

    def to_json(self) -> dict:
        return {
            "image_path": self.image_path,
            "tags": list(sorted(self.tags)),
            "notes": self.notes,
            "cluster_id": self.cluster_id,
        }

    @staticmethod
    def from_json(data: dict) -> "FrameMetadata":
        return FrameMetadata(
            image_path=data["image_path"],
            tags=set(data.get("tags", [])),
            notes=data.get("notes", ""),
            cluster_id=data.get("cluster_id"),
        )


class MetadataManager:
    """Manages per-frame metadata."""

    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.metadata: Dict[str, FrameMetadata] = {}
        self.load()

    def load(self):
        """Load metadata from disk."""
        if not self.metadata_path.exists():
            return
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.metadata = {
                img_path: FrameMetadata.from_json(meta)
                for img_path, meta in data.items()
            }
            logger.info(f"Loaded metadata for {len(self.metadata)} frames")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def save(self):
        """Save metadata to disk."""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                img_path: meta.to_json() for img_path, meta in self.metadata.items()
            }
            self.metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"Saved metadata for {len(self.metadata)} frames")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_metadata(self, image_path: str) -> FrameMetadata:
        """Get metadata for a frame, creating if needed."""
        if image_path not in self.metadata:
            self.metadata[image_path] = FrameMetadata(image_path=image_path)
        return self.metadata[image_path]

    def add_tag(self, image_path: str, tag: str):
        """Add a tag to a frame."""
        meta = self.get_metadata(image_path)
        meta.tags.add(tag)
        self.save()

    def remove_tag(self, image_path: str, tag: str):
        """Remove a tag from a frame."""
        meta = self.get_metadata(image_path)
        meta.tags.discard(tag)
        self.save()

    def set_notes(self, image_path: str, notes: str):
        """Set notes for a frame."""
        meta = self.get_metadata(image_path)
        meta.notes = notes
        self.save()

    def set_cluster_id(self, image_path: str, cluster_id: Optional[int]):
        """Set cluster ID for a frame."""
        meta = self.get_metadata(image_path)
        meta.cluster_id = cluster_id
        self.save()

    def get_frames_by_tag(self, tag: str) -> List[str]:
        """Get all frames with a specific tag."""
        return [
            img_path for img_path, meta in self.metadata.items() if tag in meta.tags
        ]


# -----------------------------
# Crash-safe recovery
# -----------------------------
class CrashSafeWriter:
    """Writes labels atomically using temp files."""

    @staticmethod
    def write_label(label_path: Path, content: str):
        """Write label file atomically using temp file + rename."""
        try:
            # Write to temp file in same directory (ensures same filesystem)
            temp_dir = label_path.parent
            temp_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=temp_dir,
                delete=False,
                prefix=f".{label_path.stem}_",
                suffix=".tmp",
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            # Atomic rename
            tmp_path.replace(label_path)
            logger.debug(f"Wrote label atomically: {label_path.name}")

        except Exception as e:
            logger.error(f"Failed to write label {label_path}: {e}")
            raise


# -----------------------------
# Versioned label backups
# -----------------------------
class LabelVersioning:
    """Manages versioned backups of labels."""

    def __init__(self, labels_dir: Path, max_versions: int = 5):
        self.labels_dir = labels_dir
        self.backup_dir = labels_dir.parent / "labels_history"
        self.max_versions = max_versions
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_label(self, label_path: Path):
        """Create a versioned backup of a label file."""
        if not label_path.exists():
            return

        try:
            stem = label_path.stem
            backup_base = self.backup_dir / stem

            # Find existing backups
            existing = sorted(self.backup_dir.glob(f"{stem}.v*.txt"))

            # Rotate old backups
            if len(existing) >= self.max_versions:
                # Remove oldest
                for old in existing[: len(existing) - self.max_versions + 1]:
                    old.unlink()
                existing = sorted(self.backup_dir.glob(f"{stem}.v*.txt"))

            # Determine next version number
            if existing:
                last_version = int(existing[-1].stem.split(".v")[1])
                next_version = last_version + 1
            else:
                next_version = 1

            # Create backup
            backup_path = self.backup_dir / f"{stem}.v{next_version:03d}.txt"
            shutil.copy2(label_path, backup_path)
            logger.debug(f"Created backup: {backup_path.name}")

        except Exception as e:
            logger.error(f"Failed to backup {label_path}: {e}")

    def restore_label(self, label_path: Path, version: Optional[int] = None):
        """Restore a label from backup."""
        stem = label_path.stem
        existing = sorted(self.backup_dir.glob(f"{stem}.v*.txt"))

        if not existing:
            logger.warning(f"No backups found for {stem}")
            return False

        if version is None:
            # Restore latest
            backup_path = existing[-1]
        else:
            backup_path = self.backup_dir / f"{stem}.v{version:03d}.txt"
            if not backup_path.exists():
                logger.warning(f"Backup version {version} not found")
                return False

        try:
            shutil.copy2(backup_path, label_path)
            logger.info(f"Restored {label_path.name} from {backup_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore: {e}")
            return False

    def list_versions(self, label_stem: str) -> List[Tuple[int, Path]]:
        """List all backup versions for a label."""
        existing = sorted(self.backup_dir.glob(f"{label_stem}.v*.txt"))
        versions = []
        for backup in existing:
            try:
                version = int(backup.stem.split(".v")[1])
                versions.append((version, backup))
            except (IndexError, ValueError):
                continue
        return versions


# -----------------------------
# Cluster-stratified splitting
# -----------------------------
def cluster_stratified_split(
    image_paths: List[Path],
    cluster_ids: List[int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_per_cluster: int = 1,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split frames into train/val/test preserving cluster distribution.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Normalize fractions
    total = train_frac + val_frac + test_frac
    train_frac /= total
    val_frac /= total
    test_frac /= total

    # Group by cluster
    clusters: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for cid, indices in clusters.items():
        n = len(indices)

        # Shuffle
        indices = list(indices)
        rng.shuffle(indices)

        if n < 3:
            # Too small to split, put all in train
            train_indices.extend(indices)
            continue

        # Calculate splits
        n_train = max(min_per_cluster, int(n * train_frac))
        n_val = max(min_per_cluster if val_frac > 0 else 0, int(n * val_frac))
        n_test = n - n_train - n_val

        if n_test < min_per_cluster and test_frac > 0:
            # Adjust to maintain minimum
            if n_train > min_per_cluster:
                n_train -= 1
                n_test += 1

        # Split
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    return train_indices, val_indices, test_indices


def cluster_kfold_split(
    image_paths: List[Path],
    cluster_ids: List[int],
    n_folds: int = 5,
    seed: int = 0,
) -> List[Tuple[List[int], List[int]]]:
    """
    Create K-fold splits preserving cluster distribution.

    Returns:
        List of (train_indices, val_indices) for each fold
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Group by cluster
    clusters: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(idx)

    # Assign each cluster's frames to folds
    fold_assignments = [[] for _ in range(n_folds)]

    for cid, indices in clusters.items():
        indices = list(indices)
        rng.shuffle(indices)

        # Distribute cluster frames across folds
        for i, idx in enumerate(indices):
            fold_assignments[i % n_folds].append(idx)

    # Create train/val splits
    splits = []
    for val_fold in range(n_folds):
        val_indices = fold_assignments[val_fold]
        train_indices = []
        for i in range(n_folds):
            if i != val_fold:
                train_indices.extend(fold_assignments[i])
        splits.append((train_indices, val_indices))

    return splits


def save_split_files(
    output_dir: Path,
    image_paths: List[Path],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    split_name: str = "split",
):
    """Save train/val/test splits to text files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_split(filename: str, indices: List[int]):
        path = output_dir / filename
        lines = [str(image_paths[i]) + "\n" for i in sorted(indices)]
        path.write_text("".join(lines), encoding="utf-8")
        logger.info(f"Wrote {len(indices)} images to {filename}")

    write_split(f"{split_name}_train.txt", train_indices)
    write_split(f"{split_name}_val.txt", val_indices)
    write_split(f"{split_name}_test.txt", test_indices)

    # Also save summary
    summary_path = output_dir / f"{split_name}_summary.json"
    summary = {
        "split_name": split_name,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "test_count": len(test_indices),
        "total_count": len(train_indices) + len(val_indices) + len(test_indices),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote summary to {summary_path}")


# -----------------------------
# Keypoint Dataclass
# -----------------------------
@dataclass
class Keypoint:
    """Represents a single keypoint with pixel coordinates and visibility."""

    x: float = 0.0
    y: float = 0.0
    v: int = 0  # 0=missing, 1=occluded, 2=visible


# -----------------------------
# Embedding/Clustering Utilities
# -----------------------------


def _stable_hash_dict(d: dict) -> str:
    """Create stable hash from dictionary for cache keys."""
    s = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _choose_device(pref: str = "auto") -> str:
    """Choose compute device for embeddings."""
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _read_image_pil(path: Path):
    """Read image as PIL RGB."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def _maybe_downscale_pil(img, max_side: int):
    """Downscale PIL image if larger than max_side."""
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), resample=2)  # BILINEAR


def _enhance_pil_for_pose(img_pil):
    """Apply CLAHE and unsharp mask enhancement to PIL image."""
    try:
        from .pose_label import enhance_for_pose
    except ImportError:
        from pose_label import enhance_for_pose

    # enhance_for_pose expects BGR np array
    arr = np.array(img_pil)  # RGB
    bgr = arr[:, :, ::-1].copy()
    bgr2 = enhance_for_pose(bgr)
    rgb2 = bgr2[:, :, ::-1].copy()
    from PIL import Image

    return Image.fromarray(rgb2)


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between L2-normalized embeddings."""
    # assumes A, B are L2-normalized
    return A @ B.T


def _farthest_point_centers(emb: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Select k centers using farthest-point sampling on embeddings."""
    # emb: (n,d) L2-normalized
    n = emb.shape[0]
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    centers = [first]
    # maintain best similarity to any chosen center (higher is closer)
    best_sim = _cosine_sim_matrix(emb, emb[[first]]).reshape(-1)
    for _ in range(1, min(k, n)):
        # pick point with lowest similarity to nearest center
        idx = int(np.argmin(best_sim))
        centers.append(idx)
        new_sim = _cosine_sim_matrix(emb, emb[[idx]]).reshape(-1)
        best_sim = np.maximum(best_sim, new_sim)
    return centers


def cluster_embeddings_cosine(
    emb: np.ndarray,
    k: int,
    method: str = "hierarchical",
    seed: int = 0,
    hierarchical_limit: int = 2500,
) -> np.ndarray:
    """
    Cluster embeddings using cosine similarity.
    Returns cluster labels in [0..k-1].
    """
    n = emb.shape[0]
    k = max(1, min(int(k), n))

    if method == "hierarchical" and n <= hierarchical_limit:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # pdist with cosine => 1 - cosine_similarity
        dist = pdist(emb, metric="cosine")
        Z = linkage(dist, method="ward")
        labels = fcluster(Z, t=k, criterion="maxclust")
        return (labels - 1).astype(np.int32)

    # fallback: farthest-point centers + nearest-center assignment
    centers = _farthest_point_centers(emb, k=k, seed=seed)
    C = emb[centers]
    sims = _cosine_sim_matrix(emb, C)  # (n,k)
    return np.argmax(sims, axis=1).astype(np.int32)


def pick_frames_stratified(
    emb: np.ndarray,
    cluster_id: np.ndarray,
    want_n: int,
    eligible_indices: List[int],
    min_per_cluster: int = 1,
    seed: int = 0,
    strategy: str = "centroid_then_diverse",  # or "centroid"
) -> List[int]:
    """
    Select frames stratified by cluster.
    emb: embeddings for eligible_indices (same order)
    cluster_id: cluster labels for eligible_indices
    returns: selected *global* indices (from eligible_indices mapping)
    """
    rng = np.random.default_rng(seed)
    want_n = max(0, min(int(want_n), len(eligible_indices)))
    if want_n == 0:
        return []

    # cluster -> local positions
    clusters: Dict[int, List[int]] = {}
    for pos, cid in enumerate(cluster_id):
        clusters.setdefault(int(cid), []).append(pos)

    # order clusters by size (largest first)
    cluster_keys = sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True)

    # If more clusters than want_n, keep only top want_n clusters (still diverse-ish)
    if len(cluster_keys) > want_n:
        cluster_keys = cluster_keys[:want_n]

    # initial quotas
    quotas = {c: 0 for c in cluster_keys}
    remaining = want_n

    # give each cluster min_per_cluster (as possible)
    for c in cluster_keys:
        give = min(min_per_cluster, len(clusters[c]), remaining)
        quotas[c] += give
        remaining -= give
        if remaining <= 0:
            break

    if remaining > 0:
        # proportional allocation
        total_sz = sum(len(clusters[c]) for c in cluster_keys)
        for c in cluster_keys:
            frac = len(clusters[c]) / max(total_sz, 1)
            extra = int(round(frac * remaining))
            # don't exceed cluster size
            cap = len(clusters[c]) - quotas[c]
            extra = min(extra, cap, remaining)
            quotas[c] += extra
            remaining -= extra
            if remaining <= 0:
                break

        # leftover
        while remaining > 0:
            for c in cluster_keys:
                if quotas[c] < len(clusters[c]):
                    quotas[c] += 1
                    remaining -= 1
                    if remaining <= 0:
                        break

    selected_local: List[int] = []

    for c in cluster_keys:
        positions = clusters[c]
        q = quotas[c]
        if q <= 0:
            continue

        if strategy == "centroid":
            # pick nearest to cluster centroid
            pos_arr = np.array(positions)
            sub_emb = emb[pos_arr]
            centroid = sub_emb.mean(axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            sims = _cosine_sim_matrix(sub_emb, centroid).reshape(-1)
            top = np.argsort(-sims)[:q]
            selected_local.extend(pos_arr[top].tolist())

        else:  # "centroid_then_diverse"
            pos_arr = np.array(positions)
            sub_emb = emb[pos_arr]
            centroid = sub_emb.mean(axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            if q == 1:
                sims = _cosine_sim_matrix(sub_emb, centroid).reshape(-1)
                best = int(np.argmax(sims))
                selected_local.append(int(pos_arr[best]))
            else:
                # pick centroid first
                sims = _cosine_sim_matrix(sub_emb, centroid).reshape(-1)
                best = int(np.argmax(sims))
                picked_local = [best]
                selected_local.append(int(pos_arr[best]))

                # then diverse
                best_sim_to_any = sims.copy()
                for _ in range(1, q):
                    cand = [i for i in range(len(pos_arr)) if i not in picked_local]
                    if not cand:
                        break
                    cand_sims = best_sim_to_any[cand]
                    # pick the one with lowest similarity to any selected
                    rel_idx = int(np.argmin(cand_sims))
                    chosen = cand[rel_idx]
                    picked_local.append(chosen)
                    selected_local.append(int(pos_arr[chosen]))
                    # update best_sim_to_any
                    new_sim = _cosine_sim_matrix(sub_emb, sub_emb[[chosen]]).reshape(-1)
                    best_sim_to_any = np.maximum(best_sim_to_any, new_sim)

    # map local positions back to global frame indices
    out = [eligible_indices[pos] for pos in selected_local]
    # ensure exact want_n (rare over/under due to edge cases)
    out = out[:want_n]
    return out


# -----------------------------
# YOLO Pose I/O
# -----------------------------


def load_yolo_pose_label(
    label_path: Path, k: int
) -> Optional[Tuple[int, List[Keypoint], Optional[Tuple[float, float, float, float]]]]:
    """Load YOLO pose label from .txt file."""
    if not label_path.exists():
        return None
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    line = txt.splitlines()[0].strip()
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, bw, bh = map(float, parts[1:5])
    rest = parts[5:]

    kpts: List[Keypoint] = []
    if len(rest) == 2 * k:
        # old format: x y
        for i in range(k):
            kpts.append(Keypoint(float(rest[2 * i]), float(rest[2 * i + 1]), 2))
    elif len(rest) == 3 * k:
        # new format: x y v
        for i in range(k):
            kpts.append(
                Keypoint(
                    float(rest[3 * i]), float(rest[3 * i + 1]), int(rest[3 * i + 2])
                )
            )
    else:
        # fallback: fill missing
        for _ in range(k):
            kpts.append(Keypoint(0.0, 0.0, 0))
        # parse what we can
        for i in range(min(k, len(rest) // 3)):
            kpts[i] = Keypoint(
                float(rest[3 * i]), float(rest[3 * i + 1]), int(rest[3 * i + 2])
            )

    return (cls, kpts, (cx, cy, bw, bh))


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, x))


def _xyxy_to_cxcywh(x1, y1, x2, y2):
    """Convert bbox from xyxy to cxcywh format."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return (cx, cy, w, h)


def save_yolo_pose_label(
    label_path: Path,
    cls: int,
    img_w: int,
    img_h: int,
    kpts_px: List[Keypoint],
    bbox_xyxy_px: Optional[Tuple[float, float, float, float]],
    pad_frac: float,
    create_backup: bool = True,
) -> None:
    """Save YOLO pose label with crash-safe writing and optional backup."""
    # Create backup of existing label before overwriting
    if create_backup and label_path.exists():
        try:
            LabelVersioning.backup(label_path)
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    if bbox_xyxy_px is None:
        # compute from keypoints
        try:
            from .pose_label import compute_bbox_from_kpts
        except ImportError:
            from pose_label import compute_bbox_from_kpts

        bbox_xyxy_px = compute_bbox_from_kpts(kpts_px, pad_frac, img_w, img_h)

    if bbox_xyxy_px is None:
        logger.warning(f"Cannot save label for {label_path}: no valid bbox")
        return

    x1, y1, x2, y2 = bbox_xyxy_px
    cx, cy, bw, bh = _xyxy_to_cxcywh(x1, y1, x2, y2)

    cxn = _clamp01(cx / img_w)
    cyn = _clamp01(cy / img_h)
    bwn = _clamp01(bw / img_w)
    bhn = _clamp01(bh / img_h)

    vals = [str(int(cls)), f"{cxn:.6f}", f"{cyn:.6f}", f"{bwn:.6f}", f"{bhn:.6f}"]
    for kp in kpts_px:
        xn = _clamp01(kp.x / img_w)
        yn = _clamp01(kp.y / img_h)
        vals.append(f"{xn:.6f}")
        vals.append(f"{yn:.6f}")
        vals.append(str(int(kp.v)))

    # Use crash-safe atomic write
    content = " ".join(vals) + "\n"
    CrashSafeWriter.write_label(label_path, content)


def migrate_labels_keypoints(
    labels_dir: Path,
    old_kpt_names: List[str],
    new_kpt_names: List[str],
    mode: str = "name",  # "name" or "index"
) -> Tuple[int, int]:
    """
    Update all .txt labels in labels_dir to match new keypoint list length/order.

    mode="name": map by unique keypoint names, fallback to 0 if missing
    mode="index": preserve positions by old index (truncate/extend)
    Returns: (files_modified, files_total)
    """
    txts = sorted(labels_dir.glob("*.txt"))
    if not txts:
        return (0, 0)

    old_k = len(old_kpt_names)
    new_k = len(new_kpt_names)

    # build mapping new_index -> old_index
    mapping: Dict[int, Optional[int]] = {}
    if mode == "index":
        for ni in range(new_k):
            mapping[ni] = ni if ni < old_k else None
    else:
        # by name
        old_map = {name: i for i, name in enumerate(old_kpt_names)}
        for ni, name in enumerate(new_kpt_names):
            mapping[ni] = old_map.get(name)

    files_modified = 0
    for lp in txts:
        try:
            txt = lp.read_text(encoding="utf-8").strip()
            if not txt:
                continue
            line = txt.splitlines()[0].strip()
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            rest = parts[5:]

            old_kpts: List[Keypoint] = []
            if len(rest) == 2 * old_k:
                for i in range(old_k):
                    old_kpts.append(
                        Keypoint(float(rest[2 * i]), float(rest[2 * i + 1]), 2)
                    )
            elif len(rest) == 3 * old_k:
                for i in range(old_k):
                    old_kpts.append(
                        Keypoint(
                            float(rest[3 * i]),
                            float(rest[3 * i + 1]),
                            int(rest[3 * i + 2]),
                        )
                    )
            else:
                # skip malformed
                continue

            # build new
            new_kpts: List[Keypoint] = []
            for ni in range(new_k):
                oi = mapping[ni]
                if oi is not None and 0 <= oi < len(old_kpts):
                    new_kpts.append(old_kpts[oi])
                else:
                    new_kpts.append(Keypoint(0.0, 0.0, 0))

            # write back
            vals = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for kp in new_kpts:
                vals.append(f"{kp.x:.6f}")
                vals.append(f"{kp.y:.6f}")
                vals.append(str(int(kp.v)))
            content = " ".join(vals) + "\n"
            CrashSafeWriter.write_label(lp, content)
            files_modified += 1

        except Exception as e:
            logger.warning(f"Failed to migrate {lp}: {e}")

    return (files_modified, len(txts))


# -----------------------------
# Embedding Worker Thread
# -----------------------------

from PySide6.QtCore import QObject, Signal, Slot


class EmbeddingWorker(QObject):
    """Worker thread for computing embeddings."""

    progress = Signal(int, int)  # done, total
    status = Signal(str)
    finished = Signal(
        object, object, object
    )  # embeddings(np.ndarray), eligible_indices(list[int]), meta(dict)
    failed = Signal(str)

    def __init__(
        self,
        image_paths: List[Path],
        eligible_indices: List[int],
        cache_dir: Path,
        model_name: str,
        device_pref: str,
        batch_size: int,
        use_enhance: bool,
        max_side: int,
        cache_ok: bool = True,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.eligible_indices = eligible_indices
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.device_pref = device_pref
        self.batch_size = batch_size
        self.use_enhance = use_enhance
        self.max_side = max_side
        self.cache_ok = cache_ok
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "model_name": self.model_name,
                "device_pref": self.device_pref,
                "batch_size": int(self.batch_size),
                "use_enhance": bool(self.use_enhance),
                "max_side": int(self.max_side),
                # include eligible image signatures to validate cache
                "images": [
                    {
                        "p": str(self.image_paths[i]),
                        "mt": int(self.image_paths[i].stat().st_mtime),
                        "sz": int(self.image_paths[i].stat().st_size),
                    }
                    for i in self.eligible_indices
                ],
            }
            key = _stable_hash_dict(meta)
            npy_path = self.cache_dir / f"{key}.npy"
            json_path = self.cache_dir / f"{key}.json"

            if self.cache_ok and npy_path.exists() and json_path.exists():
                self.status.emit("Loading cached embeddings…")
                emb = np.load(npy_path)
                self.finished.emit(emb, self.eligible_indices, meta)
                return

            self.status.emit("Loading model…")
            device = _choose_device(self.device_pref)

            import torch
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform

            model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            model.eval().to(device)
            data_config = resolve_data_config({}, model=model)
            transform = create_transform(**data_config)

            # embed
            feats = []
            total = len(self.eligible_indices)
            done = 0

            for start in range(0, total, self.batch_size):
                if self._cancel:
                    self.failed.emit("Canceled.")
                    return

                batch_idx = self.eligible_indices[start : start + self.batch_size]
                imgs = []
                for i in batch_idx:
                    img = _read_image_pil(self.image_paths[i])
                    img = _maybe_downscale_pil(img, self.max_side)
                    if self.use_enhance:
                        img = _enhance_pil_for_pose(img)
                    imgs.append(transform(img))

                batch = torch.stack(imgs).to(device)

                with torch.no_grad():
                    out = model(batch)
                    out = out.detach().float().cpu().numpy()

                # L2 normalize
                out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
                feats.append(out)

                done += len(batch_idx)
                self.progress.emit(done, total)
                self.status.emit(f"Embedding {done}/{total}")

            emb = np.vstack(feats).astype(np.float32)

            self.status.emit("Saving cache…")
            np.save(npy_path, emb)
            json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self.finished.emit(emb, self.eligible_indices, meta)

        except Exception as e:
            self.failed.emit(str(e))


# -----------------------------
# Incremental Embedding Cache
# -----------------------------


class IncrementalEmbeddingCache:
    """Manages incremental embedding computation and caching."""

    def __init__(self, cache_root: Path, model_name: str):
        self.cache_root = cache_root
        self.model_name = model_name
        self.cache_dir = (
            cache_root / "embeddings" / self._sanitize_model_name(model_name)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.index_path = self.cache_dir / "index.csv"
        self.meta_path = self.cache_dir / "metadata.json"

        self._embeddings: Optional[np.ndarray] = None
        self._index: Dict[str, int] = {}  # filename -> row index
        self._load()

    @staticmethod
    def _sanitize_model_name(name: str) -> str:
        """Convert model name to safe directory name."""
        import re

        return re.sub(r"[^\w\-.]", "_", name)

    def _load(self):
        """Load existing embeddings and index."""
        if self.embeddings_path.exists() and self.index_path.exists():
            try:
                self._embeddings = np.load(self.embeddings_path)

                # Load index CSV
                with self.index_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._index[row["path"]] = int(row["index"])

                logger.info(
                    f"Loaded {len(self._index)} cached embeddings for {self.model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._embeddings = None
                self._index = {}

    def _save(self):
        """Save embeddings and index to disk."""
        if self._embeddings is None:
            return

        try:
            np.save(self.embeddings_path, self._embeddings)

            # Save index CSV
            with self.index_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "index", "mtime", "size"])
                for path, idx in sorted(self._index.items(), key=lambda x: x[1]):
                    p = Path(path)
                    if p.exists():
                        writer.writerow(
                            [path, idx, int(p.stat().st_mtime), int(p.stat().st_size)]
                        )
                    else:
                        writer.writerow([path, idx, 0, 0])

            logger.info(f"Saved {len(self._index)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_metadata(self) -> Dict:
        """Get cache metadata."""
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def set_metadata(self, meta: Dict):
        """Save cache metadata."""
        try:
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def needs_update(self, image_paths: List[Path]) -> Tuple[List[int], List[int]]:
        """
        Check which images need embedding computation.
        Returns: (needs_compute_indices, cached_indices)
        """
        needs_compute = []
        cached = []

        for i, path in enumerate(image_paths):
            path_str = str(path)

            # Check if in cache
            if path_str in self._index:
                # Verify file hasn't changed
                try:
                    stat = path.stat()
                    # Simple check - could be enhanced with hash
                    cached.append(i)
                except Exception:
                    needs_compute.append(i)
            else:
                needs_compute.append(i)

        return needs_compute, cached

    def get_embeddings(self, image_paths: List[Path]) -> Optional[np.ndarray]:
        """Get embeddings for given paths (if all are cached)."""
        if self._embeddings is None:
            return None

        indices = []
        for path in image_paths:
            path_str = str(path)
            if path_str not in self._index:
                return None
            indices.append(self._index[path_str])

        return self._embeddings[indices]

    def add_embeddings(self, image_paths: List[Path], embeddings: np.ndarray):
        """Add new embeddings to cache."""
        if len(image_paths) != embeddings.shape[0]:
            raise ValueError("Mismatch between paths and embeddings")

        # Initialize if empty
        if self._embeddings is None:
            self._embeddings = embeddings
            for i, path in enumerate(image_paths):
                self._index[str(path)] = i
        else:
            # Append new embeddings
            start_idx = len(self._embeddings)
            self._embeddings = np.vstack([self._embeddings, embeddings])
            for i, path in enumerate(image_paths):
                self._index[str(path)] = start_idx + i

        self._save()

    def clear(self):
        """Clear all cached embeddings."""
        self._embeddings = None
        self._index = {}
        for p in [self.embeddings_path, self.index_path, self.meta_path]:
            if p.exists():
                p.unlink()


# -----------------------------
# Difficulty Score Computation
# -----------------------------


def compute_difficulty_scores(
    image_paths: List[Path], keypoint_data: Optional[List[Optional[List]]] = None
) -> np.ndarray:
    """
    Compute difficulty scores for frames based on image quality and labeling.
    Returns array of scores in [0, 1] where higher = more difficult.
    """
    scores = np.zeros(len(image_paths))

    for i, path in enumerate(image_paths):
        try:
            # Read image
            img = cv2.imread(str(path))
            if img is None:
                scores[i] = 0.5
                continue

            # 1. Blur detection (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = 1.0 / (
                1.0 + laplacian_var / 100.0
            )  # Lower variance = more blur

            # 2. Contrast (std of pixel values)
            contrast = gray.std()
            contrast_score = 1.0 / (1.0 + contrast / 50.0)  # Lower std = low contrast

            # 3. Brightness extremes
            mean_brightness = gray.mean()
            brightness_score = (
                abs(mean_brightness - 128) / 128.0
            )  # Distance from middle gray

            # 4. Occlusion rate (if keypoint data available)
            occlusion_score = 0.0
            if keypoint_data and i < len(keypoint_data) and keypoint_data[i]:
                kpts = keypoint_data[i]
                if kpts:
                    occluded = sum(1 for kp in kpts if kp.v == 1)
                    visible = sum(1 for kp in kpts if kp.v == 2)
                    total = occluded + visible
                    if total > 0:
                        occlusion_score = occluded / total

            # Combine scores
            scores[i] = (
                blur_score * 0.3
                + contrast_score * 0.3
                + brightness_score * 0.2
                + occlusion_score * 0.2
            )

        except Exception as e:
            logger.warning(f"Failed to compute difficulty for {path}: {e}")
            scores[i] = 0.5

    return scores


# -----------------------------
# UMAP Projection
# -----------------------------


def compute_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute UMAP 2D projection of embeddings.
    Returns (n, 2) array of 2D coordinates.
    """
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

        projection = reducer.fit_transform(embeddings)
        return projection.astype(np.float32)

    except ImportError:
        logger.error("UMAP not installed. Install with: pip install umap-learn")
        raise
    except Exception as e:
        logger.error(f"UMAP projection failed: {e}")
        raise


# -----------------------------
# Utility: Filter Near-Duplicates
# -----------------------------


def filter_near_duplicates(
    embeddings: np.ndarray, indices: List[int], threshold: float = 0.95
) -> List[int]:
    """
    Remove near-duplicate frames based on cosine similarity.
    Returns filtered indices.

    Args:
        embeddings: All embeddings (N x D)
        indices: List of frame indices to filter
        threshold: Cosine similarity threshold (0-1). Higher = more strict filtering.

    Returns:
        Filtered list of indices with near-duplicates removed
    """
    if len(indices) <= 1:
        return indices

    sub_emb = embeddings[indices]

    # Compute pairwise similarities
    sims = _cosine_sim_matrix(sub_emb, sub_emb)

    # Greedy filtering: keep first frame, then only add frames dissimilar to kept ones
    keep = []
    for i in range(len(indices)):
        if not keep:
            keep.append(i)
            continue

        # Check similarity to all kept frames
        max_sim = max(sims[i, j] for j in keep)
        if max_sim < threshold:
            keep.append(i)

    return [indices[i] for i in keep]
