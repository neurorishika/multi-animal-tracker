"""
Core logic for Data Sieve tool.
Includes perceptual hashing, duplicate removal, and diversity sampling.
"""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class DataSieveCore:
    def available_dedup_methods(self) -> List[str]:
        return ["phash", "dhash", "ahash", "histogram"]

    def compute_dhash(self, image: np.ndarray, hash_size: int = 8) -> int:
        """
        Compute dHash (difference hash) for an image.

        Args:
            image: BGR or grayscale image.
            hash_size: Size of hash (default 8).

        Returns:
            64-bit integer hash.
        """
        if image is None:
            return 0

        # Resize to (width, height) = (hash_size + 1, hash_size)
        resized = cv2.resize(image, (hash_size + 1, hash_size))

        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # Compare adjacent pixels
        diff = gray[:, 1:] > gray[:, :-1]

        # Convert to integer
        return sum([2**i for i, v in enumerate(diff.flatten()) if v])

    def compute_ahash(self, image: np.ndarray, hash_size: int = 8) -> int:
        """Compute average hash (aHash) as 64-bit integer."""
        if image is None:
            return 0

        resized = cv2.resize(image, (hash_size, hash_size))
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        avg_val = gray.mean()
        bits = gray > avg_val
        return sum([2**i for i, v in enumerate(bits.flatten()) if v])

    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> int:
        """Compute perceptual hash (pHash) using DCT as 64-bit integer."""
        if image is None:
            return 0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        resized = cv2.resize(gray, (32, 32)).astype(np.float32)
        dct = cv2.dct(resized)
        dct_low_freq = dct[:hash_size, :hash_size]

        flat = dct_low_freq.flatten()
        median_val = np.median(flat[1:]) if len(flat) > 1 else np.median(flat)
        bits = dct_low_freq > median_val
        return sum([2**i for i, v in enumerate(bits.flatten()) if v])

    def compute_hist_signature(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Compute normalized grayscale histogram signature for similarity checks."""
        if image is None:
            return np.zeros((bins,), dtype=np.float32)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def compute_color_signature(self, image: np.ndarray) -> np.ndarray:
        """Compute normalized HSV color histogram signature for color diversity checks."""
        if image is None:
            return np.zeros((8 * 8 * 8,), dtype=np.float32)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def color_distance(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Bhattacharyya distance over color histograms (0=similar, 1=different)."""
        return float(cv2.compareHist(sig1, sig2, cv2.HISTCMP_BHATTACHARYYA))

    def compute_signature(self, image: np.ndarray, method: str = "phash"):
        """Compute method-specific signature for deduplication."""
        method = (method or "phash").lower()
        if method == "phash":
            return self.compute_phash(image)
        if method == "dhash":
            return self.compute_dhash(image)
        if method == "ahash":
            return self.compute_ahash(image)
        if method == "histogram":
            return self.compute_hist_signature(image)
        return self.compute_phash(image)

    def signature_distance(self, sig1, sig2, method: str = "phash") -> float:
        """Method-specific distance (lower means more similar)."""
        method = (method or "phash").lower()
        if method in {"phash", "dhash", "ahash"}:
            return float(self.hamming_distance(int(sig1), int(sig2)))
        if method == "histogram":
            # Bhattacharyya distance: 0=identical, 1=different
            return float(cv2.compareHist(sig1, sig2, cv2.HISTCMP_BHATTACHARYYA))
        return float(self.hamming_distance(int(sig1), int(sig2)))

    def is_duplicate(self, sig1, sig2, threshold: float, method: str = "phash") -> bool:
        """Return True when signatures are considered duplicates for method+threshold."""
        distance = self.signature_distance(sig1, sig2, method)
        return distance <= threshold

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two 64-bit hashes."""
        x = (hash1 ^ hash2) & ((1 << 64) - 1)
        dist = 0
        while x:
            dist += 1
            x &= x - 1
        return dist

    def load_dataset(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Load images and extract metadata from filenames.
        Supports format: did{detection_id}.png
        """
        dataset = []
        folder = Path(folder_path)
        if not folder.exists():
            return []

        pattern = re.compile(r"did(\d+)\.png")

        # Sort files to ensure deterministic order
        files = sorted(folder.glob("*.png"))

        for file_path in files:
            img_path = str(file_path.resolve())
            filename = file_path.name

            match = pattern.search(filename)
            if match:
                det_id = int(match.group(1))
                frame_idx = det_id // 10000
                det_idx = det_id % 10000

                dataset.append(
                    {
                        "path": img_path,
                        "filename": filename,
                        "det_id": det_id,
                        "frame_idx": frame_idx,
                        "det_idx": det_idx,
                    }
                )
            else:
                # Fallback for other formats if needed, or skip
                # For now, skip non-matching files
                pass

        return dataset

    def temporal_subsample(
        self, dataset: List[Dict[str, Any]], interval: int
    ) -> List[Dict[str, Any]]:
        """Keep every Nth frame."""
        if interval <= 1:
            return dataset

        if not dataset:
            return []

        # Sort by frame_idx first
        sorted_data = sorted(dataset, key=lambda x: x["frame_idx"])

        # Get unique frames
        unique_frames = sorted({d["frame_idx"] for d in sorted_data})

        # Select every interval-th frame
        kept_frames = set(unique_frames[::interval])

        return [d for d in sorted_data if d["frame_idx"] in kept_frames]

    def deduplicate_by_hash(
        self,
        dataset: List[Dict[str, Any]],
        threshold: float = 0,
        method: str = "phash",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_groups: bool = False,
        color_threshold: Optional[float] = None,
    ):
        """
        Remove duplicates based on selected perceptual method.
        For hash methods (pHash/dHash/aHash), threshold is hamming bits.
        For histogram, threshold is Bhattacharyya distance in [0,1].

        Args:
            dataset: List of items (must contain 'path').
            threshold: Method-specific distance threshold.
            method: One of phash, dhash, ahash, histogram.
        """
        if not dataset:
            return []

        method = (method or "phash").lower()
        if method not in set(self.available_dedup_methods()):
            method = "phash"

        kept_items = []
        seen_exact_signatures = set()
        kept_signatures = []
        kept_color_signatures = []
        sig_to_cluster_index = {}
        duplicate_groups = []

        dataset_sorted = sorted(dataset, key=lambda x: (x["frame_idx"], x["det_id"]))

        total = len(dataset_sorted)
        for idx, item in enumerate(dataset_sorted, start=1):
            try:
                if "dedup_signature" not in item:
                    img = cv2.imread(item["path"])
                    if img is None:
                        if progress_callback and (
                            idx == 1 or idx % 250 == 0 or idx == total
                        ):
                            progress_callback(idx, total)
                        continue
                    item["dedup_signature"] = self.compute_signature(img, method)
                    if color_threshold is not None:
                        item["color_signature"] = self.compute_color_signature(img)
                elif color_threshold is not None and "color_signature" not in item:
                    img = cv2.imread(item["path"])
                    if img is not None:
                        item["color_signature"] = self.compute_color_signature(img)
            except Exception:
                if progress_callback and (idx == 1 or idx % 250 == 0 or idx == total):
                    progress_callback(idx, total)
                continue

            signature = item["dedup_signature"]
            current_color_signature = item.get("color_signature")

            if (
                method in {"phash", "dhash", "ahash"}
                and float(threshold) <= 0
                and color_threshold is None
            ):
                sig_key = int(signature)
                if sig_key in seen_exact_signatures:
                    cluster_idx = sig_to_cluster_index.get(sig_key)
                    if cluster_idx is not None:
                        duplicate_groups[cluster_idx]["paths"].append(str(item["path"]))
                    if progress_callback and (
                        idx == 1 or idx % 250 == 0 or idx == total
                    ):
                        progress_callback(idx, total)
                    continue
                seen_exact_signatures.add(sig_key)
                sig_to_cluster_index[sig_key] = len(duplicate_groups)
                kept_items.append(item)
                kept_color_signatures.append(None)
                duplicate_groups.append(
                    {
                        "hash": str(sig_key),
                        "count": 1,
                        "paths": [str(item["path"])],
                        "method": method,
                    }
                )
                if progress_callback and (idx == 1 or idx % 250 == 0 or idx == total):
                    progress_callback(idx, total)
                continue

            is_dup = False
            matched_index = -1
            for sig_idx, existing_signature in enumerate(kept_signatures):
                if self.is_duplicate(
                    signature, existing_signature, float(threshold), method
                ):
                    if color_threshold is not None:
                        existing_color_signature = kept_color_signatures[sig_idx]
                        if (
                            current_color_signature is not None
                            and existing_color_signature is not None
                            and self.color_distance(
                                current_color_signature, existing_color_signature
                            )
                            > float(color_threshold)
                        ):
                            continue
                    is_dup = True
                    matched_index = sig_idx
                    break

            if not is_dup:
                kept_signatures.append(signature)
                kept_items.append(item)
                kept_color_signatures.append(current_color_signature)
                duplicate_groups.append(
                    {
                        "hash": str(signature),
                        "count": 1,
                        "paths": [str(item["path"])],
                        "method": method,
                    }
                )
            elif 0 <= matched_index < len(duplicate_groups):
                duplicate_groups[matched_index]["paths"].append(str(item["path"]))

            if progress_callback and (idx == 1 or idx % 250 == 0 or idx == total):
                progress_callback(idx, total)

        if return_groups:
            filtered_groups = []
            for grp in duplicate_groups:
                grp["count"] = len(grp["paths"])
                if grp["count"] > 1:
                    filtered_groups.append(grp)
            filtered_groups.sort(key=lambda g: g["count"], reverse=True)
            return kept_items, filtered_groups

        return kept_items

    def diversity_sample(
        self, dataset: List[Dict[str, Any]], n_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Select diverse samples using MiniBatchKMeans on resized images.
        """
        if not dataset or len(dataset) <= n_samples:
            return dataset

        # Extract features (32x32 resized image)
        feature_size = (32, 32)
        features = []
        valid_items = []

        for item in dataset:
            try:
                img = cv2.imread(item["path"])
                if img is None:
                    continue
                # Use grayscale for feature vector to save space/time
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img

                resized = cv2.resize(gray, feature_size)
                features.append(resized.flatten())
                valid_items.append(item)
            except Exception:
                continue

        if not features:
            return []

        X = np.array(features)

        # Use MiniBatchKMeans for speed
        kmeans = MiniBatchKMeans(
            n_clusters=n_samples, random_state=42, n_init=3, batch_size=1024
        )
        kmeans.fit(X)

        # Find closest sample to each centroid
        from sklearn.metrics import pairwise_distances_argmin_min

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

        # Use a set to avoid duplicates if multiple centroids map to same sample
        selected_indices = sorted(set(closest))

        return [valid_items[i] for i in selected_indices]
