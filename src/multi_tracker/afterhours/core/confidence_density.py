"""Confidence density map core module for MAT-afterhours.

This module accumulates per-frame confidence density grids from detection
measurements, smooths them temporally, binarizes, and finds 3D connected
components (x, y, t) that represent "low-confidence high-density" regions
where identity swaps are most likely to occur.

A detection with low confidence contributes a strong Gaussian signal;
high-confidence detections contribute near-zero signal. After accumulating
all frames the volume is smoothed along the time axis, globally normalised,
and thresholded to produce a binary mask. scipy.ndimage.label then finds
connected components in that 3D mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, label

# ---------------------------------------------------------------------------
# DensityRegion
# ---------------------------------------------------------------------------


@dataclass
class DensityRegion:
    """A 3-D connected component in the confidence density volume.

    Attributes
    ----------
    label:
        Human-readable identifier, e.g. ``"region-1"``.
    frame_start:
        Inclusive first frame index covered by the region.
    frame_end:
        Inclusive last frame index covered by the region.
    pixel_bbox:
        Bounding box in image pixel coordinates ``(x1, y1, x2, y2)``.
    """

    label: str
    frame_start: int
    frame_end: int
    pixel_bbox: Tuple[int, int, int, int]

    # ------------------------------------------------------------------
    # Spatial / temporal membership
    # ------------------------------------------------------------------

    def contains(self, frame: int, cx: float, cy: float) -> bool:
        """Return True if *frame* and position ``(cx, cy)`` fall inside.

        Parameters
        ----------
        frame:
            Frame index to test.
        cx, cy:
            Detection centre in pixel coordinates.
        """
        if frame < self.frame_start or frame > self.frame_end:
            return False
        x1, y1, x2, y2 = self.pixel_bbox
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def is_boundary_frame(self, frame: int, margin: int = 3) -> bool:
        """Return True if *frame* is within *margin* frames of either edge.

        Parameters
        ----------
        frame:
            Frame index to test.
        margin:
            Number of frames from each boundary to consider as "boundary".
        """
        return (
            frame <= self.frame_start + margin - 1
            or frame >= self.frame_end - margin + 1
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "label": self.label,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "pixel_bbox": list(self.pixel_bbox),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DensityRegion":
        """Reconstruct a :class:`DensityRegion` from a dictionary."""
        return cls(
            label=d["label"],
            frame_start=int(d["frame_start"]),
            frame_end=int(d["frame_end"]),
            pixel_bbox=tuple(int(v) for v in d["pixel_bbox"]),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# ConfidenceDensityMap
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceDensityMap:
    """Container for the accumulated density volume and derived regions.

    Attributes
    ----------
    frame_grids:
        Raw per-frame float32 grids, shape ``(T, H, W)``.
    regions:
        List of :class:`DensityRegion` found after binarisation.
    frame_h:
        Height of each grid in pixels.
    frame_w:
        Width of each grid in pixels.
    """

    frame_grids: np.ndarray
    regions: List[DensityRegion]
    frame_h: int
    frame_w: int


# ---------------------------------------------------------------------------
# accumulate_frame
# ---------------------------------------------------------------------------


def accumulate_frame(
    grid: np.ndarray,
    meas: np.ndarray,
    confidences: np.ndarray,
    sizes: np.ndarray,
    sigma_scale: float,
) -> np.ndarray:
    """Add ``(1 - confidence)`` weighted Gaussians to *grid* in-place.

    Each detection contributes a Gaussian blob centred at its spatial
    location.  The weight is ``(1 - confidence)`` so that uncertain
    detections produce the strongest signal.

    Sigma is derived from the bounding-box size:
        ``sigma = sigma_scale * sqrt(size) / 2``

    Parameters
    ----------
    grid:
        Float32 array of shape ``(H, W)`` — modified in-place.
    meas:
        Detection measurements, shape ``(N, 3)`` columns ``[x, y, theta]``.
    confidences:
        Detection confidence values, shape ``(N,)``, range ``[0, 1]``.
    sizes:
        Squared bounding-box diagonal (or area proxy), shape ``(N,)``.
    sigma_scale:
        Scalar that controls the spread of each Gaussian relative to size.

    Returns
    -------
    np.ndarray
        The modified *grid* (same object that was passed in).
    """
    if meas.shape[0] == 0:
        return grid

    h, w = grid.shape

    # Pre-compute coordinate grids once (y, x order for indexing).
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)  # shape (H, W) each

    for i in range(meas.shape[0]):
        cx = float(meas[i, 0])
        cy = float(meas[i, 1])
        conf = float(confidences[i])
        size = float(sizes[i])

        weight = 1.0 - conf
        if weight <= 0.0:
            continue

        sigma = sigma_scale * np.sqrt(max(size, 1e-6)) / 2.0

        # Gaussian evaluated analytically — avoids heavy scipy calls per det.
        gauss = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2.0 * sigma**2)).astype(
            np.float32
        )

        grid += weight * gauss

    return grid


# ---------------------------------------------------------------------------
# smooth_and_binarize
# ---------------------------------------------------------------------------


def smooth_and_binarize(
    frames: np.ndarray,
    temporal_sigma: float,
    threshold: float,
) -> np.ndarray:
    """Smooth the density volume temporally, normalise globally, binarize.

    Parameters
    ----------
    frames:
        Float32 array of shape ``(T, H, W)`` — the stacked per-frame grids.
    temporal_sigma:
        Standard deviation (in frames) for the Gaussian smoothing kernel
        applied along the time axis only.
    threshold:
        Value in ``[0, 1]`` above which a voxel is marked as 1 after global
        normalisation.

    Returns
    -------
    np.ndarray
        Uint8 binary array of shape ``(T, H, W)`` with values in ``{0, 1}``.
    """
    # Apply Gaussian smoothing along axis 0 (time) only.
    smoothed = gaussian_filter(frames, sigma=(temporal_sigma, 0.0, 0.0))

    # Global normalisation so threshold is meaningful across recordings.
    global_max = smoothed.max()
    if global_max > 0.0:
        smoothed = smoothed / global_max

    binary = (smoothed >= threshold).astype(np.uint8)
    return binary


# ---------------------------------------------------------------------------
# find_regions
# ---------------------------------------------------------------------------


def find_regions(
    binary: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> List[DensityRegion]:
    """Find 3-D connected components in a binary (T, H, W) volume.

    Uses full 3-D connectivity (26-connected in 3-D) via
    ``scipy.ndimage.label``.

    Parameters
    ----------
    binary:
        Uint8 array of shape ``(T, H, W)``.
    frame_h:
        Frame height in pixels (informational; used for future normalisation).
    frame_w:
        Frame width in pixels (informational).

    Returns
    -------
    List[DensityRegion]
        One :class:`DensityRegion` per connected component, sorted by first
        occurrence frame then by pixel-space centroid x.  Empty list if no
        foreground voxels exist.
    """
    if binary.max() == 0:
        return []

    # Full 26-connectivity structure for 3-D labelling.
    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled, num_features = label(binary, structure=structure)

    regions: List[DensityRegion] = []
    for component_id in range(1, num_features + 1):
        mask = labeled == component_id  # (T, H, W) bool

        # Time extent.
        t_coords, y_coords, x_coords = np.nonzero(mask)
        frame_start = int(t_coords.min())
        frame_end = int(t_coords.max())

        # Spatial bounding box in pixel coords (x1, y1, x2, y2).
        x1 = int(x_coords.min())
        x2 = int(x_coords.max())
        y1 = int(y_coords.min())
        y2 = int(y_coords.max())

        region_label = f"region-{component_id}"
        regions.append(
            DensityRegion(
                label=region_label,
                frame_start=frame_start,
                frame_end=frame_end,
                pixel_bbox=(x1, y1, x2, y2),
            )
        )

    # Sort by frame_start, then by spatial centroid x for determinism.
    regions.sort(key=lambda r: (r.frame_start, r.pixel_bbox[0]))

    # Reassign sequential labels after sorting.
    for idx, region in enumerate(regions, start=1):
        region.label = f"region-{idx}"

    return regions


# ---------------------------------------------------------------------------
# tag_detections
# ---------------------------------------------------------------------------


def tag_detections(
    detections: List[Dict[str, Any]],
    regions: List[DensityRegion],
) -> List[Dict[str, Any]]:
    """Annotate each detection dict with region membership information.

    Each dict in *detections* must contain keys ``"frame"``, ``"cx"``,
    ``"cy"``.  Two keys are added (or overwritten) in-place:

    - ``"region_label"``: the label of the first matching
      :class:`DensityRegion`, or ``"open_field"`` if none match.
    - ``"region_boundary"``: ``True`` if the detection falls in the temporal
      boundary of its region (uses default margin of 3 frames).

    Parameters
    ----------
    detections:
        List of detection dicts.
    regions:
        List of :class:`DensityRegion` to test against.

    Returns
    -------
    List[Dict[str, Any]]
        The same list (modified in-place) for convenience.
    """
    for det in detections:
        frame = int(det["frame"])
        cx = float(det["cx"])
        cy = float(det["cy"])

        matched_region: Optional[DensityRegion] = None
        for region in regions:
            if region.contains(frame, cx, cy):
                matched_region = region
                break

        if matched_region is None:
            det["region_label"] = "open_field"
            det["region_boundary"] = False
        else:
            det["region_label"] = matched_region.label
            det["region_boundary"] = matched_region.is_boundary_frame(frame)

    return detections


# ---------------------------------------------------------------------------
# save_regions / load_regions
# ---------------------------------------------------------------------------


def save_regions(regions: List[DensityRegion], path: str | Path) -> None:
    """Persist a list of :class:`DensityRegion` objects to a JSON file.

    Parameters
    ----------
    regions:
        Regions to serialise.
    path:
        Destination file path.  Parent directories must already exist.
    """
    payload = [r.to_dict() for r in regions]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_regions(path: str | Path) -> List[DensityRegion]:
    """Load :class:`DensityRegion` objects from a JSON file.

    Parameters
    ----------
    path:
        Source file path previously written by :func:`save_regions`.

    Returns
    -------
    List[DensityRegion]
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [DensityRegion.from_dict(d) for d in payload]


# ---------------------------------------------------------------------------
# compute_density_map_from_cache
# ---------------------------------------------------------------------------


def compute_density_map_from_cache(
    detection_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_h: int,
    frame_w: int,
    sigma_scale: float,
    temporal_sigma: float,
    threshold: float,
) -> Tuple[ConfidenceDensityMap, List[np.ndarray]]:
    """Run the full density-map pipeline from a detection cache.

    Parameters
    ----------
    detection_cache:
        Mapping from ``frame_index`` to a tuple of
        ``(measurements, confidences, sizes)`` as produced by the MAT
        detection engine.  Each array has ``N`` rows (one per detection).
        *measurements* is shape ``(N, 3)`` with columns ``[x, y, theta]``.
        *confidences* is shape ``(N,)`` in ``[0, 1]``.
        *sizes* is shape ``(N,)`` (squared bbox diagonal or equivalent).
    frame_h:
        Height of the video frames in pixels.
    frame_w:
        Width of the video frames in pixels.
    sigma_scale:
        Passed through to :func:`accumulate_frame`.
    temporal_sigma:
        Passed through to :func:`smooth_and_binarize`.
    threshold:
        Passed through to :func:`smooth_and_binarize`.

    Returns
    -------
    (ConfidenceDensityMap, raw_grids)
        *raw_grids* is the list of per-frame float32 arrays before smoothing,
        in frame-index order.
    """
    if not detection_cache:
        frame_grids = np.zeros((0, frame_h, frame_w), dtype=np.float32)
        cdm = ConfidenceDensityMap(
            frame_grids=frame_grids,
            regions=[],
            frame_h=frame_h,
            frame_w=frame_w,
        )
        return cdm, []

    sorted_frames = sorted(detection_cache.keys())
    raw_grids: List[np.ndarray] = []
    for frame_idx in sorted_frames:
        meas, confs, sizes = detection_cache[frame_idx]
        grid = np.zeros((frame_h, frame_w), dtype=np.float32)
        accumulate_frame(grid, meas, confs, sizes, sigma_scale=sigma_scale)
        raw_grids.append(grid)

    frame_grids = np.stack(raw_grids, axis=0)  # (T, H, W)

    binary = smooth_and_binarize(
        frame_grids, temporal_sigma=temporal_sigma, threshold=threshold
    )
    regions = find_regions(binary, frame_h=frame_h, frame_w=frame_w)

    cdm = ConfidenceDensityMap(
        frame_grids=frame_grids,
        regions=regions,
        frame_h=frame_h,
        frame_w=frame_w,
    )
    return cdm, raw_grids
