"""Swap suspicion scorer for MAT-afterhours.

Scores pairs of trajectories for likelihood of an identity swap by combining
several independent signals: position crossing, proximity, heading
discontinuity, pose quality drop, and region context from the confidence
density map.

This is a pure Python module with no Qt dependency.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from multi_tracker.afterhours.core.confidence_density import DensityRegion

# ---------------------------------------------------------------------------
# Signal weights
# ---------------------------------------------------------------------------

W_CROSSING = 1.0
W_PROXIMITY = 0.45
W_HEADING = 0.5
W_POSE_DROP = 0.3  # amplifier only -- fires when Cr or Pr also fires
W_REGION_DISCOUNT = 0.4
W_BOUNDARY_BONUS = 0.25

# ---------------------------------------------------------------------------
# SwapSuspicionEvent
# ---------------------------------------------------------------------------


@dataclass
class SwapSuspicionEvent:
    """A single swap suspicion event between two tracks.

    Attributes
    ----------
    track_a:
        First trajectory ID.
    track_b:
        Second trajectory ID (may be ``None`` for single-track anomalies).
    frame_peak:
        Frame index where the suspicion is strongest.
    frame_range:
        Inclusive ``(start, end)`` frame range of the event.
    score:
        Combined suspicion score in ``[0, 1]``.
    signals:
        List of signal codes that fired (e.g. ``["Cr", "Hd"]``).
    region_label:
        Density region label at the peak location, or ``"open_field"``.
    region_boundary:
        Whether the peak falls on a region temporal boundary.
    """

    track_a: int
    track_b: Optional[int]
    frame_peak: int
    frame_range: Tuple[int, int]
    score: float
    signals: List[str] = field(default_factory=list)
    region_label: str = "open_field"
    region_boundary: bool = False


# ---------------------------------------------------------------------------
# SwapScorer
# ---------------------------------------------------------------------------


class SwapScorer:
    """Score trajectory pairs for identity swap suspicion.

    Parameters
    ----------
    regions:
        List of :class:`DensityRegion` from the confidence density map.
    min_score:
        Minimum combined score to include in output.
    approach_distance:
        Maximum pixel distance at which two tracks are considered to interact.
    crossing_window:
        Number of frames before/after the closest approach to check for a
        position exchange (sign flip in relative X).
    heading_reversal_deg:
        Minimum heading change (degrees) to flag as a heading discontinuity.
    """

    def __init__(
        self,
        regions: List[DensityRegion],
        min_score: float = 0.15,
        approach_distance: float = 60.0,
        crossing_window: int = 15,
        heading_reversal_deg: float = 120.0,
    ) -> None:
        self.regions = regions
        self.min_score = min_score
        self.approach_distance = approach_distance
        self.crossing_window = crossing_window
        self.heading_reversal_rad = np.deg2rad(heading_reversal_deg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        min_score: Optional[float] = None,
    ) -> List[SwapSuspicionEvent]:
        """Score all trajectory pairs and return sorted events.

        Parameters
        ----------
        df:
            DataFrame with at least columns ``TrajectoryID``, ``FrameID``,
            ``X``, ``Y``, ``Theta``.
        min_score:
            Override instance ``min_score`` for this call.

        Returns
        -------
        List[SwapSuspicionEvent]
            Events sorted by score descending.
        """
        threshold = min_score if min_score is not None else self.min_score
        track_ids = sorted(df["TrajectoryID"].unique())

        # Pre-index per-track data for fast lookup.
        track_data = {}
        for tid in track_ids:
            tdf = df[df["TrajectoryID"] == tid].sort_values("FrameID")
            track_data[tid] = tdf

        events: List[SwapSuspicionEvent] = []

        for id_a, id_b in itertools.combinations(track_ids, 2):
            df_a = track_data[id_a]
            df_b = track_data[id_b]

            # Find common frames.
            common = np.intersect1d(df_a["FrameID"].values, df_b["FrameID"].values)
            if len(common) < 3:
                continue

            # Index both tracks by frame for the common range.
            a_indexed = df_a.set_index("FrameID").loc[common]
            b_indexed = df_b.set_index("FrameID").loc[common]

            # Compute pairwise distances at each common frame.
            dx = a_indexed["X"].values - b_indexed["X"].values
            dy = a_indexed["Y"].values - b_indexed["Y"].values
            distances = np.sqrt(dx**2 + dy**2)

            # Check if they ever come within approach distance.
            min_dist = float(distances.min())
            if min_dist >= self.approach_distance:
                continue

            # Find peak frame (closest approach).
            peak_idx = int(np.argmin(distances))
            peak_frame = int(common[peak_idx])

            # Compute signals.
            signals: List[str] = []

            cr_score = self._crossing_signal(a_indexed, b_indexed, common, distances)
            if cr_score > 0:
                signals.append("Cr")

            pr_score = self._proximity_signal(distances, common)
            if pr_score > 0 and cr_score == 0:
                # Proximity only fires if no crossing detected.
                signals.append("Pr")
            elif cr_score > 0:
                # Crossing subsumes proximity -- don't double count.
                pr_score = 0.0

            hd_score = self._heading_signal(df_a, df_b, common, peak_frame)
            if hd_score > 0:
                signals.append("Hd")

            pq_score = self._pose_quality_signal(df_a, df_b, peak_frame)
            if pq_score > 0 and (cr_score > 0 or pr_score > 0):
                signals.append("Pq")
                pq_term = W_POSE_DROP * pq_score
            else:
                pq_term = 0.0

            if not signals:
                continue

            # Region context.
            cx = float(
                (a_indexed["X"].values[peak_idx] + b_indexed["X"].values[peak_idx]) / 2
            )
            cy = float(
                (a_indexed["Y"].values[peak_idx] + b_indexed["Y"].values[peak_idx]) / 2
            )
            region_label, is_boundary = self._get_region_context(cx, cy, peak_frame)

            # Region discount: reduce score if inside a known density region
            # (not open field) and NOT at the boundary.
            region_discount = 0.0
            if region_label != "open_field" and not is_boundary:
                region_discount = W_REGION_DISCOUNT

            boundary_bonus = 0.0
            if is_boundary:
                boundary_bonus = W_BOUNDARY_BONUS

            raw = (
                W_CROSSING * cr_score
                + W_PROXIMITY * pr_score
                + W_HEADING * hd_score
                + pq_term
                - region_discount
                + boundary_bonus
            )
            score_val = float(np.clip(raw, 0.0, 1.0))

            if score_val < threshold:
                continue

            # Determine frame range around the peak.
            close_mask = distances < self.approach_distance
            close_frames = common[close_mask]
            if len(close_frames) == 0:
                close_frames = common
            frame_range = (int(close_frames.min()), int(close_frames.max()))

            events.append(
                SwapSuspicionEvent(
                    track_a=int(id_a),
                    track_b=int(id_b),
                    frame_peak=peak_frame,
                    frame_range=frame_range,
                    score=score_val,
                    signals=signals,
                    region_label=region_label,
                    region_boundary=is_boundary,
                )
            )

        # Sort descending by score.
        events.sort(key=lambda e: e.score, reverse=True)
        return events

    # ------------------------------------------------------------------
    # Private signal methods
    # ------------------------------------------------------------------

    def _crossing_signal(
        self,
        a_indexed: pd.DataFrame,
        b_indexed: pd.DataFrame,
        common: np.ndarray,
        distances: np.ndarray,
    ) -> float:
        """Detect position exchange (sign flip in relative X).

        Returns a score in [0, 1] or 0 if no crossing detected.
        """
        # Find the closest approach point.
        peak_idx = int(np.argmin(distances))
        min_dist = float(distances[peak_idx])

        if min_dist >= self.approach_distance:
            return 0.0

        # Relative X before and after crossing window.
        rel_x = a_indexed["X"].values - b_indexed["X"].values
        w = self.crossing_window

        before_start = max(0, peak_idx - w)
        after_end = min(len(common), peak_idx + w + 1)

        if peak_idx - before_start < 1 or after_end - peak_idx < 2:
            return 0.0

        # Check sign of relative X before and after peak.
        rel_before = rel_x[before_start:peak_idx]
        rel_after = rel_x[peak_idx + 1 : after_end]

        if len(rel_before) == 0 or len(rel_after) == 0:
            return 0.0

        # Use median to be robust to noise.
        sign_before = np.sign(np.median(rel_before))
        sign_after = np.sign(np.median(rel_after))

        if sign_before == 0 or sign_after == 0:
            return 0.0

        if sign_before != sign_after:
            # Crossing detected -- strength based on proximity.
            strength = 1.0 - (min_dist / self.approach_distance)
            return float(np.clip(strength, 0.0, 1.0))

        return 0.0

    def _proximity_signal(
        self,
        distances: np.ndarray,
        common: np.ndarray,
    ) -> float:
        """Detect close approach without position exchange.

        Returns a score in [0, 1] or 0 if no close approach.
        """
        min_dist = float(distances.min())
        if min_dist >= self.approach_distance:
            return 0.0

        # Proximity strength with 0.6 damping factor.
        strength = 0.6 * (1.0 - min_dist / self.approach_distance)
        return float(np.clip(strength, 0.0, 1.0))

    def _heading_signal(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        common: np.ndarray,
        peak_frame: int,
    ) -> float:
        """Detect sudden heading reversal near the peak frame.

        Returns a score in [0, 1] or 0 if no reversal detected.
        """
        max_diff = 0.0

        for tdf in (df_a, df_b):
            tdf_common = tdf[tdf["FrameID"].isin(common)].sort_values("FrameID")
            if len(tdf_common) < 2:
                continue

            thetas = tdf_common["Theta"].values
            # Circular difference between consecutive frames.
            diffs = np.abs(
                np.arctan2(
                    np.sin(np.diff(thetas)),
                    np.cos(np.diff(thetas)),
                )
            )
            if len(diffs) > 0:
                max_diff = max(max_diff, float(diffs.max()))

        if max_diff >= self.heading_reversal_rad:
            strength = min(max_diff / np.pi, 1.0)
            return float(strength)

        return 0.0

    def _pose_quality_signal(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        peak_frame: int,
    ) -> float:
        """Detect pose quality drop near the peak frame.

        Only fires if ``PoseQualityScore`` column exists. Returns a score
        in [0, 1] or 0 if no significant drop.
        """
        if "PoseQualityScore" not in df_a.columns:
            return 0.0

        window = 10  # frames before/after peak
        z_threshold = 2.5

        max_drop = 0.0
        for tdf in (df_a, df_b):
            if "PoseQualityScore" not in tdf.columns:
                continue

            scores = tdf["PoseQualityScore"].dropna()
            if len(scores) < 5:
                continue

            overall_mean = float(scores.mean())
            overall_std = float(scores.std())
            if overall_std < 1e-8:
                continue

            # Quality near peak.
            near_peak = tdf[
                (tdf["FrameID"] >= peak_frame - window)
                & (tdf["FrameID"] <= peak_frame + window)
            ]
            if len(near_peak) == 0:
                continue

            local_mean = float(near_peak["PoseQualityScore"].mean())
            z_score = (overall_mean - local_mean) / overall_std

            if z_score > z_threshold:
                drop = min(z_score / 5.0, 1.0)  # normalise to [0, 1]
                max_drop = max(max_drop, drop)

        return max_drop

    def _get_region_context(
        self,
        cx: float,
        cy: float,
        frame: int,
    ) -> Tuple[str, bool]:
        """Return (region_label, is_boundary) for a point.

        Returns ``("open_field", False)`` if the point is not in any region.
        """
        for region in self.regions:
            if region.contains(frame, cx, cy):
                return region.label, region.is_boundary_frame(frame)
        return "open_field", False
