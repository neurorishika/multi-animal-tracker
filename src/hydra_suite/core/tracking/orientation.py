"""Orientation smoothing for the tracking pipeline.

Temporal smoothing of track orientations, handling both directed (pose/head-tail)
and undirected (OBB axis) heading sources with flip detection and hysteresis.
"""

import math

from hydra_suite.utils.geometry import wrap_angle_degs


def smooth_orientation(
    r,
    theta,
    speed,
    p,
    orientation_last,
    position_deques,
    directed_heading=False,
    orient_confidence=1.0,
    heading_flip_counters=None,
):
    """Smooth a track's orientation over time.

    Args:
        r: Track index.
        theta: Raw measured heading (radians).
        speed: Track speed estimate.
        p: Parameter dict (needs VELOCITY_THRESHOLD, MAX_ORIENT_DELTA_STOPPED,
           INSTANT_FLIP_ORIENTATION, DIRECTED_ORIENT_SMOOTHING,
           DIRECTED_ORIENT_FLIP_CONFIDENCE, DIRECTED_ORIENT_FLIP_PERSISTENCE).
        orientation_last: List of last committed theta per track (mutated).
        position_deques: Per-track deques of (x, y, frame) positions.
        directed_heading: Whether this heading comes from a directed source.
        orient_confidence: Confidence in the heading estimate [0, 1].
        heading_flip_counters: Per-track flip hysteresis counters (mutated).

    Returns:
        Smoothed theta in radians.
    """
    final_theta, old = theta, orientation_last[r]

    if directed_heading and p.get("DIRECTED_ORIENT_SMOOTHING", True):
        if old is None:
            if heading_flip_counters is not None:
                heading_flip_counters[r] = 0
            return theta
        flip_conf_thresh = float(p.get("DIRECTED_ORIENT_FLIP_CONFIDENCE", 0.7))
        flip_persistence = int(p.get("DIRECTED_ORIENT_FLIP_PERSISTENCE", 3))
        old_deg = math.degrees(old)
        new_deg = math.degrees(theta)
        delta = wrap_angle_degs(new_deg - old_deg)
        if abs(delta) > 90:
            # Abrupt 180-degree swaps are ambiguous without motion evidence.
            # Repeated classifier agreement alone can entrench the wrong branch,
            # so only accept these flips when track motion also supports them.
            flip_supported = False
            if speed >= p["VELOCITY_THRESHOLD"] and len(position_deques[r]) == 2:
                (x1, y1, _), (x2, y2, _) = position_deques[r]
                ang_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
                motion_favors_new = abs(wrap_angle_degs(new_deg - ang_deg)) < abs(
                    wrap_angle_degs(old_deg - ang_deg)
                )
                flip_supported = (
                    motion_favors_new and orient_confidence >= flip_conf_thresh
                )

            if flip_supported:
                if heading_flip_counters is not None:
                    heading_flip_counters[r] += 1
                    if heading_flip_counters[r] >= flip_persistence:
                        heading_flip_counters[r] = 0
                    else:
                        new_deg = (new_deg + 180.0) % 360.0
            else:
                if heading_flip_counters is not None:
                    heading_flip_counters[r] = 0
                new_deg = (new_deg + 180.0) % 360.0
        else:
            if heading_flip_counters is not None:
                heading_flip_counters[r] = 0
        return math.radians(new_deg % 360.0)

    # --- Original undirected smoothing (axis-only, no direction signal) ---
    if speed < p["VELOCITY_THRESHOLD"] and old is not None:
        old_deg, new_deg = math.degrees(old), math.degrees(theta)
        delta = wrap_angle_degs(new_deg - old_deg)
        if abs(delta) > 90:
            new_deg = (new_deg + 180) % 360
        elif abs(delta) > p["MAX_ORIENT_DELTA_STOPPED"]:
            new_deg = old_deg + math.copysign(p["MAX_ORIENT_DELTA_STOPPED"], delta)
        final_theta = math.radians(new_deg)
    elif speed >= p["VELOCITY_THRESHOLD"] and p["INSTANT_FLIP_ORIENTATION"]:
        (x1, y1, _), (x2, y2, _) = position_deques[r]
        ang = math.atan2(y2 - y1, x2 - x1)
        diff = (ang - theta + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > math.pi / 2:
            final_theta = (theta + math.pi) % (2 * math.pi)
    return final_theta
