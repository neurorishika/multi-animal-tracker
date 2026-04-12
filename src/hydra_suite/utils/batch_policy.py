"""Shared batch-size policy helpers for tracking and UI code."""

from __future__ import annotations


def is_realtime_workflow(
    realtime_enabled: object = False,
    workflow_mode: object = "non_realtime",
) -> bool:
    """Return True when the current workflow should behave as realtime."""
    if isinstance(realtime_enabled, str):
        normalized = realtime_enabled.strip().lower()
        if normalized in {"1", "true", "yes", "on", "realtime"}:
            return True
    if bool(realtime_enabled):
        return True
    return str(workflow_mode or "non_realtime").strip().lower() == "realtime"


def normalize_batch_size(value: object, default: int = 1) -> int:
    """Convert an arbitrary value into a positive integer batch size."""
    try:
        batch_size = int(value)
    except Exception:
        batch_size = int(default)
    return max(1, batch_size)


def clamp_realtime_frame_batch_size(
    requested_batch_size: object,
    *,
    realtime_enabled: object = False,
    workflow_mode: object = "non_realtime",
) -> int:
    """Force frame-level batches to 1 for realtime workflows."""
    if is_realtime_workflow(realtime_enabled, workflow_mode):
        return 1
    return normalize_batch_size(requested_batch_size)


def clamp_realtime_individual_batch_size(
    requested_batch_size: object,
    *,
    max_animals: object,
    realtime_enabled: object = False,
    workflow_mode: object = "non_realtime",
) -> int:
    """Clamp per-animal inference batches to the configured animal count in realtime."""
    batch_size = normalize_batch_size(requested_batch_size)
    if not is_realtime_workflow(realtime_enabled, workflow_mode):
        return batch_size
    return min(batch_size, normalize_batch_size(max_animals))


def estimate_padding_waste(
    requested_batch_size: object,
    *,
    upper_bound: object,
) -> tuple[int, float]:
    """Estimate unused slots when a static batch exceeds a known upper bound."""
    batch_size = normalize_batch_size(requested_batch_size)
    bound = normalize_batch_size(upper_bound)
    wasted_slots = max(0, batch_size - bound)
    waste_ratio = float(wasted_slots) / float(batch_size) if batch_size > 0 else 0.0
    return wasted_slots, waste_ratio


def should_warn_for_padding_waste(
    requested_batch_size: object,
    *,
    upper_bound: object,
    min_wasted_slots: int = 2,
    min_waste_ratio: float = 0.25,
) -> bool:
    """Return True when a configured batch is likely to waste substantial work."""
    wasted_slots, waste_ratio = estimate_padding_waste(
        requested_batch_size,
        upper_bound=upper_bound,
    )
    return wasted_slots >= int(min_wasted_slots) and waste_ratio >= float(
        min_waste_ratio
    )
