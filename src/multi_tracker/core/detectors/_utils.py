"""Shared utilities for the detectors package."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CONSERVATIVE_SPLIT_MIN_ANIMALS = 1.6


def _normalize_detection_ids(detection_ids):
    """Normalize detection IDs to runtime integers.

    Cached raw detections may come back as float-backed arrays from older NPZ
    files. Accept finite whole-number values to preserve compatibility.
    """
    if detection_ids is None:
        return None

    arr = np.asarray(detection_ids)
    if arr.size == 0:
        return []

    normalized = []
    for raw_value in arr.reshape(-1).tolist():
        if isinstance(raw_value, (np.integer, int)):
            normalized.append(int(raw_value))
            continue

        try:
            float_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid detection ID value: {raw_value!r}") from exc

        if not np.isfinite(float_value) or not float_value.is_integer():
            raise ValueError(
                f"Detection ID must be a finite whole number, got {raw_value!r}"
            )
        normalized.append(int(float_value))

    return normalized


def _advanced_config_value(params, key, default=None):
    """Read a power-user override from ADVANCED_CONFIG when present."""
    advanced_config = params.get("ADVANCED_CONFIG", {})
    if isinstance(advanced_config, dict) and key in advanced_config:
        return advanced_config.get(key)
    return default
