"""
Run-scoped lifecycle manager for pose inference backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RuntimeMetrics:  # noqa: DC03  (metrics contract; used when lifecycle profiling is enabled)
    startup_ms: float = 0.0  # noqa: DC01  (dataclass field)
    warmup_ms: float = 0.0  # noqa: DC01  (dataclass field)
    closed_ms: float = 0.0  # noqa: DC01  (dataclass field)
