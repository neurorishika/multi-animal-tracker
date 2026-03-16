"""
Run-scoped lifecycle manager for pose inference backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RuntimeMetrics:
    startup_ms: float = 0.0
    warmup_ms: float = 0.0
    closed_ms: float = 0.0
