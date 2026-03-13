"""Backward-compatibility shim — re-exports from event_scorer.

All scoring logic now lives in ``event_scorer.py``.  Import from there
for new code; this module exists so that old tests and downstream
modules keep working.
"""

from multi_tracker.afterhours.core.event_scorer import (  # noqa: F401
    EventScorer as SwapScorer,
)
from multi_tracker.afterhours.core.event_scorer import SwapSuspicionEvent

__all__ = ["SwapScorer", "SwapSuspicionEvent"]
