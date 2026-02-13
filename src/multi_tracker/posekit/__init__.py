"""Pose labeling subpackage."""

try:
    from .pose_label import main
except Exception:  # pragma: no cover - enables metadata/doc tooling without full runtime deps
    main = None

__all__ = ["main"]
