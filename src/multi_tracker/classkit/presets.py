"""Preset LabelingScheme factory functions for common animal classification tasks."""

from __future__ import annotations

from .config.schemas import Factor, LabelingScheme


def head_tail_preset() -> LabelingScheme:
    """Single-factor, 4-class head/tail direction classifier."""
    return LabelingScheme(
        name="head_tail",
        factors=[
            Factor(
                name="direction",
                labels=["left", "right", "up", "down"],
                shortcut_keys=["a", "d", "w", "s"],
            )
        ],
        training_modes=["flat_tiny", "flat_yolo"],
        description="Head/tail orientation: left, right, up, down",
    )


def color_tag_preset(n_factors: int, colors: list[str]) -> LabelingScheme:
    """Multi-factor ordered color-tag classifier.

    Args:
        n_factors: Number of ordered color tag positions (1, 2, or 3 typical).
        colors: Ordered list of color label names for each factor.
    """
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if not colors:
        raise ValueError("colors must be non-empty")

    factors = [
        Factor(name=f"tag_{i + 1}", labels=list(colors)) for i in range(n_factors)
    ]
    if n_factors == 1:
        modes = ["flat_tiny", "flat_yolo"]
    else:
        modes = ["flat_tiny", "flat_yolo", "multihead_tiny", "multihead_yolo"]

    total = len(colors) ** n_factors
    return LabelingScheme(
        name=f"color_tags_{n_factors}factor",
        factors=factors,
        training_modes=modes,
        description=f"{n_factors}-factor color tag: {len(colors)} colors each → {total} composites",
    )


def age_preset(extra_classes: list[str] | None = None) -> LabelingScheme:
    """Single-factor age classifier (young/old), extensible."""
    labels = ["young", "old"] + list(extra_classes or [])
    return LabelingScheme(
        name="age",
        factors=[Factor(name="age", labels=labels)],
        training_modes=["flat_tiny", "flat_yolo"],
        description="Age classification: " + ", ".join(labels),
    )


def apriltag_preset(family: str, max_tag_id: int) -> LabelingScheme:
    """Single-factor scheme for AprilTag ID classification.

    Args:
        family: AprilTag family string (e.g. 'tag36h11').
        max_tag_id: Highest tag ID to include. Labels will be tag_0..tag_N + no_tag.
    """
    labels = [f"tag_{i}" for i in range(max_tag_id + 1)] + ["no_tag"]
    return LabelingScheme(
        name=f"apriltag_{family}",
        factors=[Factor(name=family, labels=labels)],
        training_modes=["flat_tiny", "flat_yolo"],
        description=f"AprilTag {family} classifier: tag_0..tag_{max_tag_id} + no_tag",
    )
