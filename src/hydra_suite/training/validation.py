"""Strict non-destructive validation for MAT training datasets."""

from __future__ import annotations

from pathlib import Path

from .contracts import ValidationIssue, ValidationReport
from .dataset_inspector import DatasetInspection


def _parse_label_lines(path: Path) -> list[list[str]]:
    lines = []
    txt = path.read_text(encoding="utf-8").splitlines()
    for ln in txt:
        ln = ln.strip()
        if not ln:
            continue
        lines.append(ln.split())
    return lines


def _validate_split_counts(
    inspection: DatasetInspection,
    min_train: int,
    min_val: int,
) -> list[ValidationIssue]:
    """Check that train/val splits meet minimum item requirements."""
    issues: list[ValidationIssue] = []
    train_count = len(inspection.splits.get("train", []))
    val_count = len(inspection.splits.get("val", []))
    if train_count < min_train:
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_train",
                message=f"Train split has {train_count} items; require >= {min_train}.",
            )
        )
    if val_count < min_val:
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_val",
                message=f"Val split has {val_count} items; require >= {min_val}.",
            )
        )
    return issues


def _validate_obb_line(
    parts: list[str], lbl: Path, stats: dict[str, object]
) -> list[ValidationIssue]:
    """Validate a single OBB label line and return any issues."""
    issues: list[ValidationIssue] = []
    if len(parts) != 9:
        stats["invalid_lines"] = int(stats["invalid_lines"]) + 1
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_obb_format",
                message=f"Expected 9 fields for OBB line, got {len(parts)} fields.",
                path=str(lbl),
            )
        )
        return issues
    try:
        class_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except Exception:
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_numeric",
                message="Non-numeric OBB label values.",
                path=str(lbl),
            )
        )
        return issues
    cast = stats["class_ids"]
    if isinstance(cast, set):
        cast.add(class_id)
    for coord in coords:
        if coord < -1e-6 or coord > 1.0 + 1e-6:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="coord_out_of_range",
                    message="Normalized OBB coordinate out of [0,1] range.",
                    path=str(lbl),
                )
            )
            break
    return issues


def _validate_item_file_pair(
    item, split: str, stats: dict[str, object]
) -> list[ValidationIssue]:
    """Validate one image/label pair and return issues."""
    issues: list[ValidationIssue] = []
    img = Path(item.image_path)
    lbl = Path(item.label_path)
    if not img.exists():
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_image",
                message="Image file missing.",
                path=str(img),
            )
        )
        return issues
    if not lbl.exists():
        stats["missing_labels"] = int(stats["missing_labels"]) + 1
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_label",
                message=f"Missing label for split '{split}'.",
                path=str(lbl),
            )
        )
        return issues

    try:
        parsed = _parse_label_lines(lbl)
    except Exception as exc:
        issues.append(
            ValidationIssue(
                severity="error",
                code="label_read_error",
                message=f"Cannot read label file: {exc}",
                path=str(lbl),
            )
        )
        return issues

    if not parsed:
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_label",
                message="Label file has no objects.",
                path=str(lbl),
            )
        )
        return issues

    for parts in parsed:
        issues.extend(_validate_obb_line(parts, lbl, stats))
    return issues


def validate_obb_dataset(
    inspection: DatasetInspection,
    *,
    require_train_val: bool = True,
    min_train: int = 1,
    min_val: int = 1,
) -> ValidationReport:
    """Validate OBB-label source dataset with strict fail-fast checks."""

    issues: list[ValidationIssue] = []
    stats: dict[str, object] = {
        "root_dir": inspection.root_dir,
        "split_counts": {k: len(v) for k, v in inspection.splits.items()},
        "missing_labels": 0,
        "invalid_lines": 0,
        "class_ids": set(),
    }

    if require_train_val:
        issues.extend(_validate_split_counts(inspection, min_train, min_val))

    for split, items in inspection.splits.items():
        for item in items:
            issues.extend(_validate_item_file_pair(item, split, stats))

    class_ids = sorted(int(x) for x in stats.get("class_ids", set()))
    stats["class_ids"] = class_ids
    if len(class_ids) > 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="multi_class_source",
                message=(
                    "Source OBB dataset contains multiple classes. "
                    "Use explicit class remapping into a derived workspace copy."
                ),
            )
        )

    return ValidationReport(
        valid=not any(i.severity == "error" for i in issues), issues=issues, stats=stats
    )


def format_validation_report(report: ValidationReport) -> str:
    """Format validation report for UI logs."""

    lines = [
        f"Validation: {'PASS' if report.valid else 'FAIL'}",
        f"Stats: {report.stats}",
    ]
    for issue in report.issues:
        where = f" [{issue.path}]" if issue.path else ""
        lines.append(f"- {issue.severity.upper()} {issue.code}: {issue.message}{where}")
    return "\n".join(lines)
