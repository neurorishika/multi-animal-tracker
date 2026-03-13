"""Strict non-destructive validation for MAT training datasets."""

from __future__ import annotations

from pathlib import Path

from .contracts import ValidationIssue, ValidationReport
from .dataset_inspector import ClassifyInspection, DatasetInspection


def _parse_label_lines(path: Path) -> list[list[str]]:
    lines = []
    txt = path.read_text(encoding="utf-8").splitlines()
    for ln in txt:
        ln = ln.strip()
        if not ln:
            continue
        lines.append(ln.split())
    return lines


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

    for split, items in inspection.splits.items():
        for item in items:
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
                continue
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
                continue

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
                continue

            if not parsed:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="empty_label",
                        message="Label file has no objects.",
                        path=str(lbl),
                    )
                )
                continue

            for parts in parsed:
                if len(parts) != 9:
                    stats["invalid_lines"] = int(stats["invalid_lines"]) + 1
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="invalid_obb_format",
                            message=(
                                f"Expected 9 fields for OBB line, got {len(parts)} fields."
                            ),
                            path=str(lbl),
                        )
                    )
                    continue
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
                    continue
                cast = stats["class_ids"]
                if isinstance(cast, set):
                    cast.add(class_id)
                for coord in coords:
                    if coord < -1e-6 or coord > 1.0 + 1e-6:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                code="coord_out_of_range",
                                message=(
                                    "Normalized OBB coordinate out of [0,1] range."
                                ),
                                path=str(lbl),
                            )
                        )
                        break

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


def validate_classify_dataset(
    inspection: ClassifyInspection,
    *,
    required_classes: tuple[str, str] = ("head_left", "head_right"),
    require_val: bool = True,
) -> ValidationReport:
    """Validate classify-style head-tail datasets."""

    issues: list[ValidationIssue] = []
    stats: dict[str, object] = {
        "root_dir": inspection.root_dir,
        "split_class_counts": {},
    }

    train_split = inspection.splits.get("train", {})
    if not train_split:
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_train",
                message="Classify dataset is missing train split.",
            )
        )

    if require_val and not inspection.splits.get("val", {}):
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_val",
                message="Classify dataset is missing val split.",
            )
        )

    req = set(required_classes)
    for split, classes in inspection.splits.items():
        stats["split_class_counts"][split] = {k: len(v) for k, v in classes.items()}
        class_set = set(classes.keys())
        missing = sorted(req - class_set)
        if missing:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_required_class",
                    message=f"Split '{split}' missing required class(es): {', '.join(missing)}",
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
