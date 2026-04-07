"""High-level orchestration service for MAT role-aware training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .contracts import (
    DatasetBuildResult,
    SourceDataset,
    SplitConfig,
    TrainingRole,
    TrainingRunSpec,
    ValidationReport,
)
from .dataset_builders import merge_obb_sources, prepare_role_dataset
from .dataset_inspector import inspect_obb_or_detect_dataset
from .model_publish import publish_trained_model
from .registry import (
    create_run_record,
    dataset_fingerprint,
    finalize_run_record,
    new_run_id,
)
from .runner import run_training
from .validation import validate_obb_dataset


@dataclass(slots=True)
class RoleRunConfig:
    """Role-specific training config values."""

    role: TrainingRole
    enabled: bool = True
    base_model: str = ""
    size: str = "26s"
    species: str = "species"
    model_info: str = "model"


@dataclass(slots=True)
class TrainingSessionResult:
    """Session result summary for UI."""

    merged_dataset: str = ""  # noqa: DC01  (dataclass field)

    role_dataset_dirs: dict[str, str] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)  # noqa: DC01  (dataclass field)
    published_models: dict[str, str] = field(
        default_factory=dict
    )  # noqa: DC01  (dataclass field)
    errors: list[str] = field(default_factory=list)  # noqa: DC01  (dataclass field)


class TrainingOrchestrator:
    """Coordinates validation, dataset derivation, run registry, and publishing."""

    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def preflight_obb_sources(
        self,
        sources: list[SourceDataset],
        *,
        require_train_val: bool = False,
    ) -> ValidationReport:
        """Validate OBB/detect source datasets, checking splits, class IDs, and annotation integrity."""
        all_issues = []
        stats = {"sources": []}

        for src in sources:
            inspection = inspect_obb_or_detect_dataset(src.path)
            report = validate_obb_dataset(
                inspection,
                require_train_val=require_train_val,
                min_train=1,
                min_val=1,
            )
            stats["sources"].append(
                {
                    "path": src.path,
                    "valid": report.valid,
                    "split_counts": report.stats.get("split_counts", {}),
                    "class_ids": report.stats.get("class_ids", []),
                }
            )
            all_issues.extend(report.issues)

        valid = not any(i.severity == "error" for i in all_issues)
        return ValidationReport(valid=valid, issues=all_issues, stats=stats)

    def build_merged_obb_dataset(
        self,
        sources: list[SourceDataset],
        *,
        class_name: str,
        split_cfg: SplitConfig,
        seed: int,
        dedup: bool,
    ) -> DatasetBuildResult:
        """Merge multiple OBB source datasets into a single unified dataset with optional deduplication."""
        out_root = self.workspace_root / "datasets"
        out_root.mkdir(parents=True, exist_ok=True)
        return merge_obb_sources(
            sources=sources,
            output_root=out_root,
            class_name=class_name,
            split_cfg=split_cfg,
            seed=seed,
            dedup=dedup,
            remap_single_class=True,
        )

    def build_role_dataset(
        self,
        role: TrainingRole,
        merged_obb_dataset_dir: str,
        *,
        class_name: str,
        crop_pad_ratio: float = 0.15,
        min_crop_size_px: int = 64,
        enforce_square: bool = True,
    ) -> DatasetBuildResult:
        """Derive a role-specific dataset (detect, crop-OBB, classify) from a merged OBB dataset."""
        out_root = self.workspace_root / "derived" / role.value
        out_root.mkdir(parents=True, exist_ok=True)
        return prepare_role_dataset(
            role=role,
            merged_obb_dataset_dir=merged_obb_dataset_dir,
            role_output_root=out_root,
            class_name=class_name,
            crop_pad_ratio=crop_pad_ratio,
            min_crop_size_px=min_crop_size_px,
            enforce_square=enforce_square,
        )

    def run_role_training(
        self,
        spec: TrainingRunSpec,
        *,
        parent_run_id: str = "",
        publish_metadata: dict[str, str] | None = None,
        log_cb: Callable[[str], None] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict:
        """Execute a training run: register in the run registry, train, and optionally publish the model."""
        run_id = new_run_id(spec.role.value)
        run_dir = self.workspace_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ds_fp = dataset_fingerprint(spec.derived_dataset_dir)

        create_run_record(
            spec,
            run_id=run_id,
            run_dir=run_dir,
            dataset_fp=ds_fp,
            parent_run_id=parent_run_id,
        )

        result = run_training(
            spec,
            run_dir,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )
        result["run_id"] = run_id

        if not result.get("success", False):
            finalize_run_record(
                run_id,
                status="failed" if not result.get("canceled") else "canceled",
                command=result.get("command", []),
                metrics_paths=(
                    [result.get("metrics_path", "")]
                    if result.get("metrics_path")
                    else []
                ),
                artifact_paths=(
                    [result.get("artifact_path", "")]
                    if result.get("artifact_path")
                    else []
                ),
                error_message=(
                    "canceled"
                    if result.get("canceled")
                    else f"exit_code={result.get('exit_code', 'unknown')}"
                ),
            )
            return result

        published_key = ""
        published_path = ""
        if spec.publish_policy.auto_import and result.get("artifact_path"):
            meta = publish_metadata or {}
            published_key, published_path = publish_trained_model(
                role=spec.role,
                artifact_path=result["artifact_path"],
                size=str(meta.get("size", "") or "unknown"),
                species=str(meta.get("species", "") or "species"),
                model_info=str(
                    meta.get("model_info", "") or f"{spec.role.value}_{run_id}"
                ),
                trained_from_run_id=run_id,
                dataset_fingerprint=ds_fp,
                base_model=spec.base_model,
                training_params=meta.get("training_params"),
            )

        finalize_run_record(
            run_id,
            status="completed",
            command=result.get("command", []),
            metrics_paths=(
                [result.get("metrics_path", "")] if result.get("metrics_path") else []
            ),
            artifact_paths=(
                [result.get("artifact_path", "")] if result.get("artifact_path") else []
            ),
            published_model_path=published_path,
            published_registry_entry=published_key,
        )

        result["published_registry_key"] = published_key
        result["published_model_path"] = published_path
        result["dataset_fingerprint"] = ds_fp
        return result
