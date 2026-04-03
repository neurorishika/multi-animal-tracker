"""Training execution utilities for MAT multi-role training."""

from __future__ import annotations

import csv
import dataclasses
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .contracts import CustomCNNParams, TrainingRole, TrainingRunSpec

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]


def _safe_log(cb: LogCallback | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def _looks_like_accuracy_column(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    return ("acc" in low) or ("accuracy" in low)


def _accuracy_column_priority(name: str) -> tuple[int, str]:
    """Lower priority value means a better candidate for val/top-1 accuracy."""
    low = str(name or "").strip().lower()
    if not _looks_like_accuracy_column(low):
        return (99, low)
    if "top1" in low and "metrics" in low:
        return (0, low)
    if "top1" in low:
        return (1, low)
    if "val" in low:
        return (2, low)
    if "metrics" in low:
        return (3, low)
    return (4, low)


def _extract_best_val_acc_from_results_csv(metrics_csv_path: Path) -> float | None:
    """Extract best validation-like accuracy from Ultralytics results.csv."""
    if not metrics_csv_path.exists():
        return None
    try:
        with open(metrics_csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            field_names = list(reader.fieldnames or [])
            candidate_cols = [c for c in field_names if _looks_like_accuracy_column(c)]
            if not candidate_cols:
                return None
            candidate_cols = sorted(candidate_cols, key=_accuracy_column_priority)
            rows = list(reader)
            for col in candidate_cols:
                values: list[float] = []
                for row in rows:
                    raw = row.get(col)
                    if raw is None or str(raw).strip() == "":
                        continue
                    try:
                        values.append(float(raw))
                    except Exception:
                        continue
                if values:
                    return max(values)
            return None
    except Exception:
        return None


def _ultralytics_task_for_role(role: TrainingRole) -> str:
    if role in (TrainingRole.OBB_DIRECT, TrainingRole.SEQ_CROP_OBB):
        return "obb"
    if role == TrainingRole.SEQ_DETECT:
        return "detect"
    if role in (
        TrainingRole.CLASSIFY_FLAT_YOLO,
        TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
    ):
        return "classify"
    raise RuntimeError(f"Role {role.value} does not map to ultralytics CLI task")


def build_ultralytics_command(spec: TrainingRunSpec, run_dir: str | Path) -> list[str]:
    """Build deterministic Ultralytics train command for a role."""

    task = _ultralytics_task_for_role(spec.role)
    run_dir = Path(run_dir).expanduser().resolve()

    args = [
        task,
        "train",
        f"data={spec.derived_dataset_dir}",
        f"model={spec.base_model}",
        f"epochs={int(spec.hyperparams.epochs)}",
        f"imgsz={int(spec.hyperparams.imgsz)}",
        f"batch={int(spec.hyperparams.batch)}",
        f"lr0={float(spec.hyperparams.lr0)}",
        f"patience={int(spec.hyperparams.patience)}",
        f"workers={int(spec.hyperparams.workers)}",
        f"project={str(run_dir.parent)}",
        f"name={run_dir.name}",
        "exist_ok=True",
        f"seed={int(spec.seed)}",
    ]
    if spec.device and spec.device != "auto":
        args.append(f"device={spec.device}")
    if spec.hyperparams.cache:
        args.append("cache=True")
    if spec.augmentation_profile.enabled and spec.augmentation_profile.args:
        for k, v in sorted(spec.augmentation_profile.args.items()):
            args.append(f"{k}={v}")
    if spec.resume_from:
        args.append("resume=True")

    yolo_exe = shutil.which("yolo")
    if yolo_exe:
        return [yolo_exe, *args]
    return [sys.executable, "-m", "ultralytics", *args]


def _pick_torch_device(requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    try:
        import torch
    except Exception:
        return "cpu"

    if requested not in {"", "auto"}:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cpu":
            return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_class_to_idx(dataset_dir: Path) -> dict[str, int]:
    """Build a stable class-name→index mapping across all classify splits.

    Using one shared mapping for train/val/test prevents label-index drift when a
    split is missing one or more classes.
    """
    names: set[str] = set()
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                names.add(cls_dir.name)
    ordered = sorted(names)
    return {name: idx for idx, name in enumerate(ordered)}


def _iter_classify_samples(
    dataset_dir: Path, split: str, class_to_idx: dict[str, int] | None = None
):
    """Yield ``(image_path, class_index)`` for all images in a classify split."""
    split_dir = dataset_dir / split
    if not split_dir.exists():
        return
    if class_to_idx is None:
        class_to_idx = _build_class_to_idx(dataset_dir)
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    for cls_dir in class_dirs:
        cls_idx = class_to_idx.get(cls_dir.name)
        if cls_idx is None:
            continue
        for img in sorted(cls_dir.rglob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                yield img, cls_idx


def _train_tiny_classify(
    spec: TrainingRunSpec,
    run_dir: Path,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict:
    """Train a tiny N-class CNN classifier from an image-folder dataset."""
    try:
        import cv2
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    except Exception as exc:
        raise RuntimeError(f"Tiny classify training requires torch/cv2: {exc}") from exc

    dataset_dir = Path(spec.derived_dataset_dir).expanduser().resolve()
    device = _pick_torch_device(spec.device)
    _safe_log(log_cb, f"Tiny classify device: {device}")

    class_to_idx = _build_class_to_idx(dataset_dir)
    train_samples = list(_iter_classify_samples(dataset_dir, "train", class_to_idx))
    val_samples = list(_iter_classify_samples(dataset_dir, "val", class_to_idx))
    if len(train_samples) < 2:
        raise RuntimeError("Tiny classify training requires at least 2 train samples.")

    num_classes = len(class_to_idx)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes in train split.")
    train_label_set = {int(lbl) for _, lbl in train_samples}
    if len(train_label_set) < 2:
        raise RuntimeError(
            "Need at least 2 classes represented in train split for tiny classify."
        )

    input_w = int(spec.tiny_params.input_width)
    input_h = int(spec.tiny_params.input_height)
    rebalance_mode = (
        str(getattr(spec.tiny_params, "class_rebalance_mode", "none") or "none")
        .strip()
        .lower()
    )
    rebalance_power = float(
        max(0.0, getattr(spec.tiny_params, "class_rebalance_power", 1.0))
    )
    label_smoothing = float(
        min(0.4, max(0.0, getattr(spec.tiny_params, "label_smoothing", 0.0)))
    )

    class_counts = [0] * num_classes
    for _p, lbl in train_samples:
        if 0 <= int(lbl) < num_classes:
            class_counts[int(lbl)] += 1

    # Inverse-frequency reweighting; power controls strength.
    class_weight_values = [1.0] * num_classes
    if rebalance_mode in {"weighted_loss", "weighted_sampler", "both"}:
        max_count = max(class_counts) if class_counts else 1
        for idx in range(num_classes):
            count = max(1, class_counts[idx])
            class_weight_values[idx] = float(max_count / count) ** rebalance_power
        mean_w = sum(class_weight_values) / max(1, len(class_weight_values))
        if mean_w > 0:
            class_weight_values = [w / mean_w for w in class_weight_values]

    _safe_log(
        log_cb,
        "tiny classify options: "
        f"rebalance={rebalance_mode}, power={rebalance_power:.2f}, "
        f"label_smoothing={label_smoothing:.2f}",
    )

    class TinyDataset(Dataset):
        def __init__(self, items, augment=False, profile=None):
            self.items = items
            self.augment = augment
            self.profile = profile

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.augment and self.profile and self.profile.enabled:
                # Flip UD
                if self.profile.flipud > 0 and random.random() < self.profile.flipud:
                    img = cv2.flip(img, 0)
                # Flip LR
                if self.profile.fliplr > 0 and random.random() < self.profile.fliplr:
                    img = cv2.flip(img, 1)
                # Rotation
                if self.profile.rotate > 0:
                    angle = random.uniform(-self.profile.rotate, self.profile.rotate)
                    M = cv2.getRotationMatrix2D(
                        (img.shape[1] // 2, img.shape[0] // 2), angle, 1.0
                    )
                    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            if img.shape[1] != input_w or img.shape[0] != input_h:
                img = cv2.resize(
                    img, (input_w, input_h), interpolation=cv2.INTER_LINEAR
                )
            x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            y = torch.tensor(label, dtype=torch.long)
            return x, y

    from .tiny_model import _build_tiny_classifier_class

    _TinyClassifier = _build_tiny_classifier_class()

    class _TinyClassifierCompat(_TinyClassifier):
        """Thin wrapper accepting a params object instead of kwargs."""

        def __init__(self, n_classes, params):
            super().__init__(
                n_classes=n_classes,
                hidden_layers=params.hidden_layers,
                hidden_dim=params.hidden_dim,
                dropout=params.dropout,
            )

    model = _TinyClassifierCompat(num_classes, spec.tiny_params).to(device)
    train_sampler = None
    if rebalance_mode in {"weighted_sampler", "both"}:
        sample_weights = [class_weight_values[int(lbl)] for _p, lbl in train_samples]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
    train_loader = DataLoader(
        TinyDataset(train_samples, augment=True, profile=spec.augmentation_profile),
        batch_size=max(1, int(spec.tiny_params.batch)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            TinyDataset(val_samples, augment=False),
            batch_size=max(1, int(spec.tiny_params.batch)),
            shuffle=False,
            num_workers=0,
        )
        if val_samples
        else None
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.tiny_params.lr),
        weight_decay=float(spec.tiny_params.weight_decay),
    )
    class_weight_tensor = None
    if rebalance_mode in {"weighted_loss", "both"}:
        class_weight_tensor = torch.as_tensor(
            class_weight_values, dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=label_smoothing,
    )

    best_val_acc = -1.0
    best_state = None
    epochs = max(1, int(spec.tiny_params.epochs))
    patience = max(1, int(spec.tiny_params.patience))
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")

        model.train()
        train_loss, train_n = 0.0, 0
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            opt.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(ys)
            train_n += len(ys)

        val_acc = 0.0
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.inference_mode():
                for xs, ys in val_loader:
                    xs, ys = xs.to(device), ys.to(device)
                    preds = model(xs).argmax(dim=1)
                    correct += int((preds == ys).sum().item())
                    total += len(ys)
            val_acc = correct / max(1, total)

        mean_loss = train_loss / max(1, train_n)
        history.append(
            {"epoch": epoch + 1, "train_loss": mean_loss, "val_acc": val_acc}
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        _safe_log(
            log_cb,
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f} val_acc={val_acc:.4f}",
        )
        if progress_cb:
            progress_cb(epoch + 1, epochs)

        if val_loader and patience_counter >= patience:
            _safe_log(log_cb, f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive filename: classkit_{role}_{N}cls_{classes}_{rundir_stem}.pth
    _role_slug = (
        spec.role.value.replace("classify_", "")
        .replace("_tiny", "_tiny")
        .replace("_yolo", "_yolo")
    )
    _class_slug = "-".join(
        name for name, _idx in sorted(class_to_idx.items(), key=lambda kv: int(kv[1]))
    )
    if len(_class_slug) > 48:
        _class_slug = f"{num_classes}cls"
    _run_stem = (
        run_dir.parent.name
    )  # e.g. flat_tiny_20260309_123456 / run → parent stem
    _model_filename = f"classkit_{_role_slug}_{_class_slug}_{_run_stem}.pth"
    out_ckpt = weights_dir / _model_filename

    import json as _json

    _ckpt_dict = {
        "model_state_dict": model.state_dict(),
        "input_size": [input_w, input_h],
        "num_classes": num_classes,
        "class_names": [
            name
            for name, _idx in sorted(class_to_idx.items(), key=lambda kv: int(kv[1]))
        ],
        "best_val_acc": float(best_val_acc),
        "history": history,
    }
    torch.save(_ckpt_dict, out_ckpt)

    # Auto-export ONNX alongside .pth for runtime-flexible inference (ONNX/TensorRT).
    _onnx_path: Path | None = None
    try:
        from hydra_suite.training.tiny_model import export_tiny_to_onnx

        _onnx_candidate = out_ckpt.with_suffix(".onnx")
        export_tiny_to_onnx(model, _ckpt_dict, _onnx_candidate)
        _onnx_path = _onnx_candidate
        _safe_log(log_cb, f"ONNX exported: {_onnx_candidate.name}")
    except Exception as _onnx_exc:
        _safe_log(
            log_cb, f"ONNX export skipped ({type(_onnx_exc).__name__}: {_onnx_exc})"
        )

    metrics_path = run_dir / "tiny_metrics.json"
    metrics_path.write_text(
        _json.dumps(
            {"best_val_acc": float(best_val_acc), "history": history}, indent=2
        ),
        encoding="utf-8",
    )

    return {
        "success": True,
        "artifact_path": str(out_ckpt),
        "onnx_path": str(_onnx_path) if _onnx_path is not None else "",
        "metrics_path": str(metrics_path),
        "best_val_acc": float(best_val_acc),
        "command": ["tiny_classify_inprocess"],
        "task": "tiny_classify",
    }


def _train_custom_classify(
    spec: "TrainingRunSpec",
    run_dir: Path,
    log_cb=None,
    progress_cb=None,
    should_cancel=None,
) -> dict:
    """Train a Custom CNN classifier (TinyClassifier or torchvision backbone).

    If backbone == 'tinyclassifier', delegates entirely to _train_tiny_classify().
    Otherwise trains a pretrained torchvision model with discriminative LR.
    """
    from .torchvision_model import (
        build_torchvision_classifier,
        export_torchvision_to_onnx,
        load_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    params: CustomCNNParams = spec.custom_params or CustomCNNParams()

    if params.backbone == "tinyclassifier":
        return _train_tiny_classify(spec, run_dir, log_cb, progress_cb, should_cancel)

    # --- Torchvision training path ---
    import json

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    # Build dataset
    dataset_dir = Path(spec.derived_dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    sz = params.input_size

    train_tf = transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    class_names = train_ds.classes

    train_loader = DataLoader(
        train_ds, batch_size=params.batch, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=params.batch, shuffle=False, num_workers=0, pin_memory=True
    )

    # _pick_torch_device is defined in runner.py and handles "auto", MPS, CUDA fallback
    device = _pick_torch_device(spec.device)
    model = build_torchvision_classifier(
        params.backbone, len(class_names), params.trainable_layers
    )
    model.to(device)

    # Discriminative LR: backbone params at reduced LR, head at full LR
    head_params, backbone_params = [], []
    head_names = {"classifier", "fc", "heads"}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(h) for h in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = [{"params": head_params, "lr": params.lr}]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": params.lr * params.backbone_lr_scale}
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=params.weight_decay)
    # NOTE: params.class_rebalance_mode and class_rebalance_power are stored
    # in CustomCNNParams but not yet applied in the torchvision training path
    # (only label_smoothing is passed to CrossEntropyLoss). A WeightedRandomSampler
    # or loss weighting implementation can be added here in a future task.
    criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)

    best_val_acc = 0.0
    patience_count = 0
    history: dict = {"train_loss": [], "val_acc": []}
    best_ckpt_path = (
        weights_dir / f"classkit_custom_{params.backbone}_{len(class_names)}cls.pth"
    )

    for epoch in range(params.epochs):
        if should_cancel and should_cancel():
            _log("Training canceled.")
            break

        # Training
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        val_acc = correct / max(total, 1)
        history["val_acc"].append(val_acc)

        _log(
            f"Epoch {epoch + 1}/{params.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if progress_cb:
            progress_cb(epoch + 1, params.epochs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            save_torchvision_checkpoint(
                model=model,
                backbone=params.backbone,
                class_names=class_names,
                factor_names=[],
                input_size=(sz, sz),
                best_val_acc=best_val_acc,
                history=history,
                trainable_layers=params.trainable_layers,
                backbone_lr_scale=params.backbone_lr_scale,
                path=best_ckpt_path,
            )
        else:
            patience_count += 1
            if patience_count >= params.patience:
                _log(f"Early stopping at epoch {epoch + 1}.")
                break

    if not best_ckpt_path.exists():
        _log("No checkpoint saved (training canceled or val acc never improved).")
        metrics_path = run_dir / "custom_metrics.json"
        metrics_path.write_text(
            json.dumps({"best_val_acc": best_val_acc, "history": history}, indent=2)
        )
        return {
            "success": False,
            "artifact_path": "",
            "onnx_path": "",
            "metrics_path": str(metrics_path),
            "best_val_acc": best_val_acc,
            "command": ["custom_classify_inprocess"],
            "task": "custom_classify",
        }

    # ONNX export
    best_model, best_ckpt = load_torchvision_classifier(
        str(best_ckpt_path), device="cpu"
    )
    onnx_path = best_ckpt_path.with_suffix(".onnx")
    export_torchvision_to_onnx(best_model, best_ckpt, onnx_path)

    # Metrics
    metrics_path = run_dir / "custom_metrics.json"
    metrics_path.write_text(
        json.dumps({"best_val_acc": best_val_acc, "history": history}, indent=2)
    )
    _log(f"Training complete. Best val acc: {best_val_acc:.4f}")

    return {
        "success": True,
        "artifact_path": str(best_ckpt_path),
        "onnx_path": str(onnx_path),
        "metrics_path": str(metrics_path),
        "best_val_acc": best_val_acc,
        "command": ["custom_classify_inprocess"],
        "task": "custom_classify",
    }


def run_training(
    spec: TrainingRunSpec,
    run_dir: str | Path,
    *,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict:
    """Run one role training job and return structured result."""

    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if spec.role in (
        TrainingRole.CLASSIFY_FLAT_CUSTOM,
        TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
        TrainingRole.CLASSIFY_FLAT_TINY,
        TrainingRole.CLASSIFY_MULTIHEAD_TINY,
    ):
        # Ensure custom_params is populated; alias roles (flat_tiny, multihead_tiny)
        # inject tinyclassifier default so _train_custom_classify can dispatch correctly.
        if spec.custom_params is None:
            spec = dataclasses.replace(
                spec, custom_params=CustomCNNParams(backbone="tinyclassifier")
            )
        return _train_custom_classify(
            spec,
            run_dir,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )

    command = build_ultralytics_command(spec, run_dir)
    _safe_log(log_cb, "Running: " + " ".join(command))

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    progress_re_1 = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    progress_re_2 = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s")

    for line in proc.stdout:
        if should_cancel and should_cancel():
            try:
                proc.terminate()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            return {
                "success": False,
                "canceled": True,
                "artifact_path": "",
                "metrics_path": "",
                "command": command,
            }

        msg = ansi_re.sub("", line).rstrip()
        if msg:
            _safe_log(log_cb, msg)
        for regex in (progress_re_1, progress_re_2):
            m = regex.search(msg)
            if m and progress_cb is not None:
                try:
                    progress_cb(int(m.group(1)), int(m.group(2)))
                except Exception:
                    pass
                break

    rc = proc.wait()
    if rc != 0:
        return {
            "success": False,
            "exit_code": int(rc),
            "artifact_path": "",
            "metrics_path": "",
            "command": command,
        }

    weights_dir = run_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    artifact = best if best.exists() else last if last.exists() else None

    metrics_path = run_dir / "results.csv"
    best_val_acc = _extract_best_val_acc_from_results_csv(metrics_path)
    return {
        "success": artifact is not None,
        "artifact_path": str(artifact) if artifact is not None else "",
        "metrics_path": str(metrics_path) if metrics_path.exists() else "",
        "best_val_acc": best_val_acc,
        "command": command,
        "task": _ultralytics_task_for_role(spec.role),
    }
