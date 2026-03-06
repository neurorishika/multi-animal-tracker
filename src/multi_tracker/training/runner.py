"""Training execution utilities for MAT multi-role training."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .contracts import TrainingRole, TrainingRunSpec

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]


def _safe_log(cb: LogCallback | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def _ultralytics_task_for_role(role: TrainingRole) -> str:
    if role in (TrainingRole.OBB_DIRECT, TrainingRole.SEQ_CROP_OBB):
        return "obb"
    if role == TrainingRole.SEQ_DETECT:
        return "detect"
    if role == TrainingRole.HEADTAIL_YOLO:
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


def _iter_classify_samples(dataset_dir: Path, split: str):
    for cls_name, cls_idx in (("head_left", 0), ("head_right", 1)):
        class_dir = dataset_dir / split / cls_name
        if not class_dir.exists():
            continue
        for img in sorted(class_dir.rglob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                yield img, cls_idx


def _train_tiny_headtail(
    spec: TrainingRunSpec,
    run_dir: Path,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict:
    try:
        import cv2
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError(
            f"Tiny head-tail training requires torch/cv2: {exc}"
        ) from exc

    dataset_dir = Path(spec.derived_dataset_dir).expanduser().resolve()
    device = _pick_torch_device(spec.device)
    _safe_log(log_cb, f"Tiny trainer device: {device}")

    train_samples = list(_iter_classify_samples(dataset_dir, "train"))
    val_samples = list(_iter_classify_samples(dataset_dir, "val"))
    if len(train_samples) < 2 or len(val_samples) < 1:
        raise RuntimeError(
            "Head-tail tiny training requires non-empty train/val classify splits."
        )

    input_w = int(spec.tiny_params.input_width)
    input_h = int(spec.tiny_params.input_height)

    class TinyDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[1] != input_w or img.shape[0] != input_h:
                img = cv2.resize(
                    img, (input_w, input_h), interpolation=cv2.INTER_LINEAR
                )
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            y = torch.tensor(float(label), dtype=torch.float32)
            return x, y

    class TinyHeadClassifier(nn.Module):
        def __init__(self, input_size=(128, 64)):
            super().__init__()
            self.input_size = tuple(input_size)
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = TinyHeadClassifier(input_size=(input_w, input_h)).to(device)
    train_loader = DataLoader(
        TinyDataset(train_samples),
        batch_size=max(1, int(spec.tiny_params.batch)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TinyDataset(val_samples),
        batch_size=max(1, int(spec.tiny_params.batch)),
        shuffle=False,
        num_workers=0,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.tiny_params.lr),
        weight_decay=float(spec.tiny_params.weight_decay),
    )

    best_val_acc = -1.0
    best_state = None
    epochs = max(1, int(spec.tiny_params.epochs))
    history = []

    for epoch in range(epochs):
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")

        model.train()
        train_loss = 0.0
        train_n = 0
        for xs, ys in train_loader:
            xs = xs.to(device)
            ys = ys.to(device)
            opt.zero_grad()
            logits = model(xs).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ys)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(ys.shape[0])
            train_n += int(ys.shape[0])

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for xs, ys in val_loader:
                xs = xs.to(device)
                ys = ys.to(device)
                logits = model(xs).squeeze(1)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += int((preds == ys).sum().item())
                val_total += int(ys.shape[0])

        mean_train_loss = train_loss / max(1, train_n)
        val_acc = val_correct / max(1, val_total)
        history.append(
            {"epoch": epoch + 1, "train_loss": mean_train_loss, "val_acc": val_acc}
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        _safe_log(
            log_cb,
            f"tiny epoch {epoch + 1}/{epochs} train_loss={mean_train_loss:.4f} val_acc={val_acc:.4f}",
        )
        if progress_cb:
            progress_cb(epoch + 1, epochs)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = weights_dir / "best.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": [input_w, input_h],
            "best_val_acc": float(best_val_acc),
            "history": history,
        },
        out_ckpt,
    )

    metrics_path = run_dir / "tiny_metrics.json"
    metrics_path.write_text(
        json.dumps({"best_val_acc": float(best_val_acc), "history": history}, indent=2),
        encoding="utf-8",
    )

    return {
        "success": True,
        "artifact_path": str(out_ckpt),
        "metrics_path": str(metrics_path),
        "command": ["tiny_headtail_inprocess"],
        "task": "tiny_headtail",
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

    if spec.role == TrainingRole.HEADTAIL_TINY:
        return _train_tiny_headtail(
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
    return {
        "success": artifact is not None,
        "artifact_path": str(artifact) if artifact is not None else "",
        "metrics_path": str(metrics_path) if metrics_path.exists() else "",
        "command": command,
        "task": _ultralytics_task_for_role(spec.role),
    }
