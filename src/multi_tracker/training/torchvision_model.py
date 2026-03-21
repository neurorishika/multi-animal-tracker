"""Torchvision-based classifier: model factory, freezing, ONNX export, checkpoint I/O.

This module is the sole owner of all torchvision backbone construction and
layer-freezing logic for ClassKit's Custom CNN training mode.
All functions are pure Python / PyTorch — no Qt dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tvm

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

TORCHVISION_BACKBONES: dict[str, str] = {
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b7": "efficientnet_b7",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit_b_16": "vit_b_16",
}

# Human-readable labels for the GUI
BACKBONE_DISPLAY_NAMES: dict[str, str] = {
    "convnext_tiny": "ConvNeXt-T",
    "convnext_small": "ConvNeXt-S",
    "convnext_base": "ConvNeXt-B",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
    "efficientnet_b7": "EfficientNet-B7",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
}


def _load_pretrained(backbone: str) -> nn.Module:
    """Load a pretrained torchvision model by backbone key."""
    weights_map = {
        "convnext_tiny": tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small": tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "convnext_base": tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b3": tvm.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "efficientnet_b7": tvm.EfficientNet_B7_Weights.IMAGENET1K_V1,
        "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V1,
        "vit_b_16": tvm.ViT_B_16_Weights.IMAGENET1K_V1,
    }
    factory = getattr(tvm, backbone)
    return factory(weights=weights_map[backbone])


def _replace_head(model: nn.Module, backbone: str, num_classes: int) -> nn.Module:
    """Replace the final classifier head with a new linear layer."""
    if backbone.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone for head replacement: {backbone!r}")
    return model


def get_layer_groups(model: nn.Module, backbone: str) -> list[nn.Module]:
    """Return backbone layer groups in shallow-to-deep order.

    The caller can index from the end to unfreeze the last N groups.
    ConvNeXt and ResNet return exactly 4 groups.
    EfficientNet returns individual feature blocks.
    ViT-B/16 returns individual encoder layers.
    """
    if backbone.startswith("convnext"):
        # features[0]=stem, features[1..6]=stages (interleaved norms+stages)
        # Expose the 4 main ConvNeXt stages: indices 1, 3, 5, 7
        return [model.features[i] for i in [1, 3, 5, 7]]
    elif backbone.startswith("resnet"):
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    elif backbone.startswith("efficientnet"):
        # features is a Sequential; expose as individual blocks
        return list(model.features)
    elif backbone == "vit_b_16":
        return list(model.encoder.layers)
    else:
        raise ValueError(f"Unsupported backbone for layer groups: {backbone!r}")


def freeze_backbone(model: nn.Module, backbone: str, trainable_layers: int) -> None:
    """Freeze/unfreeze backbone parameters according to trainable_layers.

    Args:
        model: Model whose backbone parameters to freeze.
        backbone: Backbone key (used to determine head parameter names).
        trainable_layers: 0=frozen, -1=all, N>0=unfreeze last N groups.
    """
    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: always unfreeze the head
    if backbone.startswith("convnext") or backbone.startswith("efficientnet"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif backbone.startswith("resnet"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif backbone == "vit_b_16":
        for p in model.heads.parameters():
            p.requires_grad = True

    # Step 3: apply backbone unfreezing
    if trainable_layers == -1:
        for p in model.parameters():
            p.requires_grad = True
    elif trainable_layers > 0:
        groups = get_layer_groups(model, backbone)
        for group in groups[-trainable_layers:]:
            for p in group.parameters():
                p.requires_grad = True


def build_torchvision_classifier(
    backbone: str, num_classes: int, trainable_layers: int
) -> nn.Module:
    """Build a pretrained torchvision classifier with a new head.

    Args:
        backbone: One of the keys in TORCHVISION_BACKBONES.
        num_classes: Number of output classes.
        trainable_layers: 0=frozen, -1=all, N=last N groups unfrozen.

    Returns:
        nn.Module in train mode with head replaced and freezing applied.
    """
    model = _load_pretrained(backbone)
    model = _replace_head(model, backbone, num_classes)
    freeze_backbone(model, backbone, trainable_layers)
    return model


def save_torchvision_checkpoint(
    *,
    model: nn.Module,
    arch: str,
    class_names: list[str],
    factor_names: list[str],
    input_size: tuple[int, int],
    best_val_acc: float,
    history: dict[str, Any],
    trainable_layers: int,
    backbone_lr_scale: float,
    path: str | Path,
) -> Path:
    """Save a torchvision model checkpoint in the unified ClassKit .pth format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "arch": arch,
        "class_names": class_names,
        "factor_names": factor_names,
        "input_size": input_size,
        "num_classes": len(class_names),
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "trainable_layers": trainable_layers,
        "backbone_lr_scale": backbone_lr_scale,
    }
    torch.save(ckpt, str(path))
    return path


def load_torchvision_classifier(
    path: str | Path, device: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a torchvision classifier from a unified ClassKit .pth checkpoint.

    Returns:
        (model_in_eval_mode_on_device, full_ckpt_dict)
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    num_classes = ckpt["num_classes"]
    # Reconstruct with trainable_layers=-1 (all), then load state — freezing
    # state is irrelevant after loading weights for inference.
    model = build_torchvision_classifier(arch, num_classes, trainable_layers=-1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def export_torchvision_to_onnx(
    model: nn.Module, ckpt: dict[str, Any], onnx_path: str | Path
) -> Path:
    """Export a torchvision classifier to ONNX format.

    Args:
        model: Model in eval mode.
        ckpt: Checkpoint dict (used for input_size).
        onnx_path: Output path for the .onnx file.

    Returns:
        Path to the exported ONNX file.
    """
    onnx_path = Path(onnx_path)
    h, w = ckpt.get("input_size", (224, 224))
    dummy = torch.zeros(1, 3, h, w)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return onnx_path
