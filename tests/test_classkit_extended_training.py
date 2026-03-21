"""Tests for ClassKit extended training — Custom CNN models."""

from __future__ import annotations


def test_custom_cnn_params_defaults():
    from multi_tracker.training.contracts import CustomCNNParams

    p = CustomCNNParams()
    assert p.backbone == "tinyclassifier"
    assert p.trainable_layers == 0
    assert p.backbone_lr_scale == 0.1
    assert p.input_size == 224
    assert p.epochs == 50
    assert p.batch == 32
    assert p.lr == 1e-3
    assert p.weight_decay == 1e-2
    assert p.patience == 10
    assert p.label_smoothing == 0.0
    assert p.class_rebalance_mode == "none"
    assert p.class_rebalance_power == 1.0
    assert p.hidden_layers == 1
    assert p.hidden_dim == 64
    assert p.dropout == 0.2
    assert p.input_width == 128
    assert p.input_height == 64


def test_new_training_roles_exist():
    from multi_tracker.training.contracts import TrainingRole

    assert TrainingRole.CLASSIFY_FLAT_CUSTOM.value == "classify_flat_custom"
    assert TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM.value == "classify_multihead_custom"


def test_training_run_spec_has_custom_params():
    from multi_tracker.training.contracts import (
        CustomCNNParams,
        TrainingRole,
        TrainingRunSpec,
    )

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
    )
    assert hasattr(spec, "custom_params")
    assert spec.custom_params is None

    spec2 = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
        custom_params=CustomCNNParams(backbone="convnext_tiny"),
    )
    assert spec2.custom_params.backbone == "convnext_tiny"
