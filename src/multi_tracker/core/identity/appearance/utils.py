from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from multi_tracker.core.identity.runtime_types import (
    AppearanceResult,
    AppearanceRuntimeConfig,
)
from multi_tracker.core.identity.runtime_utils import (
    _artifact_meta_matches,
    _write_artifact_meta,
)
from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE, TENSORRT_AVAILABLE

logger = logging.getLogger(__name__)


def _empty_appearance_result(model_name: str, dimension: int) -> AppearanceResult:
    """Create empty result for failed predictions."""
    return AppearanceResult(
        embedding=None,
        dimension=dimension,
        model_name=model_name,
    )


def _auto_export_timm_model(
    config: AppearanceRuntimeConfig,
    runtime_flavor: str,
    runtime_device: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Export TIMM model to ONNX or TensorRT format.

    Returns:
        (exported_path, embedding_dimension)
    """
    runtime = str(runtime_flavor or "native").strip().lower()
    if runtime not in {"onnx", "tensorrt"}:
        raise RuntimeError(f"Unsupported TIMM export runtime: {runtime}")

    model_name = config.model_name
    if model_name.startswith("timm/"):
        model_name = model_name[5:]

    # Build export path and signature
    export_device = (
        str(runtime_device or config.device or "auto").strip().lower() or "auto"
    )
    sig_blob = (
        f"timm_model={model_name}|runtime={runtime}|"
        f"batch={int(config.batch_size)}|device={export_device}|"
        f"max_side={config.max_image_side}|normalize={config.normalize_embeddings}"
    ).encode("utf-8")
    sig = hashlib.sha1(sig_blob).hexdigest()[:16]

    # Export into <repo_root>/models/TIMM/
    _repo_root = Path(__file__).parents[5]
    export_root = _repo_root / "models" / "TIMM"
    export_root.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    ext = ".onnx" if runtime == "onnx" else ".engine"
    target_path = export_root / f"{safe_model_name}_{sig}{ext}"

    # Check for existing export
    if _artifact_meta_matches(target_path, sig):
        # Load dimension from metadata
        try:
            meta_path = target_path.with_suffix(target_path.suffix + ".meta")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    dimension = meta.get("embedding_dimension", 0)
                    if dimension > 0:
                        return str(target_path.resolve()), dimension
        except Exception:
            pass

    if runtime == "tensorrt" and not (TENSORRT_AVAILABLE and CUDA_AVAILABLE):
        raise RuntimeError(
            "TensorRT runtime requested but TensorRT/CUDA is unavailable."
        )

    logger.info(
        "Exporting TIMM appearance model for %s runtime: %s -> %s",
        runtime,
        model_name,
        target_path,
    )

    # Import dependencies
    try:
        import timm
        import torch
    except ImportError as e:
        raise ImportError("timm and torch required for TIMM export") from e

    # Load model for export
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    # Get data config and resolution
    data_config = timm.data.resolve_model_data_config(model)
    input_size = data_config.get("input_size", (3, 224, 224))
    H, W = input_size[1], input_size[2]

    # Get embedding dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, H, W)
        out = model(dummy)
        embedding_dim = out.shape[1]

    # Export to ONNX
    batch_size = config.batch_size
    dummy_input = torch.randn(batch_size, 3, H, W)

    onnx_path = target_path if runtime == "onnx" else target_path.with_suffix(".onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=(
            {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
            if runtime == "onnx"
            else {}
        ),
        dynamo=False,
    )

    logger.info(f"TIMM model exported to ONNX: {onnx_path}")

    # Convert to TensorRT if requested
    if runtime == "tensorrt":
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt required for TensorRT export")

        logger.info(f"Converting ONNX to TensorRT engine: {target_path}")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(parser.get_error(i))
                raise RuntimeError(f"TensorRT ONNX parsing failed: {errors}")

        config_trt = builder.create_builder_config()
        config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Enable FP16 if available
        if builder.platform_has_fast_fp16:
            config_trt.set_flag(trt.BuilderFlag.FP16)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config_trt)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        with open(target_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine created: {target_path}")

        # Clean up temporary ONNX file
        if onnx_path != target_path and onnx_path.exists():
            onnx_path.unlink()

    # Write metadata
    meta_data = {
        "signature": sig,
        "model_name": model_name,
        "runtime": runtime,
        "embedding_dimension": embedding_dim,
        "batch_size": batch_size,
        "input_height": H,
        "input_width": W,
    }
    meta_path = target_path.with_suffix(target_path.suffix + ".meta")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)

    _write_artifact_meta(target_path, sig)

    return str(target_path.resolve()), embedding_dim
