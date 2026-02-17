try:
    from multi_tracker.core.runtime.compute_runtime import (
        CANONICAL_RUNTIMES,
        allowed_runtimes_for_pipelines,
        derive_pose_runtime_settings,
        infer_compute_runtime_from_legacy,
        runtime_label,
    )
except Exception:
    CANONICAL_RUNTIMES = [
        "cpu",
        "mps",
        "cuda",
        "rocm",
        "onnx_cpu",
        "onnx_cuda",
        "onnx_rocm",
        "tensorrt",
    ]

    def runtime_label(runtime: str) -> str:
        return str(runtime or "cpu").strip().upper()

    def allowed_runtimes_for_pipelines(_pipelines):
        return ["cpu"]

    def infer_compute_runtime_from_legacy(
        yolo_device, enable_tensorrt, pose_runtime_flavor
    ):
        if enable_tensorrt:
            return "tensorrt"
        flavor = str(pose_runtime_flavor or "").lower()
        if flavor.startswith("onnx_rocm"):
            return "onnx_rocm"
        if flavor.startswith("onnx_cuda"):
            return "onnx_cuda"
        if flavor.startswith("onnx"):
            return "onnx_cpu"
        return "cpu"

    def derive_pose_runtime_settings(compute_runtime: str, backend_family: str):
        rt = str(compute_runtime or "cpu").strip().lower()
        if rt == "onnx_cpu":
            return {"pose_runtime_flavor": "onnx_cpu", "pose_sleap_device": "cpu"}
        if rt in {"onnx_cuda", "onnx_rocm"}:
            return {"pose_runtime_flavor": rt, "pose_sleap_device": "cuda:0"}
        if rt == "tensorrt":
            return {
                "pose_runtime_flavor": "tensorrt_cuda",
                "pose_sleap_device": "cuda:0",
            }
        return {"pose_runtime_flavor": rt or "cpu", "pose_sleap_device": rt or "cpu"}
