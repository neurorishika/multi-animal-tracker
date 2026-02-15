from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


@contextmanager
def _patched_modules(stubs: dict):
    sentinel = object()
    original = {}
    try:
        for name, stub in stubs.items():
            original[name] = sys.modules.get(name, sentinel)
            sys.modules[name] = stub
        yield
    finally:
        for name, old in original.items():
            if old is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_runtime_api_module(stubs: dict):
    module_name = "runtime_api_sleap_export_test"
    module_path = SRC_ROOT / "multi_tracker" / "core" / "identity" / "runtime_api.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _patched_modules(stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def test_sleap_export_backend_onnx_predicts_canonical_output(tmp_path: Path) -> None:
    calls = []

    class _FakeMetadata:
        node_names = ["head", "thorax", "abdomen"]
        crop_size = [32, 32]
        input_channels = 3

    class _FakePredictor:
        def predict(self, batch):
            if isinstance(batch, list):
                n = len(batch)
            else:
                n = int(np.asarray(batch).shape[0])
            xy = np.zeros((n, 1, 3, 2), dtype=np.float32)
            conf = np.zeros((n, 1, 3), dtype=np.float32)
            for i in range(n):
                xy[i, 0, :, :] = np.array(
                    [[10.0 + i, 20.0], [30.0 + i, 40.0], [50.0 + i, 60.0]],
                    dtype=np.float32,
                )
                conf[i, 0, :] = np.array([0.9, 0.7, 0.1], dtype=np.float32)
            return {"instance_peaks": xy, "instance_peak_vals": conf}

        def close(self):
            return None

    def _fake_load_exported_model(path, **kwargs):
        calls.append((path, kwargs))
        return _FakePredictor()

    def _fake_load_metadata(_path):
        return _FakeMetadata()

    predictors_mod = types.ModuleType("sleap_nn.export.predictors")
    predictors_mod.load_exported_model = _fake_load_exported_model
    metadata_mod = types.ModuleType("sleap_nn.export.metadata")
    metadata_mod.load_metadata = _fake_load_metadata
    export_mod = types.ModuleType("sleap_nn.export")
    export_mod.predictors = predictors_mod
    export_mod.metadata = metadata_mod
    sleap_nn_mod = types.ModuleType("sleap_nn")
    sleap_nn_mod.export = export_mod

    stubs = {
        "multi_tracker.utils.gpu_utils": types.SimpleNamespace(
            CUDA_AVAILABLE=False,
            MPS_AVAILABLE=False,
            ONNXRUNTIME_AVAILABLE=True,
            ROCM_AVAILABLE=False,
            TENSORRT_AVAILABLE=False,
            TORCH_CUDA_AVAILABLE=False,
            SLEAP_RUNTIME_ONNX_AVAILABLE=True,
            SLEAP_RUNTIME_TENSORRT_AVAILABLE=False,
        ),
        "sleap_nn": sleap_nn_mod,
        "sleap_nn.export": export_mod,
        "sleap_nn.export.predictors": predictors_mod,
        "sleap_nn.export.metadata": metadata_mod,
    }
    mod = _load_runtime_api_module(stubs)

    export_dir = tmp_path / "sleap_exported"
    export_dir.mkdir()
    cfg = mod.PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor="onnx",
        device="cpu",
        model_path=str(tmp_path / "unused_training_model"),
        exported_model_path=str(export_dir),
        min_valid_conf=0.2,
        keypoint_names=["k1", "k2", "k3"],
        sleap_batch=4,
        sleap_max_instances=1,
    )

    with _patched_modules(stubs):
        backend = mod.create_pose_backend_from_config(cfg)
        assert isinstance(backend, mod.SleapExportBackend)
        # Metadata node names should become output keypoint names.
        assert backend.output_keypoint_names == ["head", "thorax", "abdomen"]
        assert calls
        assert calls[0][0] == str(export_dir.resolve())

        crops = [
            np.zeros((24, 24, 3), dtype=np.uint8),
            np.zeros((26, 26, 3), dtype=np.uint8),
        ]
        out = backend.predict_batch(crops)
        assert len(out) == 2
        assert out[0].num_keypoints == 3
        assert out[0].num_valid == 2
        assert np.isclose(out[0].mean_conf, (0.9 + 0.7 + 0.1) / 3.0)
        assert out[0].keypoints.shape == (3, 3)
        backend.close()


def test_sleap_export_backend_falls_back_to_service_when_unavailable(
    tmp_path: Path,
) -> None:
    class _FakePoseInferenceService:
        def __init__(self, out_root, keypoint_names, skeleton_edges=None):
            self.out_root = Path(out_root)
            self.keypoint_names = list(keypoint_names)
            self.skeleton_edges = list(skeleton_edges or [])

        @classmethod
        def sleap_service_running(cls):
            return False

        @classmethod
        def start_sleap_service(cls, env_name, out_root):
            return True, "", out_root / "log.txt"

        @classmethod
        def shutdown_sleap_service(cls):
            return None

        def predict(self, model_path, image_paths, **_kwargs):
            preds = {}
            for p in image_paths:
                preds[str(p)] = [(1.0, 2.0, 0.9) for _ in self.keypoint_names]
            return preds, ""

    def _raise_load_exported_model(*_args, **_kwargs):
        raise RuntimeError("export backend unavailable")

    predictors_mod = types.ModuleType("sleap_nn.export.predictors")
    predictors_mod.load_exported_model = _raise_load_exported_model
    metadata_mod = types.ModuleType("sleap_nn.export.metadata")
    metadata_mod.load_metadata = lambda _p: {}
    export_mod = types.ModuleType("sleap_nn.export")
    export_mod.predictors = predictors_mod
    export_mod.metadata = metadata_mod
    sleap_nn_mod = types.ModuleType("sleap_nn")
    sleap_nn_mod.export = export_mod

    stubs = {
        "multi_tracker.utils.gpu_utils": types.SimpleNamespace(
            CUDA_AVAILABLE=False,
            MPS_AVAILABLE=False,
            ONNXRUNTIME_AVAILABLE=True,
            ROCM_AVAILABLE=False,
            TENSORRT_AVAILABLE=False,
            TORCH_CUDA_AVAILABLE=False,
            SLEAP_RUNTIME_ONNX_AVAILABLE=True,
            SLEAP_RUNTIME_TENSORRT_AVAILABLE=False,
        ),
        "sleap_nn": sleap_nn_mod,
        "sleap_nn.export": export_mod,
        "sleap_nn.export.predictors": predictors_mod,
        "sleap_nn.export.metadata": metadata_mod,
        "multi_tracker.posekit.pose_inference": types.SimpleNamespace(
            PoseInferenceService=_FakePoseInferenceService
        ),
    }
    mod = _load_runtime_api_module(stubs)

    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir()
    export_dir = tmp_path / "sleap_exported"
    export_dir.mkdir()
    cfg = mod.PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor="onnx",
        device="cpu",
        model_path=str(model_dir),
        exported_model_path=str(export_dir),
        min_valid_conf=0.2,
        keypoint_names=["head", "thorax", "abdomen"],
        sleap_batch=2,
        sleap_max_instances=1,
    )

    with _patched_modules(stubs):
        backend = mod.create_pose_backend_from_config(cfg)
        assert isinstance(backend, mod.SleapServiceBackend)
        backend.close()


def test_yolo_auto_runtime_exports_onnx_and_reuses_cached_artifact(
    tmp_path: Path,
) -> None:
    export_calls = []

    class _FakeYOLO:
        def __init__(self, path):
            self.path = str(path)

        def to(self, _device):
            return self

        def export(self, **kwargs):
            export_calls.append(kwargs)
            out = tmp_path / "tmp_export.onnx"
            out.write_bytes(b"onnx-bytes")
            return str(out)

    stubs = {
        "multi_tracker.utils.gpu_utils": types.SimpleNamespace(
            CUDA_AVAILABLE=False,
            MPS_AVAILABLE=False,
            ONNXRUNTIME_AVAILABLE=True,
            ROCM_AVAILABLE=False,
            TENSORRT_AVAILABLE=False,
            TORCH_CUDA_AVAILABLE=False,
            SLEAP_RUNTIME_ONNX_AVAILABLE=False,
            SLEAP_RUNTIME_TENSORRT_AVAILABLE=False,
        ),
        "ultralytics": types.SimpleNamespace(YOLO=_FakeYOLO),
    }
    mod = _load_runtime_api_module(stubs)

    model_path = tmp_path / "pose.pt"
    model_path.write_bytes(b"pt-bytes")
    cfg = mod.PoseRuntimeConfig(
        backend_family="yolo",
        runtime_flavor="auto",
        device="cpu",
        model_path=str(model_path),
        out_root=str(tmp_path),
        yolo_batch=4,
        keypoint_names=["head", "tail"],
    )

    with _patched_modules(stubs):
        b1 = mod.create_pose_backend_from_config(cfg)
        p1 = Path(b1.model_path)
        assert p1.suffix == ".onnx"
        assert p1.exists()
        b1.close()

        b2 = mod.create_pose_backend_from_config(cfg)
        p2 = Path(b2.model_path)
        b2.close()

    assert p1 == p2
    assert len(export_calls) == 1


def test_yolo_auto_runtime_reexports_when_model_changes(tmp_path: Path) -> None:
    export_calls = []

    class _FakeYOLO:
        def __init__(self, path):
            self.path = str(path)

        def to(self, _device):
            return self

        def export(self, **kwargs):
            export_calls.append(kwargs)
            out = tmp_path / f"tmp_export_{len(export_calls)}.onnx"
            out.write_bytes(f"onnx-{len(export_calls)}".encode("utf-8"))
            return str(out)

    stubs = {
        "multi_tracker.utils.gpu_utils": types.SimpleNamespace(
            CUDA_AVAILABLE=False,
            MPS_AVAILABLE=False,
            ONNXRUNTIME_AVAILABLE=True,
            ROCM_AVAILABLE=False,
            TENSORRT_AVAILABLE=False,
            TORCH_CUDA_AVAILABLE=False,
            SLEAP_RUNTIME_ONNX_AVAILABLE=False,
            SLEAP_RUNTIME_TENSORRT_AVAILABLE=False,
        ),
        "ultralytics": types.SimpleNamespace(YOLO=_FakeYOLO),
    }
    mod = _load_runtime_api_module(stubs)

    model_path = tmp_path / "pose.pt"
    model_path.write_bytes(b"pt-v1")
    cfg = mod.PoseRuntimeConfig(
        backend_family="yolo",
        runtime_flavor="auto",
        device="cpu",
        model_path=str(model_path),
        out_root=str(tmp_path),
        yolo_batch=4,
        keypoint_names=["head", "tail"],
    )

    with _patched_modules(stubs):
        b1 = mod.create_pose_backend_from_config(cfg)
        p1 = Path(b1.model_path)
        b1.close()

        # Change source model fingerprint.
        model_path.write_bytes(b"pt-v2-with-different-size")

        b2 = mod.create_pose_backend_from_config(cfg)
        p2 = Path(b2.model_path)
        b2.close()

    assert p1 != p2
    assert len(export_calls) >= 2


def test_yolo_backend_rejects_directory_model_path_with_clear_error(
    tmp_path: Path,
) -> None:
    stubs = {
        "multi_tracker.utils.gpu_utils": types.SimpleNamespace(
            CUDA_AVAILABLE=False,
            MPS_AVAILABLE=False,
            ONNXRUNTIME_AVAILABLE=True,
            ROCM_AVAILABLE=False,
            TENSORRT_AVAILABLE=False,
            TORCH_CUDA_AVAILABLE=False,
            SLEAP_RUNTIME_ONNX_AVAILABLE=False,
            SLEAP_RUNTIME_TENSORRT_AVAILABLE=False,
        ),
        "ultralytics": types.SimpleNamespace(YOLO=lambda *_a, **_k: None),
    }
    mod = _load_runtime_api_module(stubs)

    sleap_like_dir = tmp_path / "sleap_model_dir"
    sleap_like_dir.mkdir()
    cfg = mod.PoseRuntimeConfig(
        backend_family="yolo",
        runtime_flavor="native",
        device="cpu",
        model_path=str(sleap_like_dir),
        out_root=str(tmp_path),
        keypoint_names=["head", "tail"],
    )

    with _patched_modules(stubs):
        try:
            mod.create_pose_backend_from_config(cfg)
            raise AssertionError(
                "Expected RuntimeError for directory model path under yolo backend."
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            assert "pose_model_type" in msg
            assert "sleap" in msg
