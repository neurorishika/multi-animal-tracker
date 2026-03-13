from __future__ import annotations

import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _gpu_stub(**overrides):
    def _identity_decorator(*_args, **_kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    base = {
        "CUDA_AVAILABLE": False,
        "MPS_AVAILABLE": False,
        "ONNXRUNTIME_AVAILABLE": True,
        "ONNXRUNTIME_PROVIDERS": ["CPUExecutionProvider"],
        "ONNXRUNTIME_CPU_AVAILABLE": True,
        "ONNXRUNTIME_CUDA_AVAILABLE": False,
        "ONNXRUNTIME_ROCM_AVAILABLE": False,
        "ROCM_AVAILABLE": False,
        "TENSORRT_AVAILABLE": False,
        "TORCH_CUDA_AVAILABLE": False,
        "SLEAP_RUNTIME_ONNX_AVAILABLE": True,
        "SLEAP_RUNTIME_TENSORRT_AVAILABLE": False,
        "NUMBA_AVAILABLE": False,
        "GPU_AVAILABLE": False,
        "ANY_ACCELERATION": False,
        "CUPY_AVAILABLE": False,
        "TORCH_AVAILABLE": False,
        "F": None,
        "cp": None,
        "cupy_ndimage": None,
        "njit": _identity_decorator,
        "prange": range,
        "torch": None,
        "get_device_info": lambda: {},
        "log_device_info": lambda: None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


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
            return True, "", Path(out_root) / "log.txt"

        @classmethod
        def shutdown_sleap_service(cls):
            return None

        def predict(self, model_path, image_paths, **kwargs):
            calls.append(
                {
                    "model_path": str(model_path),
                    "kwargs": dict(kwargs),
                    "count": len(image_paths),
                }
            )
            preds = {}
            for i, p in enumerate(image_paths):
                preds[str(p)] = [
                    (10.0 + i, 20.0, 0.9),
                    (30.0 + i, 40.0, 0.7),
                    (50.0 + i, 60.0, 0.1),
                ]
            return preds, ""

    stubs = {
        "multi_tracker.utils.gpu_utils": _gpu_stub(),
        "multi_tracker.posekit.pose_inference": types.SimpleNamespace(
            PoseInferenceService=_FakePoseInferenceService
        ),
    }
    mod = _load_runtime_api_module(stubs)

    export_dir = tmp_path / "sleap_exported"
    export_dir.mkdir()
    (export_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.onnx").write_bytes(b"fake-onnx")
    cfg = mod.PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor="onnx",
        device="cpu",
        model_path=str(export_dir),
        exported_model_path=str(export_dir),
        min_valid_conf=0.2,
        keypoint_names=["k1", "k2", "k3"],
        sleap_batch=4,
        sleap_max_instances=1,
        sleap_experimental_features=True,
    )

    with _patched_modules(stubs):
        backend = mod.create_pose_backend_from_config(cfg)
        assert isinstance(backend, mod.SleapServiceBackend)

        crops = [
            np.zeros((24, 24, 3), dtype=np.uint8),
            np.zeros((26, 26, 3), dtype=np.uint8),
        ]
        out = backend.predict_batch(crops)
        assert calls
        assert calls[0]["kwargs"]["sleap_runtime_flavor"] == "onnx"
        assert calls[0]["kwargs"]["sleap_exported_model_path"] == str(
            export_dir.resolve()
        )
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
        "multi_tracker.utils.gpu_utils": _gpu_stub(),
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
        "multi_tracker.utils.gpu_utils": _gpu_stub(SLEAP_RUNTIME_ONNX_AVAILABLE=False),
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
        "multi_tracker.utils.gpu_utils": _gpu_stub(SLEAP_RUNTIME_ONNX_AVAILABLE=False),
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

    assert p1 == p2
    assert len(export_calls) >= 2
    assert p2.read_bytes() == b"onnx-2"


def test_yolo_backend_rejects_directory_model_path_with_clear_error(
    tmp_path: Path,
) -> None:
    stubs = {
        "multi_tracker.utils.gpu_utils": _gpu_stub(SLEAP_RUNTIME_ONNX_AVAILABLE=False),
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


def test_build_runtime_config_derives_sleap_export_hw_from_model_config(
    tmp_path: Path,
) -> None:
    stubs = {
        "multi_tracker.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_runtime_api_module(stubs)

    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = {
        "data_config": {
            "preprocessing": {
                "crop_size": None,
                "max_height": 211,
                "max_width": 216,
            }
        },
        "model_config": {"backbone_config": {"unet": {"max_stride": 32}}},
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(training_cfg), encoding="utf-8"
    )

    params = {
        "POSE_MODEL_TYPE": "sleap",
        "POSE_MODEL_DIR": str(model_dir),
        "POSE_RUNTIME_FLAVOR": "onnx",
        "POSE_SLEAP_BATCH": 4,
    }
    cfg = mod.build_runtime_config(params, out_root=str(tmp_path))
    assert cfg.sleap_export_input_hw == (224, 224)


def test_build_runtime_config_explicit_sleap_export_hw_overrides_model_config(
    tmp_path: Path,
) -> None:
    stubs = {
        "multi_tracker.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_runtime_api_module(stubs)

    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = {
        "data_config": {"preprocessing": {"max_height": 211, "max_width": 216}},
        "model_config": {"backbone_config": {"unet": {"max_stride": 32}}},
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(training_cfg), encoding="utf-8"
    )

    params = {
        "POSE_MODEL_TYPE": "sleap",
        "POSE_MODEL_DIR": str(model_dir),
        "POSE_RUNTIME_FLAVOR": "onnx",
        "POSE_SLEAP_EXPORT_INPUT_HEIGHT": 190,  # aligned up to 192
        "POSE_SLEAP_EXPORT_INPUT_WIDTH": 224,
    }
    cfg = mod.build_runtime_config(params, out_root=str(tmp_path))
    assert cfg.sleap_export_input_hw == (192, 224)


def test_attempt_sleap_cli_export_prefers_size_aware_commands_first(
    tmp_path: Path,
) -> None:
    stubs = {
        "multi_tracker.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_runtime_api_module(stubs)

    model_dir = tmp_path / "sleap_model"
    export_dir = tmp_path / "sleap_export"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    seen_cmds = []

    def _fake_run(cmd, timeout_sec=1800):
        seen_cmds.append(list(cmd))
        return False, "fail"

    mod._run_cli_command = _fake_run
    ok, _err = mod._attempt_sleap_cli_export(
        model_dir=model_dir,
        export_dir=export_dir,
        runtime_flavor="onnx",
        sleap_env="",
        input_hw=(320, 352),
    )
    assert ok is False
    assert seen_cmds
    first = seen_cmds[0]
    assert "--input-height" in first
    assert "--input-width" in first
    assert "320" in first
    assert "352" in first


def test_attempt_sleap_cli_export_includes_batch_profile(
    tmp_path: Path,
) -> None:
    stubs = {
        "multi_tracker.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_runtime_api_module(stubs)

    model_dir = tmp_path / "sleap_model"
    export_dir = tmp_path / "sleap_export"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    seen_cmds = []

    def _fake_run(cmd, timeout_sec=1800):
        _ = timeout_sec
        seen_cmds.append(list(cmd))
        return False, "fail"

    mod._run_cli_command = _fake_run
    ok, _err = mod._attempt_sleap_cli_export(
        model_dir=model_dir,
        export_dir=export_dir,
        runtime_flavor="tensorrt",
        sleap_env="",
        input_hw=(224, 224),
        batch_size=8,
    )
    assert ok is False
    assert seen_cmds
    first = seen_cmds[0]
    assert "--batch-size" in first or "--batch" in first
    assert "8" in first


def test_coerce_prediction_batch_normalizes_out_of_range_confidences() -> None:
    stubs = {
        "multi_tracker.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_runtime_api_module(stubs)

    # Mimic exported ONNX outputs that provide logits instead of probabilities.
    pred_out = {
        "instance_peaks": np.array([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]),
        "instance_peak_vals": np.array([[2.0, 0.0, -2.0]], dtype=np.float32),
    }
    kpts = mod._coerce_prediction_batch(pred_out, batch_size=1)[0]
    assert kpts is not None
    assert np.all((kpts[:, 2] >= 0.0) & (kpts[:, 2] <= 1.0))
    assert np.isclose(kpts[0, 2], 0.880797, atol=1e-5)
    assert np.isclose(kpts[1, 2], 0.5, atol=1e-6)
    assert np.isclose(kpts[2, 2], 0.119203, atol=1e-5)
