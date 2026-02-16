from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub


class _FakeContour(dict):
    def __len__(self):
        return int(self.get("n_points", 6))


def _load_engine_module():
    return load_src_module(
        "multi_tracker/core/detectors/engine.py",
        "detectors_engine_under_test",
        stubs={"cv2": make_cv2_stub()},
    )


def test_object_detector_detect_objects_filters_and_limits_count() -> None:
    mod = _load_engine_module()

    params = {
        "MAX_TARGETS": 2,
        "MAX_CONTOUR_MULTIPLIER": 20,
        "MIN_CONTOUR_AREA": 10.0,
        "ENABLE_SIZE_FILTERING": True,
        "MIN_OBJECT_SIZE": 15.0,
        "MAX_OBJECT_SIZE": 200.0,
        "MERGE_AREA_THRESHOLD": 1000.0,
        "CONSERVATIVE_KERNEL_SIZE": 3,
        "CONSERVATIVE_ERODE_ITER": 1,
    }
    detector = mod.ObjectDetector(params)

    contours = [
        _FakeContour(
            area=20.0,
            ellipse=((10.0, 10.0), (8.0, 4.0), 15.0),
            rect=(0, 0, 5, 5),
        ),
        _FakeContour(
            area=80.0,
            ellipse=((30.0, 30.0), (20.0, 8.0), 40.0),
            rect=(20, 20, 10, 10),
        ),
        _FakeContour(
            area=140.0,
            ellipse=((50.0, 50.0), (24.0, 10.0), 55.0),
            rect=(40, 40, 12, 12),
        ),
        _FakeContour(  # filtered by min size
            area=5.0,
            ellipse=((70.0, 70.0), (4.0, 2.0), 0.0),
            rect=(65, 65, 4, 4),
        ),
    ]

    meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
        contours, frame_count=1
    )
    assert yolo_results is None
    assert len(meas) == 2  # limited by MAX_TARGETS
    assert all(m.shape == (3,) for m in meas)
    assert len(shapes) == 2
    assert len(confidences) == 2
    assert all(np.isnan(c) for c in confidences)
    assert len(sizes) >= 2  # current implementation keeps original filtered size list


def test_create_detector_defaults_to_background_subtraction() -> None:
    mod = _load_engine_module()
    detector = mod.create_detector({"DETECTION_METHOD": "background_subtraction"})
    assert isinstance(detector, mod.ObjectDetector)


def test_tensorrt_engine_path_is_model_adjacent_and_stable_across_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.engine"
            out.write_bytes(b"fake-engine")
            return str(out)

    fake_ultra = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    model_dir_a = tmp_path / "a"
    model_dir_b = tmp_path / "b"
    model_dir_a.mkdir(parents=True, exist_ok=True)
    model_dir_b.mkdir(parents=True, exist_ok=True)
    model_a = model_dir_a / "best.pt"
    model_b = model_dir_b / "best.pt"
    model_a.write_bytes(b"model-a")
    model_b.write_bytes(b"model-b")

    def build_engine_path(model_path: Path, model_id: str):
        det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
        det.params = {
            "TENSORRT_MAX_BATCH_SIZE": 8,
            "INFERENCE_MODEL_ID": model_id,
        }
        det.device = "cuda:0"
        det.use_tensorrt = False
        det.tensorrt_model_path = None
        det._shapely_warning_shown = False
        det._try_load_tensorrt_model(str(model_path))
        assert det.use_tensorrt
        assert det.tensorrt_model_path is not None
        return det.tensorrt_model_path

    path_a_id1 = build_engine_path(model_a, "id-A")
    path_b_id1 = build_engine_path(model_b, "id-A")
    path_a_id2 = build_engine_path(model_a, "id-B")

    # TensorRT artifacts are co-located with source model paths.
    # Different model locations should map to different engine paths.
    assert path_a_id1 != path_b_id1
    # Same model location should keep the same engine path even if inference id changes.
    assert path_a_id1 == path_a_id2


def test_tensorrt_engine_path_is_batch_specific(tmp_path: Path, monkeypatch) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.engine"
            out.write_bytes(b"fake-engine")
            return str(out)

    fake_ultra = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    def build_engine_path(batch_size: int):
        det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
        det.params = {
            "TENSORRT_MAX_BATCH_SIZE": int(batch_size),
            "INFERENCE_MODEL_ID": "id-A",
        }
        det.device = "cuda:0"
        det.use_tensorrt = False
        det.tensorrt_model_path = None
        det.tensorrt_batch_size = 1
        det._shapely_warning_shown = False
        det._try_load_tensorrt_model(str(model_path))
        assert det.use_tensorrt
        assert det.tensorrt_model_path is not None
        return det.tensorrt_model_path, int(det.tensorrt_batch_size)

    path_b8, b8 = build_engine_path(8)
    path_b4, b4 = build_engine_path(4)
    assert path_b8 != path_b4
    assert path_b8.endswith("_b8.engine")
    assert path_b4.endswith("_b4.engine")
    assert b8 == 8
    assert b4 == 4


def test_onnx_artifact_path_is_batch_specific_and_model_adjacent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.overrides = {}
            self.model = types.SimpleNamespace(args={})

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    fake_ultra = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    def build_onnx_path(batch_size: int):
        det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
        det.params = {
            "TENSORRT_MAX_BATCH_SIZE": int(batch_size),
            "INFERENCE_MODEL_ID": "id-A",
        }
        det.device = "cpu"
        det.use_onnx = False
        det.onnx_model_path = None
        det.onnx_imgsz = None
        det.onnx_batch_size = 1
        det._shapely_warning_shown = False
        det._try_load_onnx_model(str(model_path))
        assert det.use_onnx
        assert det.onnx_model_path is not None
        return det.onnx_model_path, int(det.onnx_batch_size)

    path_b8, b8 = build_onnx_path(8)
    path_b4, b4 = build_onnx_path(4)

    assert path_b8 != path_b4
    assert path_b8.endswith("_b8.onnx")
    assert path_b4.endswith("_b4.onnx")
    assert b8 == 8
    assert b4 == 4


def test_yolo_raw_detection_cap_is_two_x_max_targets() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {"MAX_TARGETS": 6}
    assert det._raw_detection_cap() == 12


def test_resolve_onnx_imgsz_prefers_model_metadata(tmp_path: Path, monkeypatch) -> None:
    mod = _load_engine_module()

    class FakeYOLO:
        def __init__(self, _path, task=None):
            self.task = task
            self.overrides = {"imgsz": 1504}
            self.model = types.SimpleNamespace(args={"imgsz": 1504})

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"x")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {}
    imgsz = det._resolve_onnx_imgsz(model_path=model_path)
    assert imgsz == 1504


def test_filter_raw_detections_applies_conf_size_and_target_limit() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "YOLO_CONFIDENCE_THRESHOLD": 0.5,
        "YOLO_IOU_THRESHOLD": 0.7,
        "MAX_TARGETS": 2,
        "ENABLE_SIZE_FILTERING": True,
        "MIN_OBJECT_SIZE": 40.0,
        "MAX_OBJECT_SIZE": 200.0,
    }
    det._shapely_warning_shown = False

    meas = [
        np.array([10.0, 10.0, 0.0], dtype=np.float32),
        np.array([40.0, 40.0, 0.0], dtype=np.float32),
        np.array([70.0, 70.0, 0.0], dtype=np.float32),
        np.array([100.0, 100.0, 0.0], dtype=np.float32),
    ]
    sizes = [120.0, 80.0, 60.0, 20.0]
    shapes = [(120.0, 1.2), (80.0, 1.1), (60.0, 1.0), (20.0, 1.0)]
    confidences = [0.95, 0.8, 0.2, 0.9]
    obb = [
        np.array([[8, 8], [12, 8], [12, 12], [8, 12]], dtype=np.float32),
        np.array([[38, 38], [42, 38], [42, 42], [38, 42]], dtype=np.float32),
        np.array([[68, 68], [72, 68], [72, 72], [68, 72]], dtype=np.float32),
        np.array([[98, 98], [102, 98], [102, 102], [98, 102]], dtype=np.float32),
    ]
    ids = [101.0, 102.0, 103.0, 104.0]

    out = det.filter_raw_detections(
        meas, sizes, shapes, confidences, obb, roi_mask=None, detection_ids=ids
    )
    out_meas, out_sizes, _, out_conf, _, out_ids = out

    assert len(out_meas) == 2
    assert out_sizes == [120.0, 80.0]
    assert np.allclose(out_conf, [0.95, 0.8], rtol=1e-6, atol=1e-6)
    assert out_ids == [101.0, 102.0]


def test_filter_raw_detections_applies_roi_mask() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "YOLO_CONFIDENCE_THRESHOLD": 0.0,
        "YOLO_IOU_THRESHOLD": 0.7,
        "MAX_TARGETS": 4,
        "ENABLE_SIZE_FILTERING": False,
    }
    det._shapely_warning_shown = False

    roi = np.zeros((20, 20), dtype=np.uint8)
    roi[:, :10] = 255

    meas = [
        np.array([5.0, 10.0, 0.0], dtype=np.float32),
        np.array([15.0, 10.0, 0.0], dtype=np.float32),
    ]
    sizes = [50.0, 60.0]
    shapes = [(50.0, 1.0), (60.0, 1.0)]
    confidences = [0.6, 0.7]
    obb = [
        np.array([[4, 9], [6, 9], [6, 11], [4, 11]], dtype=np.float32),
        np.array([[14, 9], [16, 9], [16, 11], [14, 11]], dtype=np.float32),
    ]
    ids = [1.0, 2.0]

    out = det.filter_raw_detections(
        meas, sizes, shapes, confidences, obb, roi_mask=roi, detection_ids=ids
    )
    _, out_sizes, _, out_conf, _, out_ids = out
    assert out_sizes == [50.0]
    assert np.allclose(out_conf, [0.6], rtol=1e-6, atol=1e-6)
    assert out_ids == [1.0]
