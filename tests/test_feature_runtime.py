from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from tests.helpers.module_loader import load_src_module


class _FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = xy
        self.conf = conf


class _FakeResult:
    def __init__(self, keypoints):
        self.keypoints = keypoints


class _FakeYOLO:
    def __init__(self, _path):
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def predict(self, source=None, **_kwargs):
        out = []
        for _ in source or []:
            xy = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)
            conf = np.array([[0.9, 0.1]], dtype=np.float32)
            out.append(_FakeResult(_FakeKeypoints(xy, conf)))
        return out


class _FakePoseInferenceService:
    def __init__(self, out_root, keypoint_names, skeleton_edges=None):
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges or [])

    def predict(self, model_path, image_paths, **_kwargs):
        preds = {}
        for p in image_paths:
            preds[str(p)] = [(1.0, 2.0, 0.9) for _ in self.keypoint_names]
        return preds, ""


def _load_feature_runtime_module():
    fake_ultra = types.SimpleNamespace(YOLO=_FakeYOLO)
    fake_pose_mod = types.SimpleNamespace(
        PoseInferenceService=_FakePoseInferenceService
    )
    stubs = {
        "ultralytics": fake_ultra,
        "multi_tracker.posekit.pose_inference": fake_pose_mod,
    }
    return load_src_module(
        "multi_tracker/core/identity/feature_runtime.py",
        "feature_runtime_under_test",
        stubs=stubs,
    )


def test_yolo_backend_uses_canonical_output_schema(monkeypatch) -> None:
    mod = _load_feature_runtime_module()
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO)
    )
    backend = mod.YoloPoseBackend(
        model_path="fake.pt",
        device="cpu",
        min_valid_conf=0.2,
        keypoint_names=["head", "tail"],
    )

    crops = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((6, 6, 3), dtype=np.uint8)]
    out = backend.predict_crops(crops)
    assert len(out) == 2
    assert out[0].num_keypoints == 2
    assert out[0].num_valid == 1
    assert np.isclose(out[0].mean_conf, 0.5)
    assert np.isclose(out[0].valid_fraction, 0.5)
    assert out[0].keypoints.shape == (2, 3)


def test_sleap_factory_uses_skeleton_file(tmp_path: Path, monkeypatch) -> None:
    mod = _load_feature_runtime_module()
    monkeypatch.setitem(
        sys.modules,
        "multi_tracker.posekit.pose_inference",
        types.SimpleNamespace(PoseInferenceService=_FakePoseInferenceService),
    )

    skeleton_file = tmp_path / "skeleton.json"
    skeleton_file.write_text(
        '{"keypoint_names":["head","thorax","abdomen"],"skeleton_edges":[[0,1],[1,2]]}',
        encoding="utf-8",
    )
    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir()
    backend = mod.create_pose_backend(
        {
            "POSE_MODEL_TYPE": "sleap",
            "POSE_MODEL_DIR": str(model_dir),
            "POSE_SKELETON_FILE": str(skeleton_file),
            "POSE_MIN_KPT_CONF_VALID": 0.2,
        },
        out_root=str(tmp_path),
    )
    assert isinstance(backend, mod.SleapPoseBackend)
    assert backend.keypoint_names == ["head", "thorax", "abdomen"]
    backend.close()
