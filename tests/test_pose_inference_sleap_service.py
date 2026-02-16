from __future__ import annotations

from pathlib import Path

from tests.helpers.module_loader import load_src_module


def _load_pose_inference_module():
    return load_src_module(
        "multi_tracker/posekit/pose_inference.py",
        "pose_inference_sleap_service_under_test",
    )


def test_all_zero_pose_chunk_detection(tmp_path: Path) -> None:
    mod = _load_pose_inference_module()
    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    zero_preds = {
        str(frames[0]): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        str(frames[1]): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    }
    assert mod._all_zero_pose_chunk(frames, zero_preds, num_keypoints=2) is True

    mixed_preds = {
        str(frames[0]): [(0.0, 0.0, 0.0), (1.0, 2.0, 0.4)],
        str(frames[1]): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    }
    assert mod._all_zero_pose_chunk(frames, mixed_preds, num_keypoints=2) is False


def test_sleap_service_all_zero_onnx_triggers_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_pose_inference_module()

    class _FakeService:
        def __init__(self):
            self.log_path = None

        def start(self, _env_name: str, log_path=None):
            self.log_path = log_path
            return True, ""

        def request(self, _path: str, payload: dict, timeout: float = 3600.0):
            _ = timeout
            preds = {
                str(Path(p)): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
                for p in payload.get("images", [])
            }
            return {"ok": True, "preds": preds}

    fake_service = _FakeService()
    monkeypatch.setattr(mod, "_get_sleap_service", lambda: fake_service)

    fallback_called = {"n": 0}

    def _fake_fallback(**kwargs):
        fallback_called["n"] += 1
        images = kwargs.get("image_paths", [])
        preds = {str(Path(p)): [(10.0, 20.0, 0.8), (30.0, 40.0, 0.7)] for p in images}
        return True, preds, ""

    monkeypatch.setattr(mod, "_run_sleap_export_predict_subprocess", _fake_fallback)

    out_json = tmp_path / "posekit" / "tmp" / "preds.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images = [tmp_path / "f1.jpg", tmp_path / "f2.jpg"]
    ok, preds, err = mod._run_sleap_predict_service(
        model_dir=tmp_path / "sleap_model",
        image_paths=images,
        out_json=out_json,
        keypoint_names=["head", "tail"],
        skeleton_edges=[],
        env_name="sleap",
        device="cpu",
        batch=2,
        max_instances=1,
        runtime_flavor="onnx",
        exported_model_path=str(tmp_path / "m.onnx"),
        export_input_hw=(224, 224),
    )

    assert ok is True
    assert err == ""
    assert fallback_called["n"] == 1
    assert preds[str(images[0])][0] == (10.0, 20.0, 0.8)
    assert preds[str(images[1])][1] == (30.0, 40.0, 0.7)


def test_sleap_service_nonzero_onnx_does_not_trigger_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_pose_inference_module()

    class _FakeService:
        log_path = None

        def start(self, _env_name: str, log_path=None):
            self.log_path = log_path
            return True, ""

        def request(self, _path: str, payload: dict, timeout: float = 3600.0):
            _ = timeout
            preds = {
                str(Path(p)): [(11.0, 22.0, 0.6), (33.0, 44.0, 0.5)]
                for p in payload.get("images", [])
            }
            return {"ok": True, "preds": preds}

    monkeypatch.setattr(mod, "_get_sleap_service", lambda: _FakeService())
    fallback_called = {"n": 0}

    def _fake_fallback(**kwargs):
        _ = kwargs
        fallback_called["n"] += 1
        return True, {}, ""

    monkeypatch.setattr(mod, "_run_sleap_export_predict_subprocess", _fake_fallback)

    out_json = tmp_path / "posekit" / "tmp" / "preds.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images = [tmp_path / "f1.jpg"]
    ok, preds, err = mod._run_sleap_predict_service(
        model_dir=tmp_path / "sleap_model",
        image_paths=images,
        out_json=out_json,
        keypoint_names=["head", "tail"],
        skeleton_edges=[],
        env_name="sleap",
        runtime_flavor="onnx",
        exported_model_path=str(tmp_path / "m.onnx"),
    )

    assert ok is True
    assert err == ""
    assert fallback_called["n"] == 0
    assert preds[str(images[0])][0] == (11.0, 22.0, 0.6)
