from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "multi_tracker/core/identity/properties_cache.py",
    "individual_properties_cache_under_test",
)


def test_hashes_change_when_expected(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    model_a = tmp_path / "pose_a.pt"
    model_b = tmp_path / "pose_b.pt"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.7,
        "ENABLE_SIZE_FILTERING": True,
        "MIN_OBJECT_SIZE": 10,
        "MAX_OBJECT_SIZE": 100,
        "ROI_MASK": np.zeros((4, 4), dtype=np.uint8),
        "ENABLE_POSE_EXTRACTOR": True,
        "POSE_MODEL_TYPE": "yolo",
        "POSE_MODEL_DIR": str(model_a),
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "COMPUTE_RUNTIME": "mps",
    }

    det_hash_1 = mod.compute_detection_hash("abc", str(video), 0, 99)
    det_hash_2 = mod.compute_detection_hash("abc", str(video), 0, 99)
    det_hash_3 = mod.compute_detection_hash("def", str(video), 0, 99)
    assert det_hash_1 == det_hash_2
    assert det_hash_1 != det_hash_3

    filter_hash_1 = mod.compute_filter_settings_hash(params)
    params_changed = dict(params)
    params_changed["YOLO_CONFIDENCE_THRESHOLD"] = 0.35
    filter_hash_2 = mod.compute_filter_settings_hash(params_changed)
    assert filter_hash_1 != filter_hash_2

    ext_hash_1 = mod.compute_extractor_hash(params)
    params_model_changed = dict(params)
    params_model_changed["POSE_MODEL_DIR"] = str(model_b)
    ext_hash_2 = mod.compute_extractor_hash(params_model_changed)
    assert ext_hash_1 != ext_hash_2

    params_runtime_changed = dict(params)
    params_runtime_changed["COMPUTE_RUNTIME"] = "onnx_cpu"
    assert mod.compute_extractor_hash(params_runtime_changed) != ext_hash_1

    params_ignore_changed = dict(params)
    params_ignore_changed["POSE_IGNORE_KEYPOINTS"] = ["head"]
    assert mod.compute_extractor_hash(params_ignore_changed) == ext_hash_1

    params_ant_changed = dict(params)
    params_ant_changed["POSE_DIRECTION_ANTERIOR_KEYPOINTS"] = ["head"]
    assert mod.compute_extractor_hash(params_ant_changed) == ext_hash_1

    params_post_changed = dict(params)
    params_post_changed["POSE_DIRECTION_POSTERIOR_KEYPOINTS"] = ["tail"]
    assert mod.compute_extractor_hash(params_post_changed) == ext_hash_1

    props_id_1 = mod.compute_individual_properties_id(
        det_hash_1, filter_hash_1, ext_hash_1
    )
    props_id_2 = mod.compute_individual_properties_id(
        det_hash_1, filter_hash_2, ext_hash_1
    )
    assert props_id_1 != props_id_2


def test_cache_roundtrip_and_lookup(tmp_path: Path) -> None:
    cache_path = tmp_path / "props.npz"

    with_kpts = np.array([[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]], dtype=np.float32)
    cache_w = mod.IndividualPropertiesCache(str(cache_path), mode="w")
    cache_w.add_frame(
        10,
        [100001.0, 100002.0],
        pose_mean_conf=[0.85, 0.0],
        pose_valid_fraction=[1.0, 0.0],
        pose_num_valid=[2, 0],
        pose_num_keypoints=[2, 0],
        pose_keypoints=[with_kpts, None],
    )
    cache_w.add_frame(11, [])
    cache_w.save(metadata={"individual_properties_id": "id123"})

    cache_r = mod.IndividualPropertiesCache(str(cache_path), mode="r")
    assert cache_r.is_compatible()
    assert 10 in cache_r.get_cached_frames()

    frame = cache_r.get_frame(10)
    assert frame["detection_ids"] == [100001.0, 100002.0]
    assert frame["pose_num_valid"] == [2, 0]

    hit = cache_r.get_detection(10, 100001)
    assert hit is not None
    assert hit["pose_num_keypoints"] == 2
    assert np.asarray(hit["pose_keypoints"]).shape == (2, 3)

    miss = cache_r.get_detection(10, 999999)
    assert miss is None
    cache_r.close()
