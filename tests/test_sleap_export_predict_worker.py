from __future__ import annotations

from tests.helpers.module_loader import load_src_module


def _load_worker_module():
    return load_src_module(
        "multi_tracker/posekit/sleap_export_predict_worker.py",
        "sleap_export_predict_worker_under_test",
    )


def test_normalize_xy_conf_applies_sigmoid_for_logits() -> None:
    mod = _load_worker_module()
    raw = {
        "instance_peaks": [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]],
        "instance_peak_vals": [[2.0, 0.0, -2.0]],
    }
    xy, conf = mod._normalize_xy_conf(raw, batch_size=1)
    assert xy is not None
    assert conf is not None
    assert conf.shape == (1, 3)
    assert abs(float(conf[0, 0]) - 0.880797) < 1e-5
    assert abs(float(conf[0, 1]) - 0.5) < 1e-6
    assert abs(float(conf[0, 2]) - 0.119203) < 1e-5
