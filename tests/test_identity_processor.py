from __future__ import annotations

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub

identity_mod = load_src_module(
    "multi_tracker/core/identity/analysis.py",
    "identity_analysis_under_test",
    stubs={"cv2": make_cv2_stub()},
)
IdentityProcessor = identity_mod.IdentityProcessor


def test_identity_processor_extract_crop_handles_padding() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    processor = IdentityProcessor(
        {
            "ENABLE_IDENTITY_ANALYSIS": True,
            "IDENTITY_CROP_SIZE_MULTIPLIER": 2.0,
            "IDENTITY_CROP_MIN_SIZE": 16,
            "IDENTITY_CROP_MAX_SIZE": 64,
        }
    )
    crop, info = processor.extract_crop(frame, cx=1, cy=1, body_size=10.0)
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[0] >= 16
    assert info["padded"] is True


def test_identity_processor_disabled_mode_short_circuits() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    processor = IdentityProcessor({"ENABLE_IDENTITY_ANALYSIS": False})
    identities, confidences, crops = processor.process_frame(
        frame,
        detections=[{"cx": 10.0, "cy": 10.0, "theta": 0.0, "body_size": 10.0}],
        frame_id=0,
    )
    assert identities == [None]
    assert confidences == [0.0]
    assert crops == []
