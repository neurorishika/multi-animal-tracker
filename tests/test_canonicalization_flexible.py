import numpy as np

from hydra_suite.core.canonicalization import get_canon_transform


def test_get_canon_transform_with_annotation():
    # Mock annotation matching the expected schema
    ann = {
        "theta": 0.5,
        "canonicalization": {
            "center_px": [100.0, 100.0],
            "size_px": [50.0, 30.0],
            "angle_rad": 0.7,
        },
    }
    img_path = "mock.png"

    # Should use canonicalization.angle_rad (0.7) over theta (0.5)
    t = get_canon_transform(img_path, annotation=ann)
    assert t is not None
    assert t["canon_w"] > 0
    assert t["canon_h"] > 0

    # We can't easily check the exact affine values without math,
    # but we can verify it's a 2x3 matrix
    assert t["affine"].shape == (2, 3)
    assert t["inv_affine"].shape == (2, 3)


def test_get_canon_transform_custom_angle_field():
    ann = {
        "theta": 0.5,
        "custom_theta": 1.2,
        "canonicalization": {
            "center_px": [100.0, 100.0],
            "size_px": [50.0, 30.0],
            # No angle_rad here
        },
    }
    img_path = "mock.png"

    # Should use custom_theta (1.2)
    t = get_canon_transform(img_path, annotation=ann, angle_field="custom_theta")
    assert t is not None

    # Verify that it differs from when we don't pass custom_theta (which defaults to theta=0.5)
    t_default = get_canon_transform(img_path, annotation=ann)
    assert not np.allclose(t["affine"], t_default["affine"])


def test_get_canon_transform_theta_fallback():
    ann = {
        "theta": 0.5,
        "canonicalization": {"center_px": [100.0, 100.0], "size_px": [50.0, 30.0]},
    }
    img_path = "mock.png"

    # Should fallback to theta (0.5)
    t = get_canon_transform(img_path, annotation=ann)
    assert t is not None
