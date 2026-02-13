from __future__ import annotations

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub

kalman_mod = load_src_module(
    "multi_tracker/core/filters/kalman.py",
    "kalman_under_test",
    stubs={"cv2": make_cv2_stub()},
)
kalman_mod.NUMBA_AVAILABLE = False
KalmanFilterManager = kalman_mod.KalmanFilterManager


def _params() -> dict:
    return {
        "REFERENCE_BODY_SIZE": 20.0,
        "KALMAN_MATURITY_AGE": 5,
        "KALMAN_INITIAL_VELOCITY_RETENTION": 0.2,
        "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0,
        "KALMAN_NOISE_COVARIANCE": 0.03,
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 5.0,
        "KALMAN_LATERAL_NOISE_MULTIPLIER": 0.1,
        "KALMAN_MEASUREMENT_NOISE_COVARIANCE": 0.1,
        "KALMAN_DAMPING": 0.95,
    }


def test_kalman_predict_correct_and_velocity_constraint() -> None:
    kf = KalmanFilterManager(num_targets=2, params=_params())
    kf.initialize_filter(0, np.array([10.0, 20.0, 0.0, 200.0, 0.0], dtype=np.float32))
    kf.initialize_filter(1, np.array([30.0, 40.0, 1.0, 0.0, 0.0], dtype=np.float32))

    preds = kf.predict()
    assert preds.shape == (2, 3)

    speed = float(np.linalg.norm(kf.X[0, 3:5]))
    assert speed <= kf.max_velocity + 1e-5

    kf.correct(0, np.array([12.0, 21.0, 0.1], dtype=np.float32))
    assert int(kf.track_ages[0]) == 1


def test_kalman_theta_wrap_and_mahal_shapes() -> None:
    kf = KalmanFilterManager(num_targets=1, params=_params())
    kf.initialize_filter(0, np.array([0.0, 0.0, 6.20, 0.0, 0.0], dtype=np.float32))

    # Measurement near 0 radians should be handled as small-angle wrap.
    kf.correct(0, np.array([0.1, 0.1, 0.05], dtype=np.float32))
    assert 0.0 <= float(kf.X[0, 2]) <= (2 * np.pi + 0.1)

    mahal = kf.get_mahalanobis_matrices()
    assert mahal.shape == (1, 3, 3)
    uncertainties = kf.get_position_uncertainties()
    assert len(uncertainties) == 1
    assert uncertainties[0] > 0
