from __future__ import annotations

import pandas as pd

from hydra_suite.trackerkit.gui.workers.crops_worker import InterpolatedCropsWorker


def test_interpolated_worker_skips_backend_init_when_no_eligible_gaps(
    monkeypatch,
) -> None:
    worker = InterpolatedCropsWorker("tracks.csv", "source.mp4", "cache.npz", {})
    emitted: list[dict[str, object]] = []
    finalized = {"called": False}

    class FakeCap:
        def release(self) -> None:
            return None

    class FakeGenerator:
        def finalize(self) -> None:
            finalized["called"] = True

    worker.finished_signal.connect(lambda result: emitted.append(result))

    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_validate_and_setup",
        lambda self, profiler: (
            pd.DataFrame(
                [
                    {
                        "TrajectoryID": 1,
                        "FrameID": 0,
                        "State": "active",
                        "X": 0.0,
                        "Y": 0.0,
                        "Theta": 0.0,
                    }
                ]
            ),
            FakeCap(),
            None,
            FakeGenerator(),
            "unused-output-dir",
            False,
            True,
            1.0,
            1.0,
            2.0,
            0.1,
        ),
    )
    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_detect_interpolation_gaps",
        lambda self, df, detection_cache, position_scale, size_scale: ({}, 4, 0, 0),
    )
    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_init_interpolation_backends",
        lambda self, output_dir: (_ for _ in ()).throw(
            AssertionError("backend initialization should be skipped")
        ),
    )
    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_cleanup_backends",
        lambda self, *args, **kwargs: None,
    )

    worker.execute()

    assert finalized["called"] is True
    assert len(emitted) == 1
    assert emitted[0]["no_work_reason"] == "no_eligible_gaps"
    assert emitted[0]["occluded_rows"] == 4
    assert emitted[0]["eligible_frames"] == 0
    assert emitted[0]["eligible_rows"] == 0
    assert emitted[0]["pose_rows_produced"] == 0
    assert emitted[0]["cnn_rows_produced"] == 0
