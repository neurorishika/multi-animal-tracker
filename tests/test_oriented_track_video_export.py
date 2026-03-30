from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from multi_tracker.core.identity.dataset.oriented_video import (
    OrientedTrackVideoExporter,
    resolve_individual_dataset_dir,
)
from multi_tracker.data.detection_cache import DetectionCache


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for color in colors:
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _square(cx: float, cy: float, half: float) -> np.ndarray:
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float32,
    )


def test_resolve_individual_dataset_dir_uses_run_id(tmp_path: Path):
    root = tmp_path / "individual_crops"
    dataset_dir = root / "session_20260311"
    dataset_dir.mkdir(parents=True)

    resolved = resolve_individual_dataset_dir(
        root, dataset_name="session", run_id="20260311"
    )

    assert resolved == dataset_dir


def test_oriented_track_video_export_streams_from_source_video_and_caches(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    video_path = tmp_path / "source.mp4"
    cache_path = tmp_path / "detections.npz"
    interp_npz_path = tmp_path / "interpolated_rois.npz"
    final_csv_path = tmp_path / "tracks_final.csv"

    _write_video(
        video_path,
        [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)],
    )

    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=3) as cache:
        cache.add_frame(
            0,
            meas=[np.array([20.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[101],
        )
        cache.add_frame(
            1,
            meas=[np.array([24.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(24.0, 24.0, 6.0)],
            detection_ids=[102],
        )
        cache.add_frame(2, [], [], [], [])
        cache.add_frame(
            3,
            meas=[np.array([42.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(42.0, 24.0, 6.0)],
            detection_ids=[201],
        )
        cache.save()

    np.savez_compressed(
        str(interp_npz_path),
        frame_id=np.array([2], dtype=np.int64),
        trajectory_id=np.array([1], dtype=np.int64),
        filename=np.array([""], dtype=object),
        cx=np.array([28.0], dtype=np.float32),
        cy=np.array([24.0], dtype=np.float32),
        w=np.array([12.0], dtype=np.float32),
        h=np.array([12.0], dtype=np.float32),
        theta=np.array([0.0], dtype=np.float32),
        interp_from_start=np.array([1], dtype=np.int64),
        interp_from_end=np.array([3], dtype=np.int64),
        interp_index=np.array([1], dtype=np.int64),
        interp_total=np.array([1], dtype=np.int64),
        obb_corners=np.array([_square(28.0, 24.0, 6.0)], dtype=np.float32),
    )

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 101,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 1,
                "DetectionID": 102,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 2,
                "DetectionID": np.nan,
                "Theta": 0.0,
                "State": "occluded",
            },
            {
                "TrajectoryID": 2,
                "FrameID": 3,
                "DetectionID": 201,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=cache_path,
        interpolated_roi_npz_path=interp_npz_path,
        fps=5.0,
        padding_fraction=0.0,
    )
    result = exporter.export()

    assert result.exported_videos == 2
    assert result.exported_tracks == 2
    assert result.exported_frames == 4
    assert result.missing_rows == 0

    track1_path = Path(result.output_dir) / "trajectory_0001.mp4"
    track2_path = Path(result.output_dir) / "trajectory_0002.mp4"
    assert track1_path.exists()
    assert track2_path.exists()

    cap1 = cv2.VideoCapture(str(track1_path))
    try:
        assert cap1.isOpened()
        assert int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        cap1.release()

    cap2 = cv2.VideoCapture(str(track2_path))
    try:
        assert cap2.isOpened()
        assert int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
    finally:
        cap2.release()
