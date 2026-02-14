from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_tracker.core.identity.properties_cache import IndividualPropertiesCache
from multi_tracker.core.identity.properties_export import (
    POSE_EXPORT_COLUMNS,
    augment_trajectories_with_pose_cache,
    merge_interpolated_pose_df,
)


def test_augment_trajectories_with_pose_cache_merges_by_frame_and_detection(tmp_path):
    cache_path = tmp_path / "props.npz"
    writer = IndividualPropertiesCache(str(cache_path), mode="w")
    writer.add_frame(
        10,
        [101.0, 102.0],
        pose_mean_conf=[0.9, 0.8],
        pose_valid_fraction=[1.0, 0.5],
        pose_num_valid=[5, 3],
        pose_num_keypoints=[5, 6],
        pose_keypoints=[
            np.array([[1, 2, 0.9], [3, 4, 0.8]], dtype=np.float32),
            None,
        ],
    )
    writer.add_frame(
        11,
        [201.0],
        pose_mean_conf=[0.7],
        pose_valid_fraction=[0.25],
        pose_num_valid=[2],
        pose_num_keypoints=[8],
        pose_keypoints=[np.array([[7, 8, 0.5]], dtype=np.float32)],
    )
    writer.save(
        metadata={
            "individual_properties_id": "abc",
            "pose_keypoint_names": ["head", "tail"],
        }
    )
    writer.close()

    trajectories = pd.DataFrame(
        [
            {"FrameID": 10, "DetectionID": 101, "TrajectoryID": 1, "X": 1, "Y": 2},
            {"FrameID": 10, "DetectionID": 102, "TrajectoryID": 2, "X": 3, "Y": 4},
            {"FrameID": 11, "DetectionID": 201, "TrajectoryID": 3, "X": 5, "Y": 6},
            {"FrameID": 11, "DetectionID": 999, "TrajectoryID": 4, "X": 7, "Y": 8},
            {"FrameID": 12, "DetectionID": np.nan, "TrajectoryID": 5, "X": 9, "Y": 10},
        ]
    )
    out = augment_trajectories_with_pose_cache(trajectories, str(cache_path))

    for col in POSE_EXPORT_COLUMNS:
        assert col in out.columns

    first = out.iloc[0]
    assert first["PoseMeanConf"] == pytest.approx(0.9)
    assert first["PoseNumValid"] == 5
    assert first["PoseKpt_head_X"] == pytest.approx(1.0)
    assert first["PoseKpt_head_Y"] == pytest.approx(2.0)
    assert first["PoseKpt_head_Conf"] == pytest.approx(0.9)
    assert first["PoseKpt_tail_X"] == pytest.approx(3.0)
    assert first["PoseKpt_tail_Y"] == pytest.approx(4.0)
    assert first["PoseKpt_tail_Conf"] == pytest.approx(0.8)

    second = out.iloc[1]
    assert second["PoseMeanConf"] == pytest.approx(0.8)
    assert np.isnan(second["PoseKpt_head_X"])

    third = out.iloc[2]
    assert third["PoseNumKeypoints"] == 8

    unmatched = out.iloc[3]
    assert np.isnan(unmatched["PoseMeanConf"])
    assert np.isnan(unmatched["PoseKpt_head_X"])


def test_augment_trajectories_with_pose_cache_requires_detection_columns(tmp_path):
    cache_path = tmp_path / "props_empty.npz"
    writer = IndividualPropertiesCache(str(cache_path), mode="w")
    writer.save(metadata={"individual_properties_id": "abc"})
    writer.close()

    trajectories = pd.DataFrame([{"FrameID": 1, "TrajectoryID": 1, "X": 1, "Y": 2}])
    out = augment_trajectories_with_pose_cache(trajectories, str(cache_path))
    assert list(out.columns) == list(trajectories.columns)


def test_merge_interpolated_pose_fills_only_missing_detection_pose():
    trajectories = pd.DataFrame(
        [
            {
                "FrameID": 1,
                "TrajectoryID": 10,
                "DetectionID": 100,
                "PoseMeanConf": 0.9,
                "PoseValidFraction": 1.0,
                "PoseNumValid": 5,
                "PoseNumKeypoints": 5,
                "PoseKpt_head_X": 1.0,
                "PoseKpt_head_Y": 2.0,
                "PoseKpt_head_Conf": 0.9,
            },
            {
                "FrameID": 2,
                "TrajectoryID": 10,
                "DetectionID": np.nan,
                "PoseMeanConf": np.nan,
                "PoseValidFraction": np.nan,
                "PoseNumValid": np.nan,
                "PoseNumKeypoints": np.nan,
                "PoseKpt_head_X": np.nan,
                "PoseKpt_head_Y": np.nan,
                "PoseKpt_head_Conf": np.nan,
            },
        ]
    )
    interp_pose = pd.DataFrame(
        [
            {
                "frame_id": 1,
                "trajectory_id": 10,
                "PoseMeanConf": 0.2,
                "PoseValidFraction": 0.3,
                "PoseNumValid": 1,
                "PoseNumKeypoints": 5,
                "PoseKpt_head_X": 9.0,
                "PoseKpt_head_Y": 9.0,
                "PoseKpt_head_Conf": 0.2,
            },
            {
                "frame_id": 2,
                "trajectory_id": 10,
                "PoseMeanConf": 0.7,
                "PoseValidFraction": 0.8,
                "PoseNumValid": 4,
                "PoseNumKeypoints": 5,
                "PoseKpt_head_X": 3.0,
                "PoseKpt_head_Y": 4.0,
                "PoseKpt_head_Conf": 0.7,
            },
        ]
    )

    out = merge_interpolated_pose_df(trajectories, interp_pose)

    # Existing detection-keyed pose should be preserved.
    assert out.iloc[0]["PoseMeanConf"] == pytest.approx(0.9)
    assert out.iloc[0]["PoseKpt_head_X"] == pytest.approx(1.0)
    assert out.iloc[0]["PoseKpt_head_Y"] == pytest.approx(2.0)
    assert out.iloc[0]["PoseKpt_head_Conf"] == pytest.approx(0.9)

    # Missing pose for interpolated row should be filled from interpolated table.
    assert out.iloc[1]["PoseMeanConf"] == pytest.approx(0.7)
    assert out.iloc[1]["PoseNumValid"] == 4
    assert out.iloc[1]["PoseKpt_head_X"] == pytest.approx(3.0)
    assert out.iloc[1]["PoseKpt_head_Y"] == pytest.approx(4.0)
    assert out.iloc[1]["PoseKpt_head_Conf"] == pytest.approx(0.7)
