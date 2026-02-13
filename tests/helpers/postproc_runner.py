from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_processing_module():
    processing_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "multi_tracker"
        / "core"
        / "post"
        / "processing.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_post_processing_under_test", processing_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load processing module from {processing_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_processing = _load_processing_module()
interpolate_trajectories = _processing.interpolate_trajectories
resolve_trajectories = _processing.resolve_trajectories


FLOAT_COLUMNS = (
    "X",
    "Y",
    "Theta",
    "DetectionConfidence",
    "AssignmentConfidence",
    "PositionUncertainty",
)
INT_COLUMNS = ("TrajectoryID", "FrameID", "DetectionID")
PRIMARY_SORT = ("TrajectoryID", "FrameID")


def fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "postproc"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_into_trajectories(df: pd.DataFrame) -> list[pd.DataFrame]:
    if df is None or df.empty:
        return []
    if "TrajectoryID" not in df.columns:
        raise ValueError("Input dataframe must contain 'TrajectoryID'")

    trajectories: list[pd.DataFrame] = []
    for traj_id in sorted(df["TrajectoryID"].dropna().unique()):
        traj = df[df["TrajectoryID"] == traj_id].copy()
        if "FrameID" in traj.columns:
            traj = traj.sort_values("FrameID")
        trajectories.append(traj.reset_index(drop=True))
    return trajectories


def _concat_trajectories(trajectories: list[pd.DataFrame]) -> pd.DataFrame:
    if not trajectories:
        return pd.DataFrame(columns=["TrajectoryID", "FrameID", "X", "Y", "Theta"])
    return pd.concat(trajectories, ignore_index=True, sort=False)


def run_resolve_trajectories(input_df: Any, params: dict[str, Any]) -> pd.DataFrame:
    """
    Run resolve_trajectories and return a concatenated DataFrame.

    Accepted input_df shapes:
    - {"forward": <DataFrame>, "backward": <DataFrame>}
    - (<forward_df>, <backward_df>)
    - <forward_df> (backward treated as empty)
    """
    forward_df = None
    backward_df = None

    if isinstance(input_df, dict):
        forward_df = input_df.get("forward")
        backward_df = input_df.get("backward")
    elif isinstance(input_df, tuple) and len(input_df) == 2:
        forward_df, backward_df = input_df
    else:
        forward_df = input_df

    forward = _split_into_trajectories(
        forward_df if forward_df is not None else pd.DataFrame()
    )
    backward = _split_into_trajectories(
        backward_df if backward_df is not None else pd.DataFrame()
    )

    resolved = resolve_trajectories(forward, backward, params=params)
    return _concat_trajectories(resolved)


def run_interpolate(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return interpolate_trajectories(
        df.copy(),
        method=params.get("method", "linear"),
        max_gap=params.get("max_gap", 10),
    )


def normalize_output(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()

    for col in FLOAT_COLUMNS:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype(
                float
            )

    for col in INT_COLUMNS:
        if col in normalized.columns:
            normalized[col] = (
                pd.to_numeric(normalized[col], errors="coerce").round(0).astype("Int64")
            )

    for col in PRIMARY_SORT:
        if col not in normalized.columns:
            normalized[col] = pd.NA

    sort_cols = [c for c in PRIMARY_SORT if c in normalized.columns]
    normalized = normalized.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    first_cols = [
        c
        for c in ("TrajectoryID", "FrameID", "X", "Y", "Theta", "State")
        if c in normalized.columns
    ]
    trailing_cols = sorted([c for c in normalized.columns if c not in first_cols])
    normalized = normalized[first_cols + trailing_cols]

    return normalized


def assert_equivalent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    tol: dict[str, float] | None = None,
) -> None:
    tol = tol or {"abs": 1e-9, "rel": 1e-9}

    a = normalize_output(actual)
    e = normalize_output(expected)

    assert list(a.columns) == list(
        e.columns
    ), f"Column mismatch: {a.columns.tolist()} != {e.columns.tolist()}"
    assert len(a) == len(e), f"Row count mismatch: {len(a)} != {len(e)}"

    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(
            e[col]
        ):
            if col in INT_COLUMNS:
                left = a[col].fillna(-(2**31)).astype(np.int64).to_numpy()
                right = e[col].fillna(-(2**31)).astype(np.int64).to_numpy()
                np.testing.assert_array_equal(
                    left, right, err_msg=f"Integer mismatch in column '{col}'"
                )
            else:
                np.testing.assert_allclose(
                    a[col].to_numpy(dtype=float),
                    e[col].to_numpy(dtype=float),
                    rtol=tol["rel"],
                    atol=tol["abs"],
                    equal_nan=True,
                    err_msg=f"Float mismatch in column '{col}'",
                )
        else:
            left = a[col].fillna("<NA>").astype(str).to_numpy()
            right = e[col].fillna("<NA>").astype(str).to_numpy()
            np.testing.assert_array_equal(
                left, right, err_msg=f"Value mismatch in column '{col}'"
            )


def load_resolve_fixture(
    case_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    root = fixture_root() / "resolve" / case_name
    forward = _read_csv(root / "input_forward.csv")
    backward = _read_csv(root / "input_backward.csv")
    params = _read_json(root / "params.json")
    expected = _read_csv(root / "expected.csv")
    return forward, backward, params, expected


def load_interpolate_fixture(
    case_name: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    root = fixture_root() / "interpolate" / case_name
    input_df = _read_csv(root / "input.csv")
    params = _read_json(root / "params.json")
    expected = _read_csv(root / "expected.csv")
    return input_df, params, expected


def available_cases(kind: str) -> list[str]:
    root = fixture_root() / kind
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])
