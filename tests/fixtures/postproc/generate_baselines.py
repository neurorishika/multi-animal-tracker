from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESOLVE_ROOT = ROOT / "resolve"
INTERP_ROOT = ROOT / "interpolate"
BENCH_ROOT = ROOT / "benchmarks"


def _load_processing_module():
    processing_path = (
        ROOT.parents[2] / "src" / "multi_tracker" / "core" / "post" / "processing.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_post_processing_baseline_gen", processing_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load processing module from {processing_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_processing = _load_processing_module()
interpolate_trajectories = _processing.interpolate_trajectories
resolve_trajectories = _processing.resolve_trajectories

FLOAT_COLUMNS = {
    "X",
    "Y",
    "Theta",
    "DetectionConfidence",
    "AssignmentConfidence",
    "PositionUncertainty",
}
INT_COLUMNS = {"TrajectoryID", "FrameID", "DetectionID"}


def _split(df: pd.DataFrame) -> list[pd.DataFrame]:
    if df.empty:
        return []
    parts = []
    for traj_id in sorted(df["TrajectoryID"].dropna().unique()):
        part = df[df["TrajectoryID"] == traj_id].copy().sort_values("FrameID")
        parts.append(part.reset_index(drop=True))
    return parts


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in FLOAT_COLUMNS.intersection(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in INT_COLUMNS.intersection(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce").round(0).astype("Int64")
    if "TrajectoryID" in out.columns and "FrameID" in out.columns:
        out = out.sort_values(["TrajectoryID", "FrameID"], kind="stable")
    first_cols = [
        c
        for c in ("TrajectoryID", "FrameID", "X", "Y", "Theta", "State")
        if c in out.columns
    ]
    other_cols = sorted([c for c in out.columns if c not in first_cols])
    return out[first_cols + other_cols].reset_index(drop=True)


def _write_case(
    path: Path,
    *,
    forward: pd.DataFrame,
    backward: pd.DataFrame,
    params: dict,
    expected: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    forward.to_csv(path / "input_forward.csv", index=False)
    backward.to_csv(path / "input_backward.csv", index=False)
    (path / "params.json").write_text(
        json.dumps(params, indent=2, sort_keys=True), encoding="utf-8"
    )
    _canonical(expected).to_csv(path / "expected.csv", index=False)


def _write_interp_case(
    path: Path, *, input_df: pd.DataFrame, params: dict, expected: pd.DataFrame
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(path / "input.csv", index=False)
    (path / "params.json").write_text(
        json.dumps(params, indent=2, sort_keys=True), encoding="utf-8"
    )
    _canonical(expected).to_csv(path / "expected.csv", index=False)


def _rows(
    traj_id: int,
    frames: list[int],
    x_values: list[float],
    y_values: list[float],
    theta_values: list[float],
    *,
    state: str = "active",
    det_start: int | None = None,
) -> list[dict]:
    rows = []
    for i, frame in enumerate(frames):
        row = {
            "TrajectoryID": traj_id,
            "FrameID": frame,
            "X": float(x_values[i]),
            "Y": float(y_values[i]),
            "Theta": float(theta_values[i]),
            "State": state,
        }
        if det_start is not None:
            row["DetectionID"] = int(det_start + i)
        rows.append(row)
    return rows


def build_resolve_cases() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]:
    cases: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}

    # Case: no-overlap
    f_rows = []
    b_rows = []
    frames_a = list(range(0, 6))
    frames_b = list(range(20, 26))
    f_rows += _rows(
        0, frames_a, [10 + f for f in range(6)], [20.0] * 6, [0.1] * 6, det_start=10
    )
    f_rows += _rows(
        1, frames_b, [200 + f for f in range(6)], [60.0] * 6, [1.4] * 6, det_start=100
    )
    b_rows += _rows(
        0, frames_a, [500 + f for f in range(6)], [300.0] * 6, [0.1] * 6, det_start=1000
    )
    cases["no_overlap_small"] = (
        pd.DataFrame(f_rows),
        pd.DataFrame(b_rows),
        {
            "AGREEMENT_DISTANCE": 5.0,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )

    # Case: heavy overlap many-to-many
    rng = np.random.default_rng(42)
    f_rows = []
    b_rows = []
    frames = list(range(100, 112))
    t = np.arange(len(frames))

    # Forward trajectories
    f_rows += _rows(
        0,
        frames,
        (100 + 8 * t).tolist(),
        (220 + 0.8 * t).tolist(),
        (0.1 * t).tolist(),
        det_start=5000,
    )
    f_rows += _rows(
        1,
        frames,
        (285 - 6 * t).tolist(),
        (180 + 1.1 * t).tolist(),
        (1.2 + 0.05 * t).tolist(),
        det_start=6000,
    )
    frames_f2 = list(range(104, 116))
    t2 = np.arange(len(frames_f2))
    f_rows += _rows(
        2,
        frames_f2,
        (180 + 4.5 * t2).tolist(),
        (320 - 2.0 * t2).tolist(),
        (2.4 + 0.02 * t2).tolist(),
        det_start=7000,
    )

    # Backward trajectories close to forward tracks
    b0_x = (100 + 8 * t + rng.normal(0.0, 1.2, size=len(t))).tolist()
    b0_y = (220 + 0.8 * t + rng.normal(0.0, 1.2, size=len(t))).tolist()
    b_rows += _rows(10, frames, b0_x, b0_y, (0.15 * t).tolist(), det_start=5000)

    b1_x = (285 - 6 * t + rng.normal(0.0, 1.2, size=len(t))).tolist()
    b1_y = (180 + 1.1 * t + rng.normal(0.0, 1.2, size=len(t))).tolist()
    b_rows += _rows(11, frames, b1_x, b1_y, (1.18 + 0.03 * t).tolist(), det_start=6000)

    frames_b2 = list(range(104, 111))
    t3 = np.arange(len(frames_b2))
    b_rows += _rows(
        12,
        frames_b2,
        (188 + 4.0 * t3 + rng.normal(0.0, 0.8, size=len(t3))).tolist(),
        (250 + 2.3 * t3 + rng.normal(0.0, 0.8, size=len(t3))).tolist(),
        (0.9 + 0.1 * t3).tolist(),
        det_start=9000,
    )
    cases["heavy_overlap_many_to_many"] = (
        pd.DataFrame(f_rows),
        pd.DataFrame(b_rows),
        {
            "AGREEMENT_DISTANCE": 25.0,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )

    # Case: fragmented trajectories requiring stitch
    frames_a = list(range(300, 305))
    frames_b = list(range(307, 313))
    f_rows = []
    b_rows = []
    f_rows += _rows(
        0,
        frames_a,
        [50 + 2 * i for i in range(len(frames_a))],
        [100 + 2 * i for i in range(len(frames_a))],
        [0.2] * len(frames_a),
        det_start=10000,
    )
    f_rows += _rows(
        1,
        frames_b,
        [60 + 2 * i for i in range(len(frames_b))],
        [110 + 2 * i for i in range(len(frames_b))],
        [0.25] * len(frames_b),
        det_start=10100,
    )
    # Keep backward non-empty so resolve executes full merge/stitch pipeline.
    b_rows += _rows(
        2,
        list(range(340, 347)),
        [350 + i for i in range(7)],
        [390 + i for i in range(7)],
        [1.8] * 7,
        det_start=11000,
    )
    cases["fragmented_stitch"] = (
        pd.DataFrame(f_rows),
        pd.DataFrame(b_rows),
        {
            "AGREEMENT_DISTANCE": 6.0,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )

    # Case: forward-only subset
    f_rows = _rows(
        10,
        list(range(500, 511)),
        [700 + 3 * i for i in range(11)],
        [200 + 1.5 * i for i in range(11)],
        [0.7 + 0.02 * i for i in range(11)],
        det_start=12000,
    )
    cases["subset_forward_only"] = (
        pd.DataFrame(f_rows),
        pd.DataFrame(
            columns=[
                "TrajectoryID",
                "FrameID",
                "X",
                "Y",
                "Theta",
                "State",
                "DetectionID",
            ]
        ),
        {
            "AGREEMENT_DISTANCE": 10.0,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )

    # Case: subset bidirectional
    frames = list(range(700, 721))
    t = np.arange(len(frames))
    f_rows = []
    b_rows = []
    f_rows += _rows(
        20,
        frames,
        (100 + 4 * t).tolist(),
        (300 + 1.5 * t).tolist(),
        (0.2 + 0.01 * t).tolist(),
        det_start=13000,
    )
    f_rows += _rows(
        21,
        frames,
        (380 - 3 * t).tolist(),
        (260 + 0.8 * t).tolist(),
        (2.2 + 0.02 * t).tolist(),
        det_start=14000,
    )
    b_rows += _rows(
        30,
        frames,
        (100 + 4 * t + np.sin(t) * 0.8).tolist(),
        (300 + 1.5 * t + np.cos(t) * 0.8).tolist(),
        (0.2 + 0.015 * t).tolist(),
        det_start=13000,
    )
    b_rows += _rows(
        31,
        frames,
        (380 - 3 * t + np.sin(t) * 0.7).tolist(),
        (260 + 0.8 * t + np.cos(t) * 0.7).tolist(),
        (2.2 + 0.01 * t).tolist(),
        det_start=14000,
    )
    cases["subset_bidirectional"] = (
        pd.DataFrame(f_rows),
        pd.DataFrame(b_rows),
        {
            "AGREEMENT_DISTANCE": 12.0,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )

    # Case: duplicate rows and NaN optional columns
    f_rows = [
        {
            "TrajectoryID": 0,
            "FrameID": 0,
            "X": 10.0,
            "Y": 10.0,
            "Theta": 0.1,
            "State": "active",
            "DetectionConfidence": 0.9,
            "AssignmentConfidence": np.nan,
            "PositionUncertainty": 1.5,
        },
        {
            "TrajectoryID": 0,
            "FrameID": 1,
            "X": 11.0,
            "Y": 11.0,
            "Theta": 0.2,
            "State": "active",
            "DetectionConfidence": 0.8,
            "AssignmentConfidence": 0.7,
            "PositionUncertainty": 1.3,
        },
        {
            "TrajectoryID": 0,
            "FrameID": 1,
            "X": 11.2,
            "Y": 11.1,
            "Theta": 0.25,
            "State": "active",
            "DetectionConfidence": 0.75,
            "AssignmentConfidence": 0.8,
            "PositionUncertainty": 1.6,
        },
        {
            "TrajectoryID": 0,
            "FrameID": 2,
            "X": np.nan,
            "Y": np.nan,
            "Theta": np.nan,
            "State": "occluded",
            "DetectionConfidence": np.nan,
            "AssignmentConfidence": np.nan,
            "PositionUncertainty": np.nan,
        },
        {
            "TrajectoryID": 0,
            "FrameID": 3,
            "X": 13.0,
            "Y": 12.9,
            "Theta": 0.3,
            "State": "active",
            "DetectionConfidence": 0.85,
            "AssignmentConfidence": 0.82,
            "PositionUncertainty": 1.1,
        },
    ]
    b_rows = [
        {
            "TrajectoryID": 1,
            "FrameID": 0,
            "X": 10.4,
            "Y": 10.1,
            "Theta": 3.2,
            "State": "active",
            "DetectionConfidence": 0.88,
            "AssignmentConfidence": 0.7,
            "PositionUncertainty": 1.4,
        },
        {
            "TrajectoryID": 1,
            "FrameID": 1,
            "X": 11.4,
            "Y": 10.9,
            "Theta": 3.3,
            "State": "active",
            "DetectionConfidence": 0.77,
            "AssignmentConfidence": 0.79,
            "PositionUncertainty": 1.7,
        },
        {
            "TrajectoryID": 1,
            "FrameID": 2,
            "X": 12.1,
            "Y": 12.0,
            "Theta": 3.4,
            "State": "active",
            "DetectionConfidence": 0.81,
            "AssignmentConfidence": 0.83,
            "PositionUncertainty": 1.2,
        },
        {
            "TrajectoryID": 1,
            "FrameID": 3,
            "X": 13.2,
            "Y": 13.0,
            "Theta": 3.5,
            "State": "active",
            "DetectionConfidence": 0.9,
            "AssignmentConfidence": 0.86,
            "PositionUncertainty": 1.0,
        },
    ]
    cases["duplicates_and_nan_optional"] = (
        pd.DataFrame(f_rows),
        pd.DataFrame(b_rows),
        {
            "AGREEMENT_DISTANCE": 2.5,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )

    # Fuzz-lite deterministic random cases
    for seed in (7, 11, 19):
        rng = np.random.default_rng(seed)
        f_rows = []
        b_rows = []
        for tid in range(3):
            start = int(rng.integers(0, 12))
            length = int(rng.integers(8, 14))
            frames = list(range(900 + start, 900 + start + length))
            x0 = float(rng.uniform(50, 350))
            y0 = float(rng.uniform(50, 350))
            dx = float(rng.uniform(-3, 3))
            dy = float(rng.uniform(-3, 3))
            x = [x0 + i * dx + float(rng.normal(0, 0.6)) for i in range(length)]
            y = [y0 + i * dy + float(rng.normal(0, 0.6)) for i in range(length)]
            theta = [float((0.15 * i + tid) % (2 * np.pi)) for i in range(length)]
            f_rows += _rows(
                tid, frames, x, y, theta, det_start=15000 + seed * 10 + tid * 100
            )

            # Backward companion with slight perturbation and occasional drop
            keep_mask = rng.random(length) > 0.15
            b_frames = [f for i, f in enumerate(frames) if keep_mask[i]]
            b_x = [
                x[i] + float(rng.normal(0, 0.8)) for i in range(length) if keep_mask[i]
            ]
            b_y = [
                y[i] + float(rng.normal(0, 0.8)) for i in range(length) if keep_mask[i]
            ]
            b_theta = [
                float((theta[i] + 0.05) % (2 * np.pi))
                for i in range(length)
                if keep_mask[i]
            ]
            b_rows += _rows(
                100 + tid,
                b_frames,
                b_x,
                b_y,
                b_theta,
                det_start=15000 + seed * 10 + tid * 100,
            )

        cases[f"fuzz_seed_{seed}"] = (
            pd.DataFrame(f_rows),
            pd.DataFrame(b_rows),
            {
                "AGREEMENT_DISTANCE": 10.0,
                "MIN_OVERLAP_FRAMES": 2,
                "MIN_TRAJECTORY_LENGTH": 2,
            },
        )

    return cases


def build_interpolate_cases() -> dict[str, tuple[pd.DataFrame, dict]]:
    cases: dict[str, tuple[pd.DataFrame, dict]] = {}

    # Case: short gaps
    rows = [
        {
            "TrajectoryID": 0,
            "FrameID": 0,
            "X": 0.0,
            "Y": 0.0,
            "Theta": 0.0,
            "State": "active",
            "DetectionConfidence": 0.9,
        },
        {
            "TrajectoryID": 0,
            "FrameID": 2,
            "X": 2.0,
            "Y": 2.0,
            "Theta": 0.2,
            "State": "active",
            "DetectionConfidence": 0.8,
        },
        {
            "TrajectoryID": 0,
            "FrameID": 3,
            "X": 3.0,
            "Y": 3.2,
            "Theta": 0.3,
            "State": "active",
            "DetectionConfidence": 0.75,
        },
        {
            "TrajectoryID": 0,
            "FrameID": 6,
            "X": 6.0,
            "Y": 6.1,
            "Theta": 0.6,
            "State": "active",
            "DetectionConfidence": 0.72,
        },
        {
            "TrajectoryID": 1,
            "FrameID": 10,
            "X": 12.0,
            "Y": 12.0,
            "Theta": 1.0,
            "State": "active",
            "DetectionConfidence": 0.99,
        },
    ]
    cases["short_gaps_linear"] = (
        pd.DataFrame(rows),
        {"method": "linear", "max_gap": 2},
    )

    # Case: duplicate frame IDs + NaN optional columns
    rows = [
        {
            "TrajectoryID": 2,
            "FrameID": 5,
            "X": 100.0,
            "Y": 100.0,
            "Theta": 0.1,
            "State": "active",
            "DetectionConfidence": 0.8,
            "AssignmentConfidence": np.nan,
            "PositionUncertainty": 2.0,
        },
        {
            "TrajectoryID": 2,
            "FrameID": 5,
            "X": 102.0,
            "Y": 102.0,
            "Theta": 0.15,
            "State": "active",
            "DetectionConfidence": 0.7,
            "AssignmentConfidence": 0.7,
            "PositionUncertainty": 1.9,
        },
        {
            "TrajectoryID": 2,
            "FrameID": 7,
            "X": np.nan,
            "Y": np.nan,
            "Theta": np.nan,
            "State": "occluded",
            "DetectionConfidence": np.nan,
            "AssignmentConfidence": np.nan,
            "PositionUncertainty": np.nan,
        },
        {
            "TrajectoryID": 2,
            "FrameID": 8,
            "X": 108.0,
            "Y": 108.0,
            "Theta": 0.4,
            "State": "active",
            "DetectionConfidence": 0.85,
            "AssignmentConfidence": 0.81,
            "PositionUncertainty": 1.4,
        },
        {
            "TrajectoryID": 3,
            "FrameID": 20,
            "X": 50.0,
            "Y": 51.0,
            "Theta": 2.2,
            "State": "active",
            "DetectionConfidence": 0.95,
            "AssignmentConfidence": 0.9,
            "PositionUncertainty": 1.0,
        },
        {
            "TrajectoryID": 3,
            "FrameID": 22,
            "X": 54.0,
            "Y": 55.0,
            "Theta": 2.4,
            "State": "active",
            "DetectionConfidence": 0.92,
            "AssignmentConfidence": 0.88,
            "PositionUncertainty": 1.1,
        },
    ]
    cases["duplicate_frames_nan_optional"] = (
        pd.DataFrame(rows),
        {"method": "linear", "max_gap": 3},
    )

    return cases


def _generate_resolve_fixtures() -> list[str]:
    names = []
    for name, (forward, backward, params) in build_resolve_cases().items():
        resolved = resolve_trajectories(
            _split(forward), _split(backward), params=params
        )
        expected = (
            pd.concat(resolved, ignore_index=True, sort=False)
            if resolved
            else pd.DataFrame()
        )
        _write_case(
            RESOLVE_ROOT / name,
            forward=forward,
            backward=backward,
            params=params,
            expected=expected,
        )
        names.append(name)
    return names


def _generate_interpolate_fixtures() -> list[str]:
    names = []
    for name, (input_df, params) in build_interpolate_cases().items():
        expected = interpolate_trajectories(
            input_df.copy(), method=params["method"], max_gap=params["max_gap"]
        )
        _write_interp_case(
            INTERP_ROOT / name, input_df=input_df, params=params, expected=expected
        )
        names.append(name)
    return names


def _generate_benchmark_baseline(cases: list[str]) -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for case in cases:
        forward = pd.read_csv(RESOLVE_ROOT / case / "input_forward.csv")
        backward = pd.read_csv(RESOLVE_ROOT / case / "input_backward.csv")
        params = json.loads(
            (RESOLVE_ROOT / case / "params.json").read_text(encoding="utf-8")
        )
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            resolve_trajectories(_split(forward), _split(backward), params=params)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        baseline[case] = statistics.median(times)
    (BENCH_ROOT / "baseline_runtime_ms.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    RESOLVE_ROOT.mkdir(parents=True, exist_ok=True)
    INTERP_ROOT.mkdir(parents=True, exist_ok=True)
    resolve_names = _generate_resolve_fixtures()
    _generate_interpolate_fixtures()
    _generate_benchmark_baseline(["heavy_overlap_many_to_many", "subset_bidirectional"])
    print(
        f"Generated {len(resolve_names)} resolve fixtures and interpolation fixtures in {ROOT}"
    )


if __name__ == "__main__":
    main()
