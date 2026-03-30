"""Orientation-fixed per-track video export for MAT individual outputs."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pandas as pd

from ....data.detection_cache import DetectionCache
from ..geometry import ellipse_axes_from_area, ellipse_to_obb_corners

logger = logging.getLogger(__name__)


@dataclass
class FrameTask:
    frame_id: int
    trajectory_id: int
    affine: np.ndarray
    out_w: int
    out_h: int
    corners: np.ndarray
    expanded_corners: np.ndarray
    polygon_index: int


@dataclass
class FrameBundle:
    tasks: list[FrameTask] = field(default_factory=list)
    polygons: list[np.ndarray] = field(default_factory=list)


@dataclass
class OrientedTrackVideoExportResult:
    output_dir: str
    dataset_dir: str
    exported_videos: int
    exported_tracks: int
    exported_frames: int
    skipped_tracks: int
    missing_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_individual_dataset_dir(
    output_dir: str | Path | None,
    dataset_name: str | None = None,
    run_id: str | None = None,
) -> Optional[Path]:
    """Resolve the dataset directory used for individual outputs."""
    if not output_dir:
        return None
    root = Path(output_dir).expanduser()
    if not str(root).strip():
        return None

    name_part = str(dataset_name or "").strip()
    run_part = str(run_id or "").strip()
    if run_part:
        folder_name = f"{name_part}_{run_part}" if name_part else run_part
        return root / folder_name

    if not root.exists():
        return None

    if name_part:
        matches = sorted(root.glob(f"{name_part}_*"), key=lambda p: p.name)
    else:
        matches = sorted(
            (p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name
        )
    return matches[-1] if matches else None


class OrientedTrackVideoExporter:
    """Build per-track orientation-fixed videos directly from cached geometry."""

    def __init__(
        self,
        dataset_dir: str | Path,
        final_csv_path: str | Path,
        *,
        video_path: str | Path,
        detection_cache_path: str | Path,
        interpolated_roi_npz_path: str | Path | None = None,
        fps: float,
        padding_fraction: float = 0.1,
        background_color: tuple[int, int, int] = (0, 0, 0),
        suppress_foreign_obb: bool = False,
        output_subdir: str = "oriented_videos",
        codec: str = "mp4v",
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.final_csv_path = Path(final_csv_path).expanduser().resolve()
        self.video_path = Path(video_path).expanduser().resolve()
        self.detection_cache_path = Path(detection_cache_path).expanduser().resolve()
        self.interpolated_roi_npz_path = (
            Path(interpolated_roi_npz_path).expanduser().resolve()
            if interpolated_roi_npz_path
            else None
        )
        self.fps = max(0.1, float(fps or 0.0))
        self.padding_fraction = max(0.0, float(padding_fraction or 0.0))
        self.background_color = self._normalize_background_color(background_color)
        self.suppress_foreign_obb = bool(suppress_foreign_obb)
        self.output_dir = self.dataset_dir / str(output_subdir).strip()
        self.codec = str(codec or "mp4v")

    def export(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> OrientedTrackVideoExportResult:
        self._emit(progress_callback, 2, "Loading final cleaned trajectories...")
        trajectories_df = self._load_final_dataframe()
        self._emit(progress_callback, 8, "Loading interpolated ROI cache...")
        interp_lookup = self._load_interpolated_roi_lookup()
        self._emit(progress_callback, 12, "Building cached geometry tasks...")
        frame_bundles, track_sizes, missing_rows = self._build_frame_bundles(
            trajectories_df,
            interp_lookup,
            should_stop=should_stop,
            progress_callback=progress_callback,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        track_ids = sorted(track_sizes)
        if not track_ids:
            logger.warning(
                "No oriented track videos exported: no cached geometry matched final trajectory rows."
            )
            return OrientedTrackVideoExportResult(
                output_dir=str(self.output_dir),
                dataset_dir=str(self.dataset_dir),
                exported_videos=0,
                exported_tracks=0,
                exported_frames=0,
                skipped_tracks=0,
                missing_rows=int(missing_rows),
            )

        self._emit(progress_callback, 35, "Streaming source video for track export...")
        exported_videos, exported_frames, skipped_tracks = self._write_videos(
            frame_bundles,
            track_sizes,
            should_stop=should_stop,
            progress_callback=progress_callback,
        )
        self._emit(progress_callback, 100, "Oriented track videos complete.")
        return OrientedTrackVideoExportResult(
            output_dir=str(self.output_dir),
            dataset_dir=str(self.dataset_dir),
            exported_videos=int(exported_videos),
            exported_tracks=int(len(track_ids)),
            exported_frames=int(exported_frames),
            skipped_tracks=int(skipped_tracks),
            missing_rows=int(missing_rows),
        )

    @staticmethod
    def _emit(
        callback: Optional[Callable[[int, str], None]], pct: int, message: str
    ) -> None:
        if callback is None:
            return
        try:
            callback(int(max(0, min(100, pct))), str(message))
        except Exception:
            logger.debug(
                "Progress callback failed during oriented video export.", exc_info=True
            )

    @staticmethod
    def _normalize_background_color(color) -> tuple[int, int, int]:
        if isinstance(color, (list, tuple)) and len(color) == 3:
            return tuple(int(np.clip(v, 0, 255)) for v in color)
        fill = int(np.clip(color if color is not None else 0, 0, 255))
        return (fill, fill, fill)

    def _load_final_dataframe(self) -> pd.DataFrame:
        if not self.final_csv_path.exists():
            raise FileNotFoundError(
                f"Missing final trajectories CSV: {self.final_csv_path}"
            )
        df = pd.read_csv(self.final_csv_path)
        required = {"TrajectoryID", "FrameID"}
        if not required.issubset(df.columns):
            raise ValueError(
                "Final trajectories CSV must contain TrajectoryID and FrameID columns."
            )
        df = df.copy()
        for col in ("TrajectoryID", "FrameID", "DetectionID", "Theta"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["TrajectoryID", "FrameID"]).sort_values(
            ["FrameID", "TrajectoryID"], kind="stable"
        )

    def _load_interpolated_roi_lookup(self) -> dict[tuple[int, int], dict[str, Any]]:
        path = self.interpolated_roi_npz_path
        if path is None or not path.exists():
            return {}
        payload = np.load(str(path), allow_pickle=True)
        try:
            frame_ids = payload["frame_id"]
            trajectory_ids = payload["trajectory_id"]
            cx = payload["cx"]
            cy = payload["cy"]
            width = payload["w"]
            height = payload["h"]
            theta = payload["theta"]
            obb_corners = payload["obb_corners"]
            lookup: dict[tuple[int, int], dict[str, Any]] = {}
            total = min(
                len(frame_ids),
                len(trajectory_ids),
                len(cx),
                len(cy),
                len(width),
                len(height),
                len(theta),
                len(obb_corners),
            )
            for idx in range(total):
                lookup[(int(frame_ids[idx]), int(trajectory_ids[idx]))] = {
                    "cx": float(cx[idx]),
                    "cy": float(cy[idx]),
                    "w": float(width[idx]),
                    "h": float(height[idx]),
                    "theta": float(theta[idx]),
                    "obb_corners": np.asarray(obb_corners[idx], dtype=np.float32),
                }
            return lookup
        finally:
            try:
                payload.close()
            except Exception:
                pass

    def _build_frame_bundles(
        self,
        trajectories_df: pd.DataFrame,
        interp_lookup: dict[tuple[int, int], dict[str, Any]],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> tuple[dict[int, FrameBundle], dict[int, tuple[int, int]], int]:
        actual_rows_by_frame: dict[int, list[Any]] = defaultdict(list)
        interp_rows_by_frame: dict[int, list[Any]] = defaultdict(list)
        for row in trajectories_df.itertuples(index=False):
            frame_id = int(row.FrameID)
            det_id = getattr(row, "DetectionID", np.nan)
            if pd.notna(det_id):
                actual_rows_by_frame[frame_id].append(row)
            else:
                interp_rows_by_frame[frame_id].append(row)

        frame_bundles: dict[int, FrameBundle] = {}
        track_sizes: dict[int, tuple[int, int]] = {}
        missing_rows = 0

        if actual_rows_by_frame:
            if not self.detection_cache_path.exists():
                raise FileNotFoundError(
                    f"Missing detection cache for oriented track videos: {self.detection_cache_path}"
                )
            with DetectionCache(self.detection_cache_path, mode="r") as detection_cache:
                if not detection_cache.is_compatible():
                    raise ValueError(
                        f"Incompatible detection cache for oriented track videos: {self.detection_cache_path}"
                    )
                actual_frames = sorted(actual_rows_by_frame)
                total_actual = len(actual_frames)
                for index, frame_id in enumerate(actual_frames, start=1):
                    if should_stop and should_stop():
                        break
                    self._emit(
                        progress_callback,
                        12 + int(16 * index / max(1, total_actual)),
                        f"Preparing cached detection geometry... {index}/{total_actual}",
                    )
                    rows = actual_rows_by_frame.get(frame_id, [])
                    bundle = frame_bundles.setdefault(frame_id, FrameBundle())
                    missing_rows += self._add_actual_tasks(
                        detection_cache,
                        frame_id,
                        rows,
                        bundle,
                        track_sizes,
                    )

        interp_frames = sorted(interp_rows_by_frame)
        total_interp = len(interp_frames)
        for index, frame_id in enumerate(interp_frames, start=1):
            if should_stop and should_stop():
                break
            self._emit(
                progress_callback,
                28 + int(6 * index / max(1, total_interp)),
                f"Preparing interpolated geometry... {index}/{total_interp}",
            )
            bundle = frame_bundles.setdefault(frame_id, FrameBundle())
            for row in interp_rows_by_frame.get(frame_id, []):
                traj_id = int(row.TrajectoryID)
                record = interp_lookup.get((frame_id, traj_id))
                if record is None:
                    missing_rows += 1
                    continue
                theta = self._resolve_task_theta(row, record.get("theta", 0.0))
                task = self._build_task(
                    frame_id=frame_id,
                    trajectory_id=traj_id,
                    center_x=float(record["cx"]),
                    center_y=float(record["cy"]),
                    width=float(record["w"]),
                    height=float(record["h"]),
                    theta=theta,
                    corners=np.asarray(record["obb_corners"], dtype=np.float32),
                    polygon_index=len(bundle.polygons),
                )
                if task is None:
                    missing_rows += 1
                    continue
                bundle.polygons.append(task.corners)
                bundle.tasks.append(task)
                track_sizes[traj_id] = self._merge_canvas_size(
                    track_sizes.get(traj_id), (task.out_w, task.out_h)
                )

        frame_bundles = {k: v for k, v in frame_bundles.items() if v.tasks}
        return frame_bundles, track_sizes, missing_rows

    def _add_actual_tasks(
        self,
        detection_cache: DetectionCache,
        frame_id: int,
        rows: list[Any],
        bundle: FrameBundle,
        track_sizes: dict[int, tuple[int, int]],
    ) -> int:
        missing_rows = 0
        (
            meas,
            _sizes,
            shapes,
            _confidences,
            obb_corners,
            detection_ids,
            _heading_hints,
            _directed_mask,
            _canonical_affines,
            _canvas_dims,
            _M_inverse,
        ) = detection_cache.get_frame(frame_id)
        det_index = {}
        for idx, det_id in enumerate(detection_ids or []):
            try:
                det_index[int(det_id)] = idx
            except Exception:
                continue

        for row in rows:
            traj_id = int(row.TrajectoryID)
            det_id = getattr(row, "DetectionID", np.nan)
            if pd.isna(det_id):
                missing_rows += 1
                continue
            idx = det_index.get(int(det_id))
            if idx is None:
                missing_rows += 1
                continue
            corners = None
            if obb_corners and idx < len(obb_corners):
                corners = np.asarray(obb_corners[idx], dtype=np.float32)
            elif idx < len(meas) and idx < len(shapes):
                shape = shapes[idx]
                if shape is not None and len(shape) >= 2:
                    area = float(shape[0])
                    aspect_ratio = float(shape[1])
                    if area > 0.0 and aspect_ratio > 0.0:
                        theta = float(meas[idx][2]) if len(meas[idx]) > 2 else 0.0
                        cx = float(meas[idx][0])
                        cy = float(meas[idx][1])
                        corners = ellipse_to_obb_corners(
                            cx,
                            cy,
                            *ellipse_axes_from_area(area, aspect_ratio),
                            theta,
                        )
            if corners is None or corners.shape != (4, 2):
                missing_rows += 1
                continue
            width, height = self._edge_lengths(corners)
            center = corners.mean(axis=0)
            theta_fallback = (
                float(meas[idx][2]) if idx < len(meas) and len(meas[idx]) > 2 else 0.0
            )
            theta = self._resolve_task_theta(row, theta_fallback)
            task = self._build_task(
                frame_id=frame_id,
                trajectory_id=traj_id,
                center_x=float(center[0]),
                center_y=float(center[1]),
                width=width,
                height=height,
                theta=theta,
                corners=corners,
                polygon_index=len(bundle.polygons),
            )
            if task is None:
                missing_rows += 1
                continue
            bundle.polygons.append(task.corners)
            bundle.tasks.append(task)
            track_sizes[traj_id] = self._merge_canvas_size(
                track_sizes.get(traj_id), (task.out_w, task.out_h)
            )
        return missing_rows

    def _build_task(
        self,
        *,
        frame_id: int,
        trajectory_id: int,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        theta: float,
        corners: np.ndarray,
        polygon_index: int,
    ) -> Optional[FrameTask]:
        if not np.isfinite(center_x) or not np.isfinite(center_y):
            return None
        if not np.isfinite(width) or not np.isfinite(height):
            return None
        box_w = max(1.0, float(width))
        box_h = max(1.0, float(height))
        affine, out_w, out_h = self._compute_affine(
            center_x,
            center_y,
            box_w,
            box_h,
            float(theta),
        )
        return FrameTask(
            frame_id=int(frame_id),
            trajectory_id=int(trajectory_id),
            affine=affine,
            out_w=out_w,
            out_h=out_h,
            corners=np.asarray(corners, dtype=np.float32),
            expanded_corners=self._expand_corners(corners, self.padding_fraction),
            polygon_index=int(polygon_index),
        )

    def _write_videos(
        self,
        frame_bundles: dict[int, FrameBundle],
        track_sizes: dict[int, tuple[int, int]],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> tuple[int, int, int]:
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Missing source video for oriented track export: {self.video_path}"
            )
        needed_frames = sorted(frame_bundles)
        if not needed_frames:
            return 0, 0, 0

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open source video: {self.video_path}")

        writers: dict[int, cv2.VideoWriter] = {}
        frames_written_by_track: dict[int, int] = defaultdict(int)
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

        # Pre-allocate one reusable canvas per track to avoid per-frame np.full()
        _canvases: dict[int, np.ndarray] = {}
        for traj_id, (cw, ch) in track_sizes.items():
            _canvases[traj_id] = np.full(
                (ch, cw, 3), self.background_color, dtype=np.uint8
            )

        def _read_frame(fid):
            """Read a single frame (called from background thread)."""
            nonlocal current_pos
            if fid != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frm = cap.read()
            current_pos = int(fid) + 1
            return ok, frm

        try:
            total_frames = len(needed_frames)
            # Use a single-thread executor for 1-frame read-ahead
            with ThreadPoolExecutor(max_workers=1) as reader:
                # Submit the first read
                future = reader.submit(_read_frame, needed_frames[0])

                for index, frame_id in enumerate(needed_frames, start=1):
                    if should_stop and should_stop():
                        break
                    self._emit(
                        progress_callback,
                        35 + int(60 * index / max(1, total_frames)),
                        f"Writing oriented track videos... {index}/{total_frames}",
                    )

                    ok, frame = future.result()

                    # Submit read-ahead for next frame while processing this one
                    if index < total_frames:
                        future = reader.submit(_read_frame, needed_frames[index])

                    if not ok or frame is None:
                        continue

                    bundle = frame_bundles.get(frame_id)
                    if bundle is None:
                        continue
                    for task in bundle.tasks:
                        canvas_size = track_sizes.get(task.trajectory_id)
                        if canvas_size is None:
                            continue
                        writer = writers.get(task.trajectory_id)
                        if writer is None:
                            writer = self._open_writer(task.trajectory_id, canvas_size)
                            writers[task.trajectory_id] = writer
                        rendered = self._render_task(
                            frame,
                            task,
                            bundle.polygons,
                            canvas_size,
                            canvas=_canvases.get(task.trajectory_id),
                        )
                        if rendered is None:
                            continue
                        writer.write(rendered)
                        frames_written_by_track[task.trajectory_id] += 1
        finally:
            cap.release()
            for writer in writers.values():
                try:
                    writer.release()
                except Exception:
                    pass

        exported_videos = 0
        skipped_tracks = 0
        exported_frames = 0
        for traj_id, canvas_size in track_sizes.items():
            count = int(frames_written_by_track.get(traj_id, 0))
            if count > 0:
                exported_videos += 1
                exported_frames += count
            else:
                skipped_tracks += 1
                output_path = self.output_dir / f"trajectory_{int(traj_id):04d}.mp4"
                try:
                    output_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return exported_videos, exported_frames, skipped_tracks

    def _open_writer(
        self, trajectory_id: int, canvas_size: tuple[int, int]
    ) -> cv2.VideoWriter:
        output_path = self.output_dir / f"trajectory_{int(trajectory_id):04d}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            canvas_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create oriented track video: {output_path}")
        return writer

    def _render_task(
        self,
        frame: np.ndarray,
        task: FrameTask,
        frame_polygons: list[np.ndarray],
        canvas_size: tuple[int, int],
        canvas: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        warped = cv2.warpAffine(
            frame,
            task.affine,
            (task.out_w, task.out_h),
            flags=getattr(cv2, "INTER_LINEAR", 1),
            borderMode=cv2.BORDER_REPLICATE,
        )
        if warped is None or warped.size == 0:
            return None

        mask = np.zeros((task.out_h, task.out_w), dtype=np.uint8)
        target_poly = self._transform_polygon(
            task.expanded_corners, task.affine, task.out_w, task.out_h
        )
        if target_poly is None:
            return None
        cv2.fillPoly(mask, [target_poly], 255)

        if self.suppress_foreign_obb and len(frame_polygons) > 1:
            for idx, corners in enumerate(frame_polygons):
                if idx == task.polygon_index:
                    continue
                foreign_poly = self._transform_polygon(
                    corners, task.affine, task.out_w, task.out_h
                )
                if foreign_poly is None:
                    continue
                cv2.fillPoly(mask, [foreign_poly], 0)

        masked = np.full_like(warped, self.background_color, dtype=np.uint8)
        valid = mask > 0
        masked[valid] = warped[valid]

        canvas_w, canvas_h = canvas_size
        if masked.shape[1] == canvas_w and masked.shape[0] == canvas_h:
            return masked

        # Reuse pre-allocated canvas when available, otherwise allocate
        if (
            canvas is not None
            and canvas.shape[0] == canvas_h
            and canvas.shape[1] == canvas_w
        ):
            canvas[:] = self.background_color
        else:
            canvas = np.full(
                (canvas_h, canvas_w, 3), self.background_color, dtype=np.uint8
            )
        x0 = max(0, (canvas_w - masked.shape[1]) // 2)
        y0 = max(0, (canvas_h - masked.shape[0]) // 2)
        x1 = min(canvas_w, x0 + masked.shape[1])
        y1 = min(canvas_h, y0 + masked.shape[0])
        canvas[y0:y1, x0:x1] = masked[: y1 - y0, : x1 - x0]
        return canvas

    def _compute_affine(
        self,
        center_x: float,
        center_y: float,
        box_w: float,
        box_h: float,
        theta: float,
    ) -> tuple[np.ndarray, int, int]:
        crop_w = max(8.0, float(box_w) * (1.0 + self.padding_fraction))
        crop_h = max(8.0, float(box_h) * (1.0 + self.padding_fraction))
        out_w, out_h = self._normalize_frame_size(
            int(round(crop_w)), int(round(crop_h))
        )

        cos_a = float(math.cos(theta))
        sin_a = float(math.sin(theta))
        half_w = crop_w * 0.5
        half_h = crop_h * 0.5
        src_pts = np.array(
            [
                [
                    center_x - half_w * cos_a + half_h * sin_a,
                    center_y - half_w * sin_a - half_h * cos_a,
                ],
                [
                    center_x + half_w * cos_a + half_h * sin_a,
                    center_y + half_w * sin_a - half_h * cos_a,
                ],
                [
                    center_x - half_w * cos_a - half_h * sin_a,
                    center_y - half_w * sin_a + half_h * cos_a,
                ],
            ],
            dtype=np.float32,
        )
        dst_pts = np.array(
            [[0, 0], [out_w - 1, 0], [0, out_h - 1]],
            dtype=np.float32,
        )
        affine = cv2.getAffineTransform(src_pts, dst_pts)
        return affine, out_w, out_h

    @staticmethod
    def _normalize_frame_size(width: int, height: int) -> tuple[int, int]:
        out_w = max(8, int(width))
        out_h = max(8, int(height))
        if out_w % 2:
            out_w += 1
        if out_h % 2:
            out_h += 1
        return out_w, out_h

    @staticmethod
    def _expand_corners(corners: np.ndarray, padding_fraction: float) -> np.ndarray:
        arr = np.asarray(corners, dtype=np.float32)
        if arr.shape != (4, 2):
            return arr
        centroid = arr.mean(axis=0)
        expanded = arr.copy()
        for idx in range(4):
            direction = arr[idx] - centroid
            expanded[idx] = centroid + direction * (1.0 + float(padding_fraction))
        return expanded.astype(np.float32)

    @staticmethod
    def _transform_polygon(
        corners: np.ndarray,
        affine: np.ndarray,
        out_w: int,
        out_h: int,
    ) -> Optional[np.ndarray]:
        arr = np.asarray(corners, dtype=np.float32)
        if arr.shape != (4, 2):
            return None
        pts = cv2.transform(arr.reshape(1, 4, 2), affine).reshape(4, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, out_w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, out_h - 1))
        return pts.astype(np.int32)

    @staticmethod
    def _edge_lengths(corners: np.ndarray) -> tuple[float, float]:
        arr = np.asarray(corners, dtype=np.float32)
        width = float(np.linalg.norm(arr[1] - arr[0]))
        height = float(np.linalg.norm(arr[2] - arr[1]))
        return max(1.0, width), max(1.0, height)

    @staticmethod
    def _merge_canvas_size(
        current: Optional[tuple[int, int]],
        candidate: tuple[int, int],
    ) -> tuple[int, int]:
        if current is None:
            return candidate
        return max(int(current[0]), int(candidate[0])), max(
            int(current[1]), int(candidate[1])
        )

    @staticmethod
    def _resolve_task_theta(row: Any, fallback_theta: float) -> float:
        theta = getattr(row, "Theta", np.nan)
        if pd.notna(theta):
            return float(theta)
        return float(fallback_theta)
