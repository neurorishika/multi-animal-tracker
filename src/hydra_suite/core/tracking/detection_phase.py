"""Batched YOLO detection phase for the tracking pipeline.

Runs YOLO detection on all frames (or a specified range) and caches
the results, reporting progress via callbacks.
"""

import logging
import time
from collections import deque

import cv2

from hydra_suite.utils.batch_optimizer import BatchOptimizer

logger = logging.getLogger(__name__)


def run_batched_detection_phase(
    cap,
    detection_cache,
    detector,
    params,
    start_frame,
    end_frame,
    is_stop_requested,
    on_progress=None,
    on_stats=None,
    profiler=None,
):
    """Run batched YOLO detection on a frame range and cache results.

    Args:
        cap: OpenCV VideoCapture object.
        detection_cache: DetectionCache for writing.
        detector: YOLOOBBDetector instance.
        params: Configuration parameters.
        start_frame: Starting frame index (0-based).
        end_frame: Ending frame index (0-based).
        is_stop_requested: Callable returning True when stop is requested.
        on_progress: Optional callback ``(percentage: int, status: str) -> None``.
        on_stats: Optional callback ``(stats: dict) -> None``.
        profiler: Optional TrackingProfiler.

    Returns:
        int: Total frames processed.
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: Batched YOLO Detection")
    logger.info("=" * 80)

    advanced_config = params.get("ADVANCED_CONFIG", {}).copy()
    advanced_config["enable_tensorrt"] = params.get("ENABLE_TENSORRT", False)
    advanced_config["tensorrt_max_batch_size"] = params.get(
        "TENSORRT_MAX_BATCH_SIZE", 16
    )
    batch_optimizer = BatchOptimizer(advanced_config)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = end_frame - start_frame + 1

    logger.info(
        f"Processing frame range: {start_frame} to {end_frame} ({total_frames} frames)"
    )

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    effective_width = int(frame_width * resize_factor)
    effective_height = int(frame_height * resize_factor)

    model_name = params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt")
    batch_size = batch_optimizer.estimate_batch_size(
        effective_width, effective_height, model_name
    )

    logger.info(f"Video: {frame_width}x{frame_height}, {total_frames} frames")
    if resize_factor < 1.0:
        logger.info(
            f"Resize factor: {resize_factor} → Effective: {effective_width}x{effective_height}"
        )
    logger.info(f"Batch size: {batch_size}")

    detection_start_time = time.time()
    batch_times = deque(maxlen=30)

    frame_idx = 0
    batch_count = 0
    total_batches = (total_frames + batch_size - 1) // batch_size

    while not is_stop_requested():
        batch_start_time = time.time()

        batch_frames = []
        batch_start_idx = frame_idx

        if profiler:
            profiler.tick("batched_frame_read")
        for _ in range(batch_size):
            if is_stop_requested():
                break
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_index = start_frame + frame_idx
            if current_frame_index > end_frame:
                break

            if resize_factor < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_factor,
                    fy=resize_factor,
                    interpolation=cv2.INTER_AREA,
                )

            batch_frames.append(frame)
            frame_idx += 1
        if profiler:
            profiler.tock("batched_frame_read")

        if not batch_frames:
            break

        if is_stop_requested():
            break

        batch_count += 1
        logger.info(
            f"Processing batch {batch_count}/{total_batches} ({len(batch_frames)} frames)"
        )

        def progress_cb(
            current,
            total,
            msg,
            _batch_start=batch_start_idx,
            _batch_num=batch_count,
            _total_batches=total_batches,
        ):
            if is_stop_requested():
                return
            if total <= 0:
                return
            if current != total and current % 10 != 0:
                return
            batch_fraction = float(current) / float(total)
            overall_processed = _batch_start + current
            overall_pct = (
                int((overall_processed * 100) / total_frames) if total_frames > 0 else 0
            )
            if on_progress:
                on_progress(
                    overall_pct,
                    "Detecting objects: "
                    f"batch {_batch_num}/{_total_batches}, "
                    f"within-batch {int(batch_fraction * 100)}% "
                    f"({current}/{total})",
                )

        batch_results = detector.detect_objects_batched(
            batch_frames,
            batch_start_idx,
            progress_cb,
            return_raw=True,
            profiler=profiler,
        )

        for local_idx, (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            raw_heading_hints,
            raw_directed_mask,
            raw_canonical_affines,
        ) in enumerate(batch_results):
            relative_idx = batch_start_idx + local_idx
            actual_frame_idx = start_frame + relative_idx
            detection_ids = [actual_frame_idx * 10000 + i for i in range(len(raw_meas))]
            detection_cache.add_frame(
                actual_frame_idx,
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                detection_ids,
                raw_heading_hints,
                raw_directed_mask,
                canonical_affines=raw_canonical_affines,
            )

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        elapsed = time.time() - detection_start_time

        if len(batch_times) > 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            frames_per_batch = (
                batch_size if len(batch_frames) == batch_size else len(batch_frames)
            )
            current_fps = frames_per_batch / avg_batch_time if avg_batch_time > 0 else 0
        else:
            current_fps = 0

        if current_fps > 0:
            remaining_frames = total_frames - frame_idx
            eta = remaining_frames / current_fps
        else:
            eta = 0

        percentage = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
        status_text = (
            f"Detecting objects: batch {batch_count}/{total_batches} ({percentage}%)"
        )
        if on_progress:
            on_progress(percentage, status_text)
        if on_stats:
            on_stats({"fps": current_fps, "elapsed": elapsed, "eta": eta})

    logger.info(
        f"Detection phase complete: {frame_idx} frames processed in {batch_count} batches"
    )
    return frame_idx
