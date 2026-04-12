#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

from hydra_suite.integrations.sleap.service import _SLEAP_SERVICE_CODE


def _load_images(root: Path, count: int) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
    if not images:
        raise RuntimeError(f"No images found under {root}")
    if len(images) >= count:
        return images[:count]
    out = list(images)
    while len(out) < count:
        out.append(images[len(out) % len(images)])
    return out


def _run_native_with_breakdown(
    namespace: dict[str, object],
    labels: object,
    model_dir: str,
    device: str,
    batch: int,
    max_instances: int,
) -> tuple[object, dict[str, object]]:
    logs: list[str] = []
    namespace["_log"] = lambda msg: logs.append(str(msg))

    t0 = time.perf_counter()
    normalized_device = namespace["_normalize_device"](device)
    normalize_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    predictor = namespace["_load_predictor"](model_dir, normalized_device)
    load_predictor_ms = (time.perf_counter() - t0) * 1000.0

    path_used = None
    predictor_type = type(predictor).__name__ if predictor is not None else None
    predictor_methods = []
    predictor_call_ms = None
    predict_import_ms = None
    predict_run_ms = None
    output = None

    if predictor is not None:
        predictor_methods = [
            name
            for name in ("predict", "predict_labels", "run")
            if hasattr(predictor, name)
        ]
        t0 = time.perf_counter()
        try:
            output = namespace["_predict_with_predictor"](
                predictor, labels, batch, max_instances
            )
        except Exception as exc:
            logs.append(f"predictor_api_exception: {exc}")
            output = None
        predictor_call_ms = (time.perf_counter() - t0) * 1000.0
        if output is not None:
            path_used = "predictor_api"

    if output is None:
        t0 = time.perf_counter()
        from sleap_nn import predict as predict_mod

        predict_import_ms = (time.perf_counter() - t0) * 1000.0
        if not hasattr(predict_mod, "run_inference"):
            raise RuntimeError("sleap_nn.predict.run_inference is unavailable")

        fn = predict_mod.run_inference
        sig = inspect.signature(fn)
        kwargs = {}
        if "input_labels" in sig.parameters:
            kwargs["input_labels"] = labels
        if "model_paths" in sig.parameters:
            kwargs["model_paths"] = [model_dir]
        if (
            "device" in sig.parameters
            and normalized_device
            and normalized_device != "auto"
        ):
            kwargs["device"] = normalized_device
        if "batch_size" in sig.parameters:
            kwargs["batch_size"] = batch
        if "max_instances" in sig.parameters:
            kwargs["max_instances"] = max_instances

        t0 = time.perf_counter()
        output = fn(**kwargs)
        predict_run_ms = (time.perf_counter() - t0) * 1000.0
        path_used = "sleap_nn.predict.run_inference"

    breakdown = {
        "path_used": path_used,
        "normalize_device_ms": round(normalize_ms, 3),
        "load_predictor_ms": round(load_predictor_ms, 3),
        "predictor_loaded": predictor is not None,
        "predictor_type": predictor_type,
        "predictor_methods": predictor_methods,
        "predictor_call_ms": (
            round(predictor_call_ms, 3) if predictor_call_ms is not None else None
        ),
        "predict_import_ms": (
            round(predict_import_ms, 3) if predict_import_ms is not None else None
        ),
        "predict_run_ms": (
            round(predict_run_ms, 3) if predict_run_ms is not None else None
        ),
        "logs": logs,
    }
    return output, breakdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--skeleton-json", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=21)
    parser.add_argument("--max-instances", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    namespace: dict[str, object] = {}
    exec(_SLEAP_SERVICE_CODE, namespace)
    namespace["_log"] = lambda _msg: None

    image_root = Path(args.image_root).expanduser().resolve()
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    keypoint_names = json.loads(Path(args.skeleton_json).read_text(encoding="utf-8"))[
        "keypoint_names"
    ]
    images = _load_images(image_root, args.batch)

    sio = namespace["sio"]
    skeleton = namespace["_make_skeleton"](keypoint_names, [])
    video = namespace["_make_video"]([str(path) for path in images])
    frames = namespace["_make_labeled_frames"](video, len(images))
    try:
        labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=frames)
    except Exception:
        labels = sio.Labels(videos=[video], skeletons=[skeleton])

    namespace["_clear_predictor"]()

    timings_ms: list[float] = []
    breakdowns: list[dict[str, object]] = []
    for _ in range(max(1, args.runs)):
        start = time.perf_counter()
        if args.breakdown:
            output, breakdown = _run_native_with_breakdown(
                namespace,
                labels,
                model_dir,
                args.device,
                int(args.batch),
                int(args.max_instances),
            )
            breakdowns.append(breakdown)
        else:
            output = namespace["_run_inference"](
                labels,
                model_dir,
                args.device,
                int(args.batch),
                int(args.max_instances),
            )
        labels_out = namespace["_labels_from_output"](output)
        if labels_out is None:
            raise RuntimeError("Native inference returned no labels output")
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    state = namespace["_state"]
    result = {
        "device_requested": args.device,
        "device_used": state.get("device"),
        "batch": int(args.batch),
        "runs_ms": [round(value, 3) for value in timings_ms],
        "first_image": str(images[0]),
        "image_count": len(images),
    }
    if breakdowns:
        result["breakdowns"] = breakdowns
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
