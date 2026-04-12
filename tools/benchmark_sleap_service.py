#!/usr/bin/env python3
"""Benchmark native SLEAP service transport overhead independently of TrackerKit.

This tool focuses on the service transport path that matters for realtime pose:

- direct temp-PNG crop transport
- shared-memory transport plus native fallback materialization
- optional probe of native array-video support in a target SLEAP env

It does not require a trained model. The goal is to measure whether the service
transport itself is adding avoidable overhead before inference starts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np


def _make_crops(batch_size: int, crop_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=(crop_size, crop_size, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]


def _decode_image_payloads(payloads: list[dict]) -> tuple[list[str], list[np.ndarray]]:
    image_ids: list[str] = []
    image_arrays: list[np.ndarray] = []
    for payload in payloads:
        image_id = str(payload.get("id") or "").strip()
        shape = tuple(int(v) for v in payload["shape"])
        shm = shared_memory.SharedMemory(name=str(payload["shm_name"]))
        try:
            arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf).copy()
        finally:
            shm.close()
        image_ids.append(image_id)
        image_arrays.append(arr)
    return image_ids, image_arrays


def _materialize_image_arrays(
    image_ids: list[str], image_arrays: list[np.ndarray], tmp_dir: Path
) -> list[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for idx, arr in enumerate(image_arrays):
        path = tmp_dir / f"img_{idx:06d}.png"
        ok = cv2.imwrite(str(path), np.asarray(arr, dtype=np.uint8))
        if not ok:
            raise RuntimeError(f"Failed writing benchmark image: {path}")
        out_paths.append(path)
    return out_paths


def _bench_temp_png_write(crops: list[np.ndarray], iterations: int) -> list[float]:
    timings: list[float] = []
    with tempfile.TemporaryDirectory(prefix="sleap_bench_png_") as tmp:
        root = Path(tmp)
        for _ in range(iterations):
            created: list[Path] = []
            t0 = time.perf_counter()
            for idx, crop in enumerate(crops):
                path = root / f"crop_{idx:06d}.png"
                if not cv2.imwrite(str(path), crop):
                    raise RuntimeError(f"Failed writing benchmark crop: {path}")
                created.append(path)
            timings.append((time.perf_counter() - t0) * 1000.0)
            for path in created:
                path.unlink(missing_ok=True)
    return timings


def _bench_shared_memory_fallback(
    crops: list[np.ndarray], iterations: int
) -> dict[str, list[float]]:
    encode_ms: list[float] = []
    decode_ms: list[float] = []
    materialize_ms: list[float] = []
    total_ms: list[float] = []
    for _ in range(iterations):
        payloads: list[dict] = []
        handles: list[shared_memory.SharedMemory] = []
        t0 = time.perf_counter()
        for idx, crop in enumerate(crops):
            arr = np.ascontiguousarray(np.asarray(crop, dtype=np.uint8))
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
            payloads.append(
                {
                    "id": f"inmem_crop_{idx:06d}",
                    "shape": [int(v) for v in arr.shape],
                    "dtype": "uint8",
                    "shm_name": shm.name,
                    "nbytes": int(arr.nbytes),
                }
            )
            handles.append(shm)
        encode_ms.append((time.perf_counter() - t0) * 1000.0)
        try:
            t1 = time.perf_counter()
            image_ids, image_arrays = _decode_image_payloads(payloads)
            decode_ms.append((time.perf_counter() - t1) * 1000.0)

            with tempfile.TemporaryDirectory(prefix="sleap_bench_mat_") as tmp:
                t2 = time.perf_counter()
                _materialize_image_arrays(image_ids, image_arrays, Path(tmp))
                materialize_ms.append((time.perf_counter() - t2) * 1000.0)
        finally:
            for shm in handles:
                shm.close()
                shm.unlink()
        total_ms.append(encode_ms[-1] + decode_ms[-1] + materialize_ms[-1])
    return {
        "shm_encode_ms": encode_ms,
        "shm_decode_ms": decode_ms,
        "png_materialize_after_shm_ms": materialize_ms,
        "shm_total_with_materialize_ms": total_ms,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def _probe_sleap_env(python_executable: str) -> dict[str, object]:
    probe = (
        "import json\n"
        "import numpy as np\n"
        "import sleap_io as sio\n"
        "frames=[np.zeros((8,8,3),dtype=np.uint8) for _ in range(2)]\n"
        "stacked=np.stack(frames,axis=0)\n"
        "result={'load_video': False, 'constructors': []}\n"
        "load_video=getattr(sio,'load_video',None)\n"
        "if callable(load_video):\n"
        "  try:\n"
        "    result['load_video']=load_video(stacked) is not None\n"
        "  except Exception:\n"
        "    result['load_video']=False\n"
        "V=getattr(sio,'Video',None)\n"
        "if V is not None:\n"
        "  for attr in ('from_numpy','from_image_array','from_images','from_frames'):\n"
        "    if callable(getattr(V, attr, None)):\n"
        "      result['constructors'].append(attr)\n"
        "print(json.dumps(result))\n"
    )
    proc = subprocess.run(
        [python_executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout or "probe failed").strip(),
        }
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"ok": False, "error": f"invalid probe output: {exc}"}
    return {
        "ok": True,
        "load_video_supports_arrays": bool(data.get("load_video", False)),
        "video_array_constructors": list(data.get("constructors") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sleap-python", type=str, default="")
    args = parser.parse_args()

    crops = _make_crops(args.batch_size, args.crop_size, args.seed)
    temp_png = _bench_temp_png_write(crops, args.iterations)
    shm = _bench_shared_memory_fallback(crops, args.iterations)

    report = {
        "config": {
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "iterations": args.iterations,
        },
        "benchmarks": {
            "temp_png_write_ms": _summary(temp_png),
            **{name: _summary(values) for name, values in shm.items()},
        },
    }

    if args.sleap_python:
        report["sleap_env_probe"] = _probe_sleap_env(args.sleap_python)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
