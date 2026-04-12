#!/usr/bin/env python3
"""Benchmark direct SLEAP inference inside the target SLEAP environment.

This bypasses TrackerKit and the HTTP service transport so native runtime,
device selection, and exported-runtime behavior can be measured directly on the
same crop images TrackerKit would process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from hydra_suite.integrations.sleap.service import _SLEAP_SERVICE_CODE

_BENCHMARK_HARNESS = r"""
import argparse
import json
import statistics
import time
from pathlib import Path


def _quiet_log(_msg):
    return None


def _image_files(root):
    root = Path(root)
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    return sorted(
        p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts
    )


def _sample_images(paths, count):
    count = max(1, int(count))
    if not paths:
        raise RuntimeError('No benchmark images found.')
    if len(paths) >= count:
        return list(paths[:count])
    out = list(paths)
    while len(out) < count:
        out.append(paths[len(out) % len(paths)])
    return out


def _measure(fn, warmup, iterations):
    for _ in range(max(0, int(warmup))):
        fn()
    timings = []
    for _ in range(max(1, int(iterations))):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def _summarize(values):
    return {
        'mean_ms': round(statistics.mean(values), 3),
        'median_ms': round(statistics.median(values), 3),
        'min_ms': round(min(values), 3),
        'max_ms': round(max(values), 3),
    }


def _build_labels(images, keypoint_names):
    names = list(keypoint_names) if keypoint_names else ['k0']
    skeleton = _make_skeleton(names, [])
    video = _make_video([str(p) for p in images])
    frames = _make_labeled_frames(video, len(images))
    try:
        return sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=frames)
    except Exception:
        return sio.Labels(videos=[video], skeletons=[skeleton])


def _run_native_case(model_dir, device, batch, max_instances, images, keypoint_names, warmup, iterations):
    _clear_predictor()
    labels = _build_labels(images, keypoint_names)

    def _runner():
        out = _run_inference(labels, model_dir, device, batch, max_instances)
        labels_out = _labels_from_output(out)
        if labels_out is None:
            raise RuntimeError('Native SLEAP inference returned no labels output.')
        return labels_out

    timings = _measure(_runner, warmup, iterations)
    return {
        'runtime': 'native',
        'requested_device': device,
        'used_device': _state.get('device'),
        'batch_size': int(batch),
        'summary': _summarize(timings),
        'samples_ms': [round(v, 3) for v in timings],
    }


def _run_export_case(exported_model_path, runtime, device, batch, max_instances, images, keypoint_names, warmup, iterations):
    _clear_predictor()
    cfg = {
        'runtime_flavor': runtime,
        'exported_model_path': str(exported_model_path),
        'device': device,
        'batch': int(batch),
        'max_instances': int(max_instances),
        'export_input_hw': None,
    }
    image_paths = [str(p) for p in images]
    num_kpts = len(keypoint_names) if keypoint_names else 1

    def _runner():
        preds = _run_export_inference(cfg, image_paths, num_kpts)
        if not preds:
            raise RuntimeError('Exported SLEAP inference returned no predictions.')
        return preds

    timings = _measure(_runner, warmup, iterations)
    used_device = _state.get('device')
    used_runtime = _state.get('runtime_flavor')
    providers = None
    predictor = _state.get('predictor')
    session = None
    for attr in ('session', '_session', 'ort_session', '_ort_session', 'sess'):
        cand = getattr(predictor, attr, None) if predictor is not None else None
        if cand is not None and hasattr(cand, 'get_providers'):
            session = cand
            break
    if session is not None:
        try:
            providers = list(session.get_providers())
        except Exception:
            providers = None
    return {
        'runtime': runtime,
        'requested_device': device,
        'used_device': used_device,
        'used_runtime': used_runtime,
        'providers': providers,
        'batch_size': int(batch),
        'summary': _summarize(timings),
        'samples_ms': [round(v, 3) for v in timings],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request_json')
    args = parser.parse_args()

    globals()['_log'] = _quiet_log

    req = json.loads(Path(args.request_json).read_text(encoding='utf-8'))
    image_root = Path(req['image_root'])
    images_all = _image_files(image_root)
    if not images_all:
        raise RuntimeError(f'No images found under {image_root}')

    keypoint_names = list(req.get('keypoint_names') or [])
    cases = []
    for case in req.get('cases') or []:
        batch = int(case['batch'])
        images = _sample_images(images_all, batch)
        if case['runtime'] == 'native':
            result = _run_native_case(
                req['model_dir'],
                case['device'],
                batch,
                req.get('max_instances', 1),
                images,
                keypoint_names,
                req.get('warmup', 1),
                req.get('iterations', 5),
            )
        else:
            result = _run_export_case(
                req['exported_model_path'],
                case['runtime'],
                case['device'],
                batch,
                req.get('max_instances', 1),
                images,
                keypoint_names,
                req.get('warmup', 1),
                req.get('iterations', 5),
            )
        first = _load_image(str(images[0]))
        result['image_count'] = len(images)
        result['first_image_shape'] = list(first.shape) if first is not None else None
        result['first_image'] = str(images[0])
        cases.append(result)

    print(json.dumps({'model_dir': req['model_dir'], 'cases': cases}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
"""


def _load_keypoint_names(path: str) -> list[str]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data.get("keypoint_names") or []
    return [str(name) for name in names]


def _write_benchmark_script(path: Path) -> None:
    content = f"{_SLEAP_SERVICE_CODE}\n\n{_BENCHMARK_HARNESS}\n"
    path.write_text(content, encoding="utf-8")


def _case_label(case: dict[str, object]) -> str:
    return f"{case['runtime']}:{case['device']}:batch={case['batch']}"


def _invoke_benchmark(
    python_executable: str,
    request_path: Path,
    script_path: Path,
    timeout_seconds: float | None = None,
) -> dict:
    try:
        proc = subprocess.run(
            [python_executable, str(script_path), str(request_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise TimeoutError((stderr or stdout or "benchmark timed out").strip()) from exc
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "benchmark failed").strip())
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("benchmark produced no output")
    json_start = next(
        (idx for idx, line in enumerate(lines) if line.lstrip().startswith("{")),
        None,
    )
    if json_start is None:
        raise RuntimeError(proc.stdout.strip() or "benchmark returned no JSON payload")
    return json.loads("\n".join(lines[json_start:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--skeleton-json", required=True)
    parser.add_argument("--sleap-python", required=True)
    parser.add_argument("--exported-model-path", default="")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[21])
    parser.add_argument("--device-native", nargs="*", default=["mps", "cpu"])
    parser.add_argument("--device-onnx", nargs="*", default=["cpu"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-instances", type=int, default=1)
    parser.add_argument("--case-timeout", type=float, default=0.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    image_root = str(Path(args.image_root).expanduser().resolve())
    skeleton_json = str(Path(args.skeleton_json).expanduser().resolve())
    exported_model_path = (
        str(Path(args.exported_model_path).expanduser().resolve())
        if args.exported_model_path
        else ""
    )
    keypoint_names = _load_keypoint_names(skeleton_json)

    cases: list[dict[str, object]] = []
    for batch in args.batch_sizes:
        for device in args.device_native:
            cases.append(
                {"runtime": "native", "device": str(device), "batch": int(batch)}
            )
        if exported_model_path and not args.skip_onnx:
            for device in args.device_onnx:
                cases.append(
                    {"runtime": "onnx", "device": str(device), "batch": int(batch)}
                )

    request_base = {
        "model_dir": model_dir,
        "exported_model_path": exported_model_path,
        "image_root": image_root,
        "keypoint_names": keypoint_names,
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "max_instances": int(args.max_instances),
    }

    case_timeout = float(args.case_timeout) if float(args.case_timeout) > 0 else None
    aggregated_cases: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="sleap_runtime_bench_") as tmp:
        tmp_root = Path(tmp)
        script_path = tmp_root / "sleap_direct_benchmark.py"
        _write_benchmark_script(script_path)
        for index, case in enumerate(cases, start=1):
            label = _case_label(case)
            print(
                f"[sleap-bench] {index}/{len(cases)} {label}",
                file=sys.stderr,
                flush=True,
            )
            request = dict(request_base)
            request["cases"] = [case]
            request_path = tmp_root / f"request_{index}.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            try:
                result = _invoke_benchmark(
                    args.sleap_python,
                    request_path,
                    script_path,
                    timeout_seconds=case_timeout,
                )
                case_result = dict(result.get("cases", [{}])[0])
                case_result.setdefault("runtime", case["runtime"])
                case_result.setdefault("requested_device", case["device"])
                case_result.setdefault("batch_size", int(case["batch"]))
                case_result["status"] = "ok"
            except TimeoutError as exc:
                case_result = {
                    "runtime": case["runtime"],
                    "requested_device": case["device"],
                    "batch_size": int(case["batch"]),
                    "status": "timeout",
                    "error": str(exc) or f"Timed out after {case_timeout} seconds",
                }
            except Exception as exc:
                case_result = {
                    "runtime": case["runtime"],
                    "requested_device": case["device"],
                    "batch_size": int(case["batch"]),
                    "status": "error",
                    "error": str(exc),
                }
            aggregated_cases.append(case_result)

    result = {"model_dir": model_dir, "cases": aggregated_cases}
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).expanduser().write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
