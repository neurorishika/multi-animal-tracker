#!/usr/bin/env python3
"""Verify CUDA runtime libraries required by ONNX Runtime GPU."""

from __future__ import annotations

import glob
import os
import sys


def main() -> int:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        print("ERROR: CONDA_PREFIX is not set.", file=sys.stderr)
        return 1

    search_dirs = [
        os.path.join(prefix, "lib"),
        os.path.join(prefix, "targets", "x86_64-linux", "lib"),
    ]
    required_libs = [
        "libcublasLt.so.12",
        "libcudart.so.12",
        "libcurand.so.10",
        "libcufft.so.11",
        "libcudnn.so.9",
    ]

    missing = []
    for lib_name in required_libs:
        found = False
        for directory in search_dirs:
            matches = glob.glob(os.path.join(directory, lib_name))
            matches += glob.glob(os.path.join(directory, f"{lib_name}.*"))
            if matches:
                found = True
                break
        if not found:
            missing.append(lib_name)

    if missing:
        print(
            "ERROR: Missing CUDA runtime libraries required by ONNX Runtime:",
            file=sys.stderr,
        )
        for lib_name in missing:
            print(f"  - {lib_name}", file=sys.stderr)
        print(
            "Run `mamba env update -f environment-cuda.yml --prune` or install the "
            "missing package(s), then reactivate the environment.",
            file=sys.stderr,
        )
        return 1

    print("CUDA runtime self-check passed for ONNX Runtime GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
