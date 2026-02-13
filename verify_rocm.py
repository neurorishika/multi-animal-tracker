#!/usr/bin/env python3
"""
ROCm Verification Script for Multi-Animal-Tracker

This script checks if your ROCm installation is properly configured
and if all required components are available.

Usage:
    python verify_rocm.py
"""

import os
import subprocess
import sys


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_check(name, status, details=""):
    """Print a check result."""
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    print(f"{symbol} {name:45s} [{status_text}]")
    if details:
        print(f"  → {details}")


def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def check_system_rocm():
    """Check system-level ROCm installation."""
    print_header("System-Level ROCm Installation")

    # Check rocm-smi
    success, stdout, stderr = run_command("rocm-smi --version")
    if success and stdout:
        print_check("ROCm SMI installed", True, stdout.split("\n")[0])
    else:
        print_check("ROCm SMI installed", False, "Run: sudo apt install rocm-smi-lib")
        return False

    # Check GPU detection
    success, stdout, stderr = run_command("rocm-smi --showproductname")
    if success and stdout and "GPU" in stdout:
        gpu_lines = [line for line in stdout.split("\n") if "GPU" in line]
        print_check("GPU detected", True, gpu_lines[0] if gpu_lines else stdout)
    else:
        print_check("GPU detected", False, "No AMD GPU found")
        return False

    # Check HIP
    success, stdout, stderr = run_command("/opt/rocm/bin/hipconfig --version")
    if success and stdout:
        print_check("HIP installed", True, stdout.split("\n")[0])
    else:
        print_check("HIP installed", False, "Run: sudo apt install rocm-hip-runtime")

    # Check user groups
    success, stdout, stderr = run_command("groups")
    if success:
        groups = stdout.split()
        has_video = "video" in groups
        has_render = "render" in groups
        print_check("User in 'video' group", has_video)
        print_check("User in 'render' group", has_render)
        if not (has_video and has_render):
            print("  → Run: sudo usermod -a -G video,render $USER")
            print("  → Then log out and back in")

    # Check environment variables
    rocm_home = os.environ.get("ROCM_HOME", "")
    if rocm_home:
        print_check("ROCM_HOME set", True, rocm_home)
    else:
        print_check(
            "ROCM_HOME set", False, "Add to ~/.bashrc: export ROCM_HOME=/opt/rocm"
        )

    return True


def check_python_packages():
    """Check Python package installation."""
    print_header("Python Package Installation")

    # Check PyTorch
    try:
        import torch

        print_check("PyTorch installed", True, f"Version: {torch.__version__}")

        # Check ROCm support
        if torch.cuda.is_available():
            print_check("PyTorch ROCm support", True)
            device_name = torch.cuda.get_device_name(0)
            print_check("GPU accessible to PyTorch", True, device_name)

            # Check if ROCm backend
            if hasattr(torch.version, "hip") and torch.version.hip:
                print_check("ROCm backend detected", True, f"HIP: {torch.version.hip}")
            else:
                print_check(
                    "ROCm backend detected",
                    False,
                    "PyTorch may be using CUDA instead of ROCm",
                )
        else:
            print_check(
                "PyTorch ROCm support", False, "torch.cuda.is_available() = False"
            )
            print(
                "  → Reinstall with: uv pip install -r requirements-rocm.txt --force-reinstall"
            )

    except ImportError:
        print_check("PyTorch installed", False, "Run: uv pip install torch")
        return False

    # Check CuPy
    try:
        import cupy as cp

        print_check("CuPy installed", True, f"Version: {cp.__version__}")

        # Test CuPy-ROCm
        try:
            device = cp.cuda.Device(0)
            print_check("CuPy ROCm support", True, f"Device: {device}")

            # Test basic operation
            arr = cp.array([1, 2, 3])
            result = cp.sum(arr)
            print_check("CuPy operations working", True, f"Test sum: {result}")

        except Exception as e:
            print_check("CuPy ROCm support", False, str(e))
            print(
                "  → Check system libraries: sudo apt install rocm-dev rocrand rocblas"
            )

    except ImportError:
        print_check(
            "CuPy installed",
            False,
            "Run: uv pip install cupy-rocm-6-0 (may take 5-10 min)",
        )

    # Check Ultralytics
    try:
        from ultralytics import YOLO

        print_check("Ultralytics (YOLO) installed", True)
    except ImportError:
        print_check("Ultralytics (YOLO) installed", False)

    return True


def check_performance():
    """Run basic performance tests."""
    print_header("Performance Tests")

    try:
        import time

        import torch

        if not torch.cuda.is_available():
            print("Skipping performance tests - no GPU available")
            return

        device = torch.device("cuda:0")

        # Test tensor operations
        size = 1000
        start = time.time()
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print_check(
            f"PyTorch GPU matmul ({size}x{size})",
            True,
            f"{elapsed*1000:.2f} ms",
        )

        # Test memory
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024**3)
        print_check("GPU VRAM", True, f"{total_mem:.2f} GB")

    except Exception as e:
        print_check("Performance tests", False, str(e))

    try:
        import time

        import cupy as cp

        # Test CuPy operations
        size = 1000
        start = time.time()
        a = cp.random.randn(size, size)
        b = cp.random.randn(size, size)
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start

        print_check(
            f"CuPy GPU matmul ({size}x{size})",
            True,
            f"{elapsed*1000:.2f} ms",
        )

    except Exception as e:
        print_check("CuPy performance test", False, str(e))


def main():
    """Main verification routine."""
    print_header("Multi-Animal-Tracker - ROCm Verification")
    print("This script checks your ROCm installation and Python packages.\n")

    # System checks
    system_ok = check_system_rocm()

    # Python checks
    python_ok = check_python_packages()

    # Performance tests
    if system_ok and python_ok:
        check_performance()

    # Summary
    print_header("Summary")
    if system_ok and python_ok:
        print("✓ ROCm installation appears to be working correctly!")
        print("\nYou can now run the multi-animal tracker with ROCm acceleration.")
        print("Set in your config:")
        print('  "ENABLE_GPU_BACKGROUND": true,')
        print('  "GPU_DEVICE_ID": 0,')
        print('  "YOLO_DEVICE": "cuda:0"')
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print("\nFor detailed troubleshooting, see: ROCM_SETUP.md")
        print("Quick fixes:")
        print("  - System ROCm: sudo apt install rocm-hip-runtime rocm-hip-sdk")
        print("  - Development libs: sudo apt install rocm-dev rocrand rocblas")
        print("  - User groups: sudo usermod -a -G video,render $USER")
        print("  - Python packages: uv pip install -r requirements-rocm.txt")

    print("\n" + "=" * 70 + "\n")
    sys.exit(0 if (system_ok and python_ok) else 1)


if __name__ == "__main__":
    main()
