# ROCm Setup

This is the canonical ROCm setup guide for HYDRA Suite.

Use this page for the Linux AMD GPU path. The short installation page stays focused on package install commands; this page covers the system ROCm prerequisite, repo-specific defaults, and the failure modes that are easiest to hit in practice.

## Status

- Linux only.
- AMD's current quick-start flow uses the `amdgpu-install` package and ROCm 7.2.x.
- HYDRA's developer workflow uses `environment-rocm.yml`, `requirements-rocm.txt`, `make setup-rocm`, and `make install-rocm`.
- PyTorch ROCm is the main accelerated path in this repo.
- CuPy on ROCm is still experimental and is the least stable part of the stack.
- `onnx_rocm` is optional. The default ROCm requirements install CPU `onnxruntime`; the ROCm ONNX provider only appears if your environment supplies it separately.

## Supported Host Setup

- Supported distributions are defined by AMD's ROCm documentation. At the time of writing that includes current Ubuntu LTS releases plus selected RHEL and SLES variants.
- Check GPU and distro support before installing: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>

## 1. Install System ROCm

HYDRA does not install ROCm for you. Install ROCm system-wide first using AMD's distro-specific guide.

For Ubuntu 24.04, AMD's current quick-start pattern is:

```bash
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms rocm
sudo usermod -a -G render,video $LOGNAME
```

Notes:

- Ubuntu 22.04 uses the `jammy` package path instead of `noble`.
- Reboot after driver installation.
- For RHEL or SLES, follow AMD's detailed distro-specific instructions instead of copying Ubuntu commands.

Official references:

- Quick start: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>
- Detailed install: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/detailed-install.html>
- Radeon/Ryzen guidance: <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html>

## 2. Verify ROCm Before Installing HYDRA

Run these checks before creating the HYDRA environment:

```bash
rocm-smi --showproductname
/opt/rocm/bin/hipconfig --version
rocminfo | grep "Name:"
groups
```

You want to confirm all of the following:

- Your AMD GPU is visible.
- HIP is installed.
- Your user is in the `render` and `video` groups.

## 3. Optional Environment Variables

These are often useful for CuPy builds, multi-GPU systems, and shell sessions that do not already expose ROCm cleanly:

```bash
export ROCM_HOME=/opt/rocm
export ROCM_PATH=$ROCM_HOME
export HIP_PATH=$ROCM_HOME/hip
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
```

If your system install is healthy, PyTorch ROCm often works without all of these exports. They are still worth keeping in mind when CuPy builds or runtime discovery fail.

## 4. Create the HYDRA ROCm Environment

Preferred repo workflow:

```bash
make setup-rocm
conda activate hydra-rocm
make install-rocm
make verify-rocm
```

Equivalent manual flow:

```bash
mamba env create -f environment-rocm.yml
conda activate hydra-rocm
uv pip install -v -r requirements-rocm.txt
python verify_rocm.py
```

## 5. Repo-Specific Notes

- `requirements-rocm.txt` currently defaults to PyTorch ROCm 7.2 wheels.
- If your lab is pinned to another supported PyTorch ROCm build, change the `--extra-index-url` in `requirements-rocm.txt` to match.
- The default ROCm requirements install CPU `onnxruntime`, not a ROCm ONNX provider. That means `onnx_rocm` is only available when your environment exposes `ROCMExecutionProvider`.
- The default CuPy package in this repo is still `cupy-rocm-6-0`. If that package fails on your host, HYDRA can still run on PyTorch ROCm with CPU fallback paths for features that depend on CuPy.

## 6. Verification Commands

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('hip=', getattr(torch.version, 'hip', None)); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python -c "import cupy as cp; print(cp.__version__)"
```

Interpretation:

- If PyTorch reports `cuda_available=True` and a non-empty `torch.version.hip`, the ROCm torch install is healthy.
- If ONNX Runtime shows only `CPUExecutionProvider`, that is expected with the default repo requirements.
- If CuPy import or runtime checks fail, the rest of the ROCm stack may still be usable.

## 7. Troubleshooting

### GPU Not Detected

```bash
lsmod | grep amdgpu
dmesg | grep -i amdgpu
groups
```

If `render` or `video` is missing, re-run:

```bash
sudo usermod -a -G render,video $LOGNAME
```

Then log out fully or reboot.

### PyTorch Does Not See ROCm

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(getattr(torch.version, 'hip', None))"
```

If the torch wheel is wrong for your stack, reinstall from the ROCm index selected in `requirements-rocm.txt`.

### CuPy Build or Runtime Failures

CuPy's own AMD guidance now calls out these ROCm libraries as the common missing pieces:

```bash
sudo apt install hipblas hipsparse rocsparse rocrand hiprand rocthrust rocsolver rocfft hipfft hipcub rocprim rccl roctracer-dev
```

If CuPy still fails, keep the rest of the environment and treat CuPy-backed features as optional until you decide whether to pin a different CuPy/ROCm combination.

### Performance Is Lower Than Expected

```bash
rocm-smi --showclocks
rocm-smi --showmeminfo vram
watch -n 1 rocm-smi
```

Also verify that the app is actually using `rocm` and not falling back to `cpu` or `onnx_cpu`.

## 8. Related Pages

- [Installation](installation.md)
- [Environments and Makefile](environments.md)
- [Platform Notes](platforms.md)
- [Compute Runtimes](../user-guide/compute-runtimes.md)
