# ROCm Setup

The canonical ROCm setup guide now lives in [docs/getting-started/rocm.md](docs/getting-started/rocm.md).

Use that page for:

- current AMD system ROCm installation guidance
- HYDRA-specific ROCm environment setup
- verification commands
- troubleshooting notes

If you are working from the repository checkout, start here:

```bash
make setup-rocm
conda activate hydra-rocm
make install-rocm
make verify-rocm
```
