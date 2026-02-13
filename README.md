# 🐾 Multi-Animal-Tracker

<div align="center">

**High-performance multi-animal tracking and pose-labeling toolkit**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey.svg)](https://github.com/neurorishika/multi-animal-tracker)

[Getting Started](#-quick-start) • [Documentation](#-documentation) • [Features](#-features) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Applications](#-applications)
- [Documentation](#-documentation)
- [Development](#-development)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 🎯 **Multi-Animal Tracking** - Track multiple animals simultaneously with high accuracy
- 🦴 **Pose Estimation** - Advanced pose-labeling with keypoint detection
- 🖥️ **Dual GUI System** - Dedicated interfaces for tracking and pose labeling
- 🚀 **High Performance** - Optimized for speed with GPU acceleration support
- 📊 **Export Pipeline** - Comprehensive data export and analysis tools
- 🔧 **Extensible** - Modular architecture for easy customization
- 📚 **Well-Documented** - Complete documentation with MkDocs Material

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
mamba env create -f environment.yml

# Activate environment
conda activate multi-animal-tracker-base

# Install dependencies
uv pip install -v -r requirements.txt
```

> **Note:** For platform-specific installations (CUDA, ROCm, MPS), see [ENVIRONMENTS.md](ENVIRONMENTS.md)

### Quick Setup

```bash
make setup
make install
```

## 🎮 Applications

### Multi-Animal-Tracker (MAT)

Launch the tracking GUI for multi-animal video analysis:

```bash
mat
# or
multi-animal-tracker
```

### PoseKit Labeler

Launch the pose labeling interface:

```bash
posekit-labeler
# or
pose
```

## 📚 Documentation

This project uses **MkDocs Material** for comprehensive documentation.

### Browse Documentation

- 🏠 [Home](docs/index.md)
- 🎓 [Getting Started](docs/getting-started/)
- 📖 [User Guide](docs/user-guide/)
- 🛠️ [Developer Guide](docs/developer-guide/)
- 📘 [API & CLI Reference](docs/reference/)

### Build Documentation Locally

```bash
# Install documentation dependencies
make docs-install

# Serve documentation locally (with live reload)
make docs-serve

# Build static documentation
make docs-build
```

### Documentation Quality

Run quality checks and audits:

```bash
# Quality audit
make docs-quality

# Comprehensive check
make docs-check
```

## 🛠️ Development

### Common Commands

```bash
# Setup and installation
make setup          # Install runtime dependencies
make install        # Install package

# Documentation
make docs-install   # Install documentation tools
make docs-quality   # Run documentation quality audit
make docs-check     # Run comprehensive documentation checks

# Cleanup
make clean          # Clean build artifacts
```

### GPU Backend Support

Multiple GPU backends are supported:

- **CUDA** - NVIDIA GPUs ([environment-cuda.yml](environment-cuda.yml))
- **ROCm** - AMD GPUs ([environment-rocm.yml](environment-rocm.yml), [ROCM_SETUP.md](ROCM_SETUP.md))
- **MPS** - Apple Silicon ([environment-mps.yml](environment-mps.yml))

See [docs/developer-guide/gpu-backends.md](docs/developer-guide/gpu-backends.md) for details.

## 📁 Project Structure

```
multi-animal-tracker/
├── src/multi_tracker/
│   ├── app/              # App bootstrap & launcher
│   ├── core/             # Tracking & detection core
│   ├── data/             # Data & export pipeline
│   ├── gui/              # Multi-Animal-Tracker GUI
│   ├── posekit/          # Pose labeler app
│   └── utils/            # Shared utilities
├── docs/                 # MkDocs documentation
├── configs/              # Configuration files
├── models/               # Pre-trained models
├── training/             # Training scripts & data
└── tools/                # Development tools
```

### Key Entry Points

| Component | Path |
|-----------|------|
| App Bootstrap | `src/multi_tracker/app/launcher.py` |
| Tracking Core | `src/multi_tracker/core/` |
| Data Pipeline | `src/multi_tracker/data/` |
| Tracker GUI | `src/multi_tracker/gui/` |
| Pose Labeler | `src/multi_tracker/posekit/` |
| Utilities | `src/multi_tracker/utils/` |

## 🤝 Contributing

Contributions are welcome! Please see our [Developer Guide](docs/developer-guide/contributing.md) for details on:

- Code architecture
- Development workflow
- Testing guidelines
- Documentation standards

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by [Rishika Mohanta](https://github.com/neurorishika)**

</div>
