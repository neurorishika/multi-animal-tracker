.PHONY: env-create env-create-gpu env-create-minimal env-create-mps env-create-rocm env-update env-remove install install-gpu install-minimal install-mps install-rocm test clean docs-install docs-serve docs-build docs-quality docs-check pre-commit-install pre-commit-run pre-commit-update format lint commit-prep pre-commit-check

# Default environment name
ENV_NAME ?= multi-animal-tracker-base
ENV_NAME_GPU = multi-animal-tracker-gpu
ENV_NAME_MINIMAL = multi-animal-tracker-minimal
ENV_NAME_MPS = multi-animal-tracker-mps
ENV_NAME_ROCM = multi-animal-tracker-rocm

# Environment creation (Step 1: conda packages)
env-create:
	mamba env create -f environment.yml

env-create-gpu:
	mamba env create -f environment-gpu.yml

env-create-minimal:
	mamba env create -f environment-minimal.yml

env-create-mps:
	mamba env create -f environment-mps.yml

env-create-rocm:
	mamba env create -f environment-rocm.yml


# Package installation (Step 2: pip packages with uv)
install:
	uv pip install -v -r requirements.txt

install-mps:
	uv pip install -v -r requirements-mps.txt

install-rocm:
	uv pip install -v -r requirements-rocm.txt

install-gpu:
	uv pip install -v -r requirements-gpu.txt

install-minimal:
	uv pip install -v -r requirements-minimal.txt

# Environment update
env-update:
	mamba env update -f environment.yml --prune
	uv pip install -v -r requirements.txt --upgrade

env-update-gpu:
	mamba env update -f environment-gpu.yml --prune
env-remove-mps:
	conda env remove -n $(ENV_NAME_MPS)

env-remove-rocm:
	conda env remove -n $(ENV_NAME_ROCM)

	uv pip install -v -r requirements-gpu.txt --upgrade

# Environment removal
env-remove:
	conda env remove -n $(ENV_NAME)

env-remove-gpu:
	conda env remove -n $(ENV_NAME_GPU)

env-remove-minimal:
	conda env remove -n $(ENV_NAME_MINIMAL)

# Development commands
test:
	python -c "from multi_tracker.app.launcher import main; print('Import successful')"

verify-rocm:
	@echo "Verifying ROCm installation..."
	python verify_rocm.py

clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Quick setup (run these in sequence, activating env between steps)
setup:
	@echo "Step 1: Creating conda environment..."
	mamba env create -f environment.yml
	@echo ""
	@echo "Step 2: Activate the environment and install pip packages:"
	@echo "  mamba activate $(ENV_NAME)"
	@echo "  make install"
	@echo ""
	@echo "Or run: mamba activate $(ENV_NAME) && uv pip install -v -r requirements.txt"

setup-gpu:
	@echo "Step 1: Creating GPU conda environment..."
	mamba env create -f environment-gpu.yml
	@echo ""
	@echo "Step 2: Activate the environment and install pip packages:"
	@echo "  mamba activate $(ENV_NAME_GPU)"
setup-mps:
	@echo "Step 1: Creating Apple Silicon (MPS) conda environment..."
	mamba env create -f environment-mps.yml
	@echo ""
	@echo "Step 2: Activate the environment and install pip packages:"
	@echo "  mamba activate $(ENV_NAME_MPS)"CPU, then activate & make install)"
	@echo "  make setup-gpu       - Create NVIDIA GPU environment (CUDA, then activate & make install-gpu)"
	@echo "  make setup-mps       - Create Apple Silicon environment (M1/M2/M3, then activate & make install-mps)"
	@echo "  make setup-rocm      - Create AMD GPU environment (ROCm, then activate & make install-rocm)"
	@echo "  make setup-minimal   - Create minimal environment (lightweight, then activate & make install-minimal)"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make env-create[-gpu|-mps|rocm|-minimal]  - Step 1: Create conda env with mamba"
	@echo "  make install[-gpu|-mps|-rocm|-minimal]    - Step 2: Install pip packages with uv (run after activate)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make env-update      - Update conda and pip packages"
	@echo "  make env-remove[-gpu|-mps|-rocm|-minimal] - Remove environment"
	@echo ""
	@echo "Testing & Verification:"
	@echo "  make test            - Test package installation"
	@echo "  make verify-rocm     - Verify ROCm installation (for AMD GPUs)"
	@echo "  make clean           - Clean Python cache files"
	@echo ""
	@echo "Platform Guide:"
	@echo "  NVIDIA GPUs:   use setup-gpu (CUDA + TensorRT + CuPy)"
	@echo "  Apple Silicon: use setup-mps (Metal Performance Shaders)"
	@echo "  AMD GPUs:      use setup-rocm (ROCm + CuPy-ROCm)"
	@echo "  CPU only:      use setup (optimized NumPy/Numba)"
	@echo "  Lightweight:   use setup-minimal (essential packages only)
	@echo "  make install-rocm"
	@echo ""
	@echo "ROCm Note: Ensure ROCm 6.0+ is installed system-wide before running make install-rocm"
	@echo "Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"

	@echo "  make install-gpu"
	@echo ""
	@echo "Or run: mamba activate $(ENV_NAME_GPU) && uv pip install -v -r requirements-gpu.txt"

setup-minimal:
	@echo "Step 1: Creating minimal conda environment..."
	mamba env create -f environment-minimal.yml
	@echo ""
	@echo "Step 2: Activate the environment and install pip packages:"
	@echo "  mamba activate $(ENV_NAME_MINIMAL)"
	@echo "  make install-minimal"

# Help
help:
	@echo "Multi-Animal-Tracker - Makefile Commands"
	@echo ""
	@echo "Environment Setup (two-step process):"
	@echo "  make setup           - Create base environment (then activate & make install)"
	@echo "  make setup-gpu       - Create GPU environment (then activate & make install-gpu)"
	@echo "  make setup-minimal   - Create minimal environment (then activate & make install-minimal)"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make env-create      - Step 1: Create conda env with mamba"
	@echo "  make install         - Step 2: Install pip packages with uv (run after activate)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make env-update      - Update conda and pip packages"
	@echo "  make env-remove      - Remove environment"
	@echo "  make clean           - Remove Python cache files"
	@echo "  make test            - Test import"
	@echo ""
	@echo "Code Quality:"
	@echo "  make pre-commit-install  - Install pre-commit hooks"
	@echo "  make pre-commit-run      - Run pre-commit on all files"
	@echo "  make pre-commit-update   - Update pre-commit hook versions"
	@echo "  make commit-prep         - ⭐ Format code before committing (recommended!)"
	@echo "  make format              - Format code with black and isort"
	@echo "  make format-check        - Check code formatting without changes"
	@echo "  make lint                - Lint code with flake8"
	@echo "  make pre-commit-check    - Run all checks (format + lint)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs-install    - Install MkDocs docs dependencies"
	@echo "  make docs-serve      - Start docs dev server"
	@echo "  make docs-build      - Build docs with strict mode"
	@echo "  make docs-check      - Strict docs build + terminology checks"

# Documentation commands
docs-install:
	uv pip install -r requirements-docs.txt

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build --strict

docs-quality:
	python tools/doc_quality_check.py --baseline docs/doc-quality-baseline.json --min-module-doc 90 --min-symbol-doc 55 --min-typed-func 20

docs-check: docs-build docs-quality
	@echo "Checking docs terminology..."
	@set -e; \
	if command -v rg >/dev/null 2>&1; then \
		if rg -n "labeller|posekit-labeller" docs README.md mkdocs.yml; then \
			echo "Found non-canonical labeler spelling"; \
			exit 1; \
		fi; \
	else \
		if grep -RInE "labeller|posekit-labeller" docs README.md mkdocs.yml; then \
			echo "Found non-canonical labeler spelling"; \
			exit 1; \
		fi; \
	fi
	@echo "Docs checks passed."

# Pre-commit hooks
pre-commit-install:
	pre-commit install
	@echo "Pre-commit hooks installed. They will run automatically on git commit."

pre-commit-run:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# Code quality shortcuts
format:
	black src/ tests/ tools/
	isort src/ tests/ tools/
	@echo "Code formatted with black and isort."

lint:
	flake8 src/ tests/ tools/
	@echo "Linting complete."

format-check:
	black --check src/ tests/ tools/
	isort --check-only src/ tests/ tools/
	@echo "Format check complete."

# Pre-commit preparation (recommended before git commit)
commit-prep: format
	@echo ""
	@echo "✅ Code formatted and ready to commit!"
	@echo "📝 Next steps:"
	@echo "   git add -u"
	@echo "   git commit -m \"your message\""

# Run all checks (useful for CI or pre-push)
pre-commit-check: format lint
	@echo ""
	@echo "✅ All code quality checks passed!"
