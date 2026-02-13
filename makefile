.PHONY: env-create env-create-gpu env-create-mps env-create-rocm env-update env-update-gpu env-update-mps env-update-rocm env-remove env-remove-gpu env-remove-mps env-remove-rocm install install-gpu install-mps install-rocm setup setup-gpu setup-mps setup-rocm test pytest coverage test-cov test-cov-html verify-rocm clean docs-install docs-serve docs-build docs-quality docs-check pre-commit-install pre-commit-run pre-commit-update format format-check whitespace-fix lint lint-autofix lint-autofix-unsafe lint-moderate lint-strict lint-report commit-prep pre-commit-check help

# Environment names for different platforms
ENV_NAME = multi-animal-tracker
ENV_NAME_GPU = multi-animal-tracker-gpu
ENV_NAME_MPS = multi-animal-tracker-mps
ENV_NAME_ROCM = multi-animal-tracker-rocm

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Step 1: Create conda environment
env-create:
	@echo "Creating CPU-optimized environment..."
	mamba env create -f environment.yml

env-create-gpu:
	@echo "Creating NVIDIA GPU (CUDA) environment..."
	mamba env create -f environment-cuda.yml

env-create-mps:
	@echo "Creating Apple Silicon (MPS) environment..."
	mamba env create -f environment-mps.yml

env-create-rocm:
	@echo "Creating AMD GPU (ROCm) environment..."
	mamba env create -f environment-rocm.yml


# Step 2: Install pip packages (run after activating environment)
install:
	@echo "Installing CPU packages..."
	uv pip install -v -r requirements.txt

install-gpu:
	@echo "Installing NVIDIA GPU (CUDA) packages..."
	uv pip install -v -r requirements-gpu.txt

install-mps:
	@echo "Installing Apple Silicon (MPS) packages..."
	uv pip install -v -r requirements-mps.txt

install-rocm:
	@echo "Installing AMD GPU (ROCm) packages..."
	uv pip install -v -r requirements-rocm.txt

# =============================================================================
# ENVIRONMENT MAINTENANCE
# =============================================================================

# Update environments
env-update:
	@echo "Updating CPU environment..."
	mamba env update -f environment.yml --prune
	uv pip install -v -r requirements.txt --upgrade

env-update-gpu:
	@echo "Updating NVIDIA GPU (CUDA) environment..."
	mamba env update -f environment-cuda.yml --prune
	uv pip install -v -r requirements-gpu.txt --upgrade

env-update-mps:
	@echo "Updating Apple Silicon (MPS) environment..."
	mamba env update -f environment-mps.yml --prune
	uv pip install -v -r requirements-mps.txt --upgrade

env-update-rocm:
	@echo "Updating AMD GPU (ROCm) environment..."
	mamba env update -f environment-rocm.yml --prune
	uv pip install -v -r requirements-rocm.txt --upgrade

# Remove environments
env-remove:
	@echo "Removing CPU environment..."
	conda env remove -n $(ENV_NAME)

env-remove-gpu:
	@echo "Removing NVIDIA GPU (CUDA) environment..."
	conda env remove -n $(ENV_NAME_GPU)

env-remove-mps:
	@echo "Removing Apple Silicon (MPS) environment..."
	conda env remove -n $(ENV_NAME_MPS)

env-remove-rocm:
	@echo "Removing AMD GPU (ROCm) environment..."
	conda env remove -n $(ENV_NAME_ROCM)

# =============================================================================
# TESTING & VERIFICATION
# =============================================================================

test:
	@echo "Testing package installation..."
	python -c "from multi_tracker.app.launcher import main; print('✅ Import successful')"
	@echo "✅ All tests passed!"

pytest:
	@echo "🧪 Running pytest..."
	python -m pytest

coverage: test-cov

test-cov:
	@echo "🧪 Running pytest with coverage..."
	python -m pytest --cov=src/multi_tracker --cov-report=term

test-cov-html:
	@echo "🧪 Running pytest with HTML coverage report..."
	python -m pytest --cov=src/multi_tracker --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/index.html"

verify-rocm:
	@echo "🔍 Verifying ROCm installation..."
	python verify_rocm.py

clean:
	@echo "🧹 Cleaning Python cache files..."
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✅ Cleanup complete!"

# =============================================================================
# COMPLETE SETUP
# =============================================================================

setup:
	@echo "📦 Setting up CPU-optimized environment..."
	@echo ""
	mamba env create -f environment.yml
	@echo ""
	@echo "✅ Conda environment created!"
	@echo "📝 Next steps:"
	@echo "   1. conda activate $(ENV_NAME)"
	@echo "   2. make install"

setup-gpu:
	@echo "📦 Setting up NVIDIA GPU (CUDA) environment..."
	@echo ""
	mamba env create -f environment-cuda.yml
	@echo ""
	@echo "✅ Conda environment created!"
	@echo "📝 Next steps:"
	@echo "   1. conda activate $(ENV_NAME_GPU)"
	@echo "   2. make install-gpu"

setup-mps:
	@echo "📦 Setting up Apple Silicon (MPS) environment..."
	@echo ""
	mamba env create -f environment-mps.yml
	@echo ""
	@echo "✅ Conda environment created!"
	@echo "📝 Next steps:"
	@echo "   1. conda activate $(ENV_NAME_MPS)"
	@echo "   2. make install-mps"

setup-rocm:
	@echo "📦 Setting up AMD GPU (ROCm) environment..."
	@echo "⚠️  Note: Ensure ROCm 6.0+ is installed system-wide first!"
	@echo "   Install guide: https://rocm.docs.amd.com/"
	@echo ""
	mamba env create -f environment-rocm.yml
	@echo ""
	@echo "✅ Conda environment created!"
	@echo "📝 Next steps:"
	@echo "   1. conda activate $(ENV_NAME_ROCM)"
	@echo "   2. make install-rocm"
	@echo "   3. make verify-rocm  # Verify ROCm installation"

# =============================================================================
# DOCUMENTATION
# =============================================================================

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

whitespace-fix:
	@echo "🧹 Fixing pycodestyle whitespace issues (E226/E225/E231)..."
	uvx autopep8 --in-place --recursive --select=E226,E225,E231 src/ tests/ tools/
	black src/ tests/ tools/
	isort src/ tests/ tools/
	@echo "✅ Whitespace fix complete."

lint:
	flake8 src/ tests/ tools/
	@echo "Linting complete."

lint-autofix:
	@echo "🛠️  Auto-fixing safe lint issues (imports, unused vars, spacing, simple strings)..."
	@set +e; \
	uvx ruff check --fix --select F401,F541,F841 src/ tests/ tools/; \
	RUFF_EXIT=$$?; \
	set -e; \
	if [ $$RUFF_EXIT -ne 0 ]; then \
		echo "ℹ️  Ruff auto-fixed what it could; some issues need manual edits."; \
	fi
	@$(MAKE) whitespace-fix
	@echo "✅ Auto-fix pass complete."
	@echo "🔎 Remaining F401/F541/F841 issues:"
	@set +e; uvx ruff check --select F401,F541,F841 src/ tests/ tools/; set -e
	@echo "➡️  Next: run 'make lint-moderate' for the full gate."

lint-autofix-unsafe:
	@echo "⚠️  Running unsafe autofixes (review changes carefully)..."
	@set +e; \
	uvx ruff check --fix --unsafe-fixes --select F401,F541,F841 src/ tests/ tools/; \
	RUFF_EXIT=$$?; \
	set -e; \
	if [ $$RUFF_EXIT -ne 0 ]; then \
		echo "ℹ️  Unsafe autofix applied partially; manual edits are still needed."; \
	fi
	@$(MAKE) whitespace-fix
	@echo "✅ Unsafe auto-fix pass complete. Please review with 'git diff'."

lint-moderate:
	@echo "🔍 Running moderate linting (catches serious issues)..."
	flake8 --config=.flake8.moderate src/ tests/ tools/
	@echo "✅ Moderate linting complete."

lint-strict:
	@echo "🔍 Running strict linting (all issues)..."
	flake8 --config=.flake8.strict src/ tests/ tools/
	@echo "✅ Strict linting complete."

lint-report:
	@echo "📊 Generating linting report (all strictness levels)..."
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 CURRENT (LENIENT) - Pre-commit config"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 MODERATE - Reasonable improvements"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 --config=.flake8.moderate src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 STRICT - Best practices target"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 --config=.flake8.strict src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""
	@echo "💡 Tip: Use 'make lint-moderate > issues.txt' to save for fixing"

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

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════════════╗"
	@echo "║         Multi-Animal Tracker - Development Commands                   ║"
	@echo "╚════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 QUICK START (choose your platform):"
	@echo "  make setup           - CPU only (optimized NumPy/Numba)"
	@echo "  make setup-gpu       - NVIDIA GPUs (CUDA + TensorRT)"
	@echo "  make setup-mps       - Apple Silicon (M1/M2/M3/M4)"
	@echo "  make setup-rocm      - AMD GPUs (ROCm)"
	@echo ""
	@echo "After setup runs, follow the printed instructions to activate & install."
	@echo ""
	@echo "📦 ENVIRONMENT CREATION (Step-by-step):"
	@echo "  make env-create      - Create CPU environment"
	@echo "  make env-create-gpu  - Create NVIDIA GPU environment"
	@echo "  make env-create-mps  - Create Apple Silicon environment"
	@echo "  make env-create-rocm - Create AMD GPU environment"
	@echo ""
	@echo "  Then activate: conda activate <env-name>"
	@echo ""
	@echo "📥 PACKAGE INSTALLATION (after activating environment):"
	@echo "  make install         - Install CPU packages"
	@echo "  make install-gpu     - Install NVIDIA GPU packages"
	@echo "  make install-mps     - Install Apple Silicon packages"
	@echo "  make install-rocm    - Install AMD GPU packages"
	@echo ""
	@echo "🔄 ENVIRONMENT MAINTENANCE:"
	@echo "  make env-update      - Update CPU environment"
	@echo "  make env-update-gpu  - Update NVIDIA GPU environment"
	@echo "  make env-update-mps  - Update Apple Silicon environment"
	@echo "  make env-update-rocm - Update AMD GPU environment"
	@echo ""
	@echo "  make env-remove      - Remove CPU environment"
	@echo "  make env-remove-gpu  - Remove NVIDIA GPU environment"
	@echo "  make env-remove-mps  - Remove Apple Silicon environment"
	@echo "  make env-remove-rocm - Remove AMD GPU environment"
	@echo ""
	@echo "🧪 TESTING & VERIFICATION:"
	@echo "  make test            - Test package imports"
	@echo "  make pytest          - Run all pytest tests"
	@echo "  make coverage        - Run tests with coverage report"
	@echo "  make test-cov        - Run tests with coverage report (alias)"
	@echo "  make test-cov-html   - Run tests with HTML coverage report"
	@echo "  make verify-rocm     - Verify ROCm installation (AMD GPUs only)"
	@echo "  make clean           - Remove Python cache files"
	@echo ""
	@echo "✨ CODE QUALITY (works across all installations):"
	@echo "  make commit-prep         - ⭐ Format code before committing (recommended!)"
	@echo "  make pre-commit-install  - Install pre-commit hooks"
	@echo "  make pre-commit-run      - Run pre-commit on all files"
	@echo "  make pre-commit-update   - Update pre-commit hook versions"
	@echo "  make format              - Format code with Black & isort"
	@echo "  make format-check        - Check code formatting (no changes)"
	@echo "  make whitespace-fix      - Auto-fix pycodestyle whitespace (E226/E231)"
	@echo "  make lint                - Lint code with Flake8 (lenient, for CI)"
	@echo "  make lint-autofix        - 🛠️ Auto-fix safe lint issues before linting"
	@echo "  make lint-autofix-unsafe - ⚠️ Auto-fix with unsafe rules (review required)"
	@echo "  make lint-moderate       - 🔍 Moderate linting (serious issues only)"
	@echo "  make lint-strict         - 🎯 Strict linting (all quality issues)"
	@echo "  make lint-report         - 📊 Compare all three strictness levels"
	@echo "  make pre-commit-check    - Run all checks (format + lint)"
	@echo ""
	@echo "📚 DOCUMENTATION:"
	@echo "  make docs-install    - Install MkDocs documentation dependencies"
	@echo "  make docs-serve      - Start local docs server (http://127.0.0.1:8000)"
	@echo "  make docs-build      - Build documentation with strict mode"
	@echo "  make docs-quality    - Check documentation quality metrics"
	@echo "  make docs-check      - Full docs build + quality checks"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Platform-Specific Notes:"
	@echo "  • CPU:           Works everywhere, optimized NumPy/Numba"
	@echo "  • NVIDIA GPU:    Requires CUDA toolkit installed"
	@echo "  • Apple Silicon: M1/M2/M3/M4 with Metal Performance Shaders"
	@echo "  • AMD GPU:       Requires ROCm 6.0+installed system-wide"
	@echo "                   Install guide: https://rocm.docs.amd.com/"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
