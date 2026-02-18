.PHONY: env-create env-create-cuda env-create-mps env-create-rocm env-update env-update-cuda env-update-mps env-update-rocm env-remove env-remove-cuda env-remove-mps env-remove-rocm install install-cuda install-mps install-rocm install-dev setup setup-cuda setup-mps setup-rocm test pytest test-cov test-cov-html verify-rocm clean docs-install docs-serve docs-build docs-quality docs-check pre-commit-install pre-commit-autopep8 pre-commit-run pre-commit-update format format-check lint lint-fix lint-strict lint-report dead-code dead-code-fix dep-graph dep-graph-text type-check audit help

# Environment names for different platforms
ENV_NAME = multi-animal-tracker
ENV_NAME_GPU = multi-animal-tracker-cuda
ENV_NAME_MPS = multi-animal-tracker-mps
ENV_NAME_ROCM = multi-animal-tracker-rocm
CUDA_MAJOR ?= 13

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Step 1: Create conda environment
env-create:
	@echo "Creating CPU-optimized environment..."
	mamba env create -f environment.yml

env-create-cuda:
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

install-cuda:
	@echo "Installing NVIDIA GPU (CUDA) packages..."
	@if [ "$(CUDA_MAJOR)" != "12" ] && [ "$(CUDA_MAJOR)" != "13" ]; then \
		echo "ERROR: CUDA_MAJOR must be 12 or 13"; \
		exit 1; \
	fi
	uv pip install -v -r requirements-cuda$(CUDA_MAJOR).txt
	@if [ -z "$$CONDA_PREFIX" ]; then \
		echo "ERROR: activate the CUDA conda env first (conda activate $(ENV_NAME_GPU))"; \
		exit 1; \
	fi
	@mkdir -p "$$CONDA_PREFIX/etc/conda/activate.d" "$$CONDA_PREFIX/etc/conda/deactivate.d"
	@printf '%s\n' \
		'export _MAT_OLD_LD_LIBRARY_PATH="$${LD_LIBRARY_PATH:-}"' \
		'export LD_LIBRARY_PATH="$$CONDA_PREFIX/targets/x86_64-linux/lib:$$CONDA_PREFIX/lib$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}"' \
		> "$$CONDA_PREFIX/etc/conda/activate.d/onnxruntime-cuda12-paths.sh"
	@printf '%s\n' \
		'if [ -n "$${_MAT_OLD_LD_LIBRARY_PATH+x}" ]; then' \
		'  export LD_LIBRARY_PATH="$$_MAT_OLD_LD_LIBRARY_PATH"' \
		'  unset _MAT_OLD_LD_LIBRARY_PATH' \
		'else' \
		'  unset LD_LIBRARY_PATH' \
		'fi' \
		> "$$CONDA_PREFIX/etc/conda/deactivate.d/onnxruntime-cuda12-paths.sh"
	@echo "Configured CUDA 12 runtime library path hook for ONNX Runtime GPU."

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

env-update-cuda:
	@echo "Updating NVIDIA GPU (CUDA) environment..."
	mamba env update -f environment-cuda.yml --prune
	@if [ "$(CUDA_MAJOR)" != "12" ] && [ "$(CUDA_MAJOR)" != "13" ]; then \
		echo "ERROR: CUDA_MAJOR must be 12 or 13"; \
		exit 1; \
	fi
	uv pip install -v -r requirements-cuda$(CUDA_MAJOR).txt --upgrade

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

env-remove-cuda:
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

setup-cuda:
	@echo "📦 Setting up NVIDIA GPU (CUDA) environment..."
	@echo ""
	mamba env create -f environment-cuda.yml
	@echo ""
	@echo "✅ Conda environment created!"
	@echo "📝 Next steps:"
	@echo "   1. conda activate $(ENV_NAME_GPU)"
	@echo "   2. make install-cuda CUDA_MAJOR=13  # or CUDA_MAJOR=12"

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

install-dev:
	@echo "🔧 Installing dev & code-quality tools..."
	uv pip install -r requirements-dev.txt
	@echo ""
	@echo "⚠️  graphviz dot binary is not pip-installable."
	@echo "   If you need dep-graph, run once in your active conda env:"
	@echo "   conda install -c conda-forge graphviz"
	@echo ""
	@echo "✅ Dev tools installed. Run 'make help' to see available audit targets."

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

pre-commit-autopep8:
	@echo "🧹 Running autopep8 pre-fix for common pycodestyle issues..."
	uvx autopep8 --in-place --recursive --select=E226,E225,E231 src/ tests/ tools/ legacy/
	@set -e; \
	for f in *.py; do \
		if [ -f "$$f" ]; then \
			uvx autopep8 --in-place --select=E226,E225,E231 "$$f"; \
		fi; \
	done

pre-commit-run:
	@$(MAKE) format
	@echo "🔎 Running pre-commit hooks on all files..."
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# =============================================================================
# CODE QUALITY
# =============================================================================

# Format code: autopep8 whitespace fixes → black → isort
format:
	@echo "✨ Formatting code (autopep8 → black → isort)..."
	uvx autopep8 --in-place --recursive --select=E226,E225,E231 src/ tests/ tools/
	black src/ tests/ tools/
	isort src/ tests/ tools/
	@echo "✅ Format complete."

format-check:
	black --check src/ tests/ tools/
	isort --check-only src/ tests/ tools/
	@echo "Format check complete."

# Lint at moderate severity (default gate — catches real issues without noise)
lint:
	@echo "🔍 Linting (flake8 moderate)..."
	flake8 --config=.flake8.moderate src/ tests/ tools/
	@echo "✅ Lint complete."

# Auto-fix safe issues with ruff, then reformat
lint-fix:
	@echo "🛠️  Auto-fixing lint issues (ruff) then formatting..."
	@set +e; \
	uvx ruff check --fix --select F401,F541,F841 src/ tests/ tools/; \
	RUFF_EXIT=$$?; \
	set -e; \
	if [ $$RUFF_EXIT -ne 0 ]; then \
		echo "ℹ️  Ruff fixed what it could; remaining issues need manual edits."; \
	fi
	@$(MAKE) format
	@echo "✅ Auto-fix complete. Review with: git diff"

# Strict lint: all best-practice issues
lint-strict:
	@echo "🔍 Running strict linting (all issues)..."
	flake8 --config=.flake8.strict src/ tests/ tools/
	@echo "✅ Strict linting complete."

# Side-by-side comparison of all three strictness levels
lint-report:
	@echo "📊 Lint issue counts at all strictness levels:"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 LENIENT  — pre-commit / CI gate"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 MODERATE — make lint (recommended default)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 --config=.flake8.moderate src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 STRICT   — make lint-strict (best practices)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 --config=.flake8.strict src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""

# =============================================================================
# CODE HEALTH & DEPENDENCY AUDITING
# =============================================================================

# Find unused code (dead functions, classes, variables, imports)
dead-code:
	@echo "🔍 Scanning for dead / orphaned code (vulture, ≥80% confidence)..."
	@echo ""
	vulture src/multi_tracker --min-confidence 80
	@echo ""
	@echo "� Cross-checking with deadcode..."
	@echo ""
	deadcode src/multi_tracker
	@echo ""
	@echo "💡 To whitelist false positives (vulture):"
	@echo "   vulture src/multi_tracker --make-whitelist > vulture_whitelist.py"
	@echo "   vulture src/multi_tracker vulture_whitelist.py"
	@echo "💡 To auto-remove confirmed dead code: make dead-code-fix"

# Automatically remove dead code (caution: modifies source files — commit first!)
dead-code-fix:
	@echo "🗑️  Running deadcode --fix on src/multi_tracker ..."
	@echo "   ⚠️  This will MODIFY source files. Commit or stash changes first."
	deadcode src/multi_tracker --fix
	@echo "✅ Done. Review with: git diff"

# Generate visual dependency graph (renders to multi_tracker.svg in cwd)
dep-graph:
	@echo "🗺️  Generating dependency graph (pydeps) → multi_tracker.svg ..."
	@if ! command -v dot >/dev/null 2>&1; then \
		echo ""; \
		echo "ERROR: 'dot' (graphviz) not found on PATH."; \
		echo "Install it in your active conda environment:"; \
		echo "  conda install -c conda-forge graphviz"; \
		echo "Or update the environment and reinstall:"; \
		echo "  make env-update-mps  # (or env-update / env-update-cuda / env-update-rocm)"; \
		echo ""; \
		exit 1; \
	 fi
	pydeps src/multi_tracker \
		--max-bacon=4 \
		--cluster \
		--rankdir LR \
		--noshow \
		-o multi_tracker.svg
	@echo "✅ Graph written to multi_tracker.svg"
	@echo "   Open it in a browser or SVG viewer."

# Text-based module dependency list (no graphviz required)
dep-graph-text:
	@echo "🗺️  Module dependency map (pyreverse / pylint) ..."
	@mkdir -p .audit
	pyreverse -o dot -p multi_tracker src/multi_tracker -d .audit/ 2>/dev/null || true
	@if [ -f .audit/packages_multi_tracker.dot ]; then \
		echo ""; \
		echo "--- packages_multi_tracker.dot (raw DOT source) ---"; \
		cat .audit/packages_multi_tracker.dot; \
	else \
		echo "ℹ️  pyreverse produced no output – check pylint/graphviz installation."; \
		echo "   Falling back to pydeps text mode ..."; \
		pydeps src/multi_tracker --max-bacon=4 --noshow --nodot 2>/dev/null || true; \
	fi

# Static type checking with mypy
type-check:
	@echo "🔎 Running mypy static type check..."
	mypy src/multi_tracker --ignore-missing-imports --no-error-summary
	@echo "✅ mypy check complete."

# Full code-health audit: dead code + dep graph + type check + coverage
audit: dead-code dep-graph type-check test-cov
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Full audit complete."
	@echo "   Artifacts: multi_tracker.svg  htmlcov/index.html"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════════════╗"
	@echo "║         Multi-Animal Tracker - Development Commands                   ║"
	@echo "╚════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 QUICK START  (choose your platform, then follow printed instructions)"
	@echo "  make setup           - CPU / NumPy+Numba"
	@echo "  make setup-mps       - Apple Silicon (M1/M2/M3/M4)"
	@echo "  make setup-cuda      - NVIDIA GPU (CUDA)"
	@echo "  make setup-rocm      - AMD GPU (ROCm)"
	@echo ""
	@echo "📦 INSTALL (after activating environment)"
	@echo "  make install[-mps|-cuda|-rocm]    - Install runtime packages"
	@echo "  make install-dev                  - ⭐ Install dev & audit tools"
	@echo "  make docs-install                 - Install MkDocs dependencies"
	@echo ""
	@echo "🔄 ENVIRONMENT MAINTENANCE"
	@echo "  make env-create[-mps|-cuda|-rocm] - Create conda environment"
	@echo "  make env-update[-mps|-cuda|-rocm] - Update conda environment"
	@echo "  make env-remove[-mps|-cuda|-rocm] - Remove conda environment"
	@echo ""
	@echo "🧪 TESTING"
	@echo "  make pytest          - Run all tests"
	@echo "  make test-cov        - Run tests with coverage report (terminal)"
	@echo "  make test-cov-html   - Run tests with HTML coverage (htmlcov/index.html)"
	@echo "  make verify-rocm     - Verify ROCm GPU setup"
	@echo "  make clean           - Remove Python cache files"
	@echo ""
	@echo "✨ CODE QUALITY  (requires: make install-dev)"
	@echo "  make format          - ⭐ Format code: autopep8 → black → isort"
	@echo "  make format-check    - Check formatting without making changes"
	@echo "  make lint            - Lint at moderate severity (recommended gate)"
	@echo "  make lint-fix        - Auto-fix safe issues (ruff) then reformat"
	@echo "  make lint-strict     - Lint at maximum strictness"
	@echo "  make lint-report     - Side-by-side issue counts at all levels"
	@echo "  make pre-commit-install  - Install git pre-commit hooks"
	@echo "  make pre-commit-run      - Run pre-commit hooks on all files"
	@echo ""
	@echo "🩺 CODE HEALTH  (requires: make install-dev)"
	@echo "  make dead-code       - Find unused code (vulture)"
	@echo "  make dead-code-fix   - ⚠️  Auto-remove dead code in-place (commit first!)"
	@echo "  make dep-graph       - Visual SVG dependency graph → multi_tracker.svg"
	@echo "  make dep-graph-text  - Text module map (pyreverse, no graphviz needed)"
	@echo "  make type-check      - Static type checking (mypy)"
	@echo "  make audit           - Full sweep: dead-code + dep-graph + types + coverage"
	@echo ""
	@echo "📚 DOCUMENTATION"
	@echo "  make docs-serve      - Live preview at http://127.0.0.1:8000"
	@echo "  make docs-build      - Build (strict mode)"
	@echo "  make docs-check      - Build + quality metrics + terminology checks"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Platform notes: CPU=everywhere  MPS=Apple M-series  CUDA=NVIDIA  ROCm=AMD"
	@echo "ROCm requires ROCm 6.0+ installed system-wide: https://rocm.docs.amd.com/"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
