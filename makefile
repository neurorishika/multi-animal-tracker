.PHONY: env-create env-create-gpu env-create-minimal env-update env-remove install install-gpu install-minimal test clean

# Default environment name
ENV_NAME ?= multi-animal-tracker-base
ENV_NAME_GPU = multi-animal-tracker-gpu
ENV_NAME_MINIMAL = multi-animal-tracker-minimal

# Environment creation (Step 1: conda packages)
env-create:
	mamba env create -f environment.yml

env-create-gpu:
	mamba env create -f environment-gpu.yml

env-create-minimal:
	mamba env create -f environment-minimal.yml

# Package installation (Step 2: pip packages with uv)
install:
	uv pip install -v -r requirements.txt

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
	python -c "from multi_tracker.main import main; print('Import successful')"

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
	@echo "Multi-Animal Tracker - Makefile Commands"
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