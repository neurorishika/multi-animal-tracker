.PHONY: env-create env-update env-remove install test clean

# Environment management
env-create:
    conda env create -f environment.yml

env-update:
    conda env update -f environment.yml --prune

env-remove:
    conda env remove -n multi-animal-tracker

# Development commands
install:
    conda activate multi-animal-tracker && pip install -e . --no-deps

test:
    conda activate multi-animal-tracker && python -c "from multi_tracker.main import main; print('Import successful')"

clean:
    find . -type d -name "__pycache__" -delete
    find . -type f -name "*.pyc" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +

# Quick setup
setup: env-create install test
    @echo "Environment setup complete! Activate with: conda activate multi-animal-tracker"
    @echo "Run with: multianimaltracker"