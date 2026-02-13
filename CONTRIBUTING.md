# Contributing to Multi-Animal Tracker

Thank you for your interest in contributing to Multi-Animal Tracker! This document provides guidelines for development and code quality.

## Development Setup

### 1. Environment Setup

```bash
# Create and activate conda environment
mamba env create -f environment-mps.yml  # or environment.yml
conda activate multi-animal-tracker-mps

# Install dependencies
uv pip install -v -r requirements-mps.txt  # or requirements.txt
```

### 2. Install Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install hooks
make pre-commit-install

# Or manually
pre-commit install
```

The hooks will now run automatically on every commit.

## Code Quality Tools

### Formatting

We use **Black** and **isort** for consistent code formatting:

```bash
# Auto-format all code
make format

# Check formatting without changes
make format-check
```

**Configuration:**
- Line length: 88 characters (Black default)
- Import sorting: Black-compatible profile
- Configuration in: `pyproject.toml`

### Linting

We use **Flake8** for linting with extensions for better code quality:

```bash
# Run linting
make lint
```

**Plugins:**
- `flake8-docstrings` - Docstring conventions
- `flake8-bugbear` - Common bugs
- `flake8-comprehensions` - Better comprehensions
- `flake8-simplify` - Code simplification

**Configuration in:** `.flake8`

### Pre-commit Hooks

Run all pre-commit hooks manually:

```bash
# Run on all files
make pre-commit-run

# Run on staged files only
pre-commit run

# Update hook versions
make pre-commit-update
```

**Hooks include:**
- ✅ Black (code formatting)
- ✅ isort (import sorting)
- ✅ Flake8 (linting)
- ✅ Trailing whitespace removal
- ✅ End-of-file fixer
- ✅ YAML/JSON validation
- ✅ Large file check
- ✅ Notebook formatting (nbQA)

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_postproc_equivalence.py

# Run with coverage
pytest --cov=src/multi_tracker

# Run in VSCode
# Use the Testing panel (flask icon in sidebar)
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files: `test_*.py`
- Name test functions: `test_*()`
- Use pytest fixtures and parametrize for efficiency

## Code Style Guidelines

### Python Style

- Follow **PEP 8** conventions (enforced by Flake8)
- Use **type hints** for function signatures
- Write **docstrings** for public functions and classes (Google style)
- Maximum line length: **88 characters**
- Use **meaningful variable names**

### Import Order

1. Standard library imports
2. Third-party package imports
3. Local application imports

isort handles this automatically.

### Example

```python
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from multi_tracker.core.tracking import Tracker
from multi_tracker.utils import load_config


def process_tracking_data(
    data: pd.DataFrame,
    config: dict,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Process tracking data with configuration.

    Args:
        data: Input DataFrame with tracking data
        config: Configuration dictionary
        output_path: Optional path to save results

    Returns:
        Processed DataFrame
    """
    # Implementation
    pass
```

## Commit Guidelines

### Commit Messages

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

**Example:**
```
feat: Add GPU acceleration for pose estimation

Implement CUDA-based inference pipeline for ViTPose model.
Improves processing speed by 10x on NVIDIA GPUs.

Closes #42
```

### Pre-commit Workflow

**⭐ Recommended Approach (avoids re-commit cycle):**

```bash
# 1. Format your code BEFORE committing
make commit-prep

# 2. Stage your changes
git add -u  # or git add <specific-files>

# 3. Commit (hooks will pass since code is already formatted)
git commit -m "your commit message"
```

**Alternative: Standard Git Workflow (may require re-commit):**

1. Stage your changes: `git add <files>`
2. Commit: `git commit -m "message"`
3. Pre-commit hooks run automatically
4. **If Black/isort auto-fix files:**
   - The commit will be blocked (this is expected)
   - The files are now formatted, but unstaged
   - Stage the auto-fixes: `git add -u`
   - Commit again: `git commit -m "message"`
   - Hooks will pass this time
5. **If flake8 fails:**
   - Fix the linting errors manually
   - Stage the fixes: `git add <fixed-files>`
   - Commit again

**Why does Black fail the first time?**

When Black or isort reformat your code during the commit, they modify the files. Pre-commit then blocks the commit to prevent you from accidentally committing unformatted code. You need to re-stage the auto-formatted files and commit again. Using `make commit-prep` before committing avoids this entirely.

## Pull Request Process

1. **Create a branch** for your feature/fix
2. **Make your changes** following the style guidelines
3. **Run tests** to ensure nothing breaks
4. **Run pre-commit** on all files: `make pre-commit-run`
5. **Update documentation** if needed
6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

## Development Commands

```bash
# Environment
make setup              # Create environment
make env-update         # Update environment
make clean              # Clean cache files

# Code Quality
make pre-commit-install # Install hooks
make commit-prep        # ⭐ Format before committing (recommended!)
make format             # Format code manually
make lint               # Lint code
make pre-commit-check   # Run format + lint

# Testing
pytest                  # Run tests

# Documentation
make docs-serve         # Serve docs locally
make docs-build         # Build docs
make docs-quality       # Check doc quality
```

## Questions?

- Check the [documentation](docs/)
- Open an [issue](https://github.com/neurorishika/multi-animal-tracker/issues)
- Ask in discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
