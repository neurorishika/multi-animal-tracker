# Code Quality Improvement Strategy

This document outlines the approach to gradually improve code quality in the Multi-Animal Tracker codebase.

## Current Status

The codebase currently uses **lenient** Flake8 configuration (`.flake8`) to allow all existing code to pass pre-commit hooks. This was necessary for the initial setup but should be improved over time.

### What We're Currently Ignoring

The lenient config ignores **30+ error codes** including:
- **E226, E231**: Whitespace issues
- **E402**: Module imports not at top of file
- **E722**: Bare `except:` clauses (dangerous!)
- **E741**: Ambiguous variable names
- **F401**: Unused imports
- **F811**: Redefinition of unused names
- **F841**: Unused local variables
- **B001, B007, B014**: Bugbear issues (common bugs)
- **C901**: McCabe complexity (set to 110, should be <10)
- **D100-D107**: Missing docstrings (all types)
- **SIM102-SIM910**: Code simplification recommendations

## Three-Tier Linting Strategy

We've created three Flake8 configurations to improve quality gradually:

### 1. **Lenient** (`.flake8`) - Current CI/Pre-commit
- **Purpose**: Don't break existing workflows
- **Complexity**: max 110
- **Ignores**: 30+ error types
- **Usage**: `make lint` (default)
- **When**: Pre-commit hooks, CI pipelines

### 2. **Moderate** (`.flake8.moderate`) - Next Target
- **Purpose**: Catch serious issues without overwhelming
- **Complexity**: max 25
- **Ignores**: Only cosmetic issues (docstrings, minor simplifications)
- **Usage**: `make lint-moderate`
- **When**: Weekly code quality sessions

### 3. **Strict** (`.flake8.strict`) - Long-term Goal
- **Purpose**: Best practices and publication-ready code
- **Complexity**: max 10
- **Ignores**: Only Black compatibility (E203, W503)
- **Usage**: `make lint-strict`
- **When**: Final review before major releases

## How to Use

### Step 1: See What Needs Fixing

Run the comparison report:
```bash
make lint-report
```

This shows you issue counts at all three levels:
```
📋 CURRENT (LENIENT) - Pre-commit config
0 issues

📋 MODERATE - Reasonable improvements
127 issues

📋 STRICT - Best practices target
843 issues
```

### Step 2: Focus on One Level at a Time

**Start with moderate linting:**
```bash
make lint-moderate > issues_moderate.txt
```

This creates a file with all issues. Review and prioritize.

### Step 3: Fix Issues Systematically

Pick a category and fix all instances:

#### Priority Order (Moderate):
1. **F841**: Unused variables → Remove or use them
2. **F401**: Unused imports → Clean up imports
3. **E402**: Import position → Move to top
4. **E722**: Bare except → Change to `except Exception:`
5. **B001, B007**: Bugbear → Fix common bugs
6. **C901**: Complexity → Refactor complex functions

#### Example Workflow:
```bash
# Find all unused variables
make lint-moderate | grep F841

# Fix them in the code (use editor)

# Verify fixed
make lint-moderate | grep F841  # Should show fewer/none

# Commit the improvements
make commit-prep
git add -u
git commit -m "Remove unused variables (F841)"
```

### Step 4: Update Pre-commit Config (Eventually)

Once you've fixed all moderate issues:

1. **Update `.flake8`** to match `.flake8.moderate`
2. **Update `.pre-commit-config.yaml`** to use cleaner config
3. **Run**: `pre-commit run --all-files` to verify
4. Commit the stricter config

### Step 5: Move to Strict (Long-term)

After moderate is clean, tackle strict issues:
- Add docstrings (D100-D107)
- Simplify code (SIM codes)
- Reduce complexity (refactor complex functions)

## Useful Commands

```bash
# Quick comparison
make lint-report

# Run moderate linting
make lint-moderate

# Run strict linting
make lint-strict

# Save issues to file
make lint-moderate > issues.txt
make lint-strict > issues_strict.txt

# Count issues by type
make lint-moderate | cut -d: -f4 | cut -d' ' -f2 | sort | uniq -c | sort -rn

# Find specific error type
make lint-moderate | grep E722  # Bare except clauses
make lint-moderate | grep F841  # Unused variables
```

## File-Specific Improvements

You can also lint specific files or directories:

```bash
# Single file
flake8 --config=.flake8.moderate src/multi_tracker/core/detection.py

# Specific module
flake8 --config=.flake8.strict src/multi_tracker/posekit/

# Exclude tests while focusing on src
flake8 --config=.flake8.moderate src/
```

## Tracking Progress

Create a simple tracking file:

```bash
# Save baseline
echo "$(date): Moderate=$(make lint-moderate 2>/dev/null | wc -l) Strict=$(make lint-strict 2>/dev/null | wc -l)" >> quality_progress.txt

# Check weekly
cat quality_progress.txt
```

## Tips for Fixing

### Unused Imports (F401)
```python
# Before
import numpy as np
import matplotlib.pyplot as plt  # Never used

# After
import numpy as np
```

### Bare Except (E722)
```python
# Before (dangerous)
try:
    risky_operation()
except:
    pass

# After (safer)
try:
    risky_operation()
except Exception as e:
    logger.warning(f"Operation failed: {e}")
```

### Complexity (C901)
```python
# Before (complexity > 25)
def complex_function(data):
    if condition1:
        if condition2:
            if condition3:
                # ... deeply nested logic

# After (refactored)
def complex_function(data):
    if not _validate_input(data):
        return None
    return _process_data(data)

def _validate_input(data):
    return condition1 and condition2 and condition3

def _process_data(data):
    # ... extracted logic
```

### Unused Variables (F841)
```python
# Before
def process():
    result = expensive_computation()
    # result never used
    return None

# After
def process():
    # Removed unused computation
    return None
```

## Integration with Workflow

### Daily Development
- Use `make lint` (lenient) for normal work
- Pre-commit hooks continue using lenient config

### Weekly Quality Sessions
- Run `make lint-moderate`
- Pick one error type to fix
- Commit improvements incrementally

### Before Releases
- Run `make lint-strict`
- Review and address critical issues
- Document technical debt if needed

## Long-term Goals

1. **Month 1-2**: Fix all moderate issues in `src/multi_tracker/core/`
2. **Month 3-4**: Fix moderate issues in `src/multi_tracker/posekit/`
3. **Month 5-6**: Fix moderate issues in `src/multi_tracker/gui/`
4. **Month 7+**: Switch to moderate config for pre-commit
5. **Month 12+**: Begin strict improvements

## Resources

- [Flake8 Error Codes](https://flake8.pycqa.org/en/latest/user/error-codes.html)
- [Flake8-Bugbear](https://github.com/PyCQA/flake8-bugbear)
- [Flake8-Simplify](https://github.com/MartinThoma/flake8-simplify)
- [McCabe Complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity)

## Questions?

See [CONTRIBUTING.md](CONTRIBUTING.md) for the overall development workflow.
