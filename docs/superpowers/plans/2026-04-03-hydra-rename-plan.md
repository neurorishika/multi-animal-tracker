# HYDRA Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project from multi-animal-tracker/multi_tracker/MAT to HYDRA/hydra-suite/hydra_suite across the entire codebase.

**Architecture:** This is a mechanical rename with no logic changes. The directory `src/multi_tracker/` becomes `src/hydra_suite/`, the submodule `mat/` becomes `tracker/`, and all references throughout the codebase are updated to match. No migration code is added.

**Tech Stack:** Python, setuptools, conda, mkdocs, flake8, pytest

---

### Task 1: Rename source directories

**Files:**
- Rename: `src/multi_tracker/` → `src/hydra_suite/`
- Rename: `src/hydra_suite/mat/` → `src/hydra_suite/tracker/`

- [ ] **Step 1: Rename multi_tracker to hydra_suite**

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git mv src/multi_tracker src/hydra_suite
```

- [ ] **Step 2: Rename mat to tracker**

```bash
git mv src/hydra_suite/mat src/hydra_suite/tracker
```

- [ ] **Step 3: Remove old brand SVGs**

```bash
rm src/hydra_suite/resources/brand/multianimaltracker.svg
```

Also remove any old .png brand files that are no longer needed (classkit.png, filterkit.png, multianimaltracker.png, posekit.png, refinekit.png were already deleted per git status).

- [ ] **Step 4: Clean build directory**

```bash
rm -rf build/
```

- [ ] **Step 5: Commit directory renames**

```bash
git add -A
git commit -m "refactor: rename src/multi_tracker → src/hydra_suite, mat → tracker"
```

---

### Task 2: Global find-replace in Python source files

**Files:**
- Modify: All `.py` files under `src/hydra_suite/`

The order of replacements matters — more specific patterns first:

- [ ] **Step 1: Replace `multi_tracker.mat` → `hydra_suite.tracker` in all .py files**

```bash
find src/hydra_suite -name '*.py' -exec sed -i '' 's/multi_tracker\.mat/hydra_suite.tracker/g' {} +
```

- [ ] **Step 2: Replace `multi_tracker` → `hydra_suite` in all .py files**

```bash
find src/hydra_suite -name '*.py' -exec sed -i '' 's/multi_tracker/hydra_suite/g' {} +
```

- [ ] **Step 3: Replace `multi-animal-tracker` → `hydra-suite` in all .py files**

```bash
find src/hydra_suite -name '*.py' -exec sed -i '' 's/multi-animal-tracker/hydra-suite/g' {} +
```

- [ ] **Step 4: Replace MAT-specific references**

In `src/hydra_suite/paths.py`:
- `APP_NAME = "multi-animal-tracker"` → `APP_NAME = "hydra-suite"`
- `MAT_CONFIG_DIR` → `HYDRA_CONFIG_DIR` (3 occurrences)
- `MAT_DATA_DIR` → `HYDRA_DATA_DIR` (3 occurrences)
- Docstring references to MAT_CONFIG_DIR/MAT_DATA_DIR
- `from multi_tracker.paths import print_paths` in docstring → `from hydra_suite.paths import print_paths`

- [ ] **Step 5: Verify no remaining multi_tracker references in source**

```bash
grep -r "multi_tracker" src/hydra_suite/ --include="*.py" | head -20
grep -r "multi-animal-tracker" src/hydra_suite/ --include="*.py" | head -20
```

- [ ] **Step 6: Commit source replacements**

```bash
git add src/
git commit -m "refactor: update all Python imports multi_tracker → hydra_suite"
```

---

### Task 3: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update project metadata**

- Line 1 comment: `multi-animal-tracker` → `hydra-suite`
- Line 8: `name = "hydra-suite"`
- Line 10: description update to mention HYDRA
- Line 70: `"multi-animal-tracker[cuda12]"` → `"hydra-suite[cuda12]"`

- [ ] **Step 2: Update URLs**

```
Homepage = "https://github.com/neurorishika/hydra-suite"
Documentation = "https://github.com/neurorishika/hydra-suite/tree/main/docs"
Repository = "https://github.com/neurorishika/hydra-suite"
Issues = "https://github.com/neurorishika/hydra-suite/issues"
```

- [ ] **Step 3: Update entry points**

```toml
[project.scripts]
hydra = "hydra_suite.tracker.app.launcher:main"
posekit = "hydra_suite.posekit.ui.main:main"
filterkit = "hydra_suite.filterkit.gui:main"
classkit = "hydra_suite.classkit.app:main"
refinekit = "hydra_suite.refinekit.app:main"
detectkit = "hydra_suite.detectkit.app:main"
```

- [ ] **Step 4: Update coverage and setuptools config**

- Line 185: `source = ["src/hydra_suite"]`
- Line 241: `"hydra_suite" = [...]`

- [ ] **Step 5: Update publish commands in comments if any**

Replace remaining `multi-animal-tracker` references in publish-test echo.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "refactor: update pyproject.toml for hydra-suite"
```

---

### Task 4: Update Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Update environment names**

```makefile
ENV_NAME = hydra
ENV_NAME_GPU = hydra-cuda
ENV_NAME_MPS = hydra-mps
ENV_NAME_ROCM = hydra-rocm
```

- [ ] **Step 2: Update _MAT_ variable prefix**

Replace `_MAT_OLD_LD_LIBRARY_PATH` → `_HYDRA_OLD_LD_LIBRARY_PATH` (lines 50, 54, 55)

- [ ] **Step 3: Update test target import**

Line 123: `from multi_tracker.mat.app.launcher` → `from hydra_suite.tracker.app.launcher`

- [ ] **Step 4: Update coverage paths**

Lines 132-136: `src/multi_tracker` → `src/hydra_suite`

- [ ] **Step 5: Update dead-code/dep-graph/type-check paths**

Replace all `src/multi_tracker` → `src/hydra_suite` and `multi_tracker` → `hydra_suite` in tool targets.

- [ ] **Step 6: Update help text**

Line 492: `Multi-Animal Tracker` → `HYDRA Suite`
All references to `multi_tracker.svg` → `hydra_suite.svg`

- [ ] **Step 7: Update publish echo lines**

Lines 260, 269: `multi-animal-tracker` → `hydra-suite`

- [ ] **Step 8: Commit**

```bash
git add Makefile
git commit -m "refactor: update Makefile for hydra-suite"
```

---

### Task 5: Update environment and requirements files

**Files:**
- Modify: `environment.yml`, `environment-cuda.yml`, `environment-mps.yml`, `environment-rocm.yml`
- Modify: Any `requirements*.txt` that reference `multi-animal-tracker`

- [ ] **Step 1: Update environment.yml names**

- `environment.yml`: `name: hydra-base`
- `environment-cuda.yml`: `name: hydra-cuda`
- `environment-mps.yml`: `name: hydra-mps`
- `environment-rocm.yml`: `name: hydra-rocm`

- [ ] **Step 2: Update any multi-animal-tracker references in requirements files**

```bash
grep -l "multi-animal-tracker" requirements*.txt
```

Replace any found references.

- [ ] **Step 3: Commit**

```bash
git add environment*.yml requirements*.txt
git commit -m "refactor: update environment and requirements files for hydra"
```

---

### Task 6: Update test files

**Files:**
- Modify: All `tests/*.py` and `tests/**/*.py` files

- [ ] **Step 1: Replace imports in test files**

```bash
find tests -name '*.py' -exec sed -i '' 's/multi_tracker\.mat/hydra_suite.tracker/g' {} +
find tests -name '*.py' -exec sed -i '' 's/multi_tracker/hydra_suite/g' {} +
find tests -name '*.py' -exec sed -i '' 's/multi-animal-tracker/hydra-suite/g' {} +
```

- [ ] **Step 2: Update test_packaging.py brand assertions**

- Replace `"multianimaltracker.svg"` with `"hydra.svg"` in `test_brand_svgs_exist_in_package`
- Replace `version("multi-animal-tracker")` with `version("hydra-suite")`
- Replace `from multi_tracker import` with `from hydra_suite import`
- Replace `files("multi_tracker.resources.brand")` with `files("hydra_suite.resources.brand")`
- Similarly for configs and skeletons resource paths

- [ ] **Step 3: Verify no remaining old references in tests**

```bash
grep -r "multi_tracker\|multi-animal-tracker\|multianimaltracker" tests/ --include="*.py" | head -20
```

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: update test imports and assertions for hydra-suite"
```

---

### Task 7: Update documentation

**Files:**
- Modify: All `docs/**/*.md` files
- Modify: `mkdocs.yml`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `.github/*.md`, `.github/workflows/*.yml`

- [ ] **Step 1: Global replace in all markdown files**

```bash
# Order matters — specific patterns first
find docs -name '*.md' -exec sed -i '' 's/multi_tracker\.mat/hydra_suite.tracker/g' {} +
find docs -name '*.md' -exec sed -i '' 's/multi_tracker/hydra_suite/g' {} +
find docs -name '*.md' -exec sed -i '' 's/multi-animal-tracker/hydra-suite/g' {} +
find docs -name '*.md' -exec sed -i '' 's/Multi-Animal-Tracker/HYDRA Suite/g' {} +
find docs -name '*.md' -exec sed -i '' 's/Multi-Animal Tracker/HYDRA Suite/g' {} +
```

- [ ] **Step 2: Replace MAT acronym in docs (contextual)**

Replace `MAT` when used as the tracker acronym with `HYDRA` throughout docs. This requires care — only replace when it refers to the Multi-Animal Tracker app, not when part of other words.

```bash
# Common patterns
find docs -name '*.md' -exec sed -i '' 's/\bMAT\b/HYDRA/g' {} +
find docs -name '*.md' -exec sed -i '' 's/\bmat\b command/hydra command/g' {} +
```

- [ ] **Step 3: Update mkdocs.yml**

- `site_name: HYDRA Suite Docs`
- `site_description: User and developer documentation for HYDRA Suite`
- `site_url: https://neurorishika.github.io/hydra-suite/`
- `repo_url: https://github.com/neurorishika/hydra-suite`
- `repo_name: neurorishika/hydra-suite`
- Nav references to MAT → HYDRA

- [ ] **Step 4: Update README.md**

Replace all `multi-animal-tracker` → `hydra-suite`, `multi_tracker` → `hydra_suite`, `MAT` → `HYDRA`, `mat` CLI → `hydra` CLI.

- [ ] **Step 5: Update CLAUDE.md**

Full update: package name, imports, CLI commands, architecture table, env names, env vars, file paths, all references.

- [ ] **Step 6: Update .github/ files**

- `.github/workflows/docs-pages.yml`: update any references
- `.github/CODE_QUALITY_GUIDE.md`: update references
- `.github/PRE_COMMIT_GUIDE.md`: update references

- [ ] **Step 7: Verify no remaining old references in docs**

```bash
grep -r "multi_tracker\|multi-animal-tracker\|Multi-Animal" docs/ mkdocs.yml README.md CLAUDE.md .github/ | head -30
```

- [ ] **Step 8: Commit**

```bash
git add docs/ mkdocs.yml README.md CLAUDE.md .github/
git commit -m "docs: update all documentation for HYDRA rename"
```

---

### Task 8: Update remaining config files

**Files:**
- Modify: `.flake8`, `.flake8.moderate`, `.flake8.strict` (if they reference multi_tracker)
- Modify: `ROCM_SETUP.md`
- Modify: Any other files with old references

- [ ] **Step 1: Check and update flake8 configs**

```bash
grep -l "multi_tracker\|multi-animal-tracker" .flake8*
```

Update any found references.

- [ ] **Step 2: Update ROCM_SETUP.md**

Replace all old name references.

- [ ] **Step 3: Check for any remaining references repo-wide**

```bash
grep -r "multi_tracker\|multi-animal-tracker\|multianimaltracker" --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml" --include="*.toml" --include="*.txt" --include="*.cfg" . | grep -v ".git/" | grep -v "build/" | grep -v "__pycache__" | head -40
```

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A
git commit -m "refactor: update remaining config files for hydra-suite"
```

---

### Task 9: Move data directories on dev machine

- [ ] **Step 1: Move config/data directories**

```bash
mv ~/Library/Application\ Support/multi-animal-tracker ~/Library/Application\ Support/hydra-suite
```

(Only if the directory exists)

---

### Task 10: Format, lint, and validate

- [ ] **Step 1: Reinstall package in dev mode**

```bash
pip install -e .
```

- [ ] **Step 2: Run formatter**

```bash
make format
```

- [ ] **Step 3: Run linter**

```bash
make lint
```

- [ ] **Step 4: Run tests**

```bash
make pytest
```

- [ ] **Step 5: Fix any failures and re-run**

Address any test failures or lint issues from the rename.

- [ ] **Step 6: Final commit if formatting changed anything**

```bash
git add -A
git commit -m "style: format after hydra rename"
```
