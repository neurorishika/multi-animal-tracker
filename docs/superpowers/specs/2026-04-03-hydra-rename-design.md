# HYDRA Rename Design Spec

**Date:** 2026-04-03
**Branch:** `refactor/src-reorganization`
**Scope:** Rename project from "hydra-suite" / "MAT" / "hydra_suite" to "HYDRA" (Holistic YOLO-based Detection Recognition and Analysis Suite)

## Naming Map

| Old | New | Context |
|-----|-----|---------|
| `hydra-suite` | `hydra-suite` | pip package name, conda env base, GitHub repo, platformdirs app name |
| `hydra_suite` | `hydra_suite` | Python package (imports) |
| `hydra_suite.tracker` | `hydra_suite.tracker` | Main tracker app module |
| `hydra` | `hydra` | CLI entry point |
| `MAT` | `HYDRA` | Acronym in docs/comments |
| `MAT_DATA_DIR` / `MAT_CONFIG_DIR` | `HYDRA_DATA_DIR` / `HYDRA_CONFIG_DIR` | Env var overrides |
| `hydra-suite` (conda) | `hydra` / `hydra-mps` / `hydra-cuda` / `hydra-rocm` | Conda environment names |
| `neurorishika/hydra-suite` | `neurorishika/hydra-suite` | GitHub repo URL |
| `~/...hydra-suite/` | `~/...hydra-suite/` | User data/config dirs |

## CLI Entry Points

| Command | Target |
|---------|--------|
| `hydra` | `hydra_suite.tracker.app.launcher:main` |
| `posekit` | `hydra_suite.posekit.ui.main:main` |
| `detectkit` | `hydra_suite.detectkit.app:main` |
| `classkit` | `hydra_suite.classkit.app:main` |
| `filterkit` | `hydra_suite.filterkit.gui:main` |
| `refinekit` | `hydra_suite.refinekit.app:main` |

No aliases. One command per tool.

## Directory Structure

```
src/hydra_suite/     -->  src/hydra_suite/
src/hydra_suite/mat/ -->  src/hydra_suite/tracker/
```

All other subdirectories (core, posekit, classkit, filterkit, refinekit, detectkit, runtime, data, training, utils, resources, paths, integrations) keep their names, just move under `hydra_suite/`.

## Brand Assets

- Remove obsolete SVGs: `hydra.svg` and any other old brand files
- `hydra.svg` already exists and is the new primary brand asset
- Update tests that assert on old brand file names

## Data Directory Migration

No migration code in the runtime. Data dirs on the developer's machine are moved manually during implementation. Other users have an existing migration script in the repo.

## Scope Summary

- ~3,350 import/reference changes across ~185 .py files
- ~149 occurrences of `hydra-suite` in config/docs
- ~138 `MAT` acronym references across ~60 files
- ~22 brand-related references
- 8 entry points in pyproject.toml
- 4 conda environment definitions
- 4 environment*.yml files
- ~25 GitHub URLs
- ~41 documentation .md files

## Execution Order

1. Rename directory `src/hydra_suite/` to `src/hydra_suite/` (including `mat/` to `tracker/`)
2. Remove old brand SVGs (`hydra.svg`, etc.)
3. Global find-replace across all .py files: `hydra_suite.tracker` to `hydra_suite.tracker`, then `hydra_suite` to `hydra_suite`
4. Update `pyproject.toml` (package name, entry points, URLs, coverage, package data)
5. Update `Makefile` (env names, variable prefixes)
6. Update `environment*.yml` (env names)
7. Update all docs (.md files, mkdocs.yml, README)
8. Update `CLAUDE.md`
9. Update test files (imports, hardcoded string assertions)
10. Update `.flake8.strict` and other config referencing old names
11. Move data dirs on dev machine
12. Run `make format` and `make lint`
13. Run tests to validate
