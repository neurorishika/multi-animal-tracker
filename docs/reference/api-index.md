# API Index

This section mixes narrative notes with auto-generated API documentation.

## Scope

- Auto docs are generated with `mkdocstrings` from source under `src/`.
- The focus is runtime modules that are intended as stable entry surfaces.

## Intentionally Limited Areas

- Some very large UI internals are documented at module level only.
- For complex GUI behavior details, use the User/Developer guides in addition to API pages.

## API Sections

- Launcher: bootstrap and entry point APIs
- TrackerKit GUI: tracking interface, orchestrators, panels, workers
- Core: tracking pipeline algorithms
- Data: outputs/cache/dataset APIs
- PoseKit: pose-labeling app modules
- Widgets: shared UI base classes (BaseWorker, BaseDialog, WelcomePage)
- Utils: shared helper utilities
