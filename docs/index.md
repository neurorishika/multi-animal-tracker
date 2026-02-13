# Multi-Animal-Tracker Documentation

This documentation covers two applications in this repository:

- `mat` / `multianimaltracker`: the tracking GUI for detection, tracking, merging, export, and analysis.
- `posekit-labeler` / `pkl`: the PoseKit labeling and model-assist workflow.

## What You Can Find Here

- **Getting Started**: installation and first run paths for each app.
- **User Guide**: workflow-level explanations of features, parameters, outputs, and failure modes.
- **Developer Guide**: system architecture, module boundaries, extension points, and performance notes.
- **Reference**: API docs (mkdocstrings), CLI flags, changelog, and FAQ.

## Canonical Commands

```bash
# tracker GUI
mat
# or
multianimaltracker

# pose labeling UI
posekit-labeler
# or
pkl
```

## Documentation Build

```bash
make docs-install
make docs-serve
# or strict static build
make docs-build
```

## Documentation Scope

This site documents the current package layout:

- `multi_tracker.app`
- `multi_tracker.core`
- `multi_tracker.data`
- `multi_tracker.gui`
- `multi_tracker.posekit`
- `multi_tracker.utils`

Legacy notes from the previous docs layout are archived under `docs/archive/legacy-flat/`.
