# HYDRA Suite Documentation

![HYDRA Suite Banner](assets/banner.png)

Welcome to the central documentation for:

- `hydra`/ `hydra` (tracking GUI)
- `posekit-labeler` / `pose` (pose labeling GUI)

!!! info "Use This Site as the Source of Truth"
    This docs site is the canonical guide for setup, workflows, feature behavior, and reference material.

## Quick Navigation

<div class="grid cards" markdown>

- :material-rocket-launch: **Getting Started**

    ---

    Installation, first launch, and platform setup.

    [Open Getting Started](getting-started/installation.md)

- :material-play-circle: **User Guide**

    ---

    End-to-end workflow for tracking, post-processing, datasets, and identity analysis.

    [Open User Guide](user-guide/overview.md)

- :material-source-branch: **Developer Guide**

    ---

    Architecture, module map, data flow, extension points, and performance notes.

    [Open Developer Guide](developer-guide/architecture.md)

- :material-book-open-page-variant: **Reference**

    ---

    API docs, CLI docs, UI component references, FAQ, and changelog.

    [Open Reference](reference/api-index.md)

- :material-file-document-outline: **Technical Reference**

    ---

    Publication-style algorithm writeup plus LaTeX manuscript source for the current tracker.

    [Open Technical Reference](reference/technical-reference.md)

</div>

## Launch Commands

```bash
# Tracking GUI
hydra
# or
hydra

# Pose labeling GUI
posekit-labeler
# or
pose
```

## Local Docs Workflow

=== "Serve locally"

    ```bash
    make docs-install
    make docs-serve
    ```

=== "Strict build"

    ```bash
    make docs-build
    make docs-check
    ```

## Scope

This documentation maps to the current package layout:

- `hydra_suite.app`
- `hydra_suite.core`
- `hydra_suite.data`
- `hydra_suite.gui`
- `hydra_suite.posekit`
- `hydra_suite.utils`

Legacy flat docs are archived under `docs/archive/legacy-flat/`.
