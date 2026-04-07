# HYDRA Suite Documentation

![HYDRA Suite Banner](assets/banner.png)

Welcome to the central documentation for:

- `hydra` (launcher / tool selector)
- `trackerkit` (multi-animal tracking)
- `posekit` (pose labeling)
- `classkit` (classification / embedding)
- `detectkit` (detection model training)
- `filterkit` (dataset filtering)
- `refinekit` (interactive proofreading)

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
# HYDRA launcher (tool selector)
hydra

# Individual tools
trackerkit         # Multi-animal tracking
posekit            # Pose labeling
classkit           # Classification / embedding
detectkit          # Detection model training
filterkit          # Dataset filtering
refinekit          # Interactive proofreading
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

- `hydra_suite.launcher`
- `hydra_suite.trackerkit`
- `hydra_suite.posekit`
- `hydra_suite.classkit`
- `hydra_suite.detectkit`
- `hydra_suite.filterkit`
- `hydra_suite.refinekit`
- `hydra_suite.core`
- `hydra_suite.data`
- `hydra_suite.training`
- `hydra_suite.runtime`
- `hydra_suite.integrations`
- `hydra_suite.widgets`
- `hydra_suite.utils`
- `hydra_suite.paths`
- `hydra_suite.resources`
