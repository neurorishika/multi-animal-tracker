# Technical Reference (LaTeX)

The repository now includes a standalone LaTeX technical reference package for the current tracking system.

[Download the latest PDF](../assets/technical-reference/hydra-suite-technical-reference.pdf){ .md-button .md-button--primary }

Location in the repo:

```text
technical-reference/
```

The package is intended for:

- publication-style internal references,
- lab handoffs,
- methods-section drafting,
- figure reuse,
- and deeper technical review than the MkDocs site should usually carry inline.

## What Is Included

- a structured `main.tex`,
- a published PDF mirrored into the docs site for direct download,
- publication-style title, abstract, contents, and appendices,
- equations for the state and cost models,
- TikZ schematics for pipeline, lifecycle, and post-processing flow,
- and build instructions for `latexmk`.

## Build

From the repository root:

```bash
make techref-build
```

Directly from the technical-reference folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean intermediates:

```bash
make techref-clean
```

## Audience Split

The LaTeX document is organized so it can serve both:

- readers who need a polished narrative overview,
- and technical readers who need implementation-level detail.

In practice this means the manuscript includes:

- an executive overview,
- a system pipeline description,
- formal state-estimation and association sections,
- post-processing and identity continuity logic,
- operational guidance,
- and appendices that map the narrative back to code modules and configuration keys.

## Recommended Workflow

Use the docs site for day-to-day usage and code-adjacent navigation. Use the LaTeX reference when you need a citable, printable, review-friendly technical artifact.

Direct links:

- [Technical reference PDF](../assets/technical-reference/hydra-suite-technical-reference.pdf)

Related pages:

- [Tracking, Identity Continuity, and Merging](../user-guide/tracking-and-merging.md)
- [Tracking Algorithm Deep Dive](../developer-guide/tracking-algorithm-deep-dive.md)
