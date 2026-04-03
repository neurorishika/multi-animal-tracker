# Tracking System Technical Reference

This directory contains a standalone LaTeX reference for the current Multi-Animal-Tracker pipeline.

## Build

From the repository root:

```bash
make techref-build
```

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean auxiliary files:

```bash
make techref-clean
```

## Contents

- `main.tex`: main manuscript
- `figures/`: TikZ schematics used by the manuscript

## Intended Use

Use this document when you need:

- a publication-style methods reference,
- a review-friendly PDF for collaborators,
- or a stable technical narrative that stays closer to the code than a lightweight README.
