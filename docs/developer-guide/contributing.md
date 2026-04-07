# Contributing

## Development Basics

```bash
# install runtime deps
make setup
conda activate hydra
make install

# docs + dev-tool deps (vulture, pydeps, pylint, mypy)
make docs-install
make install-dev
```

## Documentation Requirements

- Use current package paths (`hydra_suite.posekit`, new `core/*` layout).
- Keep command names canonical (`posekit`) and avoid legacy alternate spellings.
- Run strict docs build before opening PR:

```bash
make docs-build
```

## Code Change Expectations

- Keep architecture boundaries clear (GUI vs core vs data).
- Add/update docs whenever behavior or configuration semantics change.

## Code Health

Run a full health check before submitting a large PR:

```bash
make audit     # dead-code + dep-graph + type-check + coverage
```

See the [Code Health & Auditing](code-health.md) page for details on interpreting
results and the whitelist workflow for false-positive dead-code findings.

## Recommended Pre-PR Checklist

```bash
make commit-prep     # format (black + isort)
make lint-moderate   # catch serious issues
make audit           # full health snapshot (optional for small PRs)
make docs-check      # verify docs build cleanly
```
