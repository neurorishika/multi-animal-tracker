# Contributing

## Development Basics

```bash
# install runtime deps
mamba env create -f environment.yml
conda activate multi-animal-tracker-base
uv pip install -r requirements.txt

# docs deps
uv pip install -r requirements-docs.txt
```

## Documentation Requirements

- Use current package paths (`multi_tracker.posekit`, new `core/*` layout).
- Keep command names canonical (`posekit-labeler`) and avoid legacy alternate spellings.
- Run strict docs build before opening PR:

```bash
make docs-build
```

## Code Change Expectations

- Keep architecture boundaries clear (GUI vs core vs data).
- Add/update docs whenever behavior or configuration semantics change.
