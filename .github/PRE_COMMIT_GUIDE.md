# Pre-commit Quick Reference

## TL;DR - Recommended Workflow

```bash
# Before every commit:
make commit-prep    # Auto-formats your code
git add -u          # Stage the formatted files
git commit -m "..."  # Commit (hooks will pass!)
```

## Why?

When you commit without formatting first, this happens:

```
❌ git commit -m "fix bug"
   → Black reformats files
   → Commit blocked!
   → You must: git add -u && git commit -m "fix bug" again
```

Using `make commit-prep` formats BEFORE committing, so hooks pass on first try!

## Alternative: Git Aliases

Add to your `~/.gitconfig`:

```ini
[alias]
  cprep = !git diff --name-only --cached | grep '\\.py$' && make format && git add -u || echo "No Python files to format"
  cfmt = !make commit-prep && git add -u
```

Then use:
```bash
git cfmt                    # Format + stage changes
git commit -m "your message"  # Commit
```

## What Pre-commit Does

On every `git commit`, these hooks run automatically:

1. **Black** - Auto-formats Python code (88 char line length)
2. **isort** - Sorts imports
3. **Flake8** - Lints for code quality issues
4. **Trailing whitespace** - Removes trailing spaces
5. **End-of-file** - Ensures newline at EOF
6. **YAML/JSON** - Validates syntax
7. **nbQA** - Formats Jupyter notebooks

## Manual Commands

```bash
make commit-prep         # ⭐ Format code (recommended before commit)
make format              # Run Black + isort
make lint                # Run Flake8
make pre-commit-run      # Run ALL hooks manually
make pre-commit-check    # Run format + lint checks
```

## Troubleshooting

**"Black failed - files were modified by this hook"**
- ✅ This is expected! Black auto-formatted your code
- Solution: `git add -u && git commit -m "..."`
- Or next time: Use `make commit-prep` first

**"Flake8 failed"**
- ❌ Code quality issues that need manual fixes
- Solution: Fix the errors, then `git add -u && git commit -m "..."`

**"Want to skip hooks temporarily"**
- `git commit --no-verify -m "..."` (NOT recommended)
- Better: Fix issues properly

## Configuration Files

- `.pre-commit-config.yaml` - Hook configuration
- `pyproject.toml` - Black, isort, pytest config
- `.flake8` - Linting rules

## See Also

- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Full development guide
- [Pre-commit documentation](https://pre-commit.com/)
