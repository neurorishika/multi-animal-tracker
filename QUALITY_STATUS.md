# Code Quality Issues - Current Status

**Generated:** $(date)
**Branch:** reorganization

## Summary by Strictness Level

| Level | Issues | Description |
|-------|--------|-------------|
| **Lenient** (current CI) | 0 | Pre-commit passes ✅ |
| **Moderate** (next target) | 179 | Serious code issues to fix |
| **Strict** (long-term goal) | 710 | Best practices target |

## Moderate Issues Breakdown (179 total)

| Count | Code | Description | Priority | Effort |
|-------|------|-------------|----------|--------|
| 41 | F401 | Unused imports | 🔥 High | Easy |
| 37 | F541 | f-string without placeholders | 🔥 High | Easy |
| 22 | C901 | Function too complex | 🟡 Medium | Hard |
| 21 | E226 | Missing whitespace around arithmetic operator | 🟢 Low | Easy |
| 19 | F841 | Unused local variables | 🔥 High | Easy |
| 12 | F811 | Redefinition of unused name | 🟡 Medium | Medium |
| 10 | E402 | Module import not at top | 🟡 Medium | Easy |
| 8 | E231 | Missing whitespace after ',' | 🟢 Low | Easy |
| 4 | E741 | Ambiguous variable name (I, l, O) | 🔥 High | Easy |
| 4 | E722 | Bare except clause | 🔥 High | Easy |

## Recommended Fix Order

### Phase 1: Quick Wins (65 issues, ~2-3 hours)
Fix the easy, high-impact issues first:

1. **E722 (4 issues)** - Bare except clauses
   - These are **dangerous** (can hide KeyboardInterrupt, SystemExit)
   - Change `except:` to `except Exception:`
   - Command: `make lint-moderate | grep E722`

2. **E741 (4 issues)** - Ambiguous variable names
   - Variables named `I`, `l`, or `O` are confusing
   - Rename to descriptive names
   - Command: `make lint-moderate | grep E741`

3. **F541 (37 issues)** - f-strings without placeholders
   - Using `f"text"` instead of `"text"`
   - Remove the `f` prefix
   - Command: `make lint-moderate | grep F541`

4. **F841 (19 issues)** - Unused variables
   - Variables assigned but never used
   - Either use them or remove the assignment
   - Command: `make lint-moderate | grep F841`

### Phase 2: Import Cleanup (41 issues, ~1-2 hours)

5. **F401 (41 issues)** - Unused imports
   - Remove imports that aren't used
   - Some might be used indirectly (check carefully)
   - Command: `make lint-moderate | grep F401`

### Phase 3: Code Organization (18 issues, ~1 hour)

6. **E402 (10 issues)** - Imports not at top
   - Move imports to the top of the file
   - May need to handle dynamic imports specially
   - Command: `make lint-moderate | grep E402`

7. **E226 + E231 (29 issues)** - Whitespace formatting
   - Black should handle these, might be in comments
   - Run `make format` first, then check remaining
   - Command: `make lint-moderate | grep -E "E226|E231"`

### Phase 4: Refactoring (34 issues, ~1-2 weeks)

8. **F811 (12 issues)** - Redefinition of names
   - Same function/variable defined multiple times
   - Often from conditional imports (numba/jit)
   - Requires careful review
   - Command: `make lint-moderate | grep F811`

9. **C901 (22 issues)** - Complex functions
   - Functions with too many branches/logic paths
   - Requires refactoring into smaller functions
   - Start with highest complexity first
   - Command: `make lint-moderate | grep C901`

## How to Fix: Step-by-Step

### Example Session

```bash
# 1. Pick a category (start with E722)
make lint-moderate | grep E722 > bare_except_issues.txt

# 2. Review the file
cat bare_except_issues.txt
# Output:
# src/multi_tracker/gui/main_window.py:1234:5: E722 do not use bare 'except'

# 3. Edit the file
# Before:
#   try:
#       risky_operation()
#   except:
#       pass
#
# After:
#   try:
#       risky_operation()
#   except Exception as e:
#       logger.warning(f"Operation failed: {e}")

# 4. Verify fixed
make lint-moderate | grep E722  # Should show 3 instead of 4

# 5. Test that code still works
make test

# 6. Commit the fix
make commit-prep
git add -u
git commit -m "Fix bare except clauses (E722)"

# 7. Move to next category
```

### Batch Fixing Tips

For simple replacements, you can fix multiple at once:

```bash
# Fix all f-strings without placeholders
# Find them first:
make lint-moderate | grep F541

# Edit files and remove 'f' prefix from strings like:
#   f"Loading data..."  →  "Loading data..."

# Verify all fixed:
make lint-moderate | grep F541  # Should be 0
```

## Tracking Progress

Create a quality log:

```bash
# Initial baseline
echo "$(date +%Y-%m-%d) Moderate: 179  Strict: 710" >> .quality_log

# After each fix session
make lint-report 2>&1 | grep "issues" | xargs -I {} echo "$(date +%Y-%m-%d) {}" >> .quality_log

# View progress
cat .quality_log
```

## Weekly Quality Goals

A reasonable pace:

- **Week 1**: Fix E722, E741 (8 issues) → Down to 171
- **Week 2**: Fix 20 F541 issues → Down to 151
- **Week 3**: Fix remaining F541 + F841 → Down to 95
- **Week 4**: Fix F401 imports → Down to 54
- **Week 5-6**: Fix E402, whitespace → Down to 15
- **Week 7-8**: Fix F811 → Down to 3
- **Month 3+**: Refactor C901 complex functions → Down to 0 🎉

## Files Needing Most Attention

Based on the sample, these files have multiple issues:

1. `src/multi_tracker/core/post/processing.py` (8+ issues)
   - Unused imports (F401)
   - Unused variables (F841)
   - High complexity (C901)

2. `src/multi_tracker/core/identity/analysis.py` (many F541)
   - f-strings without placeholders

3. `src/multi_tracker/core/background/model.py` (F811)
   - Redefinition issues

4. `src/multi_tracker/core/filters/kalman.py` (E741)
   - Ambiguous variable names

## Commands Reference

```bash
# Generate this report
make lint-report

# Run moderate linting
make lint-moderate

# Save to file
make lint-moderate > issues.txt

# Filter by error type
make lint-moderate | grep F401

# Count by type
make lint-moderate | cut -d: -f4 | cut -d' ' -f2 | sort | uniq -c | sort -rn

# Fix a specific file
flake8 --config=.flake8.moderate src/multi_tracker/core/post/processing.py
```

## Next Steps

1. ✅ Start with **Phase 1** (Quick Wins)
2. ✅ Commit fixes incrementally
3. ✅ Run tests after each fix
4. ✅ Track progress weekly
5. ⏰ When moderate reaches 0, update `.flake8` to moderate config
6. ⏰ Then start on strict issues

## Resources

- See [CODE_QUALITY_GUIDE.md](.github/CODE_QUALITY_GUIDE.md) for detailed strategies
- [Flake8 docs](https://flake8.pycqa.org/en/latest/)
- [Error code reference](https://flake8.pycqa.org/en/latest/user/error-codes.html)
