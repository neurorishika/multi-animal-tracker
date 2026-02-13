# Extending Detection

## Main Extension Surface

`multi_tracker.core.detectors.engine`

## Recommended Pattern

1. Add detector class with `detect_objects`-compatible output shape.
2. Register selection logic in factory/creation path.
3. Keep measurement format consistent with assignment and Kalman expectations.
4. Update user-facing options/documentation for new mode.

## Validation Checklist

- Produces stable center/orientation estimates.
- Handles empty/noisy frames gracefully.
- Works with existing tracking/post-processing without schema changes.
