# Confidence Metrics

Confidence-related outputs combine detection and tracking signals.

## Typical Metrics

- Detection confidence (detector quality signal)
- Assignment confidence (match quality signal)
- Position uncertainty (state covariance-derived signal)

## Why They Matter

- Identify hard frames for active learning.
- Detect parameter regimes that overfit or underfit scene dynamics.
- Prioritize manual review where trajectory quality is weakest.

## Integration Points

- Collected during tracking worker pipeline.
- Optionally persisted to CSV when enabled.
- Consumed by dataset generation/scoring workflows.
