# Dataset Generation

Dataset generation supports active learning loops for improving detector models.

## What It Does

- Scores frames using configurable quality metrics.
- Selects challenging frames for annotation.
- Exports image/label artifacts for downstream training.

## Quality Metrics (Conceptual)

- Low confidence detections
- Detection count mismatch vs expected target count
- High assignment costs
- Track loss events
- Optional uncertainty-based triggers

## When to Use

- Detector underperforms on specific lighting or behaviors.
- You need focused retraining data instead of random frame sampling.

## Tradeoffs

- Aggressive selection can bias toward outliers.
- Conservative selection may miss edge cases.

## Practical Loop

1. Run MAT and generate candidate frames.
2. Validate/annotate selected frames.
3. Retrain YOLO model.
4. Re-run MAT on representative videos.
5. Compare confidence and identity continuity metrics.
