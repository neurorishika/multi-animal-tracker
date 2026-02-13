# Performance Tuning

## Highest-Impact Controls

- `resize_factor`: dominant speed/accuracy tradeoff.
- Visualization/overlay toggles: large runtime impact in GUI mode.
- Detection mode and model size: major compute driver.
- Batch sizing/device selection: critical for YOLO throughput.

## Throughput Strategies

- For CPU-only workflows: use background subtraction and conservative visualization.
- For GPU workflows: verify backend, then tune model/batch settings.
- For long videos: enable caching to accelerate iterative backward/post-processing runs.

## Stability Strategies

- Use conservative thresholds first.
- Validate on representative frames before full-scale runs.
- Keep session logs and config snapshots for regression comparisons.
