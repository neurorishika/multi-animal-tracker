## Unified Pose Inference Runtime (Doc-Aligned, No Double-Implementation)

### Summary
We will unify MAT + PoseKit pose inference behind one shared runtime interface, while reusing official acceleration paths from Ultralytics and SLEAP-NN instead of building custom ONNX/TensorRT stacks where upstream already provides them.

### Documentation-Locked Decisions
1. YOLO26 acceleration will use Ultralytics-native export/load flow (`model.export(...)`, then `YOLO("*.onnx"|"*.engine")`) and not a custom ONNXRuntime wrapper for YOLO.
Source: [Ultralytics Export Mode](https://docs.ultralytics.com/modes/export/), [YOLO ONNX Integration](https://docs.ultralytics.com/integrations/onnx/), [YOLO TensorRT Integration](https://docs.ultralytics.com/integrations/tensorrt/).
2. SLEAP acceleration will use `sleap-nn` exported-model APIs (`load_exported_model`, `ONNXPredictor`, `TensorRTPredictor`) as optional runtime modes, with safe fallback to current SLEAP service.
Source: [SLEAP Export Guide](https://nn.sleap.ai/dev/export/), [SLEAP Export Predictors](https://nn.sleap.ai/dev/api/export/predictors/), [SLEAP ONNX Predictor API](https://nn.sleap.ai/dev/api/export/predictors/onnx/).
3. SLEAP export path is treated as feature-flagged because upstream marks export as experimental and documents limitations (bottom-up grouping CPU-side, fixed H/W at export, TRT portability).
Source: [SLEAP Export Guide](https://nn.sleap.ai/dev/export/).

### Public Interfaces / Types Changes
1. Add shared runtime API in `/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/core/identity/runtime_api.py`.
2. Define `PoseInferenceBackend` protocol with `warmup()`, `predict_batch(crops)`, `close()`, `benchmark()`.
3. Define backends:
- `YoloNativeBackend` (Ultralytics `.pt/.onnx/.engine`).
- `SleapServiceBackend` (existing service path).
- `SleapExportBackend` (`onnx`/`tensorrt` predictors).
4. Add run-scoped lifecycle manager `InferenceRuntimeManager` in `/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/core/identity/runtime_manager.py`.
5. Add explicit runtime config model:
- `backend_family`: `yolo | sleap`
- `runtime_flavor`: `native | onnx | tensorrt | auto`
- `device`: `auto | cpu | cuda | mps`
- `batch_size`
- `model_path`
- `exported_model_path` (for SLEAP export mode)

### Implementation Plan
1. Refactor `/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/core/identity/feature_runtime.py` to call only the shared API.
2. Refactor PoseKit workers in `/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/posekit/pose_label.py` to use the same API.
3. Keep current per-run precompute flow, but add runtime warmup and persistent session reuse for full run.
4. Move SLEAP service start/stop into run manager enter/exit only; remove ad-hoc global lifetime coupling.
5. Route GPU/provider capability checks through `/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/utils/gpu_utils.py` and selected backend capabilities.
6. Add backend selection policy:
- YOLO: prefer Ultralytics native for `.pt`; use exported `.onnx/.engine` if provided.
- SLEAP: default service backend; export backend only when explicitly enabled and dependencies available.
7. Add metrics hooks for per-backend throughput, startup latency, and cache hit/miss time in runtime logs.

### Test Cases and Scenarios
1. API parity tests:
- Same output schema across `YoloNativeBackend`, `SleapServiceBackend`, `SleapExportBackend`.
2. Lifecycle tests:
- SLEAP service starts once per tracking run and always closes on completion/error/cancel.
3. Backend selection tests:
- Deterministic backend choice for model extension + runtime flavor + device availability.
4. Equivalence tests:
- For fixed crops and model, backend swaps do not change exported column contract or keypoint indexing.
5. Failure tests:
- Missing ORT/TensorRT deps degrades gracefully to supported backend with explicit warning.
6. Performance tests:
- Batch inference throughput improves or remains neutral vs current baseline on same hardware.
7. Integration tests:
- MAT tracking + interpolated-frame analysis + final wide CSV export still complete with pose columns.
- PoseKit preview/inference path works via shared runtime API.

### Rollout
1. Phase 1: Shared API + adapters + no behavior change.
2. Phase 2: Run-scoped lifecycle manager + logging/metrics.
3. Phase 3: Enable SLEAP export backend behind feature flag.
4. Phase 4: Tune defaults per hardware profile and expose minimal UI runtime selectors.

### Assumptions and Defaults
1. YOLO acceleration stays Ultralytics-native by default.
2. SLEAP export acceleration is opt-in until stability is proven.
3. No custom YOLO ONNXRuntime/TensorRT engine implementation will be added.
4. Existing cache semantics remain unless explicitly revised in a separate cache-focused change.
5. Runtime manager is authoritative owner of start/stop for long-lived inference resources.
