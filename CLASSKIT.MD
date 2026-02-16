# ClassKit: PySide6 Active-Learning Dataset Builder (PoseKit Sister Framework)

> **Context:** We already have a pose training kit called **PoseKit** in this repo at `src/multi_tracker/posekit` on the `mat-pose-integration` branch. ([GitHub][1])
> **Goal:** Copy PoseKit’s “training kit” pattern and adapt it into a **smart sister framework** for **image classification dataset generation + active learning + fast iteration**, implemented as a **Python backend + PySide6 desktop UI**.

---

## Current ClassKit status (February 2026)

This section reflects what is currently implemented in the codebase (not just planned).

### Implemented now

* **Desktop app shell + workflows:** PySide6 app entrypoint, project create/open, ingestion, embedding, clustering, UMAP, and labeling flow are wired.
* **Explorer + labeling UX:** Explore vs Labeling mode behavior is enforced, candidate sampling works, keyboard labeling (1-9) works, and selected-point styling is high-contrast.
* **Performance/stability hardening:** coalesced redraw/preview/context timers, command cooldown gating, auto-repeat suppression, and preallocated history slots are in place.
* **Labeling completion UX:** when a sampled set is fully labeled, user is prompted to either sample another set or train immediately.
* **Hover-mode fix after completion:** selection clears and hover is restored when candidate set is exhausted.
* **Autosave + heartbeat:** buffered label writes autosave on timer, heartbeat indicator shows state, autosave interval is configurable in UI and persisted in `project.json`.
* **Training UI integration:** embedding-head training dialog is implemented (linear/MLP + hyperparameters), worker supports those settings, and training success stores checkpoints and metadata in project `models/`.
* **Export workflow (P0):** export dialog now supports `ImageFolder`, `CSV`, `Parquet`, and `Ultralytics classify` targets with validation, worker-backed progress, and success/error dialogs.
* **kNN labeling assist (P0):** labeling panel now shows nearest neighbors for the selected anchor point, includes quick jump-to-nearest, and guarded bulk-label assist for nearest unlabeled items.
* **Checkpoint restore + inference loop (P0):** UI can load latest/snapshot classifier checkpoints and run predict-on-unlabeled; predictions/confidences are persisted to DB metadata and exported to `models/predictions_latest.csv`.
* **Caching infrastructure:** embeddings, clusters, and UMAP caches are persisted and can be reloaded.

### Partially implemented / still missing

* **Active-learning batch builder UI:** strategy composition UI and reason-based basket building are not yet exposed in desktop views.
* **Metrics dashboard UI:** confusion matrix, per-class metrics, and reliability plot UI are not yet implemented.
* **PoseKit parity layer:** shared run/config/log abstractions are still incomplete and need standardization.

---

## Next-feature TODOs (prioritized)

### P0 — Complete user-critical labeling loop

- [x] Finish **export workflow** in UI (`ImageFolder`, `CSV/Parquet`, `Ultralytics classify`) with validation + progress + success dialogs.
- [x] Add **kNN neighbors panel** in labeling view (top-k nearest, quick jump, optional guarded bulk-label assist).
- [x] Add **training checkpoint loader** in UI (choose latest/snapshot checkpoint, restore trainer for inference).
- [x] Add **predict-on-unlabeled** post-training action and persist prediction/confidence outputs.

### P1 — Active learning features promised in spec

- [ ] Build **Batch Builder view** with sliders for uncertainty/diversity/representative/audit weights.
- [ ] Wire `al/acquisition.py` recipe to real UI actions and dataset state.
- [ ] Enforce **per-cluster caps** and expose composition summary (by cluster, class, reason).
- [ ] Add **cluster browser actions**: “add N uncertain/diverse from cluster”.

### P1 — Evaluation and QA

- [ ] Implement **Metrics view**: confusion matrix, class PR/F1, calibration summary (pre/post temperature scaling).
- [ ] Add **Audit mode** queue injection and disagreement surfacing.
- [ ] Track round-level stopping heuristics (plateau, disagreement stability, novelty drop).

### P2 — Robustness + scalability

- [ ] Add cancellable long-running job controls (cancel/retry, clearer error surfaces).
- [ ] Add project-level settings panel for cache policies and autosave defaults.
- [ ] Improve >100k image explorer behavior with explicit LOD/downsampling policy and benchmarks.

### P2 — PoseKit sister-framework parity

- [ ] Align ClassKit run/artifact folder conventions with PoseKit (`runs/classkit/...`).
- [ ] Align config schema + defaults style with PoseKit conventions.
- [ ] Add optional PoseKit-to-ClassKit import bridge (tracklet/frame ingestion source).

---

## 1) High-level product requirements

### 1.1 What ClassKit does

* Ingest a large image corpus (folders / videos-to-frames / existing datasets).
* Compute image **embeddings**, build a fast **nearest-neighbor index**, and compute a 2D **UMAP** map for exploration (UMAP is *visualization-only*). ([umap-learn.readthedocs.io][2])
* Provide a **PySide6 UI** to:

  * explore corpus (UMAP map, clusters, kNN neighbors)
  * rapidly label images (keyboard-first queue)
  * build **active-learning batches** (uncertainty + diversity + representativeness + audits)
* Train a **classifier head on embeddings** (fast) each round; optionally export + train an end-to-end classifier later.
* Export versioned datasets in common formats (ImageFolder / CSV/Parquet / Ultralytics classify).

### 1.2 Non-goals

* No web app / multi-tenant auth / distributed MLOps.
* No “UMAP looks better” as a training metric.
* No heavy refactors inside PoseKit; ClassKit should **reuse PoseKit conventions**, not break them.

---

## 2) Key design decisions (opinionated)

### 2.1 Use FAISS for similarity + kNN

* **FAISS** provides efficient similarity search and clustering for dense vectors and supports large-scale indexing. ([GitHub][3])

### 2.2 Use UMAP only for UI navigation

* UMAP is a dimensionality reduction technique useful for visualization and non-linear reduction, but should not be treated as the “truth” for selection/metrics. ([umap-learn.readthedocs.io][2])
* Selection and evaluation run in **embedding space**, not in UMAP space.

### 2.3 Use Qt background workers (don’t freeze UI)

* Use `QThreadPool` + `QRunnable` for background tasks; this is the standard Qt pattern for responsive apps. ([doc.qt.io][4])

### 2.4 Calibrate confidence before using it for active learning

* Use **temperature scaling** for post-hoc calibration; it’s a simple, effective method for modern neural nets. ([arXiv][5])

---

## 3) Repo structure: mirror PoseKit, add ClassKit

### 3.1 Location

Create a new package alongside PoseKit:

```
src/multi_tracker/
  posekit/                  # existing (do not break)
  classkit/                 # new sister framework
```

### 3.2 “Copy PoseKit” rule

ClassKit should follow PoseKit’s conventions for:

* config schema style (YAML/JSON conventions, defaults, validation)
* run/artifact directories (versioned runs, logs, weights)
* CLI entrypoints + project initialization patterns
* internal “trainer” API shape (fit/eval/export)
* logging + progress reporting

> **Important:** Do not invent a parallel universe. Reuse PoseKit patterns and extend them for classification + active learning.

---

## 4) ClassKit architecture (Python backend + PySide6 UI)

### 4.1 Core modules (backend)

**Backend is still “local,”** but structured as a set of services called by the UI.

```
src/multi_tracker/classkit/
  __init__.py
  config/
    schemas.py             # pydantic/dataclasses; mirror PoseKit approach
    defaults.yaml
  data/
    ingest.py              # scan folders, hash, metadata
    thumbs.py              # thumbnail cache
    splits.py              # train/val/test assignment
  embed/
    embedder_base.py       # interface: embed(paths)->(N,D)
    embedders/             # CLIP, DINOv2, custom
    store.py               # memmap/parquet storage + versioning
  index/
    faiss_index.py         # build/load/query knn (FAISS) :contentReference[oaicite:6]{index=6}
  viz/
    umap_reduce.py         # compute coords for UI :contentReference[oaicite:7]{index=7}
  cluster/
    clusterer.py           # overclustering for “modes”, not classes
    prototypes.py          # medoids/prototypes per cluster
  al/
    acquisition.py         # batch selection recipe (uncert/diverse/rep/audit)
    density.py             # density proxies (kNN distance stats)
  train/
    head_models.py         # linear/MLP heads on embeddings
    trainer.py             # fit/eval/predict
    calibrate.py           # temperature scaling :contentReference[oaicite:8]{index=8}
    metrics.py             # confusion, per-class, calibration
  store/
    db.py                  # sqlite schema + migrations
    artifacts.py           # run folders, versioning, export
  export/
    imagefolder.py
    ultralytics_classify.py
    parquet_csv.py
  jobs/
    worker.py              # QRunnable wrappers + signals :contentReference[oaicite:9]{index=9}
```

### 4.2 Data persistence

* SQLite DB for metadata + labels + versions.
* Filesystem artifact folders for:

  * thumbnails
  * embeddings (memmap recommended)
  * FAISS indices
  * UMAP coordinates
  * clusters + prototypes
  * model checkpoints + calibration params
  * exports

**Rule:** never overwrite; create new version folders each run.

---

## 5) Active-learning loop (concrete and implementable)

### 5.1 Overclustering strategy (modes)

* Cluster embeddings into **many clusters** (e.g., hundreds to thousands) to represent *visual modes*.
* Do **not** set `k = num_classes`. Clusters ≠ classes.

Store per cluster:

* size
* density proxy (mean kNN distance)
* “purity” once labels exist
* disagreement rate

### 5.2 Batch acquisition recipe (default)

For batch size **B**:

* 40% **uncertainty** (highest entropy / smallest margin)
* 35% **diversity** (k-center / farthest-first in embedding space)
* 15% **representativeness** (dense clusters with low label coverage)
* 10% **audits** (random + targeted high-disagreement clusters)

Hard constraints:

* per-cluster cap (avoid one mode dominating)
* minimum per-class (if class imbalance already exists)

### 5.3 Training each round (fast path)

* Train embedding-head classifier (linear → optional small MLP).
* Calibrate via temperature scaling on val. ([arXiv][5])
* Predict on unlabeled pool → compute uncertainty.
* Generate next batch.

Stop conditions:

* val metric plateaus for N rounds
* audit accuracy/disagreement stabilizes
* selection “novelty” drops (new batch too similar to prior batches)

---

## 6) PySide6 UI specification

### 6.1 Main window layout

* Left: navigation (Explorer / Clusters / Label / Batch / Metrics / Settings)
* Center: active view
* Right: context (image details, neighbors, class definitions, batch basket)

### 6.2 Views

#### A) Explorer (UMAP Map)

* Render scatter of UMAP coords (LOD/downsample when zoomed out).
* Color modes:

  * labeled/unlabeled
  * predicted class
  * uncertainty heatmap
  * cluster id
* Interactions:

  * click point → image detail + neighbors
  * lasso select → add to batch basket
  * filters: cluster, label status, predicted class, uncertainty range

#### B) Cluster Browser

* Table/list of clusters with:

  * prototypes thumbnails grid
  * size, density, label coverage, disagreement, avg uncertainty
* Buttons:

  * “Add N uncertain from this cluster”
  * “Add N diverse from this cluster”
  * “Open cluster grid”

#### C) Label Workbench (keyboard-first)

* Big image preview + next/prev queue
* Keybinds:

  * 1..9 assign class
  * `U` unsure, `J` junk, `S` skip
  * `A` accept suggested label (optional)
* Side panel:

  * kNN neighbors (from FAISS) ([GitHub][3])
  * “label similar N” (guarded: requires manual review before bulk apply)

#### D) Batch Builder

* Composition charts (by cluster, by predicted class, by reason)
* Sliders for recipe weights (uncert/diverse/rep/audit)
* “Regenerate batch” button

#### E) Metrics

* Confusion matrix
* per-class precision/recall/F1
* calibration reliability plot (pre/post temperature scaling)

#### F) QA / Audit

* inject audit items into queue (default 10%)
* track annotator agreement (if multi-annotator later)
* flag clusters with high disagreement (taxonomy issues)

---

## 7) Background jobs (Qt threading)

### 7.1 Must-run-in-worker tasks

* ingest + hash
* thumbnail generation
* embedding extraction
* FAISS index build/load
* UMAP compute
* clustering + prototypes
* model training + prediction
* export

### 7.2 Implementation pattern

* `QRunnable` job classes emitting signals: progress/status/result/error
* `QThreadPool.globalInstance().start(job)` for execution ([doc.qt.io][6])
* UI receives signals and updates progress overlays without blocking.

---

## 8) Integration points with PoseKit (explicit tasks)

### 8.1 Shared “kit” conventions

* Create a small internal convention layer (or reuse PoseKit’s if it exists):

  * `ProjectConfig` base class
  * `RunContext` / artifact manager
  * logging + progress abstraction
  * common CLI layout

### 8.2 Shared artifact layout

* Ensure PoseKit and ClassKit exports can coexist under the same dataset/project root:

  * `runs/posekit/...`
  * `runs/classkit/...`

### 8.3 Optional: “Pose → Class” bridge

If useful for your tracking pipeline:

* allow importing PoseKit outputs (cropped individuals / tracklets) into ClassKit as a “corpus source”
* label at the tracklet-level or frame-level

---

## 9) Milestones (Codex implementation plan)

### Milestone 1 — MVP labeling + embeddings

* dataset ingest (folder)
* thumbnail cache
* embedding compute + storage
* FAISS kNN query
* label workbench + SQLite labels
* export ImageFolder

### Milestone 2 — Explorer + clustering + batch builder

* UMAP coords compute + map view
* overclustering + prototypes view
* active-learning batch builder v1

### Milestone 3 — Training loop + calibration + metrics

* embedding-head training
* temperature scaling calibration ([arXiv][5])
* prediction + uncertainty computation
* metrics dashboard + plateau detection

### Milestone 4 — Polish + “sister framework” parity with PoseKit

* unify configs / runs / logging to match PoseKit conventions
* robust version switching (embedding vN, model vM)
* audit mode, disagreement surfacing, export formats

---

## 10) Acceptance criteria

* Can handle ≥100k images with responsive UI (LOD + caching).
* kNN neighbors load fast and reliably via FAISS.
* Active-learning batch includes mixed reasons and respects per-cluster caps.
* Training round finishes quickly (head-only) and produces calibrated confidence.
* Exports are reproducible and versioned.
* ClassKit follows PoseKit “kit” conventions (config/run/log layout) without breaking PoseKit.
