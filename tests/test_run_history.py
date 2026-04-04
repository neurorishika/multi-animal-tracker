from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.trackerkit.gui.dialogs.run_history_dialog import load_run_history


def test_load_run_history_empty(tmp_path: Path):
    reg_path = tmp_path / "registry.json"
    reg_path.write_text('{"runs": []}', encoding="utf-8")
    runs = load_run_history(str(reg_path))
    assert runs == []


def test_load_run_history_with_entries(tmp_path: Path):
    reg_path = tmp_path / "registry.json"
    data = {
        "runs": [
            {
                "run_id": "20260401-120000_obb_direct_abc12345",
                "started_at": "2026-04-01T12:00:00",
                "finished_at": "2026-04-01T12:30:00",
                "status": "completed",
                "role": "obb_direct",
                "spec": {
                    "hyperparams": {"epochs": 100, "imgsz": 640, "batch": 16},
                    "base_model": "yolo26s-obb.pt",
                },
            },
            {
                "run_id": "20260401-130000_seq_detect_def67890",
                "started_at": "2026-04-01T13:00:00",
                "finished_at": "",
                "status": "failed",
                "role": "seq_detect",
                "spec": {
                    "hyperparams": {"epochs": 50, "imgsz": 640, "batch": 8},
                    "base_model": "yolo26s.pt",
                },
            },
        ]
    }
    reg_path.write_text(json.dumps(data), encoding="utf-8")
    runs = load_run_history(str(reg_path))
    assert len(runs) == 2
    assert runs[0]["status"] == "completed"
    assert runs[1]["status"] == "failed"
