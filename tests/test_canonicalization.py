from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from multi_tracker.core.canonicalization import MatMetadataCanonicalizer
from multi_tracker.core.identity.dataset.generator import IndividualDatasetGenerator


def _write_dataset(root: Path, annotations: list[dict]) -> Path:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / "f000000.jpg"
    arr = np.zeros((48, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 100
    arr[:, :, 1] = 150
    Image.fromarray(arr).save(image_path)

    metadata = {
        "schema_version": 2,
        "frames": [
            {
                "frame_id": 0,
                "image_file": image_path.name,
                "annotations": annotations,
            }
        ],
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return image_path


def _write_individual_dataset(root: Path, image_name: str, image_meta: dict) -> Path:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / image_name
    arr = np.zeros((40, 60, 3), dtype=np.uint8)
    arr[:, :, 0] = 80
    arr[:, :, 1] = 120
    Image.fromarray(arr).save(image_path)

    metadata = {"images": [{**image_meta, "filename": image_name}]}
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return image_path


def test_mat_metadata_canonicalizer_applies_single_annotation(tmp_path: Path):
    image_path = _write_dataset(
        tmp_path / "ds",
        [
            {
                "canonicalization": {
                    "center_px": [32.0, 24.0],
                    "size_px": [20.0, 10.0],
                    "angle_rad": 0.0,
                }
            }
        ],
    )
    img = Image.open(image_path).convert("RGB")
    canonicalizer = MatMetadataCanonicalizer(enabled=True, margin=1.2)

    out = canonicalizer(image_path, img)

    assert out.size == (24, 12)
    summary = canonicalizer.summary()
    assert summary["applied_count"] == 1
    assert summary["skipped_count"] == 0


def test_mat_metadata_canonicalizer_skips_ambiguous_frames(tmp_path: Path):
    image_path = _write_dataset(
        tmp_path / "ds",
        [
            {
                "canonicalization": {
                    "center_px": [20.0, 20.0],
                    "size_px": [10.0, 6.0],
                    "angle_rad": 0.0,
                }
            },
            {
                "canonicalization": {
                    "center_px": [40.0, 20.0],
                    "size_px": [10.0, 6.0],
                    "angle_rad": 0.0,
                }
            },
        ],
    )
    img = Image.open(image_path).convert("RGB")
    canonicalizer = MatMetadataCanonicalizer(enabled=True)

    out = canonicalizer(image_path, img)

    assert out.size == img.size
    summary = canonicalizer.summary()
    assert summary["applied_count"] == 0
    assert summary["skip_reasons"]["ambiguous_annotation_count"] == 1


def test_mat_metadata_canonicalizer_reads_individual_dataset_metadata(tmp_path: Path):
    image_path = _write_individual_dataset(
        tmp_path / "individual_ds",
        "did17.png",
        {
            "canonicalization": {
                "center_px": [30.0, 20.0],
                "size_px": [18.0, 8.0],
                "angle_rad": 0.0,
            }
        },
    )
    img = Image.open(image_path).convert("RGB")
    canonicalizer = MatMetadataCanonicalizer(enabled=True, margin=1.0)

    out = canonicalizer(image_path, img)

    assert out.size == (18, 8)
    summary = canonicalizer.summary()
    assert summary["applied_count"] == 1


def test_individual_dataset_generator_writes_canonicalization_metadata(tmp_path: Path):
    generator = IndividualDatasetGenerator(
        {
            "ENABLE_INDIVIDUAL_DATASET": True,
            "INDIVIDUAL_DATASET_RUN_ID": "testrun",
            "INDIVIDUAL_OUTPUT_FORMAT": "png",
            "INDIVIDUAL_CROP_PADDING": 0.0,
        },
        output_dir=str(tmp_path),
        video_name="video.mp4",
        dataset_name="individual_dataset",
    )
    frame = np.full((50, 80, 3), 180, dtype=np.uint8)
    corners = np.array([[24.0, 18.0], [40.0, 18.0], [40.0, 26.0], [24.0, 26.0]])

    saved = generator.process_frame(
        frame=frame,
        frame_id=0,
        meas=[[32.0, 22.0, 0.0]],
        obb_corners=[corners],
        confidences=[0.9],
        track_ids=[3],
    )
    assert saved == 1

    dataset_dir = Path(generator.finalize())
    raw = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    entry = raw["images"][0]

    assert entry["canonicalization"]["center_px"] == [8.0, 4.0]
    assert entry["canonicalization"]["size_px"] == [16.0, 8.0]
    assert entry["canonicalization"]["orientation_source"] == "tracking_theta"
    assert len(entry["obb_corners_local"]) == 4
