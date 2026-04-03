from pathlib import Path

from hydra_suite.classkit.store.db import ClassKitDB


def test_db():
    db_path = Path("test_classkit.db")
    if db_path.exists():
        db_path.unlink()

    db = ClassKitDB(db_path)
    paths = [Path(f"img_{i}.png") for i in range(5)]
    db.add_images(paths)

    # Check initial labels
    labels = db.get_all_labels()
    print(f"Initial labels: {labels}")
    assert all(l is None for l in labels)

    # Update one label (use resolved path to match what add_images stores)
    db.update_labels_batch({str(paths[2].resolve()): "class_A"})

    # Reload and check
    labels = db.get_all_labels()
    print(f"Labels after update: {labels}")
    assert labels[2] == "class_A"

    db_path.unlink()
    print("Test passed!")


if __name__ == "__main__":
    test_db()
