from __future__ import annotations

from collections import Counter, defaultdict
from math import floor
from typing import Hashable, Sequence


def _normalize_split_strategy(strategy: str | None) -> str:
    normalized = str(strategy or "stratified").strip().lower()
    return normalized if normalized in {"stratified", "random"} else "stratified"


def _allocate_holdout_counts(
    counts_by_label: dict[Hashable, int],
    *,
    fraction: float,
    total_items: int,
    preallocated: dict[Hashable, int] | None = None,
) -> dict[Hashable, int]:
    """Allocate a stratified holdout count per class.

    The allocator preserves at least one train sample per class whenever
    possible and uses largest-remainder apportionment so the overall holdout
    count stays close to the requested fraction.
    """

    if total_items <= 0 or fraction <= 0.0:
        return {label: 0 for label in counts_by_label}

    preallocated = preallocated or {}
    capacities: dict[Hashable, int] = {}
    for label, count in counts_by_label.items():
        already = int(preallocated.get(label, 0))
        remaining = max(0, int(count) - already)
        capacities[label] = max(0, remaining - 1)

    desired_total = min(
        int(round(float(total_items) * float(fraction))),
        sum(capacities.values()),
    )
    if desired_total <= 0:
        return {label: 0 for label in counts_by_label}

    allocations: dict[Hashable, int] = {}
    remainders: list[tuple[float, int, str, Hashable]] = []
    for label, count in counts_by_label.items():
        raw = float(count) * float(fraction)
        base = min(int(floor(raw)), capacities[label])
        allocations[label] = base
        remainders.append((raw - floor(raw), int(count), str(label), label))

    allocated_total = sum(allocations.values())
    for _remainder, _count, _label_text, label in sorted(
        remainders,
        key=lambda item: (-item[0], -item[1], item[2]),
    ):
        if allocated_total >= desired_total:
            break
        if allocations[label] >= capacities[label]:
            continue
        allocations[label] += 1
        allocated_total += 1

    return allocations


def build_stratified_splits(
    labels: Sequence[Hashable],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[str]:
    """Return deterministic stratified train/val/test split labels.

    The split preserves class proportions as closely as practical while keeping
    at least one training example per class whenever a class has multiple
    samples.
    """

    import numpy as np

    n_items = len(labels)
    if n_items == 0:
        return []

    val_fraction = min(max(float(val_fraction), 0.0), 1.0)
    test_fraction = min(max(float(test_fraction), 0.0), 1.0)

    labels_list = list(labels)
    counts_by_label = dict(Counter(labels_list))
    test_allocations = _allocate_holdout_counts(
        counts_by_label,
        fraction=test_fraction,
        total_items=n_items,
    )
    val_allocations = _allocate_holdout_counts(
        counts_by_label,
        fraction=val_fraction,
        total_items=n_items,
        preallocated=test_allocations,
    )

    indices_by_label: dict[Hashable, list[int]] = defaultdict(list)
    for index, label in enumerate(labels_list):
        indices_by_label[label].append(index)

    rng = np.random.default_rng(seed)
    split_by_index = ["train"] * n_items
    for label in sorted(indices_by_label, key=str):
        indices = list(indices_by_label[label])
        rng.shuffle(indices)

        n_test = min(len(indices), int(test_allocations.get(label, 0)))
        n_val = min(
            max(0, len(indices) - n_test),
            int(val_allocations.get(label, 0)),
        )

        for index in indices[:n_test]:
            split_by_index[index] = "test"
        for index in indices[n_test : n_test + n_val]:
            split_by_index[index] = "val"

    if val_fraction > 0.0 and "val" not in split_by_index and n_items > 1:
        train_counts = Counter(
            labels_list[index]
            for index, split in enumerate(split_by_index)
            if split == "train"
        )
        for label in sorted(
            counts_by_label, key=lambda item: (-counts_by_label[item], str(item))
        ):
            if train_counts.get(label, 0) > 1:
                for index, split in enumerate(split_by_index):
                    if split == "train" and labels_list[index] == label:
                        split_by_index[index] = "val"
                        return split_by_index

    return split_by_index


def build_random_splits(
    labels: Sequence[Hashable],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[str]:
    """Return deterministic random train/val/test split labels."""

    import numpy as np

    n_items = len(labels)
    if n_items == 0:
        return []

    val_fraction = min(max(float(val_fraction), 0.0), 1.0)
    test_fraction = min(max(float(test_fraction), 0.0), 1.0)

    desired_test = int(round(float(n_items) * test_fraction))
    desired_val = int(round(float(n_items) * val_fraction))

    if test_fraction > 0.0 and desired_test <= 0 and n_items > 1:
        desired_test = 1
    desired_test = min(max(0, desired_test), max(0, n_items - 1))

    remaining_after_test = n_items - desired_test
    if val_fraction > 0.0 and desired_val <= 0 and remaining_after_test > 1:
        desired_val = 1
    desired_val = min(max(0, desired_val), max(0, remaining_after_test - 1))

    indices = list(range(n_items))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_by_index = ["train"] * n_items
    for index in indices[:desired_test]:
        split_by_index[index] = "test"
    for index in indices[desired_test : desired_test + desired_val]:
        split_by_index[index] = "val"

    return split_by_index


def build_dataset_splits(
    labels: Sequence[Hashable],
    *,
    strategy: str = "stratified",
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[str]:
    """Return deterministic split labels using the requested strategy."""

    normalized_strategy = _normalize_split_strategy(strategy)
    if normalized_strategy == "random":
        return build_random_splits(
            labels,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    return build_stratified_splits(
        labels,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
