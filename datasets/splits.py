from __future__ import annotations

from collections import defaultdict
import random
from typing import Mapping, Sequence

from dataset_utils import SampleRecord

SplitDict = dict[str, list[SampleRecord]]

_SPLIT_NAMES = ("train", "val", "test")


def group_samples_by_id(samples: Sequence[SampleRecord]) -> dict[str, list[SampleRecord]]:
    grouped: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_id].append(sample)

    return {
        group_id: sorted(group_samples, key=lambda sample: sample.sample_id)
        for group_id, group_samples in sorted(grouped.items())
    }


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    for ratio_name, ratio in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{ratio_name} must be between 0.0 and 1.0 inclusive, got {ratio!r}")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio, val_ratio, and test_ratio must sum to 1.0 "
            f"within tolerance, got {ratio_sum!r}"
        )


def _group_type(group_samples: Sequence[SampleRecord]) -> str:
    labels = {sample.label for sample in group_samples}
    if labels == {"forged", "authentic"}:
        return "paired"
    if labels == {"forged"}:
        return "forged_only"
    if labels == {"authentic"}:
        return "authentic_only"
    raise ValueError(f"Unsupported label composition for group {group_samples[0].group_id!r}: {sorted(labels)!r}")


def _allocate_group_counts(total_groups: int, ratios: Sequence[float]) -> list[int]:
    if total_groups == 0:
        return [0 for _ in ratios]

    raw_counts = [ratio * total_groups for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = total_groups - sum(counts)
    fractions = sorted(
        ((index, raw_counts[index] - counts[index]) for index in range(len(ratios))),
        key=lambda item: (-item[1], item[0]),
    )

    for index, _fraction in fractions[:remainder]:
        counts[index] += 1

    return counts


def _shuffled_groups(group_samples: list[list[SampleRecord]], seed: int) -> list[list[SampleRecord]]:
    shuffled = list(group_samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def make_grouped_stratified_splits(
    samples: Sequence[SampleRecord],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitDict:
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    grouped_samples = group_samples_by_id(samples)
    groups_by_type: dict[str, list[list[SampleRecord]]] = {
        "paired": [],
        "forged_only": [],
        "authentic_only": [],
    }

    for group_id in sorted(grouped_samples):
        group_samples = grouped_samples[group_id]
        group_kind = _group_type(group_samples)
        groups_by_type[group_kind].append(group_samples)

    split_to_samples: SplitDict = {name: [] for name in _SPLIT_NAMES}
    split_ratios = (train_ratio, val_ratio, test_ratio)

    for group_kind in ("paired", "forged_only", "authentic_only"):
        groups = _shuffled_groups(groups_by_type[group_kind], seed)
        split_counts = _allocate_group_counts(len(groups), split_ratios)

        start = 0
        for split_name, split_count in zip(_SPLIT_NAMES, split_counts, strict=True):
            stop = start + split_count
            for group in groups[start:stop]:
                split_to_samples[split_name].extend(sample.with_split(split_name) for sample in group)
            start = stop

    for split_name in _SPLIT_NAMES:
        split_to_samples[split_name].sort(key=lambda sample: (sample.group_id, sample.label, sample.sample_id))

    return split_to_samples


def count_samples_by_split_and_label(splits: Mapping[str, Sequence[SampleRecord]]) -> dict:
    total = 0
    by_label: defaultdict[str, int] = defaultdict(int, {"forged": 0, "authentic": 0})
    by_split: dict[str, dict[str, int]] = {}

    for split_name, split_samples in splits.items():
        split_total = 0
        split_label_counts: defaultdict[str, int] = defaultdict(int)
        for sample in split_samples:
            total += 1
            split_total += 1
            by_label[sample.label] += 1
            split_label_counts[sample.label] += 1

        by_split[split_name] = {
            "total": split_total,
            "forged": split_label_counts["forged"],
            "authentic": split_label_counts["authentic"],
        }

    return {
        "total": total,
        "by_label": {"forged": by_label["forged"], "authentic": by_label["authentic"]},
        "by_split": by_split,
    }
