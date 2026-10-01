from collections import Counter

from onwordly.datasets.arithmetic import (
    build_static_arithmetic_dataset,
    read_arithmetic_jsonl,
    write_arithmetic_jsonl,
)


def test_static_dataset_is_deterministic_and_balanced(tmp_path) -> None:
    first = build_static_arithmetic_dataset(seed=5, size=18, digit_levels=(1, 2))
    second = build_static_arithmetic_dataset(seed=5, size=18, digit_levels=(1, 2))
    assert first == second

    counts = Counter((task.operation, task.digits) for task in first)
    assert max(counts.values()) - min(counts.values()) <= 1

    path = write_arithmetic_jsonl(first, tmp_path / "dataset.jsonl")
    assert read_arithmetic_jsonl(path) == first
