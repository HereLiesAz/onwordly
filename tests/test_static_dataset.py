from collections import Counter

from onwordly.datasets.arithmetic import (
    arithmetic_dataset_stats,
    build_static_arithmetic_dataset,
    read_arithmetic_jsonl,
    write_arithmetic_jsonl,
)
from onwordly.tasks.arithmetic import arithmetic_partition


def test_static_dataset_is_deterministic_and_balanced(tmp_path) -> None:
    first = build_static_arithmetic_dataset(seed=5, size=18, digit_levels=(1, 2))
    second = build_static_arithmetic_dataset(seed=5, size=18, digit_levels=(1, 2))
    assert first == second

    counts = Counter((task.operation, task.digits) for task in first)
    assert max(counts.values()) - min(counts.values()) <= 1

    path = write_arithmetic_jsonl(first, tmp_path / "dataset.jsonl")
    assert read_arithmetic_jsonl(path) == first


def test_partitioned_datasets_do_not_share_operand_keys() -> None:
    train = build_static_arithmetic_dataset(
        seed=5,
        size=100,
        digit_levels=(2,),
        partition="train",
    )
    evaluation = build_static_arithmetic_dataset(
        seed=6,
        size=100,
        digit_levels=(2,),
        partition="eval",
    )

    assert all(
        arithmetic_partition(task.left, task.right, task.operation) == "train"
        for task in train
    )
    assert all(
        arithmetic_partition(task.left, task.right, task.operation) == "eval"
        for task in evaluation
    )


def test_dataset_balances_withheld_prompt_styles() -> None:
    tasks = build_static_arithmetic_dataset(
        seed=9,
        size=18,
        operations=("add",),
        digit_levels=(2,),
        prompt_styles=("question", "words", "expression"),
        partition="eval",
    )
    counts = Counter(task.prompt_style for task in tasks)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_dataset_stats_report_duplicate_rate() -> None:
    tasks = build_static_arithmetic_dataset(
        seed=4,
        size=30,
        operations=("add",),
        digit_levels=(1,),
        partition="train",
    )
    stats = arithmetic_dataset_stats(tasks)
    assert stats["examples"] == 30
    assert 0 < stats["unique_logical_tasks"] <= 30
    assert stats["duplicate_logical_examples"] == (
        stats["examples"] - stats["unique_logical_tasks"]
    )
    assert 0.0 <= stats["duplicate_logical_rate"] <= 1.0
