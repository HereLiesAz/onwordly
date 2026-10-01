from collections import Counter
from random import Random

from onwordly.datasets.string_manipulation import build_static_string_dataset
from onwordly.tasks.string_manipulation import (
    apply_string_operation,
    generate_string_task,
    make_composed_string_task,
    make_string_task,
    string_partition,
)
from onwordly.verifiers.string_manipulation import verify_string_answer


def test_string_operations_are_exact() -> None:
    assert apply_string_operation("ABCDE", "remove_vowels") == "BCD"
    assert apply_string_operation("ABCDE", "take_even_indices") == "ACE"
    assert apply_string_operation("ABCDE", "reverse_pairs") == "BADCE"
    assert apply_string_operation("AB3", "duplicate_each") == "AABB33"
    assert apply_string_operation("AEIOU", "remove_vowels") == "<EMPTY>"


def test_generated_string_tasks_verify() -> None:
    rng = Random(12)
    for operation in (
        "remove_vowels",
        "take_even_indices",
        "reverse_pairs",
        "duplicate_each",
    ):
        task = generate_string_task(rng, operation, 8)
        assert verify_string_answer(task, task.answer)
        assert not verify_string_answer(task, f"answer: {task.answer}")


def test_string_partition_separates_datasets() -> None:
    train = build_static_string_dataset(
        seed=1,
        size=40,
        lengths=(6,),
        partition="train",
    )
    evaluation = build_static_string_dataset(
        seed=2,
        size=40,
        lengths=(6,),
        partition="eval",
    )
    assert all(string_partition(task) == "train" for task in train)
    assert all(string_partition(task) == "eval" for task in evaluation)


def test_string_dataset_is_balanced() -> None:
    tasks = build_static_string_dataset(seed=3, size=32, lengths=(4, 8))
    counts = Counter((task.operation, task.length) for task in tasks)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_string_task_normalizes_input() -> None:
    task = make_string_task("ab12", "reverse_pairs")
    assert task.text == "AB12"
    assert task.answer == "BA21"


def test_composed_string_task_applies_rules_in_order() -> None:
    task = make_composed_string_task(
        "ABCDE",
        "remove_vowels",
        "reverse_pairs",
    )
    assert task.answer == "CBD"
    assert task.bucket_key == "remove_vowels>reverse_pairs:5"
    assert verify_string_answer(task, "CBD")


def test_composed_string_handles_empty_intermediate() -> None:
    task = make_composed_string_task(
        "AEIOU",
        "remove_vowels",
        "duplicate_each",
    )
    assert task.answer == "<EMPTY>"
