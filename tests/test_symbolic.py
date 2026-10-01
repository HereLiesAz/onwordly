from collections import Counter
from random import Random

from onwordly.datasets.symbolic import (
    build_static_symbolic_dataset,
    read_symbolic_jsonl,
    write_symbolic_jsonl,
)
from onwordly.tasks.symbolic import (
    apply_symbolic_operation,
    generate_symbolic_task,
    make_symbolic_task,
)
from onwordly.verifiers.symbolic import parse_symbolic_answer, verify_symbolic_answer


def test_symbolic_operations_are_exact() -> None:
    assert apply_symbolic_operation("ABCA", "reverse") == "ACBA"
    assert apply_symbolic_operation("CBAC", "sort") == "ABCC"
    assert apply_symbolic_operation("AAABBCCA", "dedupe_adjacent") == "ABCA"
    assert apply_symbolic_operation("ABCD", "rotate_left") == "BCDA"


def test_generated_symbolic_tasks_verify() -> None:
    rng = Random(9)
    for operation in ("reverse", "sort", "dedupe_adjacent", "rotate_left"):
        for length in (4, 8, 12):
            task = generate_symbolic_task(rng, operation, length)
            assert verify_symbolic_answer(task, task.answer)


def test_symbolic_verifier_is_strict() -> None:
    task = make_symbolic_task("ABCD", "reverse")
    assert verify_symbolic_answer(task, "DCBA")
    assert not verify_symbolic_answer(task, "The answer is DCBA")
    assert parse_symbolic_answer(" dcba ") == "DCBA"
    assert parse_symbolic_answer("DC BA") is None


def test_symbolic_dataset_is_deterministic_balanced_and_roundtrips(tmp_path) -> None:
    first = build_static_symbolic_dataset(seed=4, size=24, lengths=(4, 8))
    second = build_static_symbolic_dataset(seed=4, size=24, lengths=(4, 8))
    assert first == second

    counts = Counter((task.operation, task.length) for task in first)
    assert max(counts.values()) - min(counts.values()) <= 1

    path = write_symbolic_jsonl(first, tmp_path / "symbolic.jsonl")
    assert read_symbolic_jsonl(path) == first
