from collections import Counter
from random import Random

from onwordly.datasets.program_execution import (
    build_program_transition_dataset,
    build_static_program_dataset,
)
from onwordly.tasks.program_execution import (
    Instruction,
    execute_program,
    execute_program_trace,
    generate_program_task,
    make_program_task,
    make_program_trace_task,
    program_contains_transition,
    program_partition,
)
from onwordly.verifiers.program_execution import (
    parse_program_trace,
    verify_program_answer,
    verify_program_trace,
)


def test_program_execution_is_exact() -> None:
    program = (
        Instruction("SET", 3),
        Instruction("ADD", 4),
        Instruction("MUL", 2),
        Instruction("NEG"),
        Instruction("SUB", 1),
    )
    assert execute_program(program) == -15


def test_generated_programs_verify() -> None:
    rng = Random(14)
    for length in (3, 5, 8):
        task = generate_program_task(rng, length)
        assert verify_program_answer(task, str(task.answer))
        assert not verify_program_answer(task, f"result={task.answer}")


def test_program_partition_separates_datasets() -> None:
    train = build_static_program_dataset(
        seed=1,
        size=40,
        lengths=(5,),
        partition="train",
    )
    evaluation = build_static_program_dataset(
        seed=2,
        size=40,
        lengths=(5,),
        partition="eval",
    )
    assert all(program_partition(task) == "train" for task in train)
    assert all(program_partition(task) == "eval" for task in evaluation)


def test_program_dataset_balances_lengths() -> None:
    tasks = build_static_program_dataset(seed=3, size=30, lengths=(3, 5, 8))
    counts = Counter(task.length for task in tasks)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_make_program_task_renders_prompt() -> None:
    task = make_program_task((Instruction("SET", 2), Instruction("MUL", 3)))
    assert task.answer == 6
    assert "SET 2; MUL 3" in task.prompt


def test_program_trace_is_exact_and_step_aligned() -> None:
    program = (
        Instruction("SET", 3),
        Instruction("ADD", 4),
        Instruction("MUL", 2),
        Instruction("NEG"),
        Instruction("SUB", 1),
    )
    assert execute_program_trace(program) == (3, 7, 14, -14, -15)
    task = make_program_trace_task(program)
    assert task.answer == "3,7,14,-14,-15"
    assert verify_program_trace(task, task.answer)


def test_program_trace_verifier_rejects_explanation_and_missing_steps() -> None:
    task = make_program_trace_task(
        (Instruction("SET", 2), Instruction("ADD", 3), Instruction("NEG"))
    )
    assert parse_program_trace("2,5,-5") == (2, 5, -5)
    assert parse_program_trace("2, 5,-5") is None
    assert not verify_program_trace(task, "5,-5")
    assert not verify_program_trace(task, "states: 2,5,-5")


def test_withheld_transition_dataset_requires_requested_pair() -> None:
    tasks = build_program_transition_dataset(
        seed=8,
        size=20,
        transition=("MUL", "NEG"),
        lengths=(5,),
        partition="eval",
    )
    assert all(
        program_contains_transition(task, ("MUL", "NEG"))
        for task in tasks
    )


def test_static_program_dataset_can_exclude_transition() -> None:
    tasks = build_static_program_dataset(
        seed=9,
        size=40,
        lengths=(5,),
        forbidden_transitions=(("MUL", "NEG"),),
    )
    assert all(
        not program_contains_transition(task, ("MUL", "NEG"))
        for task in tasks
    )
