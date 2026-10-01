from collections import Counter
from random import Random

from onwordly.datasets.program_execution import build_static_program_dataset
from onwordly.tasks.program_execution import (
    Instruction,
    execute_program,
    generate_program_task,
    make_program_task,
    program_partition,
)
from onwordly.verifiers.program_execution import verify_program_answer


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
