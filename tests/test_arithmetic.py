from random import Random

from onwordly.tasks.arithmetic import generate_arithmetic_task
from onwordly.verifiers.arithmetic import parse_integer_answer, verify_arithmetic_answer


def test_generated_answers_are_correct() -> None:
    rng = Random(7)
    for operation in ("add", "subtract", "multiply"):
        for digits in (1, 2, 3):
            for _ in range(20):
                task = generate_arithmetic_task(rng, operation, digits)
                assert verify_arithmetic_answer(task, str(task.answer))


def test_subtraction_never_requires_negative_answer() -> None:
    rng = Random(11)
    for _ in range(100):
        task = generate_arithmetic_task(rng, "subtract", 3)
        assert task.answer >= 0


def test_verifier_is_strict() -> None:
    rng = Random(13)
    task = generate_arithmetic_task(rng, "add", 2)

    assert verify_arithmetic_answer(task, str(task.answer))
    assert not verify_arithmetic_answer(task, f"The answer is {task.answer}.")
    assert parse_integer_answer("  -12  ") == -12
    assert parse_integer_answer("12.0") is None
