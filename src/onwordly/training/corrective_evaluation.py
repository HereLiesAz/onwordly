"""Held-out evaluation of the corrective arithmetic language game (Experiment 008)."""
from __future__ import annotations

from random import Random
from typing import Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.tasks.corrective import make_corrective_task, synthetic_wrong_answer
from onwordly.verifiers.arithmetic import parse_integer_answer, verify_arithmetic_answer


def evaluate_corrective(
    adapter: ModelAdapter,
    heldout: Sequence[ArithmeticTask],
    *,
    seed: int,
) -> dict[str, object]:
    """Three exact measures on held-out problems.

    - correction: shown a synthetic wrong answer, return the right one;
    - confirmation: shown the right answer, return it unchanged;
    - self_correction: answer, then see your own answer in a corrective prompt.
      Unparseable first answers keep their first-pass result in pass 2.
    """
    if not heldout:
        raise ValueError("corrective evaluation needs held-out tasks")
    rng = Random(seed)
    correction = sum(
        verify_arithmetic_answer(task, adapter.generate(make_corrective_task(task, synthetic_wrong_answer(task, rng)).prompt))
        for task in heldout
    )
    confirmation = sum(
        verify_arithmetic_answer(task, adapter.generate(make_corrective_task(task, task.answer).prompt))
        for task in heldout
    )
    first = second = fixed = broken = second_pass_calls = 0
    for task in heldout:
        response = adapter.generate(task.prompt)
        first_ok = verify_arithmetic_answer(task, response)
        parsed = parse_integer_answer(response)
        if parsed is None:
            second_ok = first_ok
        else:
            second_pass_calls += 1
            second_ok = verify_arithmetic_answer(
                task, adapter.generate(make_corrective_task(task, parsed).prompt)
            )
        first += first_ok
        second += second_ok
        fixed += (not first_ok) and second_ok
        broken += first_ok and not second_ok
    n = len(heldout)
    return {
        "examples": n,
        "correction_accuracy": correction / n,
        "confirmation_accuracy": confirmation / n,
        "self_correction": {
            "first_pass_accuracy": first / n,
            "second_pass_accuracy": second / n,
            "fixed": fixed,
            "broken": broken,
            "second_pass_calls": second_pass_calls,
        },
        "generation_calls": 3 * n + second_pass_calls,
    }
