"""Held-out evaluation of the verdict game (Experiment 009)."""
from __future__ import annotations

from random import Random
from typing import Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.tasks.corrective import synthetic_wrong_answer
from onwordly.tasks.verdict import make_verdict_task, parse_verdict
from onwordly.verifiers.arithmetic import parse_integer_answer, verify_arithmetic_answer


def evaluate_verdict(
    adapter: ModelAdapter,
    heldout: Sequence[ArithmeticTask],
    *,
    seed: int,
) -> dict[str, object]:
    """Exact measures on held-out problems.

    - right_shown: proposal correct; ``right`` is the only accepted reply.
    - wrong_shown: proposal a synthetic near miss; ``wrong: <answer>`` required.
      ``judgement`` counts the verdict alone (said wrong), ``repair`` also the number.
    - self_check: answer, then judge your own answer; the final answer is the
      first one if judged right, else the repair. Fixed/broken vs pass 1.
    Always ``right`` scores 50% on the balanced pair; always ``wrong`` without
    the right number scores 0% on repair.
    """
    if not heldout:
        raise ValueError("verdict evaluation needs held-out tasks")
    rng = Random(seed)
    right = judged_wrong = repaired = 0
    for task in heldout:
        right += parse_verdict(adapter.generate(make_verdict_task(task, task.answer).prompt)) == (True, None)
        verdict = parse_verdict(adapter.generate(make_verdict_task(task, synthetic_wrong_answer(task, rng)).prompt))
        if verdict is not None and verdict[0] is False:
            judged_wrong += 1
            repaired += verdict[1] == task.answer
    first = final = fixed = broken = second_calls = 0
    for task in heldout:
        response = adapter.generate(task.prompt)
        first_ok = verify_arithmetic_answer(task, response)
        parsed = parse_integer_answer(response)
        final_ok = first_ok
        if parsed is not None:
            second_calls += 1
            verdict = parse_verdict(adapter.generate(make_verdict_task(task, parsed).prompt))
            if verdict is not None:
                final_value = parsed if verdict[0] else verdict[1]
                final_ok = final_value == task.answer
        first += first_ok
        final += final_ok
        fixed += (not first_ok) and final_ok
        broken += first_ok and not final_ok
    n = len(heldout)
    return {
        "examples": n,
        "right_shown_accuracy": right / n,
        "wrong_shown_judgement_accuracy": judged_wrong / n,
        "wrong_shown_repair_accuracy": repaired / n,
        "balanced_verdict_accuracy": (right + judged_wrong) / (2 * n),
        "self_check": {
            "first_pass_accuracy": first / n,
            "final_accuracy": final / n,
            "fixed": fixed,
            "broken": broken,
            "second_pass_calls": second_calls,
        },
        "generation_calls": 3 * n + second_calls,
    }
