"""Held-out evaluation of the verdict game (Experiments 009–010)."""
from __future__ import annotations

from random import Random
from typing import Callable, Literal, Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.tasks.corrective import synthetic_wrong_answer
from onwordly.tasks.solve_judge import make_solve_judge_task, parse_solve_judge
from onwordly.tasks.verdict import make_verdict_task, parse_verdict
from onwordly.verifiers.arithmetic import parse_integer_answer, verify_arithmetic_answer

VerdictFormat = Literal["verdict", "solve-judge"]

# A judged reply, normalised across formats: (said_right, stated_answer).
# ``stated_answer`` is None when the reply carries no number (009 ``right``).
Judged = tuple[bool, "int | None"]


def _verdict_reader(fmt: VerdictFormat) -> tuple[Callable[[ArithmeticTask, int], object], Callable[[str], Judged | None]]:
    if fmt == "verdict":
        return make_verdict_task, parse_verdict
    if fmt == "solve-judge":

        def read(response: str) -> Judged | None:
            parsed = parse_solve_judge(response)
            return None if parsed is None else (parsed[1], parsed[0])

        return make_solve_judge_task, read
    raise ValueError(f"unknown verdict format: {fmt}")


def evaluate_verdict(
    adapter: ModelAdapter,
    heldout: Sequence[ArithmeticTask],
    *,
    seed: int,
    format: VerdictFormat = "verdict",
) -> dict[str, object]:
    """Exact measures on held-out problems.

    - right_shown: proposal correct; ``right`` is the only accepted reply.
    - wrong_shown: proposal a synthetic near miss; ``wrong: <answer>`` required.
      ``judgement`` counts the verdict alone (said wrong), ``repair`` also the number.
    - self_check: answer, then judge your own answer; the final answer is the
      first one if judged right, else the repair. Fixed/broken vs pass 1.
    Always ``right`` scores 50% on the balanced pair; always ``wrong`` without
    the right number scores 0% on repair.

    ``format="solve-judge"`` (Experiment 010) asks for ``<n>; right`` or
    ``<n>; wrong`` instead. Right-shown accuracy then counts the verdict alone
    (comparable with 009); ``right_shown_with_answer_accuracy`` also requires
    the correct ``n``. Repair = said wrong with the correct ``n``. In the
    self-check a ``right`` keeps the first answer and a ``wrong`` takes ``n``.
    The default format reproduces 009's prompts, calls and scores exactly;
    ``format`` and the per-class count blocks are added keys.
    """
    if not heldout:
        raise ValueError("verdict evaluation needs held-out tasks")
    make_move, read = _verdict_reader(format)
    rng = Random(seed)
    right = right_with_answer = judged_wrong = repaired = 0
    classes = {
        "right_kept": 0,
        "right_rejected": 0,
        "right_unparseable": 0,
        "wrong_caught_repaired": 0,
        "wrong_caught_misrepaired": 0,
        "wrong_accepted": 0,
        "wrong_unparseable": 0,
    }
    for task in heldout:
        verdict = read(adapter.generate(make_move(task, task.answer).prompt))
        if verdict is None:
            classes["right_unparseable"] += 1
        elif verdict[0]:
            right += 1
            classes["right_kept"] += 1
            right_with_answer += verdict[1] in (None, task.answer)
        else:
            classes["right_rejected"] += 1
        verdict = read(adapter.generate(make_move(task, synthetic_wrong_answer(task, rng)).prompt))
        if verdict is None:
            classes["wrong_unparseable"] += 1
        elif verdict[0]:
            classes["wrong_accepted"] += 1
        else:
            judged_wrong += 1
            fixed_number = verdict[1] == task.answer
            repaired += fixed_number
            classes["wrong_caught_repaired" if fixed_number else "wrong_caught_misrepaired"] += 1
    first = final = fixed = broken = second_calls = 0
    own = {
        "own_right_kept": 0,
        "own_right_rejected": 0,
        "own_right_unparseable_verdict": 0,
        "own_wrong_caught_repaired": 0,
        "own_wrong_caught_misrepaired": 0,
        "own_wrong_accepted": 0,
        "own_wrong_unparseable_verdict": 0,
        "unparseable_first_pass": 0,
    }
    for task in heldout:
        response = adapter.generate(task.prompt)
        first_ok = verify_arithmetic_answer(task, response)
        parsed = parse_integer_answer(response)
        final_ok = first_ok
        if parsed is None:
            own["unparseable_first_pass"] += 1
        else:
            second_calls += 1
            verdict = read(adapter.generate(make_move(task, parsed).prompt))
            prefix = "own_right_" if first_ok else "own_wrong_"
            if verdict is None:
                own[prefix + "unparseable_verdict"] += 1
            else:
                final_value = parsed if verdict[0] else verdict[1]
                final_ok = final_value == task.answer
                if first_ok:
                    own["own_right_kept" if verdict[0] else "own_right_rejected"] += 1
                elif verdict[0]:
                    own["own_wrong_accepted"] += 1
                else:
                    own["own_wrong_caught_repaired" if final_ok else "own_wrong_caught_misrepaired"] += 1
        first += first_ok
        final += final_ok
        fixed += (not first_ok) and final_ok
        broken += first_ok and not final_ok
    n = len(heldout)
    result: dict[str, object] = {
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
            "class_counts": own,
        },
        "generation_calls": 3 * n + second_calls,
        "format": format,
        "class_counts": classes,
    }
    if format == "solve-judge":
        result["right_shown_with_answer_accuracy"] = right_with_answer / n
    return result
