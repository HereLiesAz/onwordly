"""Held-out evaluation for Experiment 011 (constrained strings), mirroring 010's
per-class verdict counts. Every check is the exact rule checker."""
from __future__ import annotations

from random import Random
from statistics import fmean
from typing import Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.constrained_strings import (
    ConstrainedStringTask,
    make_constrained_verdict_task,
    parse_string_answer,
    parse_string_verdict,
    satisfaction,
    satisfies_all,
    synthetic_violation,
)


def evaluate_constrained_generation(adapter: ModelAdapter, tasks: Sequence[ConstrainedStringTask]) -> dict[str, object]:
    """Plain generation: fraction fully valid and mean rule satisfaction."""
    if not tasks:
        raise ValueError("evaluation set cannot be empty")
    valid: list[bool] = []
    scores: list[float] = []
    for task in tasks:
        candidate = parse_string_answer(adapter.generate(task.prompt))
        valid.append(candidate is not None and satisfies_all(task, candidate))
        scores.append(satisfaction(task, candidate))
    return {
        "examples": len(tasks),
        "correct": sum(valid),
        "accuracy": sum(valid) / len(tasks),
        "mean_satisfaction": fmean(scores),
    }


def evaluate_constrained_verdict(
    adapter: ModelAdapter, heldout: Sequence[ConstrainedStringTask], *, seed: int
) -> dict[str, object]:
    """Shown-right / shown-wrong judgements and the self-check, with per-class counts.

    - right_shown: the witness is shown; ``right`` is the only accepted reply.
    - wrong_shown: a synthetic violation is shown; judgement counts ``wrong``,
      repair also requires the stated string to satisfy every rule.
    - self_check: produce, then judge your own string; the final string is the
      first one if judged right, else the repair. ``mean_final_satisfaction``
      is mean s(final) (0 when there is no final string).
    """
    if not heldout:
        raise ValueError("verdict evaluation needs held-out tasks")
    rng = Random(seed)
    right = judged_wrong = repaired = 0
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
        verdict = parse_string_verdict(adapter.generate(make_constrained_verdict_task(task, task.witness).prompt))
        if verdict is None:
            classes["right_unparseable"] += 1
        elif verdict[0]:
            right += 1
            classes["right_kept"] += 1
        else:
            classes["right_rejected"] += 1
        shown = synthetic_violation(task, rng)
        verdict = parse_string_verdict(adapter.generate(make_constrained_verdict_task(task, shown).prompt))
        if verdict is None:
            classes["wrong_unparseable"] += 1
        elif verdict[0]:
            classes["wrong_accepted"] += 1
        else:
            judged_wrong += 1
            ok = satisfies_all(task, str(verdict[1]))
            repaired += ok
            classes["wrong_caught_repaired" if ok else "wrong_caught_misrepaired"] += 1

    first = final = fixed = broken = second_calls = 0
    first_scores: list[float] = []
    final_scores: list[float] = []
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
        parsed = parse_string_answer(adapter.generate(task.prompt))
        first_ok = parsed is not None and satisfies_all(task, parsed)
        first_scores.append(satisfaction(task, parsed))
        final_ok = first_ok
        final_string: str | None = parsed
        if parsed is None:
            own["unparseable_first_pass"] += 1
        else:
            second_calls += 1
            verdict = parse_string_verdict(adapter.generate(make_constrained_verdict_task(task, parsed).prompt))
            prefix = "own_right_" if first_ok else "own_wrong_"
            if verdict is None:
                # No final string; as in 010, the first pass stands for accuracy.
                own[prefix + "unparseable_verdict"] += 1
            else:
                final_string = parsed if verdict[0] else verdict[1]
                final_ok = satisfies_all(task, str(final_string))
                if first_ok:
                    own["own_right_kept" if verdict[0] else "own_right_rejected"] += 1
                elif verdict[0]:
                    own["own_wrong_accepted"] += 1
                else:
                    own["own_wrong_caught_repaired" if final_ok else "own_wrong_caught_misrepaired"] += 1
        final_scores.append(satisfaction(task, final_string))
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
            "mean_first_satisfaction": fmean(first_scores),
            "mean_final_satisfaction": fmean(final_scores),
            "fixed": fixed,
            "broken": broken,
            "second_pass_calls": second_calls,
            "class_counts": own,
        },
        "generation_calls": 3 * n + second_calls,
        "class_counts": classes,
    }
