from __future__ import annotations

from typing import Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.symbolic import SymbolicTask
from onwordly.training.evaluation import EvaluationResult
from onwordly.verifiers.symbolic import verify_symbolic_answer


def evaluate_symbolic(
    adapter: ModelAdapter,
    tasks: Sequence[SymbolicTask],
) -> EvaluationResult:
    if not tasks:
        raise ValueError("evaluation set cannot be empty")

    correct = 0
    buckets: dict[str, list[int]] = {}
    for task in tasks:
        response = adapter.generate(task.prompt)
        is_correct = verify_symbolic_answer(task, response)
        correct += int(is_correct)
        bucket = buckets.setdefault(task.bucket_key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(is_correct)

    by_bucket = {
        key: {
            "examples": values[0],
            "correct": values[1],
            "accuracy": values[1] / values[0],
        }
        for key, values in sorted(buckets.items())
    }
    return EvaluationResult(
        examples=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks),
        by_bucket=by_bucket,
        by_prompt_style={
            "canonical": {
                "examples": len(tasks),
                "correct": correct,
                "accuracy": correct / len(tasks),
            }
        },
    )
