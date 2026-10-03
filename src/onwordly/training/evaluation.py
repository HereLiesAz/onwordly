from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.verifiers.arithmetic import verify_arithmetic_answer


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    examples: int
    correct: int
    accuracy: float
    by_bucket: dict[str, dict[str, float | int]]
    by_prompt_style: dict[str, dict[str, float | int]]
    # Diagnostic only, never used for training: the expected answer appears as a
    # standalone token anywhere in the response. An upper bound on capability
    # that ignores output format; restated prompts can inflate it.
    lenient_correct: int = 0
    lenient_accuracy: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def lenient_match(response: str, answer: object) -> bool:
    """Answer present as a standalone token, ignoring case and surrounding text."""
    expected = str(answer)
    if not expected:
        return False
    pattern = rf"(?<![A-Za-z0-9-]){re.escape(expected)}(?![A-Za-z0-9])"
    return re.search(pattern, response, flags=re.IGNORECASE) is not None


def evaluate_arithmetic(
    adapter: ModelAdapter,
    tasks: Sequence[ArithmeticTask],
) -> EvaluationResult:
    if not tasks:
        raise ValueError("evaluation set cannot be empty")

    correct = 0
    lenient = 0
    buckets: dict[str, list[int]] = {}
    styles: dict[str, list[int]] = {}

    for task in tasks:
        response = adapter.generate(task.prompt)
        is_correct = verify_arithmetic_answer(task, response)
        correct += int(is_correct)
        lenient += int(lenient_match(response, task.answer))

        bucket_key = f"{task.operation}:{task.digits}"
        bucket = buckets.setdefault(bucket_key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(is_correct)

        style = styles.setdefault(task.prompt_style, [0, 0])
        style[0] += 1
        style[1] += int(is_correct)

    def summarize(records: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "examples": values[0],
                "correct": values[1],
                "accuracy": values[1] / values[0],
            }
            for key, values in sorted(records.items())
        }

    return EvaluationResult(
        examples=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks),
        lenient_correct=lenient,
        lenient_accuracy=lenient / len(tasks),
        by_bucket=summarize(buckets),
        by_prompt_style=summarize(styles),
    )
