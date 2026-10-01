from __future__ import annotations

from dataclasses import asdict, dataclass, field
from random import Random
from statistics import fmean
from typing import Callable

from onwordly.models.base import ModelAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.training.sources import ArithmeticTaskSource
from onwordly.verifiers.arithmetic import verify_arithmetic_answer

Verifier = Callable[[ArithmeticTask, str], bool]
CheckpointCallback = Callable[[int, int], dict[str, object]]


@dataclass(slots=True)
class BucketRunStats:
    attempts: int = 0
    correct_before_train: int = 0

    @property
    def accuracy(self) -> float:
        return 0.0 if self.attempts == 0 else self.correct_before_train / self.attempts


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    regime: str
    token_budget: int
    training_tokens: int
    examples_trained: int
    generation_calls: int
    verifier_calls: int
    correct_before_train: int
    mean_loss: float | None
    bucket_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    checkpoints: tuple[dict[str, object], ...] = ()

    @property
    def pretrain_accuracy(self) -> float:
        if self.generation_calls == 0:
            return 0.0
        return self.correct_before_train / self.generation_calls

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pretrain_accuracy"] = self.pretrain_accuracy
        return payload


def run_equal_token_training(
    *,
    regime: str,
    adapter: ModelAdapter,
    source: ArithmeticTaskSource,
    token_budget: int,
    seed: int,
    verifier: Verifier = verify_arithmetic_answer,
    checkpoint_interval_tokens: int | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TrainingRunResult:
    """Train without ever exceeding the requested model-token budget.

    Every regime performs the same pre-update generation and verification step.
    Adaptive sources may use that observation to choose future tasks. Optional
    checkpoint evaluation is read-only: its results are never fed to the source.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be at least 1")
    if checkpoint_callback is not None and (
        checkpoint_interval_tokens is None or checkpoint_interval_tokens < 1
    ):
        raise ValueError("checkpoint callback requires a positive checkpoint interval")

    rng = Random(seed)
    training_tokens = 0
    examples_trained = 0
    generation_calls = 0
    verifier_calls = 0
    correct_before_train = 0
    losses: list[float] = []
    buckets: dict[str, BucketRunStats] = {}
    checkpoints: list[dict[str, object]] = []

    next_checkpoint = checkpoint_interval_tokens
    if checkpoint_callback is not None:
        baseline = checkpoint_callback(0, 0)
        checkpoints.append(
            {
                "scheduled_tokens": 0,
                "actual_tokens": 0,
                **baseline,
            }
        )

    while True:
        task = source.next_task(rng)
        target = str(task.answer)
        planned_tokens = adapter.count_training_tokens(task.prompt, target)
        if planned_tokens < 1:
            raise RuntimeError("adapter reported a non-positive training token count")
        if training_tokens + planned_tokens > token_budget:
            break

        response = adapter.generate(task.prompt)
        generation_calls += 1
        correct = verifier(task, response)
        verifier_calls += 1
        correct_before_train += int(correct)
        source.observe(task, correct)

        bucket_key = f"{task.operation}:{task.digits}"
        bucket = buckets.setdefault(bucket_key, BucketRunStats())
        bucket.attempts += 1
        bucket.correct_before_train += int(correct)

        step = adapter.train_example(task.prompt, target)
        if step.tokens != planned_tokens:
            raise RuntimeError(
                "adapter token accounting changed between planning and training: "
                f"planned={planned_tokens}, actual={step.tokens}"
            )

        training_tokens += step.tokens
        examples_trained += 1
        losses.append(step.loss)

        if (
            checkpoint_callback is not None
            and next_checkpoint is not None
            and training_tokens >= next_checkpoint
        ):
            scheduled = next_checkpoint
            while next_checkpoint <= training_tokens:
                next_checkpoint += checkpoint_interval_tokens  # type: ignore[operator]
            metrics = checkpoint_callback(scheduled, training_tokens)
            checkpoints.append(
                {
                    "scheduled_tokens": scheduled,
                    "actual_tokens": training_tokens,
                    **metrics,
                }
            )

    if checkpoint_callback is not None:
        last_actual = int(checkpoints[-1]["actual_tokens"]) if checkpoints else -1
        if last_actual != training_tokens:
            metrics = checkpoint_callback(training_tokens, training_tokens)
            checkpoints.append(
                {
                    "scheduled_tokens": training_tokens,
                    "actual_tokens": training_tokens,
                    **metrics,
                }
            )

    bucket_payload = {
        key: {
            "attempts": value.attempts,
            "correct_before_train": value.correct_before_train,
            "accuracy": value.accuracy,
        }
        for key, value in sorted(buckets.items())
    }

    return TrainingRunResult(
        regime=regime,
        token_budget=token_budget,
        training_tokens=training_tokens,
        examples_trained=examples_trained,
        generation_calls=generation_calls,
        verifier_calls=verifier_calls,
        correct_before_train=correct_before_train,
        mean_loss=fmean(losses) if losses else None,
        bucket_stats=bucket_payload,
        checkpoints=tuple(checkpoints),
    )
