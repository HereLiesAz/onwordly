from __future__ import annotations

from dataclasses import asdict, dataclass, field
from random import Random
from statistics import fmean
from time import perf_counter
from typing import Callable, Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.base import TrainableTask
from onwordly.verifiers.arithmetic import verify_arithmetic_answer

Verifier = Callable[[TrainableTask, str], bool]
CheckpointCallback = Callable[[int, int], dict[str, object]]


class TaskSource:
    def next_task(self, rng: Random) -> TrainableTask:
        raise NotImplementedError

    def observe(self, task: TrainableTask, correct: bool) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class OnPolicyConfig:
    """Opt-in on-policy update for a task (Experiment 010, ``verdict-rl``).

    A source opts a task in by returning this from ``on_policy_config(task)``.
    Instead of one SFT step on the target, the harness samples ``samples``
    completions at ``temperature``, rewards each 1/0 by the exact verifier,
    and trains each with weight ``reward - mean(rewards)`` via
    ``adapter.train_weighted`` (advantage-weighted negative log-likelihood).
    This is REINFORCE with a group-mean baseline: GRPO-style group advantages
    without ratio clipping, a KL penalty or reward-std normalisation. It is an
    established method, not an Onwordly invention. A group whose rewards are
    all equal has zero advantage everywhere and is skipped without an update.
    """

    samples: int = 4
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.samples < 2:
            raise ValueError("on-policy groups need at least two samples")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


def group_advantages(rewards: Sequence[float]) -> list[float] | None:
    """``reward - group mean`` per sample; ``None`` when every reward is equal."""
    if not rewards:
        raise ValueError("rewards must be non-empty")
    if all(reward == rewards[0] for reward in rewards):
        return None
    mean = fmean(rewards)
    return [reward - mean for reward in rewards]


def _synchronize_adapter(adapter: ModelAdapter) -> None:
    synchronize = getattr(adapter, "synchronize", None)
    if callable(synchronize):
        synchronize()


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
    # Wall-clock split so inference/verifier work is never billed as training.
    training_core_seconds: float
    generation_seconds: float = 0.0
    verifier_seconds: float = 0.0
    generated_characters: int = 0
    unique_examples: int = 0
    # Shared format warm-up, trained before the regime's source and counted in
    # training_tokens; identical for every regime.
    warmup_tokens: int = 0
    warmup_examples: int = 0
    bucket_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    checkpoints: tuple[dict[str, object], ...] = ()
    # On-policy accounting (Experiment 010); None, and omitted from to_dict,
    # for every run without on-policy tasks. Its sampling calls/seconds are
    # also included in generation_calls/generation_seconds.
    on_policy: dict[str, float | int | None] | None = None

    @property
    def pre_update_attempts(self) -> int:
        """Greedy pre-update attempts (generation calls minus on-policy samples)."""
        samples = int(self.on_policy["sample_generation_calls"] or 0) if self.on_policy else 0
        return self.generation_calls - samples

    @property
    def pretrain_accuracy(self) -> float:
        if self.pre_update_attempts == 0:
            return 0.0
        return self.correct_before_train / self.pre_update_attempts

    @property
    def mean_training_tokens_per_example(self) -> float | None:
        if self.examples_trained == 0:
            return None
        return self.training_tokens / self.examples_trained

    @property
    def token_budget_utilization(self) -> float:
        return self.training_tokens / self.token_budget

    @property
    def unused_token_budget(self) -> int:
        """Tokens left when the next task no longer fit; compare across regimes."""
        return self.token_budget - self.training_tokens

    @property
    def repeated_examples(self) -> int:
        """Examples seen more than once (static pool cycling vs fresh online data)."""
        return self.examples_trained - self.unique_examples

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.on_policy is None:
            del payload["on_policy"]
        payload["pretrain_accuracy"] = self.pretrain_accuracy
        payload["mean_training_tokens_per_example"] = self.mean_training_tokens_per_example
        payload["token_budget_utilization"] = self.token_budget_utilization
        payload["unused_token_budget"] = self.unused_token_budget
        payload["repeated_examples"] = self.repeated_examples
        return payload


def run_equal_token_training(
    *,
    regime: str,
    adapter: ModelAdapter,
    source: TaskSource,
    token_budget: int,
    seed: int,
    verifier: Verifier = verify_arithmetic_answer,
    checkpoint_interval_tokens: int | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    warmup_tasks: Sequence[TrainableTask] = (),
    warmup_token_budget: int = 0,
) -> TrainingRunResult:
    """Train without ever exceeding the requested model-token budget.

    Every regime performs the same pre-update generation and verification step.
    Adaptive sources may use that observation to choose future tasks. Optional
    checkpoint evaluation is read-only: its results are never fed to the source.
    Optional format warm-up: ``warmup_tasks`` are trained in order, without
    generation, verification or source observation, until the next one would
    exceed ``warmup_token_budget``. Those tokens count toward ``token_budget``,
    so every regime keeps the same total. Pass the same tasks to every regime.
    Timing is split: generation_seconds, verifier_seconds and training_core_seconds
    (the update step alone) are recorded separately.

    On-policy opt-in: when ``source.on_policy_config(task)`` returns an
    ``OnPolicyConfig``, that task is trained by a group of sampled completions
    (see ``OnPolicyConfig``) instead of SFT on its target. Every trained
    (prompt, completion) pair counts its tokens toward ``token_budget``; a
    group that would not fit ends training, as an SFT task would. Sampling is
    billed as generation calls/seconds and sample checks as verifier calls,
    and is also reported separately under ``on_policy``.
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
    training_core_seconds = 0.0
    generation_seconds = 0.0
    verifier_seconds = 0.0
    generated_characters = 0
    seen_examples: set[tuple[str, str]] = set()
    losses: list[float] = []
    buckets: dict[str, BucketRunStats] = {}
    checkpoints: list[dict[str, object]] = []
    on_policy_for = getattr(source, "on_policy_config", None)
    if not callable(on_policy_for):
        on_policy_for = None
    on_policy_stats: dict[str, float | int | None] | None = None
    on_policy_losses: list[float] = []

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

    if warmup_token_budget < 0 or warmup_token_budget > token_budget:
        raise ValueError("warmup_token_budget must be between 0 and token_budget")
    if warmup_token_budget and not warmup_tasks:
        raise ValueError("warmup_token_budget requires warmup_tasks")
    warmup_tokens = 0
    warmup_examples = 0
    for warmup_task in warmup_tasks:
        planned = adapter.count_training_tokens(warmup_task.prompt, warmup_task.target_text)
        if warmup_tokens + planned > warmup_token_budget:
            break
        _synchronize_adapter(adapter)
        update_started = perf_counter()
        step = adapter.train_example(warmup_task.prompt, warmup_task.target_text)
        _synchronize_adapter(adapter)
        training_core_seconds += perf_counter() - update_started
        warmup_tokens += step.tokens
        warmup_examples += 1
        losses.append(step.loss)
        seen_examples.add((warmup_task.prompt, warmup_task.target_text))
    training_tokens = warmup_tokens
    examples_trained = warmup_examples

    while True:
        task = source.next_task(rng)
        target = task.target_text
        policy = on_policy_for(task) if on_policy_for is not None else None
        if policy is None:
            planned_tokens = adapter.count_training_tokens(task.prompt, target)
            if planned_tokens < 1:
                raise RuntimeError("adapter reported a non-positive training token count")
            if training_tokens + planned_tokens > token_budget:
                break

        _synchronize_adapter(adapter)
        started = perf_counter()
        response = adapter.generate(task.prompt)
        _synchronize_adapter(adapter)
        generated = perf_counter()
        generation_seconds += generated - started
        generation_calls += 1
        generated_characters += len(response)
        correct = verifier(task, response)
        verified = perf_counter()
        verifier_seconds += verified - generated
        verifier_calls += 1
        correct_before_train += int(correct)
        observe_response = getattr(source, "observe_response", None)
        if callable(observe_response):
            # Sources that need the model's actual answer (Experiment 008).
            observe_response(task, response, correct)
        else:
            source.observe(task, correct)

        bucket_key = task.bucket_key
        bucket = buckets.setdefault(bucket_key, BucketRunStats())
        bucket.attempts += 1
        bucket.correct_before_train += int(correct)

        if policy is not None:
            if on_policy_stats is None:
                on_policy_stats = {
                    "samples_per_group": policy.samples,
                    "temperature": policy.temperature,
                    "groups": 0,
                    "groups_trained": 0,
                    "groups_skipped_equal_rewards": 0,
                    "sample_generation_calls": 0,
                    "sample_generation_seconds": 0.0,
                    "sample_verifier_calls": 0,
                    "samples_correct": 0,
                    "weighted_updates": 0,
                    "update_tokens": 0,
                }
            stats = on_policy_stats
            _synchronize_adapter(adapter)
            started = perf_counter()
            completions = list(adapter.sample(task.prompt, policy.samples, policy.temperature))
            _synchronize_adapter(adapter)
            sampled = perf_counter()
            if len(completions) != policy.samples:
                raise RuntimeError("adapter returned the wrong number of samples")
            generation_seconds += sampled - started
            generation_calls += len(completions)
            generated_characters += sum(len(completion) for completion in completions)
            rewards = [float(verifier(task, completion)) for completion in completions]
            verifier_seconds += perf_counter() - sampled
            verifier_calls += len(completions)
            stats["groups"] += 1
            stats["sample_generation_calls"] += len(completions)
            stats["sample_generation_seconds"] += sampled - started
            stats["sample_verifier_calls"] += len(completions)
            stats["samples_correct"] += int(sum(rewards))
            advantages = group_advantages(rewards)
            if advantages is None:
                stats["groups_skipped_equal_rewards"] += 1
                continue
            pairs = [
                (completion, advantage)
                for completion, advantage in zip(completions, advantages)
                if advantage != 0.0
            ]
            counts = [adapter.count_training_tokens(task.prompt, completion) for completion, _ in pairs]
            if min(counts) < 1:
                raise RuntimeError("adapter reported a non-positive training token count")
            if training_tokens + sum(counts) > token_budget:
                break
            for (completion, advantage), planned in zip(pairs, counts):
                update_started = perf_counter()
                step = adapter.train_weighted(task.prompt, completion, advantage)
                if step.tokens != planned:
                    raise RuntimeError(
                        "adapter token accounting changed between planning and training: "
                        f"planned={planned}, actual={step.tokens}"
                    )
                _synchronize_adapter(adapter)
                training_core_seconds += perf_counter() - update_started
                seen_examples.add((task.prompt, completion))
                training_tokens += step.tokens
                examples_trained += 1
                on_policy_losses.append(step.loss)
                stats["weighted_updates"] += 1
                stats["update_tokens"] += step.tokens
            stats["groups_trained"] += 1
        else:
            update_started = perf_counter()
            step = adapter.train_example(task.prompt, target)
            if step.tokens != planned_tokens:
                raise RuntimeError(
                    "adapter token accounting changed between planning and training: "
                    f"planned={planned_tokens}, actual={step.tokens}"
                )
            _synchronize_adapter(adapter)
            training_core_seconds += perf_counter() - update_started
            seen_examples.add((task.prompt, target))

            training_tokens += step.tokens
            examples_trained += 1
            losses.append(step.loss)

        if (
            checkpoint_callback is not None
            and next_checkpoint is not None
            and checkpoint_interval_tokens is not None
            and training_tokens >= next_checkpoint
        ):
            scheduled = next_checkpoint
            while next_checkpoint <= training_tokens:
                next_checkpoint += checkpoint_interval_tokens
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

    if on_policy_stats is not None:
        on_policy_stats["mean_sample_nll"] = fmean(on_policy_losses) if on_policy_losses else None

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
        training_core_seconds=training_core_seconds,
        generation_seconds=generation_seconds,
        verifier_seconds=verifier_seconds,
        generated_characters=generated_characters,
        unique_examples=len(seen_examples),
        warmup_tokens=warmup_tokens,
        warmup_examples=warmup_examples,
        bucket_stats=bucket_payload,
        checkpoints=tuple(checkpoints),
        on_policy=on_policy_stats,
    )
