from dataclasses import dataclass

from onwordly.models.base import TrainStepMetrics
from onwordly.tasks.arithmetic import make_arithmetic_task
from onwordly.tasks.symbolic import make_symbolic_task
from onwordly.verifiers.symbolic import verify_symbolic_answer
from onwordly.training.harness import run_equal_token_training
from onwordly.training.sources import StaticArithmeticSource


@dataclass
class FakeAdapter:
    tokens_per_example: int = 7
    train_calls: int = 0

    def generate(self, prompt: str) -> str:
        del prompt
        return "wrong"

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt, target
        return self.tokens_per_example

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        del prompt, target
        self.train_calls += 1
        return TrainStepMetrics(loss=1.0 / self.train_calls, tokens=self.tokens_per_example)

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        del temperature
        return [self.generate(prompt) for _ in range(n)]

    def train_weighted(self, prompt: str, completion: str, weight: float) -> TrainStepMetrics:
        del weight
        return self.train_example(prompt, completion)


def test_equal_token_harness_never_overshoots() -> None:
    source = StaticArithmeticSource((make_arithmetic_task(2, 3, "add"),))
    adapter = FakeAdapter(tokens_per_example=7)

    result = run_equal_token_training(
        regime="test",
        adapter=adapter,
        source=source,
        token_budget=20,
        seed=1,
    )

    assert result.training_tokens == 14
    assert result.examples_trained == 2
    assert result.generation_calls == 2
    assert result.verifier_calls == 2
    assert result.training_tokens <= result.token_budget
    assert result.mean_training_tokens_per_example == 7
    assert result.token_budget_utilization == 14 / 20
    assert result.unused_token_budget == 6
    assert result.unique_examples == 1
    assert result.repeated_examples == 1
    assert result.generated_characters == 2 * len("wrong")
    payload = result.to_dict()
    for key in ("generation_seconds", "verifier_seconds", "training_core_seconds"):
        assert payload[key] >= 0


def test_checkpoints_include_baseline_and_final_state() -> None:
    source = StaticArithmeticSource((make_arithmetic_task(2, 3, "add"),))
    adapter = FakeAdapter(tokens_per_example=7)
    calls: list[tuple[int, int]] = []

    def checkpoint(scheduled: int, actual: int) -> dict[str, object]:
        calls.append((scheduled, actual))
        return {"evaluation": {"accuracy": actual / 20}}

    result = run_equal_token_training(
        regime="test",
        adapter=adapter,
        source=source,
        token_budget=20,
        seed=1,
        checkpoint_interval_tokens=10,
        checkpoint_callback=checkpoint,
    )

    assert calls == [(0, 0), (10, 14)]
    assert len(result.checkpoints) == 2
    assert result.checkpoints[0]["actual_tokens"] == 0
    assert result.checkpoints[-1]["actual_tokens"] == 14


class SynchronizedAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1


def test_harness_synchronizes_optional_accelerator_timing() -> None:
    adapter = SynchronizedAdapter()
    source = StaticArithmeticSource(
        [make_arithmetic_task(2, 3, "add")]
    )
    result = run_equal_token_training(
        regime="sync",
        adapter=adapter,
        source=source,
        token_budget=10,
        seed=1,
    )
    assert result.examples_trained > 0
    assert adapter.sync_calls >= result.examples_trained * 2


def test_harness_accepts_non_arithmetic_trainable_tasks() -> None:
    adapter = FakeAdapter()
    task = make_symbolic_task("ABCD", "reverse")

    class Source:
        def next_task(self, rng):
            del rng
            return task

        def observe(self, task, correct):
            del task, correct

    result = run_equal_token_training(
        regime="symbolic",
        adapter=adapter,
        source=Source(),
        token_budget=10,
        seed=1,
        verifier=verify_symbolic_answer,
    )
    assert result.examples_trained > 0
    assert "reverse:4" in result.bucket_stats
