from dataclasses import dataclass

from onwordly.models.base import TrainStepMetrics
from onwordly.tasks.arithmetic import make_arithmetic_task
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
