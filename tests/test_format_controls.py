"""Lenient diagnostic score and shared format warm-up."""
import pytest

from onwordly.tasks.arithmetic import make_arithmetic_task
from onwordly.training.evaluation import evaluate_arithmetic, lenient_match
from onwordly.training.harness import run_equal_token_training
from onwordly.training.sources import StaticArithmeticSource
from onwordly.models.base import TrainStepMetrics


class FakeAdapter:
    def __init__(self, tokens_per_example: int = 7) -> None:
        self.tokens_per_example = tokens_per_example

    def generate(self, prompt: str) -> str:
        del prompt
        return "wrong"

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt, target
        return self.tokens_per_example

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        del prompt, target
        return TrainStepMetrics(loss=1.0, tokens=self.tokens_per_example)


def test_lenient_match_finds_standalone_answer_only() -> None:
    assert lenient_match("The answer is 2.", 2)
    assert lenient_match("result: -5", -5)
    assert not lenient_match("The answer is 12.", 2)
    assert not lenient_match("x-5", 5)
    assert lenient_match("It is TRUE", "true")
    assert not lenient_match("anything", "")


def test_evaluation_reports_lenient_separately() -> None:
    class Chatty:
        def generate(self, prompt: str) -> str:
            del prompt
            return "The answer is 5."

    result = evaluate_arithmetic(Chatty(), [make_arithmetic_task(2, 3, "add")])
    assert result.accuracy == 0.0
    assert result.lenient_accuracy == 1.0


def test_warmup_counts_in_budget_and_skips_generation() -> None:
    warmup = tuple(make_arithmetic_task(1, value, "add") for value in range(5))
    source = StaticArithmeticSource((make_arithmetic_task(2, 3, "add"),))
    result = run_equal_token_training(
        regime="test",
        adapter=FakeAdapter(tokens_per_example=7),
        source=source,
        token_budget=35,
        seed=1,
        warmup_tasks=warmup,
        warmup_token_budget=20,
    )
    assert result.warmup_examples == 2
    assert result.warmup_tokens == 14
    assert result.training_tokens == 35
    assert result.examples_trained == 5
    assert result.generation_calls == 3
    assert result.unique_examples == 3


def test_warmup_budget_validation() -> None:
    source = StaticArithmeticSource((make_arithmetic_task(2, 3, "add"),))
    with pytest.raises(ValueError):
        run_equal_token_training(
            regime="test",
            adapter=FakeAdapter(),
            source=source,
            token_budget=10,
            seed=1,
            warmup_token_budget=5,
        )
