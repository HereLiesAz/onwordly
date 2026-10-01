from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.experiments.suite import run_suite
from onwordly.models.base import TrainStepMetrics


class TinyLearningAdapter:
    def __init__(self) -> None:
        self.answers: dict[str, str] = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "0")

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt, target
        return 5

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.answers[prompt] = target
        return TrainStepMetrics(loss=0.5, tokens=5)


def _manifest() -> ArithmeticExperimentManifest:
    return ArithmeticExperimentManifest(
        model_name="fake",
        token_budget=20,
        static_dataset_size=12,
        evaluation_size=12,
        generalization_size=9,
        checkpoint_evaluation_size=6,
        checkpoint_interval_tokens=10,
        dataset_seed=1,
        evaluation_seed=2,
        training_seed=3,
        learning_rate=2e-5,
        max_new_tokens=4,
        digit_levels=(1,),
        out_of_range_digit_levels=(2,),
        operations=("add", "subtract", "multiply"),
        withheld_prompt_styles=("question", "words", "expression"),
        variants_per_failure=2,
        holdout_modulus=5,
    )


def test_suite_aggregates_repeated_runs(tmp_path) -> None:
    result = run_suite(
        _manifest(),
        seeds=(3, 4),
        output_dir=tmp_path,
        create_adapter_for_seed=lambda seed: TinyLearningAdapter(),
    )

    assert result["seeds"] == [3, 4]
    assert result["aggregate"]["runs"] == 2
    assert set(result["aggregate"]["regimes"]) == {
        "static",
        "adaptive",
        "error-focused",
    }
    assert (tmp_path / "aggregate.json").exists()
    assert (tmp_path / "checkpoints.csv").exists()
    assert (tmp_path / "seed-3" / "summary.json").exists()
    assert (tmp_path / "seed-4" / "summary.json").exists()
    for regime in result["aggregate"]["regimes"].values():
        assert "prompt_transfer_only" in regime["accuracy"]
