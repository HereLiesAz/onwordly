import json

from onwordly.experiments.arithmetic import run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest
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


def test_experiment_serializes_all_three_regimes(tmp_path) -> None:
    manifest = ArithmeticExperimentManifest(
        model_name="fake",
        token_budget=25,
        static_dataset_size=9,
        evaluation_size=9,
        dataset_seed=1,
        evaluation_seed=2,
        training_seed=3,
        learning_rate=2e-5,
        max_new_tokens=4,
        digit_levels=(1,),
        operations=("add", "subtract", "multiply"),
        variants_per_failure=2,
    )

    result = run_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinyLearningAdapter,
    )

    assert set(result["regimes"]) == {"static", "adaptive", "error-focused"}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "static-train.jsonl").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["manifest"]["token_budget"] == 25
