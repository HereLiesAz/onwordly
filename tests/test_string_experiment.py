from onwordly.experiments.string_manipulation import run_string_experiment
from onwordly.experiments.string_manifest import StringExperimentManifest
from onwordly.models.base import TrainStepMetrics


class TinyAdapter:
    def __init__(self) -> None:
        self.answers = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "A")

    def count_training_tokens(self, prompt: str, target: str) -> int:
        return 5

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.answers[prompt] = target
        return TrainStepMetrics(loss=0.1, tokens=5)

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        del temperature
        return [self.generate(prompt) for _ in range(n)]

    def train_weighted(self, prompt: str, completion: str, weight: float) -> TrainStepMetrics:
        del weight
        return self.train_example(prompt, completion)


def test_string_experiment_runs(tmp_path) -> None:
    manifest = StringExperimentManifest(
        model_name="fake",
        token_budget=10,
        static_dataset_size=8,
        evaluation_size=6,
        composition_evaluation_size=4,
        checkpoint_evaluation_size=3,
        checkpoint_interval_tokens=5,
        dataset_seed=1,
        evaluation_seed=2,
        training_seed=3,
        learning_rate=2e-5,
        max_new_tokens=8,
        lengths=(4,),
        out_of_range_lengths=(6,),
        operations=("reverse_pairs", "duplicate_each"),
        alphabet=("A", "B", "1"),
        variants_per_failure=2,
        holdout_modulus=5,
    )
    result = run_string_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinyAdapter,
    )
    assert set(result["regimes"]) == {"static", "adaptive", "error-focused"}
    assert (tmp_path / "summary.json").exists()
    for regime in result["regimes"].values():
        assert set(regime["evaluation"]) == {
            "heldout",
            "longer_strings",
            "composition",
        }
