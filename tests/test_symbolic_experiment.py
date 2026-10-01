from onwordly.experiments.symbolic import run_symbolic_experiment
from onwordly.experiments.symbolic_manifest import SymbolicExperimentManifest
from onwordly.models.base import TrainStepMetrics


class TinySymbolicAdapter:
    def __init__(self) -> None:
        self.answers: dict[str, str] = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "A")

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt, target
        return 5

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.answers[prompt] = target
        return TrainStepMetrics(loss=0.5, tokens=5)


def test_symbolic_experiment_runs_equal_token_regimes(tmp_path) -> None:
    manifest = SymbolicExperimentManifest(
        model_name="fake",
        token_budget=20,
        static_dataset_size=12,
        evaluation_size=8,
        checkpoint_evaluation_size=4,
        checkpoint_interval_tokens=10,
        dataset_seed=1,
        evaluation_seed=2,
        training_seed=3,
        learning_rate=2e-5,
        max_new_tokens=8,
        lengths=(4,),
        out_of_range_lengths=(6,),
        operations=("reverse", "sort"),
        alphabet=("A", "B", "C"),
        variants_per_failure=2,
        holdout_modulus=5,
    )
    result = run_symbolic_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinySymbolicAdapter,
    )
    assert set(result["regimes"]) == {"static", "adaptive", "error-focused"}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "evaluation-longer-sequences.jsonl").exists()
    for regime in result["regimes"].values():
        assert set(regime["evaluation"]) == {"heldout", "longer_sequences"}
        assert regime["training"]["training_tokens"] <= manifest.token_budget
