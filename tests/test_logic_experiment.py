from onwordly.experiments.formal_logic import run_logic_experiment
from onwordly.experiments.logic_manifest import LogicExperimentManifest
from onwordly.models.base import TrainStepMetrics


class TinyAdapter:
    def __init__(self) -> None:
        self.answers = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "false")

    def count_training_tokens(self, prompt: str, target: str) -> int:
        return 5

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.answers[prompt] = target
        return TrainStepMetrics(loss=0.1, tokens=5)


def test_logic_experiment_runs(tmp_path) -> None:
    manifest = LogicExperimentManifest(
        model_name="fake", token_budget=10, static_dataset_size=8,
        evaluation_size=6, checkpoint_evaluation_size=3, checkpoint_interval_tokens=5,
        dataset_seed=1, evaluation_seed=2, training_seed=3, learning_rate=2e-5,
        max_new_tokens=4, depths=(1,), out_of_range_depths=(2,),
        variables=("A","B","C"), withheld_composition=("xor","not"),
        composition_evaluation_size=4, variants_per_failure=2, holdout_modulus=5,
    )
    result = run_logic_experiment(
        manifest, output_dir=tmp_path, create_adapter=TinyAdapter
    )
    assert set(result["regimes"]) == {"static","adaptive","error-focused"}
    for regime in result["regimes"].values():
        assert set(regime["evaluation"]) == {
            "heldout",
            "deeper_formulas",
            "withheld_composition",
        }
