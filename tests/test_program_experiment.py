from onwordly.experiments.program_execution import run_program_experiment
from onwordly.experiments.program_manifest import ProgramExperimentManifest
from onwordly.models.base import TrainStepMetrics


class TinyAdapter:
    def __init__(self) -> None:
        self.answers = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "0")

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


def test_program_experiment_runs(tmp_path) -> None:
    manifest = ProgramExperimentManifest(
        model_name="fake", token_budget=10, static_dataset_size=8,
        evaluation_size=6, checkpoint_evaluation_size=3, checkpoint_interval_tokens=5,
        dataset_seed=1, evaluation_seed=2, training_seed=3, learning_rate=2e-5,
        max_new_tokens=8, lengths=(3,), out_of_range_lengths=(5,),
        operations=("SET","ADD","SUB","MUL","NEG"),
        withheld_transition=("MUL","NEG"),
        argument_min=-3, argument_max=3,
        variants_per_failure=2, holdout_modulus=5,
    )
    result = run_program_experiment(
        manifest, output_dir=tmp_path, create_adapter=TinyAdapter
    )
    assert set(result["regimes"]) == {"static","adaptive","error-focused"}
    for regime in result["regimes"].values():
        assert set(regime["evaluation"]) == {
            "heldout",
            "longer_programs",
            "withheld_transition",
        }
