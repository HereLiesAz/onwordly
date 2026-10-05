from onwordly.experiments.logic_manifest import LogicExperimentManifest
from onwordly.experiments.logic_suite import run_logic_suite
from onwordly.models.base import TrainStepMetrics


class TinyAdapter:
    def __init__(self) -> None:
        self.answers = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "false")

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt, target
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


def test_logic_suite_aggregates_composition_transfer(tmp_path) -> None:
    manifest = LogicExperimentManifest(
        model_name="fake",
        token_budget=10,
        static_dataset_size=12,
        evaluation_size=6,
        checkpoint_evaluation_size=3,
        checkpoint_interval_tokens=5,
        dataset_seed=1,
        evaluation_seed=2,
        training_seed=3,
        learning_rate=2e-5,
        max_new_tokens=4,
        depths=(1, 2),
        out_of_range_depths=(3,),
        variables=("A", "B", "C"),
        withheld_composition=("xor", "not"),
        composition_evaluation_size=4,
        variants_per_failure=2,
        holdout_modulus=5,
    )
    result = run_logic_suite(
        manifest,
        seeds=(3, 4),
        output_dir=tmp_path,
        create_adapter_for_seed=lambda seed: TinyAdapter(),
    )
    assert result["seeds"] == [3, 4]
    assert result["aggregate"]["runs"] == 2
    assert (tmp_path / "aggregate.json").exists()
    for regime in result["aggregate"]["regimes"].values():
        assert "withheld_composition" in regime["accuracy"]
