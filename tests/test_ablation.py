from onwordly.experiments.arithmetic import ABLATION_REGIMES, run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import TrainStepMetrics


class TinyAdapter:
    def generate(self, prompt: str) -> str:
        del prompt
        return "0"

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt, target
        return 5

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        del prompt, target
        return TrainStepMetrics(loss=0.5, tokens=5)


def test_ablation_runs_all_component_regimes(tmp_path) -> None:
    manifest = ArithmeticExperimentManifest(
        model_name="fake",
        token_budget=10,
        static_dataset_size=10,
        evaluation_size=9,
        generalization_size=6,
        checkpoint_evaluation_size=3,
        checkpoint_interval_tokens=5,
        dataset_seed=1,
        evaluation_seed=2,
        training_seed=3,
        learning_rate=2e-5,
        max_new_tokens=4,
        digit_levels=(1,),
        out_of_range_digit_levels=(2,),
        operations=("add",),
        withheld_prompt_styles=("question",),
        variants_per_failure=2,
        holdout_modulus=5,
    )
    result = run_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinyAdapter,
        regimes=ABLATION_REGIMES,
    )
    assert tuple(result["regime_order"]) == ABLATION_REGIMES
    assert set(result["regimes"]) == set(ABLATION_REGIMES)
