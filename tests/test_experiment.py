import json

from onwordly.experiments.arithmetic import run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import TrainStepMetrics


class TinyLearningAdapter:
    def __init__(self) -> None:
        self.answers: dict[str, str] = {}
        self.closed = False

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "0")

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt, target
        return 5

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.answers[prompt] = target
        return TrainStepMetrics(loss=0.5, tokens=5)

    def synchronize(self) -> None:
        pass

    def reset_peak_memory_stats(self) -> None:
        pass

    def peak_memory_bytes(self) -> int:
        return 1024

    def close(self) -> None:
        self.closed = True


def test_experiment_serializes_all_three_regimes(tmp_path) -> None:
    manifest = ArithmeticExperimentManifest(
        model_name="fake",
        token_budget=25,
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

    result = run_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinyLearningAdapter,
    )

    assert set(result["regimes"]) == {"static", "adaptive", "error-focused"}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "static-train.jsonl").exists()
    assert (tmp_path / "evaluation-heldout.jsonl").exists()
    assert (tmp_path / "evaluation-prompt-transfer-only.jsonl").exists()
    assert (tmp_path / "evaluation-withheld-prompts.jsonl").exists()
    assert (tmp_path / "evaluation-out-of-range.jsonl").exists()

    for regime in result["regimes"].values():
        assert set(regime["evaluation"]) == {
            "heldout",
            "prompt_transfer_only",
            "withheld_prompts",
            "out_of_range",
        }
        assert len(regime["training"]["checkpoints"]) >= 2
        assert set(regime["tokens_to_threshold"]) == {"0.70", "0.80", "0.90", "0.95"}
        assert regime["model"]["peak_memory_bytes"] == 1024
        assert regime["measurement_overhead"]["regime_wall_seconds"] >= 0
        assert "capability_gain_per_million_training_tokens" in regime["measurement_overhead"]

    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["manifest"]["token_budget"] == 25
    assert set(saved["datasets"]) == {
        "static_train",
        "heldout",
        "prompt_transfer_only",
        "withheld_prompts",
        "out_of_range",
    }
