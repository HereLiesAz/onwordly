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

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        del temperature
        return [self.generate(prompt) for _ in range(n)]

    def train_weighted(self, prompt: str, completion: str, weight: float) -> TrainStepMetrics:
        del weight
        return self.train_example(prompt, completion)

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


def _resume_manifest(**overrides) -> ArithmeticExperimentManifest:
    values = dict(
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
    values.update(overrides)
    return ArithmeticExperimentManifest(**values)


def test_experiment_resumes_finished_regimes(tmp_path) -> None:
    import pytest

    first = run_experiment(
        _resume_manifest(), output_dir=tmp_path, create_adapter=TinyLearningAdapter
    )
    (tmp_path / "adaptive.json").unlink()  # simulate an interruption before this regime
    created: list[TinyLearningAdapter] = []

    def counting_factory() -> TinyLearningAdapter:
        adapter = TinyLearningAdapter()
        created.append(adapter)
        return adapter

    second = run_experiment(
        _resume_manifest(), output_dir=tmp_path, create_adapter=counting_factory
    )
    assert len(created) == 1  # only the missing regime re-ran
    assert second["regimes"]["static"] == json.loads(json.dumps(first["regimes"]["static"]))
    assert not list(tmp_path.glob("*.partial"))

    with pytest.raises(RuntimeError, match="different manifest"):
        run_experiment(
            _resume_manifest(learning_rate=1e-4),
            output_dir=tmp_path,
            create_adapter=TinyLearningAdapter,
        )


def test_resume_rejects_changed_code_before_touching_datasets(tmp_path, monkeypatch) -> None:
    import pytest

    import onwordly.experiments.arithmetic as arithmetic

    run_experiment(_resume_manifest(), output_dir=tmp_path, create_adapter=TinyLearningAdapter)
    frozen = (tmp_path / "static-train.jsonl").read_bytes()

    monkeypatch.setattr(arithmetic, "_code_digest", lambda: "different-code")
    with pytest.raises(RuntimeError, match="code version"):
        run_experiment(_resume_manifest(), output_dir=tmp_path, create_adapter=TinyLearningAdapter)

    monkeypatch.undo()
    with pytest.raises(RuntimeError):
        run_experiment(
            _resume_manifest(dataset_seed=99),
            output_dir=tmp_path,
            create_adapter=TinyLearningAdapter,
        )
    assert (tmp_path / "static-train.jsonl").read_bytes() == frozen


def test_corrective_regimes_run_and_report(tmp_path) -> None:
    from onwordly.reporting.arithmetic import render_result

    manifest = _resume_manifest(token_budget=60, corrective_evaluation_size=6)
    result = run_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinyLearningAdapter,
        regimes=("static", "corrective-own", "corrective-synthetic"),
    )
    own = result["regimes"]["corrective-own"]
    assert own["corrective_tasks_queued"]["correct"] > 0  # "0" is a parseable wrong answer
    assert own["corrective_evaluation"]["examples"] == 6
    assert result["regimes"]["static"]["corrective_tasks_queued"] is None
    assert "Corrective language game" in render_result(tmp_path / "summary.json")


def test_verdict_regimes_run_and_report(tmp_path) -> None:
    from onwordly.reporting.arithmetic import render_result

    manifest = _resume_manifest(token_budget=60, verdict_evaluation_size=6, verdict_rate=1.0)
    result = run_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinyLearningAdapter,
        regimes=("static", "verdict-synthetic", "verdict-mixed"),
    )
    mixed = result["regimes"]["verdict-mixed"]
    assert sum(mixed["corrective_tasks_queued"].values()) > 0
    assert mixed["verdict_evaluation"]["examples"] == 6
    assert "Verdict game" in render_result(tmp_path / "summary.json")
