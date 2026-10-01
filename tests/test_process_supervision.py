from onwordly.experiments.process_supervision import run_process_supervision_experiment
from onwordly.experiments.program_manifest import ProgramExperimentManifest
from onwordly.models.base import TrainStepMetrics
from onwordly.tasks.program_execution import (
    Instruction,
    make_program_supervision_task,
)
from onwordly.verifiers.program_execution import (
    parse_supervised_program_final,
    verify_supervised_program_final,
    verify_supervised_program_trace,
)


class TinyAdapter:
    def __init__(self) -> None:
        self.answers = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "FINAL=0")

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt
        return 5 + len(target.split(","))

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.answers[prompt] = target
        return TrainStepMetrics(loss=0.1, tokens=self.count_training_tokens(prompt, target))


def test_common_prompt_differs_only_in_supervision_target() -> None:
    program = (Instruction("SET", 2), Instruction("ADD", 3), Instruction("NEG"))
    outcome = make_program_supervision_task(program, supervision="outcome")
    trace = make_program_supervision_task(program, supervision="trace")
    assert outcome.prompt == trace.prompt
    assert outcome.target == "FINAL=-5"
    assert trace.target == "TRACE=2,5,-5;FINAL=-5"
    assert verify_supervised_program_final(trace, trace.target)
    assert verify_supervised_program_trace(trace, trace.target)
    assert parse_supervised_program_final("TRACE=2,5,-5;FINAL=-5") == -5


def test_process_supervision_ablation_uses_equal_token_ceiling(tmp_path) -> None:
    manifest = ProgramExperimentManifest(
        model_name="fake", token_budget=30, static_dataset_size=12,
        evaluation_size=8, checkpoint_evaluation_size=4, checkpoint_interval_tokens=10,
        dataset_seed=1, evaluation_seed=2, training_seed=3, learning_rate=2e-5,
        max_new_tokens=8, lengths=(3,), out_of_range_lengths=(5,),
        operations=("SET","ADD","SUB","MUL","NEG"),
        withheld_transition=("MUL","NEG"),
        argument_min=-3, argument_max=3,
        variants_per_failure=2, holdout_modulus=5,
    )
    result = run_process_supervision_experiment(
        manifest,
        output_dir=tmp_path,
        create_adapter=TinyAdapter,
    )
    assert set(result["regimes"]) == {"outcome-only", "trace-supervised"}
    for regime in result["regimes"].values():
        assert regime["training"]["training_tokens"] <= manifest.token_budget
        assert regime["training"]["mean_training_tokens_per_example"] is not None
        assert regime["training_exposure"]["program_pool_size"] == manifest.static_dataset_size
        assert regime["training_exposure"]["examples_trained"] == regime["training"]["examples_trained"]
        assert set(regime["evaluation"]) == {"final_answer", "exact_trace"}

    comparison = result["comparison"]
    assert comparison["training_token_budget_equal"] is True
    assert comparison["trace_to_outcome_first_pass_token_ratio"] > 1
    assert comparison["trace_examples_trained"] < comparison["outcome_examples_trained"]
