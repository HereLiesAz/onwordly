"""Experiment 010 (verdict repair): solve-judge move, dense/RL arms, evaluation, plan."""
from pathlib import Path
from random import Random

import pytest

from onwordly.experiments.arithmetic import run_experiment
from onwordly.experiments.kaggle import VERDICT_REPAIR_REGIMES, _job_units, load_run_plan
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import TrainStepMetrics
from onwordly.tasks.arithmetic import make_arithmetic_task
from onwordly.tasks.solve_judge import SolveJudgeTask, make_solve_judge_task, parse_solve_judge
from onwordly.tasks.verdict import VerdictTask, verify_task
from onwordly.training.harness import OnPolicyConfig, group_advantages, run_equal_token_training
from onwordly.training.verdict_evaluation import evaluate_verdict
from onwordly.training.verdict_sources import VerdictArithmeticSource

ADD_TASKS = [make_arithmetic_task(a, b, "add") for a in range(1, 9) for b in range(1, 9)]


def test_solve_judge_parse_and_exact_verify() -> None:
    base = make_arithmetic_task(47, 6, "multiply")
    right = make_solve_judge_task(base, 282)
    wrong = make_solve_judge_task(base, 272)
    assert right.target_text == "282; right" and wrong.target_text == "282; wrong"
    assert verify_task(right, " 282; right ") and verify_task(right, "282;right")
    assert not verify_task(right, "282; wrong") and not verify_task(right, "272; right")
    assert verify_task(wrong, "282; wrong") and not verify_task(wrong, "272; wrong")
    assert not verify_task(wrong, "right") and not verify_task(wrong, "wrong: 282")
    assert parse_solve_judge("-5; wrong") == (-5, False)
    assert parse_solve_judge("007; right") is None
    assert parse_solve_judge("282; Right") is None
    assert parse_solve_judge("282 right") is None
    assert right.bucket_key.startswith("solve-judge-right:")


def _feed(source: VerdictArithmeticSource, count: int) -> list[object]:
    rng = Random(0)
    for _ in range(count):
        task = source.base.next_task(rng)
        source.observe_response(task, str(task.answer), True)
    moves = []
    while source.pending:
        moves.append(source.next_task(rng))
    return moves


def test_solve_judge_source_is_balanced_with_synthetic_wrongs() -> None:
    source = VerdictArithmeticSource(
        ADD_TASKS, wrong_source="synthetic", verdict_rate=1.0, seed=0, make_move=make_solve_judge_task
    )
    moves = _feed(source, 400)
    assert len(moves) == 400 and all(isinstance(move, SolveJudgeTask) for move in moves)
    right = sum(move.shown_is_right for move in moves)
    assert 160 < right < 240 and source.queued["wrong_own"] == 0
    assert source.queued["right"] == right and source.queued["wrong_synthetic"] == 400 - right
    assert source.on_policy_config(moves[0]) is None


def test_default_source_reproduces_009_stream() -> None:
    plain = VerdictArithmeticSource(ADD_TASKS, wrong_source="synthetic", verdict_rate=0.3, seed=5)
    rl = VerdictArithmeticSource(
        ADD_TASKS, wrong_source="synthetic", verdict_rate=0.3, seed=5, on_policy=OnPolicyConfig()
    )
    first, second = _feed(plain, 200), _feed(rl, 200)
    assert first == second and all(isinstance(move, VerdictTask) for move in first)
    assert rl.on_policy_config(first[0]) == OnPolicyConfig()
    assert rl.on_policy_config(ADD_TASKS[0]) is None


def test_dense_rate_queues_more_moves() -> None:
    sparse = VerdictArithmeticSource(ADD_TASKS, wrong_source="synthetic", verdict_rate=0.3, seed=1)
    dense = VerdictArithmeticSource(ADD_TASKS, wrong_source="synthetic", verdict_rate=1.0, seed=1)
    assert len(_feed(dense, 300)) == 300 > 2.5 * len(_feed(sparse, 300))


def test_group_advantages() -> None:
    assert group_advantages([1.0, 0.0, 1.0, 0.0]) == [0.5, -0.5, 0.5, -0.5]
    assert group_advantages([1.0, 0.0, 0.0, 0.0]) == [0.75, -0.25, -0.25, -0.25]
    assert group_advantages([1.0, 1.0, 1.0]) is None
    assert group_advantages([0.0, 0.0]) is None
    with pytest.raises(ValueError):
        OnPolicyConfig(samples=1)


class ScriptedPolicyAdapter:
    """Arithmetic: always right. Samples: 'right' / 'junk' alternately."""

    def __init__(self) -> None:
        self.sft: list[tuple[str, str]] = []
        self.weighted: list[tuple[str, str, float]] = []
        self.sample_calls = 0

    def generate(self, prompt: str) -> str:
        if "proposed answer" in prompt:
            return "right"
        left, rest = prompt.split("Compute ")[1].split(" + ")
        return str(int(left) + int(rest.split(".")[0]))

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt
        return 3 + len(target)

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.sft.append((prompt, target))
        return TrainStepMetrics(loss=1.0, tokens=self.count_training_tokens(prompt, target))

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        assert temperature == 0.7
        self.sample_calls += 1
        return ["right" if index % 2 == 0 else "junk" for index in range(n)]

    def train_weighted(self, prompt: str, completion: str, weight: float) -> TrainStepMetrics:
        self.weighted.append((prompt, completion, weight))
        return TrainStepMetrics(loss=2.0, tokens=self.count_training_tokens(prompt, completion))


def test_on_policy_harness_weights_skips_and_stays_in_budget() -> None:
    source = VerdictArithmeticSource(
        ADD_TASKS,
        wrong_source="synthetic",
        verdict_rate=1.0,
        seed=0,
        on_policy=OnPolicyConfig(samples=4, temperature=0.7),
    )
    adapter = ScriptedPolicyAdapter()
    result = run_equal_token_training(
        regime="verdict-rl", adapter=adapter, source=source, token_budget=400, seed=0, verifier=verify_task
    )
    stats = result.on_policy
    assert stats is not None
    # Every verdict move is a group; right-shown groups have mixed rewards
    # ('right' correct, 'junk' not), wrong-shown groups are all 0 and skipped.
    assert stats["groups"] == adapter.sample_calls
    assert stats["groups_trained"] + stats["groups_skipped_equal_rewards"] in (stats["groups"], stats["groups"] - 1)
    assert stats["groups_trained"] > 0 and stats["groups_skipped_equal_rewards"] > 0
    assert stats["weighted_updates"] == 4 * stats["groups_trained"] == len(adapter.weighted)
    for prompt, completion, weight in adapter.weighted:
        assert "proposed answer" in prompt
        assert weight == (0.5 if completion == "right" else -0.5)
    assert not any("proposed answer" in prompt for prompt, _ in adapter.sft)
    # Budget: SFT targets plus every trained sampled pair, never above it.
    sft_tokens = sum(adapter.count_training_tokens(p, t) for p, t in adapter.sft)
    rl_tokens = sum(adapter.count_training_tokens(p, c) for p, c, _ in adapter.weighted)
    assert stats["update_tokens"] == rl_tokens
    assert result.training_tokens == sft_tokens + rl_tokens <= 400
    assert result.examples_trained == len(adapter.sft) + len(adapter.weighted)
    # Sampling is billed as generation/verifier work and reported separately.
    assert stats["sample_generation_calls"] == 4 * stats["groups"]
    assert result.generation_calls == result.pre_update_attempts + stats["sample_generation_calls"]
    assert result.verifier_calls == result.generation_calls
    assert stats["samples_correct"] == 2 * stats["groups_trained"]
    assert stats["mean_sample_nll"] == 2.0 and result.mean_loss == 1.0
    assert result.to_dict()["on_policy"] == stats


def test_all_equal_rewards_never_update() -> None:
    class Unanimous(ScriptedPolicyAdapter):
        def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
            self.sample_calls += 1
            return ["junk"] * n

    source = VerdictArithmeticSource(
        ADD_TASKS, wrong_source="synthetic", verdict_rate=1.0, seed=0,
        on_policy=OnPolicyConfig(samples=3, temperature=0.7),
    )
    adapter = Unanimous()
    result = run_equal_token_training(
        regime="verdict-rl", adapter=adapter, source=source, token_budget=200, seed=0, verifier=verify_task
    )
    assert adapter.weighted == [] and result.on_policy["weighted_updates"] == 0
    assert result.on_policy["groups_skipped_equal_rewards"] == result.on_policy["groups"] > 0
    assert result.on_policy["mean_sample_nll"] is None


def test_sft_runs_omit_on_policy_block() -> None:
    source = VerdictArithmeticSource(ADD_TASKS, wrong_source="synthetic", verdict_rate=1.0, seed=0)
    result = run_equal_token_training(
        regime="verdict-synthetic", adapter=ScriptedPolicyAdapter(), source=source,
        token_budget=200, seed=0, verifier=verify_task,
    )
    assert result.on_policy is None and "on_policy" not in result.to_dict()
    assert result.pre_update_attempts == result.generation_calls


class Scripted:
    """Deterministic judge: right if shown equals a fixed guess, else a reply by rule."""

    def __init__(self, solve_judge: bool = False) -> None:
        self.solve_judge = solve_judge
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        left, rest = prompt.split("Compute ")[1].split(" + ")
        right = int(rest.split(".")[0])
        answer = int(left) + right
        if "proposed answer" not in prompt:
            return str(answer) if left != "3" else str(answer + 1)  # 3 + x answered wrong
        shown = int(prompt.split("A proposed answer is ")[1].split(".")[0])
        if left == "7":
            return "garbage"
        if self.solve_judge:
            return f"{answer}; right" if shown == answer or left == "5" else f"{answer + (left == '1')}; wrong"
        if shown == answer or left == "5":
            return "right"
        return f"wrong: {answer + (left == '1')}"


HELDOUT = [make_arithmetic_task(a, 2, "add") for a in (1, 3, 5, 7, 9)]


def test_evaluate_verdict_default_output_unchanged() -> None:
    adapter = Scripted()
    result = evaluate_verdict(adapter, HELDOUT, seed=0)
    legacy = {
        "examples": 5,
        "right_shown_accuracy": 0.8,
        "wrong_shown_judgement_accuracy": 0.6,
        "wrong_shown_repair_accuracy": 0.4,
        "balanced_verdict_accuracy": 0.7,
        "generation_calls": 20,
    }
    assert {key: result[key] for key in legacy} == legacy
    check = dict(result["self_check"])
    own = check.pop("class_counts")
    assert check == {
        "first_pass_accuracy": 0.8,
        "final_accuracy": 1.0,
        "fixed": 1,
        "broken": 0,
        "second_pass_calls": 5,
    }
    # Only additive keys beyond 009's output.
    assert set(result) - set(legacy) == {"self_check", "format", "class_counts"}
    assert result["format"] == "verdict"
    assert result["class_counts"] == {
        "right_kept": 4, "right_rejected": 0, "right_unparseable": 1,
        "wrong_caught_repaired": 2, "wrong_caught_misrepaired": 1,
        "wrong_accepted": 1, "wrong_unparseable": 1,
    }
    # Own answers: 3 + 2 answered wrong then caught and repaired; 7 + 2's verdict
    # is unparseable (first answer kept); 1, 5, 9 kept.
    assert own == {
        "own_right_kept": 3, "own_right_rejected": 0, "own_right_unparseable_verdict": 1,
        "own_wrong_caught_repaired": 1, "own_wrong_caught_misrepaired": 0,
        "own_wrong_accepted": 0, "own_wrong_unparseable_verdict": 0,
        "unparseable_first_pass": 0,
    }
    # Same prompts, same order as 009 (verdict prompts carry 009's wording).
    assert all("reply exactly: wrong: <the correct integer>" in p for p in adapter.prompts if "proposed" in p)


def test_evaluate_verdict_solve_judge_format() -> None:
    result = evaluate_verdict(Scripted(solve_judge=True), HELDOUT, seed=0, format="solve-judge")
    assert result["format"] == "solve-judge"
    assert result["right_shown_accuracy"] == 0.8
    assert result["right_shown_with_answer_accuracy"] == 0.8
    assert result["wrong_shown_judgement_accuracy"] == 0.6
    assert result["wrong_shown_repair_accuracy"] == 0.4
    assert sum(result["class_counts"].values()) == 10
    assert sum(result["self_check"]["class_counts"].values()) == 5
    with pytest.raises(ValueError):
        evaluate_verdict(Scripted(), HELDOUT, seed=0, format="other")


class RepairAdapter(ScriptedPolicyAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.answers: dict[str, str] = {}

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, "0")

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.answers[prompt] = target
        return super().train_example(prompt, target)

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        del temperature
        return ["right" if index % 2 == 0 else "junk" for index in range(n)]


def test_verdict_repair_regimes_run_and_report(tmp_path: Path) -> None:
    from onwordly.reporting.arithmetic import render_result

    manifest = ArithmeticExperimentManifest.from_json("experiments/010-verdict-repair/smoke-manifest.json")
    result = run_experiment(
        manifest, output_dir=tmp_path, create_adapter=RepairAdapter, regimes=VERDICT_REPAIR_REGIMES
    )
    regimes = result["regimes"]
    assert regimes["solve-judge-synthetic"]["verdict_evaluation"]["format"] == "solve-judge"
    assert regimes["verdict-dense"]["verdict_evaluation"]["format"] == "verdict"
    assert "on_policy" in regimes["verdict-rl"]["training"]
    assert all("on_policy" not in regimes[name]["training"] for name in VERDICT_REPAIR_REGIMES if name != "verdict-rl")
    for name in VERDICT_REPAIR_REGIMES:
        assert regimes[name]["training"]["training_tokens"] <= manifest.token_budget
    queued = {name: sum(regimes[name]["corrective_tasks_queued"].values()) for name in ("verdict-synthetic", "verdict-dense")}
    assert queued["verdict-dense"] > queued["verdict-synthetic"]
    report = render_result(tmp_path / "summary.json")
    assert "Wrong caught + repaired" in report and "On-policy verdict updates" in report


def test_kaggle_plan_for_010(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text("experiment: 010\nmode: single\n", encoding="utf-8")
    plan = load_run_plan(plan_path)
    assert plan.manifest == "experiments/010-verdict-repair/manifest.json"
    plan_path.write_text("experiment: 010\nmode: suite\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gated"):
        load_run_plan(plan_path)
    plan_path.write_text(
        "experiment: 010\nmode: single\nmanifest: experiments/001-arithmetic-curriculum/manifest.json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verdict_evaluation_size"):
        load_run_plan(plan_path)
    plan_path.write_text("experiment: batch\njobs: 010\n", encoding="utf-8")
    assert dict(load_run_plan(plan_path).options)["jobs"] == "010"
    units, _ = _job_units("010", (3303,), tmp_path)
    assert [unit[2] for unit in units] == list(VERDICT_REPAIR_REGIMES)
    assert {unit[1] for unit in units} == {str(tmp_path / "010-verdict-repair")}
    assert VERDICT_REPAIR_REGIMES == (
        "static", "verdict-synthetic", "verdict-dense", "solve-judge-synthetic", "verdict-rl",
    )


def test_010_manifest_is_009_plus_new_fields() -> None:
    import json

    for name in ("manifest.json", "smoke-manifest.json"):
        old = json.loads(Path(f"experiments/009-verdict-language-game/{name}").read_text())
        new = json.loads(Path(f"experiments/010-verdict-repair/{name}").read_text())
        assert {key: new[key] for key in old} == old
        assert set(new) - set(old) == {"dense_verdict_rate", "rl_samples", "rl_temperature"}


def test_hf_adapter_sample_and_weighted_update_with_lora() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from onwordly.models.huggingface import HuggingFaceCausalLMAdapter

    try:
        adapter = HuggingFaceCausalLMAdapter(
            "sshleifer/tiny-gpt2", device="cpu", seed=0, max_new_tokens=4,
            lora={"r": 2, "alpha": 4, "target_modules": ["c_attn"]},
        )
    except OSError:
        pytest.skip("model download unavailable")
    samples = adapter.sample("Compute 2 + 3.", 3, 1.0)
    assert len(samples) == 3 and all(isinstance(text, str) for text in samples)
    before = {n: p.detach().clone() for n, p in adapter.model.named_parameters() if p.requires_grad}
    step = adapter.train_weighted("Compute 2 + 3.", "5", -0.5)
    assert step.tokens == adapter.count_training_tokens("Compute 2 + 3.", "5") and step.loss > 0
    assert any(not p.detach().equal(before[n]) for n, p in adapter.model.named_parameters() if n in before)
