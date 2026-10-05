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
    rl_arms = ("verdict-rl", "verdict-rl-graded", "selfcheck-rl-binary")
    assert all("on_policy" in regimes[name]["training"] for name in rl_arms)
    assert all("on_policy" not in regimes[name]["training"] for name in VERDICT_REPAIR_REGIMES if name not in rl_arms)
    assert set(regimes["verdict-rl"]["training"]["on_policy"]["reward_tiers"]) <= {"correct", "incorrect"}
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
        "static", "verdict-synthetic", "verdict-dense", "solve-judge-synthetic", "verdict-rl", "verdict-rl-graded",
        "selfcheck-rl-binary",
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



# --- Two-turn self-check episodes (verdict-rl-graded / selfcheck-rl-binary) ---

from onwordly.training.selfcheck_episodes import (  # noqa: E402
    GRADED_EPISODE_REWARDS,
    SelfCheckEpisode,
    SelfCheckEpisodeConfig,
    classify_episode,
    episode_reward,
    is_close_repair,
)
from onwordly.training.verdict_sources import SelfCheckEpisodeSource  # noqa: E402

MUL = make_arithmetic_task(47, 6, "multiply")  # 282


@pytest.mark.parametrize(
    ("turn1", "turn2", "tier", "graded", "binary"),
    (
        ("282", "right", "right_kept", 1.0, 1.0),
        ("272", "wrong: 282", "wrong_repaired", 0.6, 1.0),
        ("272", "wrong: 290", "wrong_caught_close_repair", 0.4, 0.0),
        ("272", "wrong: 300", "wrong_caught_far_repair", 0.3, 0.0),
        ("272", "wrong: 272", "wrong_caught_close_repair", 0.4, 0.0),
        ("272", "right", "close_accepted", 0.2, 0.0),
        ("268", "right", "close_accepted", 0.2, 0.0),
        ("267", "right", "far_accepted", 0.0, 0.0),
        ("300", "wrong: 282", "wrong_repaired", 0.6, 1.0),
        ("300", "wrong: 1", "wrong_caught_far_repair", 0.3, 0.0),
        ("282", "wrong: 282", "right_rejected", -0.5, 1.0),
        ("282", "wrong: 5", "right_rejected", -0.5, 0.0),
        ("282", "maybe", "unparseable_verdict", 0.0, 0.0),
        ("two hundred", None, "unparseable_answer", 0.0, 0.0),
    ),
)
def test_episode_tiers_and_rewards(turn1, turn2, tier, graded, binary) -> None:
    got_tier, final = classify_episode(MUL, turn1, turn2)
    assert got_tier == tier
    assert episode_reward("graded", got_tier, final, MUL.answer) == graded == GRADED_EPISODE_REWARDS[tier]
    assert episode_reward("binary", got_tier, final, MUL.answer) == binary


def test_close_repair_boundary() -> None:
    # Tolerance max(1, 5% of |answer|): 282 -> 14.1.
    assert is_close_repair(282 + 14, 282) and is_close_repair(282 - 14, 282)
    assert not is_close_repair(282 + 15, 282)
    assert is_close_repair(-268, -282) and not is_close_repair(-267, -282)
    # The absolute floor of 1 for small answers.
    assert is_close_repair(4, 3) and is_close_repair(-1, 0) and not is_close_repair(5, 3)
    from onwordly.training.selfcheck_episodes import CLOSE_ACCEPTED_REWARD

    assert GRADED_EPISODE_REWARDS["close_accepted"] == CLOSE_ACCEPTED_REWARD == 0.2


class EpisodeAdapter(ScriptedPolicyAdapter):
    """Turn 1 alternates a right and a wrong answer; turn 2 is scripted."""

    def __init__(self, verdicts: dict[bool, str], turn1: list[str] | None = None) -> None:
        super().__init__()
        self.verdicts = verdicts
        self.turn1 = turn1
        self.calls: list[tuple[str, int]] = []

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        self.calls.append((prompt, n))
        if "proposed answer" in prompt:
            assert n == 1
            shown = int(prompt.split("A proposed answer is ")[1].split(".")[0])
            left, rest = prompt.split("Compute ")[1].split(" + ")
            answer = int(left) + int(rest.split(".")[0])
            return [self.verdicts[shown == answer].format(answer=answer)]
        if self.turn1 is not None:
            return list(self.turn1)[:n]
        left, rest = prompt.split("Compute ")[1].split(" + ")
        answer = int(left) + int(rest.split(".")[0])
        return [str(answer) if index % 2 == 0 else str(answer + 1) for index in range(n)]


def _episodes(reward: str, adapter: EpisodeAdapter, budget: int = 600):
    source = SelfCheckEpisodeSource(
        ADD_TASKS, config=SelfCheckEpisodeConfig(reward=reward, samples=4, temperature=0.7),
        verdict_rate=1.0, seed=0,
    )
    result = run_equal_token_training(
        regime=reward, adapter=adapter, source=source, token_budget=budget, seed=0, verifier=verify_task
    )
    return result


def test_graded_episodes_train_both_turns_with_episode_advantage() -> None:
    # Right answers kept (1.0); wrong answers caught and exactly repaired (0.6).
    adapter = EpisodeAdapter({True: "right", False: "wrong: {answer}"})
    result = _episodes("graded", adapter)
    stats = result.on_policy
    assert stats["groups_trained"] > 0 and stats["reward_tiers"] == {
        "right_kept": 2 * stats["groups"], "wrong_repaired": 2 * stats["groups"],
    }
    # Advantages: 1.0 - 0.8 = +0.2 and 0.6 - 0.8 = -0.2, applied to both turns.
    assert {round(w, 10) for _, _, w in adapter.weighted} == {0.2, -0.2}
    by_prompt = [(p, c, round(w, 10)) for p, c, w in adapter.weighted]
    answer_turns = [x for x in by_prompt if "proposed answer" not in x[0]]
    verdict_turns = [x for x in by_prompt if "proposed answer" in x[0]]
    assert len(answer_turns) == len(verdict_turns) == 4 * stats["groups_trained"]
    assert stats["weighted_updates"] == 8 * stats["groups_trained"]
    # Budget and generation accounting: 4 turn-1 samples + 4 verdicts per group.
    assert stats["sample_generation_calls"] == 8 * stats["groups"]
    assert result.generation_calls == result.pre_update_attempts + stats["sample_generation_calls"]
    sft = sum(adapter.count_training_tokens(p, t) for p, t in adapter.sft)
    rl = sum(adapter.count_training_tokens(p, c) for p, c, _ in adapter.weighted)
    assert stats["update_tokens"] == rl and result.training_tokens == sft + rl <= 600
    assert stats["samples_correct"] == 4 * stats["groups"]


def test_binary_episodes_skip_when_finals_all_correct() -> None:
    # Same episodes as above: every final answer is correct -> binary skips all.
    adapter = EpisodeAdapter({True: "right", False: "wrong: {answer}"})
    result = _episodes("binary", adapter)
    assert adapter.weighted == [] and result.on_policy["groups_trained"] == 0
    assert result.on_policy["groups_skipped_equal_rewards"] == result.on_policy["groups"] > 0
    # Tiers are recorded identically for both rewards.
    assert set(result.on_policy["reward_tiers"]) == {"right_kept", "wrong_repaired"}


def test_binary_episodes_reward_final_answer() -> None:
    # Wrong answers accepted -> final wrong (0) vs right kept (1): ±0.5.
    adapter = EpisodeAdapter({True: "right", False: "right"})
    result = _episodes("binary", adapter)
    assert result.on_policy["groups_trained"] > 0
    assert {w for _, _, w in adapter.weighted} == {0.5, -0.5}


def test_unparseable_turn1_trains_only_turn1() -> None:
    adapter = EpisodeAdapter({True: "right", False: "right"}, turn1=["7", "x", "y", "z"])
    result = _episodes("graded", adapter)
    stats = result.on_policy
    # Only parseable turn-1 answers get a verdict call: 4 + 1 per group.
    assert stats["sample_generation_calls"] == 5 * stats["groups"]
    assert stats["reward_tiers"]["unparseable_answer"] == 3 * stats["groups"]
    assert all(p.count("proposed answer") <= 1 for p, _, _ in adapter.weighted)
    trained = [(p, c) for p, c, _ in adapter.weighted if "proposed answer" not in p]
    assert {c for _, c in trained} <= {"7", "x", "y", "z"}
    verdict_turns = [c for p, c, _ in adapter.weighted if "proposed answer" in p]
    assert len(verdict_turns) * 4 == len(trained)


def test_selfcheck_source_queues_episodes_only_for_arithmetic() -> None:
    config = SelfCheckEpisodeConfig(reward="graded")
    source = SelfCheckEpisodeSource(ADD_TASKS, config=config, verdict_rate=0.5, seed=0)
    rng = Random(0)
    for _ in range(200):
        source.observe_response(source.base.next_task(rng), "0", False)
    assert 60 < source.queued["episode_groups"] < 140
    episode = source.next_task(rng)
    assert isinstance(episode, SelfCheckEpisode)
    assert source.on_policy_config(episode) is config and source.on_policy_config(ADD_TASKS[0]) is None
    source.observe_response(episode, "0", False)  # episodes never queue more episodes
    with pytest.raises(ValueError):
        SelfCheckEpisodeConfig(reward="shaped")


def test_verdict_rl_binary_path_unchanged() -> None:
    assert OnPolicyConfig() == OnPolicyConfig(samples=4, temperature=1.0)
    adapter = ScriptedPolicyAdapter()
    source = VerdictArithmeticSource(
        ADD_TASKS, wrong_source="synthetic", verdict_rate=1.0, seed=0,
        on_policy=OnPolicyConfig(samples=4, temperature=0.7),
    )
    result = run_equal_token_training(
        regime="verdict-rl", adapter=adapter, source=source, token_budget=400, seed=0, verifier=verify_task
    )
    tiers = result.on_policy["reward_tiers"]
    assert set(tiers) == {"correct", "incorrect"}
    assert tiers["correct"] == result.on_policy["samples_correct"]
    assert result.on_policy["episodes"] == result.on_policy["sample_generation_calls"]
    assert {(c, w) for _, c, w in adapter.weighted} == {("right", 0.5), ("junk", -0.5)}


def test_on_policy_tasks_skip_greedy_attempt() -> None:
    class Counting(EpisodeAdapter):
        def __init__(self) -> None:
            super().__init__({True: "right", False: "right"})
            self.greedy: list[str] = []

        def generate(self, prompt: str) -> str:
            self.greedy.append(prompt)
            return super().generate(prompt)

    adapter = Counting()
    result = _episodes("graded", adapter)
    # Greedy calls are exactly the SFT arithmetic tasks; episodes go straight to sampling.
    assert len(adapter.greedy) == len(adapter.sft) == result.pre_update_attempts
    assert result.generation_calls == len(adapter.greedy) + result.on_policy["sample_generation_calls"]
    assert sum(b["attempts"] for b in result.bucket_stats.values()) == len(adapter.sft)
    assert not any(key.startswith("selfcheck:") for key in result.bucket_stats)

    single = ScriptedPolicyAdapter()
    calls: list[str] = []
    original = single.generate
    single.generate = lambda prompt: calls.append(prompt) or original(prompt)
    source = VerdictArithmeticSource(
        ADD_TASKS, wrong_source="synthetic", verdict_rate=1.0, seed=0,
        on_policy=OnPolicyConfig(samples=4, temperature=0.7),
    )
    run_equal_token_training(
        regime="verdict-rl", adapter=single, source=source, token_budget=400, seed=0, verifier=verify_task
    )
    assert calls and not any("proposed answer" in prompt for prompt in calls)
