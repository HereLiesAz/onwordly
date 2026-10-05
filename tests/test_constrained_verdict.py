"""Experiment 011: constrained-string task, verdict move, episodes, evaluation, runner, plan."""
import json
from pathlib import Path
from random import Random

import pytest

from onwordly.experiments.constrained_strings import (
    CONSTRAINED_VERDICT_REGIMES,
    ConstrainedExperimentManifest,
    render_constrained_result,
    run_constrained_experiment,
)
from onwordly.experiments.kaggle import execute_run_plan, load_run_plan
from onwordly.models.base import TrainStepMetrics
from onwordly.tasks.constrained_strings import (
    build_constrained_dataset,
    check_rule,
    constrained_partition,
    generate_constrained_task,
    make_constrained_task,
    make_constrained_verdict_task,
    parse_string_answer,
    parse_string_verdict,
    satisfaction,
    satisfies_all,
    synthetic_violation,
)
from onwordly.training.constrained_evaluation import evaluate_constrained_verdict
from onwordly.training.constrained_sources import (
    PARTIAL_REPAIR_BASE,
    RIGHT_KEPT_REWARD,
    RIGHT_REJECTED_REWARD,
    WRONG_ACCEPTED_SCALE,
    WRONG_REPAIRED_REWARD,
    ConstrainedSelfCheckConfig,
    ConstrainedSelfCheckEpisode,
    ConstrainedSelfCheckSource,
    ConstrainedVerdictSource,
    classify_constrained_episode,
    constrained_episode_reward,
    verify_constrained,
)
from onwordly.training.harness import run_equal_token_training

SMOKE = "experiments/011-verdict-constrained-strings/smoke-manifest.json"
# allowed ABCDE12345, length 5, starts A, ends 3, excludes E: 5 rules.
TASK = make_constrained_task(
    [("allowed", "ABCDE12345"), ("length", 5), ("starts", "A"), ("ends", "3"), ("excludes", "E")], "AB123"
)


def test_rules_and_satisfaction() -> None:
    assert satisfies_all(TASK, "AB123") and satisfies_all(TASK, "A4443")
    assert satisfaction(TASK, "AB123") == 1.0
    assert satisfaction(TASK, "EB123") == pytest.approx(3 / 5)  # breaks starts and excludes
    assert satisfaction(TASK, "AB12") == pytest.approx(3 / 5)  # length and ends
    assert satisfaction(TASK, None) == 0.0
    assert not check_rule(("no_repeat", None), "AAB") and check_rule(("no_repeat", None), "ABA")
    assert not check_rule(("allowed", "AB"), "ABC") and check_rule(("contains", "C"), "ABC")
    assert TASK.verify(" A4443 ") and not TASK.verify("A4443 x") and not TASK.verify("")
    assert parse_string_answer("AB 12") is None and parse_string_answer(" AB12\n") == "AB12"
    with pytest.raises(ValueError):
        make_constrained_task([("allowed", "AB"), ("length", 2)], "ABC")
    with pytest.raises(ValueError):
        make_constrained_task([("length", 2)], "AB")


def test_generation_is_satisfiable_deterministic_and_partitioned() -> None:
    rng = Random(3)
    for _ in range(300):
        task = generate_constrained_task(rng, rng.choice([5, 7, 9]))
        assert satisfies_all(task, task.witness) and len(task.rules) == 5
        assert synthetic_violation(task, rng) and not satisfies_all(task, synthetic_violation(task, rng))
    first = build_constrained_dataset(seed=1, size=40, lengths=[5, 7], partition="train")
    assert first == build_constrained_dataset(seed=1, size=40, lengths=[5, 7], partition="train")
    evals = build_constrained_dataset(seed=2, size=40, lengths=[5, 7], partition="eval")
    assert all(constrained_partition(t) == "train" for t in first)
    assert all(constrained_partition(t) == "eval" for t in evals)
    assert not {t.rules for t in first} & {t.rules for t in evals}


def test_verdict_move_targets_and_exact_verify() -> None:
    right = make_constrained_verdict_task(TASK, "A4443")
    wrong = make_constrained_verdict_task(TASK, "EB123")
    assert right.shown_is_right and right.target_text == "right"
    assert not wrong.shown_is_right and wrong.target_text == "wrong: AB123"
    assert right.verify("right") and not right.verify("wrong: AB123")
    # Any valid repair is accepted, not only the witness.
    assert wrong.verify("wrong: A1113") and wrong.verify("wrong: AB123")
    assert not wrong.verify("wrong: EB123") and not wrong.verify("right")
    assert parse_string_verdict("wrong:  AB1") == (False, "AB1") and parse_string_verdict("Right") is None
    assert verify_constrained(wrong, "wrong: A1113") and wrong.bucket_key.startswith("verdict-wrong:")


@pytest.mark.parametrize(
    ("turn1", "turn2", "tier", "graded", "binary"),
    (
        ("AB123", "right", "right_kept", 1.0, 1.0),
        ("AB123", "wrong: A1113", "right_rejected", -0.5, 1.0),
        ("AB123", "wrong: EEEEE", "right_rejected", -0.5, 0.0),
        ("EB123", "wrong: A1113", "wrong_repaired", 0.6, 1.0),
        ("EB123", "wrong: AB12", "wrong_caught_partial_repair", 0.3 + 0.1 * 3 / 5, 0.0),
        ("EB123", "wrong: A1E13", "wrong_caught_partial_repair", 0.3 + 0.1 * 4 / 5, 0.0),
        ("EB123", "right", "wrong_accepted", 0.2 * 3 / 5, 0.0),
        ("A1E13", "right", "wrong_accepted", 0.2 * 4 / 5, 0.0),
        ("AB123", "maybe", "unparseable_verdict", 0.0, 0.0),
        ("A B", None, "unparseable_answer", 0.0, 0.0),
    ),
)
def test_episode_tiers_and_rewards(turn1, turn2, tier, graded, binary) -> None:
    got, final = classify_constrained_episode(TASK, turn1, turn2)
    assert got == tier
    assert constrained_episode_reward("graded", TASK, got, turn1, final) == pytest.approx(graded)
    assert constrained_episode_reward("binary", TASK, got, turn1, final) == binary


def test_graded_reward_bounds() -> None:
    assert (RIGHT_KEPT_REWARD, RIGHT_REJECTED_REWARD, WRONG_REPAIRED_REWARD) == (1.0, -0.5, 0.6)
    # Invalid strings have s < 1, so partial repair < 0.4 and accepted wrong < 0.2.
    rng = Random(0)
    for _ in range(200):
        task = generate_constrained_task(rng, 6)
        bad = synthetic_violation(task, rng)
        partial = constrained_episode_reward("graded", task, "wrong_caught_partial_repair", "x", bad)
        accepted = constrained_episode_reward("graded", task, "wrong_accepted", bad, bad)
        assert PARTIAL_REPAIR_BASE <= partial < 0.4 < WRONG_REPAIRED_REWARD
        assert 0.0 <= accepted < WRONG_ACCEPTED_SCALE


class StringAdapter:
    """Greedy: the witness-shaped answer for even-length tasks, else 'ZZ'.
    Samples alternate valid / invalid; verdicts are scripted."""

    def __init__(self, verdict_valid: str = "right", verdict_invalid: str = "wrong: {repair}") -> None:
        self.verdict_valid = verdict_valid
        self.verdict_invalid = verdict_invalid
        self.sft: list[tuple[str, str]] = []
        self.weighted: list[tuple[str, str, float]] = []
        self.greedy: list[str] = []
        self.tasks: dict[str, object] = {}

    def _task(self, prompt: str):
        return self.tasks[prompt.split(" A proposed string")[0]]

    def _challenge(self, prompt: str) -> str:
        task = self.tasks[prompt.split(" Your answer was ")[0]]
        first = prompt.split(" Your answer was ")[1].split(". ")[0]
        return f"hold: {first}" if satisfies_all(task, first) else f"change: {task.witness}"

    def generate(self, prompt: str) -> str:
        self.greedy.append(prompt)
        if " Your answer was " in prompt:
            return self._challenge(prompt)
        if " A proposed string is " in prompt:
            task = self._task(prompt)
            shown = prompt.split(" A proposed string is ")[1].split(". If")[0]
            return "right" if satisfies_all(task, shown) else f"wrong: {task.witness}"
        task = self.tasks[prompt]
        return task.witness if task.length % 2 == 0 else "ZZ"

    def count_training_tokens(self, prompt: str, target: str) -> int:
        del prompt
        return 5 + len(target)

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        self.sft.append((prompt, target))
        return TrainStepMetrics(loss=1.0, tokens=self.count_training_tokens(prompt, target))

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        if " Your answer was " in prompt:
            return [self._challenge(prompt)] * n
        if " A proposed string is " in prompt:
            task = self._task(prompt)
            shown = prompt.split(" A proposed string is ")[1].split(". If")[0]
            template = self.verdict_valid if satisfies_all(task, shown) else self.verdict_invalid
            return [template.format(repair=task.witness)]
        task = self.tasks[prompt]
        return [task.witness if i % 2 == 0 else task.witness + "Q" for i in range(n)]

    def train_weighted(self, prompt: str, completion: str, weight: float) -> TrainStepMetrics:
        self.weighted.append((prompt, completion, weight))
        return TrainStepMetrics(loss=2.0, tokens=self.count_training_tokens(prompt, completion))


def _adapter_for(tasks, **kwargs) -> StringAdapter:
    adapter = StringAdapter(**kwargs)
    adapter.tasks = {t.prompt: t for t in tasks} | {t.prompt.rstrip(): t for t in tasks}
    return adapter


TRAIN = build_constrained_dataset(seed=5, size=60, lengths=[5, 6], partition="train")


def test_verdict_source_is_balanced_and_sft() -> None:
    source = ConstrainedVerdictSource(TRAIN, verdict_rate=1.0, seed=0)
    rng = Random(0)
    for _ in range(400):
        source.observe_response(source.base.next_task(rng), "x", False)
    moves = list(source.pending)
    assert len(moves) == 400 and 160 < sum(m.shown_is_right for m in moves) < 240
    assert source.queued["right"] == sum(m.shown_is_right for m in moves)
    assert not hasattr(source, "on_policy_config")
    source.observe_response(moves[0], "right", True)  # verdict moves never queue more
    assert len(source.pending) == 400


def test_selfcheck_episodes_in_harness_graded_and_binary() -> None:
    def run(reward: str, **kwargs):
        adapter = _adapter_for(TRAIN, **kwargs)
        source = ConstrainedSelfCheckSource(
            TRAIN, config=ConstrainedSelfCheckConfig(reward=reward, samples=4, temperature=0.7), verdict_rate=1.0, seed=0
        )
        result = run_equal_token_training(
            regime=reward, adapter=adapter, source=source, token_budget=3000, seed=0, verifier=verify_constrained
        )
        return adapter, result

    adapter, result = run("graded")
    stats = result.on_policy
    assert stats["groups"] > 0 and stats["reward_tiers"] == {
        "right_kept": 2 * stats["groups"], "wrong_repaired": 2 * stats["groups"]
    }
    assert {round(w, 10) for _, _, w in adapter.weighted} == {0.2, -0.2}
    assert stats["sample_generation_calls"] == 8 * stats["groups"]
    # Episodes skip the greedy attempt: greedy calls == SFT string tasks.
    assert len(adapter.greedy) == len(adapter.sft) == result.pre_update_attempts
    assert result.training_tokens <= 3000
    # Binary on identical episodes: every final valid -> all groups skipped.
    adapter, result = run("binary")
    assert adapter.weighted == [] and result.on_policy["groups_skipped_equal_rewards"] == result.on_policy["groups"]
    # Accepting invalid strings: binary separates them, graded gives 0.2*s.
    adapter, result = run("binary", verdict_invalid="right")
    assert {w for _, _, w in adapter.weighted} == {0.5, -0.5}
    with pytest.raises(TypeError):
        ConstrainedSelfCheckConfig(reward="graded").rollout(adapter, TRAIN[0], verify_constrained)
    episode = ConstrainedSelfCheckEpisode(TRAIN[0])
    assert episode.bucket_key.startswith("selfcheck:") and episode.verify(TRAIN[0].witness)


def test_evaluate_verdict_counts() -> None:
    heldout = build_constrained_dataset(seed=9, size=12, lengths=[5, 6], partition="eval")
    adapter = _adapter_for(heldout)
    result = evaluate_constrained_verdict(adapter, heldout, seed=1)
    assert result["right_shown_accuracy"] == 1.0 and result["wrong_shown_repair_accuracy"] == 1.0
    assert result["balanced_verdict_accuracy"] == 1.0
    assert result["class_counts"]["right_kept"] == result["class_counts"]["wrong_caught_repaired"] == 12
    sc = result["self_check"]
    even = sum(t.length % 2 == 0 for t in heldout)
    # 'ZZ' first passes are caught and repaired with the witness.
    assert sc["first_pass_accuracy"] == even / 12 and sc["final_accuracy"] == 1.0
    assert sc["fixed"] == 12 - even and sc["broken"] == 0 and sc["mean_final_satisfaction"] == 1.0
    assert sc["class_counts"]["own_right_kept"] == even
    assert sc["class_counts"]["own_wrong_caught_repaired"] == 12 - even
    assert result["generation_calls"] == 3 * 12 + 12
    assert sum(result["class_counts"].values()) == 24
    assert sum(sc["class_counts"].values()) == 12


def test_runner_all_arms_matched_budget_and_report(tmp_path: Path) -> None:
    manifest = ConstrainedExperimentManifest.from_json(SMOKE)
    adapters: list[StringAdapter] = []

    def factory() -> StringAdapter:
        common = dict(alphabet=manifest.alphabet, optional_rules=manifest.optional_rules,
                      partition_modulus=manifest.holdout_modulus)
        tasks = (
            build_constrained_dataset(seed=manifest.dataset_seed, size=manifest.static_dataset_size,
                                      lengths=manifest.lengths, partition="train", **common)
            + build_constrained_dataset(seed=manifest.evaluation_seed, size=manifest.evaluation_size,
                                        lengths=manifest.lengths, partition="eval", **common)
            + build_constrained_dataset(seed=manifest.evaluation_seed + 1, size=manifest.longer_evaluation_size,
                                        lengths=manifest.out_of_range_lengths, partition="eval", **common)
        )
        adapters.append(_adapter_for(tasks))
        return adapters[-1]

    result = run_constrained_experiment(manifest, output_dir=tmp_path, create_adapter=factory)
    regimes = result["regimes"]
    assert tuple(regimes) == CONSTRAINED_VERDICT_REGIMES
    for name in CONSTRAINED_VERDICT_REGIMES:
        assert regimes[name]["training"]["training_tokens"] <= manifest.token_budget
        assert regimes[name]["training"]["token_budget"] == manifest.token_budget
        assert "mean_final_satisfaction" in regimes[name]["verdict_evaluation"]["self_check"]
    assert "on_policy" in regimes["selfcheck-rl-graded"]["training"]
    assert "on_policy" not in regimes["static"]["training"]
    assert regimes["challenge-rl-graded"]["challenge_evaluation"]["by_error_rate"]["0.3"]["final_accuracy"] >= 0
    assert "challenge_evaluation" not in regimes["selfcheck-rl-graded"]
    assert sum(regimes["verdict-synthetic"]["tasks_queued"].values()) > 0
    # Resume: rerun loads saved regimes without creating adapters.
    count = len(adapters)
    run_constrained_experiment(manifest, output_dir=tmp_path, create_adapter=factory)
    assert len(adapters) == count
    report = render_constrained_result(tmp_path / "summary.json")
    assert "Fallible challenge" in report
    assert "Mean s(final)" in report and "selfcheck-rl-binary" in report and "On-policy" in report
    # Held-out rule sets never appear in training data.
    train_rules = {json.dumps(json.loads(l)["rules"]) for l in (tmp_path / "static-train.jsonl").read_text().splitlines()}
    eval_rules = {json.dumps(json.loads(l)["rules"]) for l in (tmp_path / "evaluation-heldout.jsonl").read_text().splitlines()}
    assert not train_rules & eval_rules


def test_manifests_validate_and_match() -> None:
    full = json.loads(Path("experiments/011-verdict-constrained-strings/manifest.json").read_text())
    smoke = json.loads(Path(SMOKE).read_text())
    assert set(full) == set(smoke)
    ConstrainedExperimentManifest.from_json("experiments/011-verdict-constrained-strings/manifest.json")
    # Same RL settings as 010.
    ten = json.loads(Path("experiments/010-verdict-repair/manifest.json").read_text())
    for key in ("token_budget", "verdict_rate", "rl_samples", "rl_temperature", "training_seed", "learning_rate"):
        assert full[key] == ten[key]


def test_kaggle_plan_for_011(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_path = tmp_path / "plan"
    plan_path.write_text("experiment: 011\nmode: single\n", encoding="utf-8")
    assert load_run_plan(plan_path).manifest == "experiments/011-verdict-constrained-strings/manifest.json"
    plan_path.write_text("experiment: 011\nmode: suite\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gated"):
        load_run_plan(plan_path)
    plan_path.write_text(f"experiment: 011\nmanifest: {SMOKE}\n", encoding="utf-8")
    plan = load_run_plan(plan_path)

    import onwordly.experiments.constrained_strings as module
    import onwordly.experiments.kaggle as kaggle

    seen = {}
    monkeypatch.setattr(kaggle, "run_units", lambda units, devices, module: seen.update(units=units, module=module))

    def fake_run(manifest, *, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        seen["ran"] = True

    monkeypatch.setattr(module, "run_constrained_experiment", fake_run)
    monkeypatch.setattr(module, "render_constrained_result", lambda path: "report\n")
    output = execute_run_plan(plan, tmp_path / "results")
    assert output.name == "011-verdict-constrained-strings" and (output / "RESULTS.md").read_text() == "report\n"
    assert [u[2] for u in seen["units"]] == list(CONSTRAINED_VERDICT_REGIMES)
    assert seen["module"] == "onwordly.experiments.constrained_strings" and seen["ran"]
