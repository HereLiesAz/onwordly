"""Verdict prior diagnostic (no training) with a fake scoring adapter."""
import json
from pathlib import Path

import pytest

from onwordly.diagnostics.verdict_prior import auroc, heldout_tasks, main, run_models, run_verdict_prior
from onwordly.experiments.kaggle import execute_run_plan, load_run_plan
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.tasks.verdict import make_verdict_task

SMOKE = "experiments/009-verdict-language-game/smoke-manifest.json"


def _shown(prompt: str) -> int:
    return int(prompt.split("A proposed answer is ")[1].split(".")[0])


class FakeScorer:
    """Prefers 'right' by +2 when the shown answer is correct, +0.5 otherwise.

    Greedy answers: correct when the answer is even, else off by one."""

    def __init__(self, truth: dict[str, int]) -> None:
        self.truth = truth
        self.scored: list[tuple[str, str]] = []

    def generate(self, prompt: str) -> str:
        answer = self.truth[prompt]
        return str(answer) if answer % 2 == 0 else str(answer + 1)

    def continuation_logprob(self, prompt: str, continuation: str) -> float:
        self.scored.append((prompt, continuation))
        base = prompt.split(" A proposed answer")[0]
        correct = _shown(prompt) == self.truth[base]
        if continuation == "right":
            return -1.0
        if continuation == "wrong":
            return -3.0 if correct else -1.5
        return -4.0 if correct else -1.25  # full "wrong: <n>"


def _fake(tasks):
    truth = {task.prompt: task.answer for task in tasks}
    truth.update({task.prompt.rstrip(): task.answer for task in tasks})
    return FakeScorer(truth)


def test_auroc() -> None:
    assert auroc([1, 2], [0, 0]) == 1.0
    assert auroc([0], [0]) == 0.5
    assert auroc([0, 0], [1, 1]) == 0.0
    assert auroc([], [1]) is None


def test_heldout_matches_009_construction() -> None:
    manifest = ArithmeticExperimentManifest.from_json(SMOKE)
    tasks = heldout_tasks(manifest, 5)
    assert len(tasks) == 5
    with pytest.raises(ValueError):
        heldout_tasks(manifest, manifest.evaluation_size + 1)


def test_prior_conditions_and_discrimination() -> None:
    manifest = ArithmeticExperimentManifest.from_json(SMOKE)
    tasks = heldout_tasks(manifest, 10)
    adapter = _fake(tasks)
    result = run_verdict_prior(adapter, tasks, seed=manifest.evaluation_seed)
    odd = sum(task.answer % 2 for task in tasks)
    assert result["greedy"] == {"correct": 10 - odd, "unparseable": 0, "wrong_scored": odd}
    c = result["conditions"]
    assert c["correct_shown"]["n"] == c["near_miss_shown"]["n"] == 10 and c["own_wrong_shown"]["n"] == odd
    assert c["correct_shown"]["margin_bare"] == {"prefers_right": 1.0, "mean": 2.0}
    assert c["correct_shown"]["margin_full"]["mean"] == 3.0
    assert c["near_miss_shown"]["margin_bare"] == {"prefers_right": 1.0, "mean": 0.5}
    assert c["near_miss_shown"]["margin_full"]["mean"] == 0.25
    d = result["discrimination"]["correct_vs_near_miss_shown"]
    assert d["margin_bare"]["auroc"] == 1.0 and d["margin_bare"]["paired_correct_margin_higher"] == 1.0
    assert d["margin_full"]["mean_margin_difference"] == pytest.approx(2.75)
    # 3 scores per scored proposal; one greedy call per problem.
    assert result["scoring_calls"] == len(adapter.scored) == 3 * (20 + odd)
    continuations = {c for _, c in adapter.scored}
    assert "right" in continuations and "wrong" in continuations
    # Prompts are exactly the 009 verdict prompt.
    assert adapter.scored[0][0] == make_verdict_task(tasks[0], tasks[0].answer).prompt


def test_no_discrimination_gives_half() -> None:
    manifest = ArithmeticExperimentManifest.from_json(SMOKE)
    tasks = heldout_tasks(manifest, 6)

    class Flat(FakeScorer):
        def continuation_logprob(self, prompt: str, continuation: str) -> float:
            return -1.0 if continuation == "right" else -2.0

    result = run_verdict_prior(Flat({t.prompt: t.answer for t in tasks}), tasks, seed=1)
    assert result["discrimination"]["correct_vs_near_miss_shown"]["margin_bare"]["auroc"] == 0.5
    assert result["conditions"]["near_miss_shown"]["margin_full"]["prefers_right"] == 1.0


def test_run_models_writes_report(tmp_path: Path) -> None:
    manifest = ArithmeticExperimentManifest.from_json(SMOKE)
    tasks = heldout_tasks(manifest, 4)
    results = run_models(
        ["fake-a", "fake-b"], manifest_path=SMOKE, n=4, output_dir=tmp_path, create_adapter=lambda name: _fake(tasks)
    )
    assert set(results) == {"fake-a", "fake-b"}
    payload = json.loads((tmp_path / "verdict-prior.json").read_text())
    assert payload["n"] == 4 and set(payload["models"]) == {"fake-a", "fake-b"}
    report = (tmp_path / "RESULTS.md").read_text()
    assert "## fake-a" in report and "AUROC" in report and "near_miss_shown" in report


def test_prior_kaggle_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_path = tmp_path / "plan"
    plan_path.write_text("experiment: prior\n", encoding="utf-8")
    plan = load_run_plan(plan_path)
    assert plan.manifest == "experiments/009-verdict-language-game/manifest.json"
    assert dict(plan.options) == {"models": "Qwen/Qwen2.5-0.5B", "n": "200"}
    plan_path.write_text(
        f"experiment: prior\nmanifest: {SMOKE}\nmodels: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-0.5B-Instruct\nn: 3\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    calls = {}

    def fake_run_models(models, **kwargs):
        calls["models"] = models
        calls.update(kwargs)
        return {}

    import onwordly.diagnostics.verdict_prior as module

    monkeypatch.setattr(module, "run_models", fake_run_models)
    output = execute_run_plan(plan, tmp_path / "results")
    assert output == tmp_path / "results" / "verdict-prior"
    assert calls["models"] == ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"] and calls["n"] == 3
    for bad in ("experiment: prior\nn: 0\n", "experiment: prior\nseeds_x: 1\n"):
        plan_path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_run_plan(plan_path)


def test_cli_uses_factory_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_run_models(models, **kwargs):
        seen["models"] = models
        seen.update(kwargs)

    import onwordly.diagnostics.verdict_prior as module

    monkeypatch.setattr(module, "run_models", fake_run_models)
    main(["--manifest", SMOKE, "--n", "5", "--out", str(tmp_path), "--model", "a", "--model", "b"])
    assert seen["models"] == ["a", "b"] and seen["n"] == 5
    main(["--manifest", SMOKE, "--out", str(tmp_path)])
    assert seen["models"] == ["Qwen/Qwen2.5-0.5B"]


def test_hf_continuation_logprob() -> None:
    pytest.importorskip("torch")
    from onwordly.models.huggingface import HuggingFaceCausalLMAdapter

    try:
        adapter = HuggingFaceCausalLMAdapter("sshleifer/tiny-gpt2", device="cpu", seed=0, max_new_tokens=4)
    except OSError:
        pytest.skip("model download unavailable")
    before = [p.detach().clone() for p in adapter.model.parameters()]
    right = adapter.continuation_logprob("Is it right?", "right")
    longer = adapter.continuation_logprob("Is it right?", "wrong: 12345")
    assert right < 0 and longer < 0
    assert all(p.detach().equal(b) for p, b in zip(adapter.model.parameters(), before))
