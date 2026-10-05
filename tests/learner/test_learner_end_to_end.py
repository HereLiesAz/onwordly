"""Tiny CPU end-to-end run of Experiment 000 and its Kaggle plan."""
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from onwordly.learner.experiment import run_learner_experiment
from onwordly.learner.kaggle import load_plan
from onwordly.learner.manifest import ARMS, LearnerManifest


def test_tiny_end_to_end(tmp_path) -> None:
    manifest = LearnerManifest.from_json("experiments/000-onwordly-learner/smoke-manifest.json")
    manifest = replace(manifest, train_steps=3, eval_size=20, train_size=60)
    summary = run_learner_experiment(manifest, output_dir=tmp_path, device=torch.device("cpu"))
    assert set(summary["arms"]) == set(ARMS)
    budget = summary["matched_budget"]
    params = {arm: r["parameters"] for arm, r in summary["arms"].items()}
    assert abs(params["plain"] - params["learner"]) / params["learner"] < 0.05
    for arm, result in summary["arms"].items():
        training = result["training"]
        assert training["optimizer_steps"] == budget["optimizer_steps"] == 3
        assert training["training_problems"] == budget["training_problems"]
        assert len(result["solve"]["per_step_accuracy"]) == (1 if arm in ("plain", "handcoded") else manifest.revision_steps)
    for arm in ("learner", "learner-flat", "learner-no-memory", "learner-no-trust", "handcoded"):
        challenge = summary["arms"][arm]["challenge"]
        assert set(challenge["by_corrector"]) == {"A", "B", "C"}
        assert not challenge["by_corrector"]["C"]["seen_in_training"]
        memory = summary["arms"][arm]["memory"]
        assert memory["chain_verified"]
        # Ledger saw training episodes only: one self + one corrector update each.
        assert memory["record_counts"]["trust_update"] == 2 * budget["training_problems"]
        assert "C" not in memory["ledger"]["corrector"]
    assert (tmp_path / "RESULTS.md").read_text().startswith("# Experiment 000")
    assert json.loads((tmp_path / "summary.json").read_text())["experiment"] == "000-onwordly-learner"


def test_kaggle_plan(tmp_path) -> None:
    path = tmp_path / ".kaggle-run"
    path.write_text("experiment: 000\nmode: single\narms: learner,plain\n")
    plan = load_plan(path)
    assert plan.manifest.endswith("000-onwordly-learner/manifest.json") and plan.arms == ("learner", "plain")
    path.write_text("experiment: 000\nmode: suite\n")
    with pytest.raises(ValueError, match="gated"):
        load_plan(path)
    path.write_text("experiment: 010\n")
    with pytest.raises(ValueError):
        load_plan(path)
