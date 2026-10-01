from pathlib import Path

import pytest

from onwordly.experiments.kaggle import load_run_plan


def test_load_single_kaggle_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: 001\n"
        "mode: single\n"
        "manifest: experiments/001-arithmetic-curriculum/manifest.json\n"
        "seeds: 3303,4404,5505\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.experiment == "001"
    assert plan.mode == "single"
    assert plan.seeds == (3303, 4404, 5505)


def test_experiment_002_suite_stays_gated(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: 002\nmode: suite\nseeds: 1,2,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gated"):
        load_run_plan(plan_path)
