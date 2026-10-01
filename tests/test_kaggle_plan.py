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


def test_experiment_003_suite_is_prepared(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: 003\n"
        "mode: suite\n"
        "manifest: experiments/003-symbolic-transformations/manifest.json\n"
        "seeds: 3303,4404,5505\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.experiment == "003"
    assert plan.mode == "suite"


def test_experiment_004_suite_is_prepared(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: 004\n"
        "mode: suite\n"
        "manifest: experiments/004-string-manipulation/manifest.json\n"
        "seeds: 3303,4404,5505\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.experiment == "004"
    assert plan.mode == "suite"


def test_experiment_005_suite_is_prepared(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: 005\n"
        "mode: suite\n"
        "manifest: experiments/005-program-execution/manifest.json\n"
        "seeds: 3303,4404,5505\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.experiment == "005"
    assert plan.mode == "suite"


def test_experiment_006_suite_is_prepared(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: 006\n"
        "mode: suite\n"
        "manifest: experiments/006-formal-logic/manifest.json\n"
        "seeds: 3303,4404,5505\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.experiment == "006"
    assert plan.mode == "suite"


@pytest.mark.parametrize(
    ("experiment", "expected"),
    (
        ("001", "experiments/001-arithmetic-curriculum/manifest.json"),
        ("002", "experiments/002-adaptive-ablation/manifest.json"),
        ("003", "experiments/003-symbolic-transformations/manifest.json"),
        ("004", "experiments/004-string-manipulation/manifest.json"),
        ("005", "experiments/005-program-execution/manifest.json"),
        ("006", "experiments/006-formal-logic/manifest.json"),
    ),
)
def test_default_manifest_matches_experiment(
    tmp_path: Path,
    experiment: str,
    expected: str,
) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        f"experiment: {experiment}\nmode: single\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.manifest == expected
