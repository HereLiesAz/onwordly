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


def test_experiment_007_single_is_prepared_but_suite_is_gated(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: 007\n"
        "mode: single\n"
        "manifest: experiments/007-program-process-supervision/smoke-manifest.json\n"
        "seeds: 3303\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.experiment == "007"
    assert plan.mode == "single"

    plan_path.write_text("experiment: 007\nmode: suite\nseeds: 3303,4404,5505\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gated"):
        load_run_plan(plan_path)


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
        ("007", "experiments/007-program-process-supervision/manifest.json"),
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


def test_baseline_plan_parses_options(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: baseline\n"
        "backend: kaggle\n"
        "models: Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-0.5B-Instruct\n"
        "chat_template: both\n"
        "per_split: 200\n"
        "run: 1\n",
        encoding="utf-8",
    )
    plan = load_run_plan(plan_path)
    assert plan.experiment == "baseline"
    assert dict(plan.options) == {
        "models": "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-0.5B-Instruct",
        "chat_template": "both",
        "per_split": "200",
    }


def test_baseline_plan_rejects_unknown_keys(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text("experiment: baseline\nchat_template: maybe\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_run_plan(plan_path)


def test_batch_plan_parses_jobs(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text(
        "experiment: batch\njobs: 001-suite, 002\nseeds: 3303,4404,5505\n", encoding="utf-8"
    )
    plan = load_run_plan(plan_path)
    assert plan.mode == "batch"
    assert dict(plan.options)["jobs"] == "001-suite,002"
    assert plan.seeds == (3303, 4404, 5505)


def test_batch_plan_rejects_unknown_job(tmp_path: Path) -> None:
    plan_path = tmp_path / ".kaggle-run"
    plan_path.write_text("experiment: batch\njobs: 003\n", encoding="utf-8")
    with pytest.raises(ValueError, match="batch jobs"):
        load_run_plan(plan_path)


def test_batch_units_cover_every_seed_and_regime(tmp_path: Path) -> None:
    from onwordly.experiments.kaggle import _job_units

    units, _ = _job_units("001-suite", (1, 2), tmp_path)
    assert len(units) == 6
    assert {unit[2] for unit in units} == {"static", "adaptive", "error-focused"}
    assert (tmp_path / "001-arithmetic-curriculum-suite" / "seed-2" / "manifest.json").exists()
    units, _ = _job_units("002", (1,), tmp_path)
    assert len(units) == 5
