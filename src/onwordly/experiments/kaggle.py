from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from onwordly.experiments.ablation import ABLATION_REGIMES
from onwordly.experiments.arithmetic import run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.experiments.suite import run_suite
from onwordly.reporting.arithmetic import render_result
from onwordly.experiments.symbolic import run_symbolic_experiment
from onwordly.experiments.symbolic_manifest import SymbolicExperimentManifest
from onwordly.experiments.symbolic_suite import run_symbolic_suite
from onwordly.reporting.symbolic import render_symbolic_result
from onwordly.experiments.string_manipulation import run_string_experiment
from onwordly.experiments.string_manifest import StringExperimentManifest
from onwordly.experiments.string_suite import run_string_suite
from onwordly.reporting.string_manipulation import render_string_result
from onwordly.experiments.program_execution import run_program_experiment
from onwordly.experiments.program_manifest import ProgramExperimentManifest
from onwordly.experiments.program_suite import run_program_suite
from onwordly.reporting.program_execution import render_program_result
from onwordly.experiments.formal_logic import run_logic_experiment
from onwordly.experiments.logic_manifest import LogicExperimentManifest
from onwordly.experiments.logic_suite import run_logic_suite
from onwordly.reporting.formal_logic import render_logic_result
from onwordly.experiments.process_supervision import run_process_supervision_experiment
from onwordly.reporting.process_supervision import render_process_supervision_result


DEFAULT_MANIFESTS: dict[str, str] = {
    "001": "experiments/001-arithmetic-curriculum/manifest.json",
    "002": "experiments/002-adaptive-ablation/manifest.json",
    "003": "experiments/003-symbolic-transformations/manifest.json",
    "004": "experiments/004-string-manipulation/manifest.json",
    "005": "experiments/005-program-execution/manifest.json",
    "006": "experiments/006-formal-logic/manifest.json",
    "007": "experiments/007-program-process-supervision/manifest.json",
}


@dataclass(frozen=True, slots=True)
class KaggleRunPlan:
    experiment: str
    mode: str
    manifest: str
    seeds: tuple[int, ...]


def load_run_plan(path: str | Path) -> KaggleRunPlan:
    source = Path(path)
    values: dict[str, str] = {}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid run-plan line {line_number}: {raw}")
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    experiment = values.get("experiment", "001")
    mode = values.get("mode", "single")
    manifest = values.get("manifest", DEFAULT_MANIFESTS.get(experiment, ""))
    seeds_text = values.get("seeds", "3303,4404,5505")
    seeds = tuple(int(value.strip()) for value in seeds_text.split(",") if value.strip())

    if experiment not in DEFAULT_MANIFESTS:
        raise ValueError(f"unsupported experiment: {experiment}")
    if mode not in {"single", "suite"}:
        raise ValueError(f"unsupported mode: {mode}")
    if not seeds:
        raise ValueError("at least one seed is required")
    if experiment == "002" and mode == "suite":
        raise ValueError("Experiment 002 suite is gated until Experiment 001 is interpreted")
    if experiment == "007" and mode == "suite":
        raise ValueError("Experiment 007 suite is gated until its single-run accounting is inspected")

    return KaggleRunPlan(
        experiment=experiment,
        mode=mode,
        manifest=manifest,
        seeds=seeds,
    )


def execute_run_plan(plan: KaggleRunPlan, output_root: str | Path) -> Path:
    root = Path(output_root)

    if plan.experiment == "007":
        program_manifest = ProgramExperimentManifest.from_json(plan.manifest)
        output = root / "007-program-process-supervision"
        run_process_supervision_experiment(program_manifest, output_dir=output)
        result_path = output / "summary.json"
        (output / "RESULTS.md").write_text(
            render_process_supervision_result(result_path),
            encoding="utf-8",
        )
        return output

    if plan.experiment == "006":
        logic_manifest = LogicExperimentManifest.from_json(plan.manifest)
        if plan.mode == "single":
            output = root / "006-formal-logic"
            run_logic_experiment(logic_manifest, output_dir=output)
            result_path = output / "summary.json"
        else:
            output = root / "006-formal-logic-suite"
            run_logic_suite(logic_manifest, seeds=plan.seeds, output_dir=output)
            result_path = output / "aggregate.json"
        (output / "RESULTS.md").write_text(
            render_logic_result(result_path),
            encoding="utf-8",
        )
        return output

    if plan.experiment == "005":
        program_manifest = ProgramExperimentManifest.from_json(plan.manifest)
        if plan.mode == "single":
            output = root / "005-program-execution"
            run_program_experiment(program_manifest, output_dir=output)
            result_path = output / "summary.json"
        else:
            output = root / "005-program-execution-suite"
            run_program_suite(program_manifest, seeds=plan.seeds, output_dir=output)
            result_path = output / "aggregate.json"
        (output / "RESULTS.md").write_text(
            render_program_result(result_path),
            encoding="utf-8",
        )
        return output

    if plan.experiment == "004":
        string_manifest = StringExperimentManifest.from_json(plan.manifest)
        if plan.mode == "single":
            output = root / "004-string-manipulation"
            run_string_experiment(string_manifest, output_dir=output)
            result_path = output / "summary.json"
        else:
            output = root / "004-string-manipulation-suite"
            run_string_suite(
                string_manifest,
                seeds=plan.seeds,
                output_dir=output,
            )
            result_path = output / "aggregate.json"
        (output / "RESULTS.md").write_text(
            render_string_result(result_path),
            encoding="utf-8",
        )
        return output

    if plan.experiment == "003":
        symbolic_manifest = SymbolicExperimentManifest.from_json(plan.manifest)
        if plan.mode == "single":
            output = root / "003-symbolic-transformations"
            run_symbolic_experiment(symbolic_manifest, output_dir=output)
            result_path = output / "summary.json"
        else:
            output = root / "003-symbolic-transformations-suite"
            run_symbolic_suite(
                symbolic_manifest,
                seeds=plan.seeds,
                output_dir=output,
            )
            result_path = output / "aggregate.json"
        (output / "RESULTS.md").write_text(
            render_symbolic_result(result_path),
            encoding="utf-8",
        )
        return output

    manifest = ArithmeticExperimentManifest.from_json(plan.manifest)

    if plan.experiment == "001" and plan.mode == "single":
        output = root / "001-arithmetic-curriculum"
        run_experiment(manifest, output_dir=output)
        result_path = output / "summary.json"
    elif plan.experiment == "001" and plan.mode == "suite":
        output = root / "001-arithmetic-curriculum-suite"
        run_suite(manifest, seeds=plan.seeds, output_dir=output)
        result_path = output / "aggregate.json"
    elif plan.experiment == "002":
        output = root / "002-adaptive-ablation"
        run_experiment(manifest, output_dir=output, regimes=ABLATION_REGIMES)
        result_path = output / "summary.json"
    else:
        raise ValueError(f"unsupported plan: {plan}")

    (output / "RESULTS.md").write_text(
        render_result(result_path) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute an Onwordly Kaggle run plan")
    parser.add_argument("--plan", default=".kaggle-run")
    parser.add_argument("--output-root", default="/kaggle/working/results")
    args = parser.parse_args()

    plan = load_run_plan(args.plan)
    execute_run_plan(plan, args.output_root)


if __name__ == "__main__":
    main()
