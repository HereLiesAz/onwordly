from __future__ import annotations

import argparse
import json
from typing import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from onwordly.experiments.ablation import ABLATION_REGIMES
from onwordly.experiments.arithmetic import DEFAULT_REGIMES, run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.experiments.multi_gpu import gpu_count, run_parallel, run_units
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
    "008": "experiments/008-corrective-language-game/manifest.json",
    "009": "experiments/009-verdict-language-game/manifest.json",
    "010": "experiments/010-verdict-repair/manifest.json",
    "011": "experiments/011-verdict-constrained-strings/manifest.json",
}

# Verdict prior diagnostic (no training): default model and problem count.
PRIOR_DEFAULT_MODELS = "Qwen/Qwen2.5-0.5B"
PRIOR_DEFAULT_N = "200"

# Experiment 010: repair levers for 009's accept collapse (budget,
# answer-before-verdict ordering, on-policy reward, graded two-turn self-check
# reward with its binary ablation) against static and 009's verdict-synthetic
# arm, rerun for seed comparability.
VERDICT_REPAIR_REGIMES: tuple[str, ...] = (
    "static",
    "verdict-synthetic",
    "verdict-dense",
    "solve-judge-synthetic",
    "verdict-rl",
    "verdict-rl-graded",
    "selfcheck-rl-binary",
    # Fallible-challenge test of earned self-trust (on-policy, graded) and its SFT control.
    "challenge-rl-graded",
    "challenge-sft",
)

# Experiment 009: static reference, two verdict arms, and 008's synthetic
# corrective arm to show that copying fails a balanced verdict test.
VERDICT_REGIMES: tuple[str, ...] = (
    "static",
    "verdict-synthetic",
    "verdict-mixed",
    "corrective-synthetic",
)

# Experiment 008: static and error-focused as references, two corrective arms.
CORRECTIVE_REGIMES: tuple[str, ...] = (
    "static",
    "corrective-own",
    "corrective-synthetic",
    "error-focused",
)


@dataclass(frozen=True, slots=True)
class KaggleRunPlan:
    experiment: str
    mode: str
    manifest: str
    seeds: tuple[int, ...]
    # Extra plan keys, used by the "baseline" plan (models, chat_template, per_split).
    options: tuple[tuple[str, str], ...] = ()


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
    # Seeds drive suite mode only; single mode uses the manifest training_seed.
    seeds_text = values.get("seeds", "3303,4404,5505")
    seeds = tuple(int(value.strip()) for value in seeds_text.split(",") if value.strip())

    if experiment == "batch":
        jobs = tuple(job.strip() for job in values.get("jobs", "").split(",") if job.strip())
        unknown = set(jobs) - set(BATCH_JOBS)
        if not jobs or unknown:
            raise ValueError(f"batch jobs must be chosen from {sorted(BATCH_JOBS)}; got {jobs}")
        return KaggleRunPlan(
            experiment=experiment,
            mode="batch",
            manifest="",
            seeds=seeds,
            options=(("jobs", ",".join(jobs)),),
        )
    if experiment == "baseline":
        known = {"experiment", "mode", "backend", "accelerator", "run", "models", "chat_template", "per_split"}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unsupported baseline plan keys: {sorted(unknown)}")
        if values.get("chat_template", "no") not in {"no", "yes", "both"}:
            raise ValueError("chat_template must be no, yes or both")
        return KaggleRunPlan(
            experiment=experiment,
            mode="single",
            manifest="",
            seeds=seeds,
            options=tuple(
                (key, values[key]) for key in ("models", "chat_template", "per_split") if key in values
            ),
        )
    if experiment == "prior":
        # Untrained verdict prior on 009's held-out problems; one GPU, minutes.
        known = {"experiment", "mode", "backend", "accelerator", "run", "manifest", "models", "n"}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unsupported prior plan keys: {sorted(unknown)}")
        n = values.get("n", PRIOR_DEFAULT_N)
        if not n.isdigit() or int(n) < 1:
            raise ValueError("n must be a positive integer")
        return KaggleRunPlan(
            experiment=experiment,
            mode="single",
            manifest=values.get("manifest", DEFAULT_MANIFESTS["009"]),
            seeds=seeds,
            options=(("models", values.get("models", PRIOR_DEFAULT_MODELS)), ("n", n)),
        )
    if experiment not in DEFAULT_MANIFESTS:
        raise ValueError(f"unsupported experiment: {experiment}")
    if mode not in {"single", "suite"}:
        raise ValueError(f"unsupported mode: {mode}")
    if not seeds:
        raise ValueError("at least one seed is required")
    if experiment == "002" and mode == "suite":
        raise ValueError("Experiment 002 suite is gated until Experiment 001 is interpreted")
    if experiment == "008" and manifest:
        # A 001 manifest trains the same regimes but silently skips the
        # corrective evaluation, which is the point of 008.
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if not payload.get("corrective_evaluation_size"):
            raise ValueError(
                f"Experiment 008 needs a manifest with corrective_evaluation_size > 0; "
                f"{manifest} has none (use {DEFAULT_MANIFESTS['008']})"
            )
    if experiment in ("009", "010") and manifest:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if not payload.get("verdict_evaluation_size"):
            raise ValueError(
                f"Experiment {experiment} needs a manifest with verdict_evaluation_size > 0; "
                f"{manifest} has none (use {DEFAULT_MANIFESTS[experiment]})"
            )
    if experiment == "011" and mode == "suite":
        raise ValueError("Experiment 011 suite is gated until a single run is inspected")
    if experiment == "010" and mode == "suite":
        raise ValueError("Experiment 010 suite is gated until a single run is inspected")
    if experiment == "009" and mode == "suite":
        raise ValueError("Experiment 009 suite is gated until a single run is inspected")
    if experiment == "008" and mode == "suite":
        raise ValueError("Experiment 008 suite is gated until a single run is inspected")
    if experiment == "007" and mode == "suite":
        raise ValueError("Experiment 007 suite is gated until its single-run accounting is inspected")

    return KaggleRunPlan(
        experiment=experiment,
        mode=mode,
        manifest=manifest,
        seeds=seeds,
    )


def _execute_baseline(plan: KaggleRunPlan, root: Path) -> Path:
    """Untrained-model baselines: build frozen eval data, then score each model."""
    from onwordly.diagnostics import audit, baseline

    options = dict(plan.options)
    models = [model.strip() for model in options.get("models", "Qwen/Qwen2.5-0.5B").split(",") if model.strip()]
    chat = options.get("chat_template", "no")
    variants = {"no": [False], "yes": [True], "both": [False, True]}[chat]
    per_split = options.get("per_split", "200")

    output = root / "baseline"
    data = output / "data"
    audit.main(["--out", str(data), "--report", str(output / "dataset-audit.md")])
    for model in models:
        for use_chat in variants:
            label = model.replace("/", "--") + ("-chat" if use_chat else "-raw")
            args = ["--data", str(data), "--model", model, "--per-split", per_split,
                    "--report", str(output / f"baseline-{label}.md")]
            if use_chat:
                args.append("--chat-template")
            baseline.main(args)
            (data / "baseline.json").rename(output / f"baseline-{label}.json")
    return output


# Arithmetic jobs a batch plan can queue across GPUs: name -> manifest.
BATCH_JOBS: dict[str, str] = {
    "001": DEFAULT_MANIFESTS["001"],
    "001-suite": DEFAULT_MANIFESTS["001"],
    "002": DEFAULT_MANIFESTS["002"],
    "008": DEFAULT_MANIFESTS["008"],
    "009": DEFAULT_MANIFESTS["009"],
    "010": DEFAULT_MANIFESTS["010"],
}


def _job_units(job: str, seeds: tuple[int, ...], root: Path) -> tuple[list[tuple[str, str, str]], Callable[[], Path]]:
    """Regime-level work units for one arithmetic job, and how to assemble its report."""
    manifest_path = BATCH_JOBS[job]
    manifest = ArithmeticExperimentManifest.from_json(manifest_path)
    if job == "001-suite":
        output = root / "001-arithmetic-curriculum-suite"
        units = []
        for seed in seeds:
            seed_dir = output / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            seeded_path = seed_dir / "manifest.json"
            seeded_path.write_text(
                json.dumps(asdict(replace(manifest, training_seed=seed)), indent=2) + "\n",
                encoding="utf-8",
            )
            units += [(str(seeded_path), str(seed_dir), regime) for regime in DEFAULT_REGIMES]

        def finish() -> Path:
            run_suite(manifest, seeds=seeds, output_dir=output)
            (output / "RESULTS.md").write_text(render_result(output / "aggregate.json") + "\n", encoding="utf-8")
            return output

        return units, finish
    regimes = {"002": ABLATION_REGIMES, "008": CORRECTIVE_REGIMES, "009": VERDICT_REGIMES, "010": VERDICT_REPAIR_REGIMES}.get(job, DEFAULT_REGIMES)
    output = root / {
        "002": "002-adaptive-ablation",
        "008": "008-corrective-language-game",
        "009": "009-verdict-language-game",
        "010": "010-verdict-repair",
    }.get(job, "001-arithmetic-curriculum")
    units = [(manifest_path, str(output), regime) for regime in regimes]

    def finish() -> Path:
        run_experiment(manifest, output_dir=output, regimes=regimes)
        (output / "RESULTS.md").write_text(render_result(output / "summary.json") + "\n", encoding="utf-8")
        return output

    return units, finish


def _execute_batch(plan: KaggleRunPlan, root: Path) -> Path:
    """Queue every regime of every job over all visible GPUs, then assemble reports."""
    jobs = dict(plan.options)["jobs"].split(",")
    planned = [_job_units(job, plan.seeds, root) for job in jobs]
    # Interleave jobs so each GPU sees a mix rather than one job hogging both.
    queues = [list(units) for units, _ in planned]
    units: list[tuple[str, str, str]] = []
    while any(queues):
        for queue in queues:
            if queue:
                units.append(queue.pop(0))
    run_units(units, devices=max(1, gpu_count()))
    for _, finish in planned:
        finish()
    return root


def execute_run_plan(plan: KaggleRunPlan, output_root: str | Path) -> Path:
    root = Path(output_root)

    if plan.experiment == "batch":
        return _execute_batch(plan, root)

    if plan.experiment == "baseline":
        return _execute_baseline(plan, root)

    if plan.experiment == "prior":
        from onwordly.diagnostics.verdict_prior import run_models

        options = dict(plan.options)
        output = root / "verdict-prior"
        run_models(
            [model.strip() for model in options["models"].split(",") if model.strip()],
            manifest_path=plan.manifest,
            n=int(options["n"]),
            output_dir=output,
        )
        return output

    if plan.experiment == "011":
        from onwordly.experiments.constrained_strings import (
            CONSTRAINED_VERDICT_REGIMES,
            ConstrainedExperimentManifest,
            render_constrained_result,
            run_constrained_experiment,
        )

        output = root / "011-verdict-constrained-strings"
        # One worker process per regime, queued over every visible GPU; the
        # in-process call then loads the finished regimes and writes the summary.
        run_units(
            [(plan.manifest, str(output), regime) for regime in CONSTRAINED_VERDICT_REGIMES],
            devices=max(1, gpu_count()),
            module="onwordly.experiments.constrained_strings",
        )
        run_constrained_experiment(ConstrainedExperimentManifest.from_json(plan.manifest), output_dir=output)
        (output / "RESULTS.md").write_text(render_constrained_result(output / "summary.json"), encoding="utf-8")
        return output

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
        # Regimes are independent; spread them over every visible GPU.
        run_parallel(plan.manifest, output_dir=output)
        result_path = output / "summary.json"
    elif plan.experiment == "001" and plan.mode == "suite":
        units, finish = _job_units("001-suite", plan.seeds, root)
        run_units(units, devices=max(1, gpu_count()))
        output = finish()
        return output
    elif plan.experiment == "008":
        output = root / "008-corrective-language-game"
        run_parallel(plan.manifest, output_dir=output, regimes=CORRECTIVE_REGIMES)
        result_path = output / "summary.json"
    elif plan.experiment == "009":
        output = root / "009-verdict-language-game"
        run_parallel(plan.manifest, output_dir=output, regimes=VERDICT_REGIMES)
        result_path = output / "summary.json"
    elif plan.experiment == "010":
        output = root / "010-verdict-repair"
        run_parallel(plan.manifest, output_dir=output, regimes=VERDICT_REPAIR_REGIMES)
        result_path = output / "summary.json"
    elif plan.experiment == "002":
        output = root / "002-adaptive-ablation"
        run_parallel(plan.manifest, output_dir=output, regimes=ABLATION_REGIMES)
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
    output = execute_run_plan(plan, args.output_root)
    # One archive per run so the central finalizer can attach it to a GitHub
    # release with a simple glob (Kaggle keeps /kaggle/working as kernel output).
    import shutil

    archive = shutil.make_archive(str(output), "gztar", output)
    print(f"results archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
