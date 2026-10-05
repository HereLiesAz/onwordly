"""Standalone Kaggle entry point for Experiment 000 (no dependency on the
archived LLM experiment runner).

Plan file (``.kaggle-run``), same ``key: value`` format as before:

    experiment: 000
    manifest: experiments/000-onwordly-learner/manifest.json   # optional
    arms: learner,plain,handcoded                               # optional

``python -m onwordly.learner.kaggle --plan .kaggle-run --output-root /kaggle/working/results``
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MANIFEST = "experiments/000-onwordly-learner/manifest.json"
KNOWN_KEYS = {"experiment", "mode", "manifest", "arms", "backend", "accelerator", "run", "seeds"}


@dataclass(frozen=True, slots=True)
class LearnerPlan:
    experiment: str
    manifest: str
    arms: tuple[str, ...]


def load_plan(path: str | Path) -> LearnerPlan:
    values: dict[str, str] = {}
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid run-plan line {number}: {raw}")
        key, value = line.split(":", 1)
        values[key.strip()] = value.split("#", 1)[0].strip()
    if values.get("experiment") != "000":
        raise ValueError("this entry point runs experiment 000 only")
    if values.get("mode", "single") != "single":
        raise ValueError("Experiment 000 suite is gated until a single run is inspected")
    unknown = set(values) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"unsupported plan keys: {sorted(unknown)}")
    arms = tuple(a.strip() for a in values.get("arms", "").split(",") if a.strip())
    return LearnerPlan("000", values.get("manifest", DEFAULT_MANIFEST), arms)


def execute(plan: LearnerPlan, output_root: str | Path) -> Path:
    from onwordly.learner.experiment import run_learner_experiment
    from onwordly.learner.manifest import LearnerManifest

    output = Path(output_root) / "000-onwordly-learner"
    run_learner_experiment(LearnerManifest.from_json(plan.manifest), output_dir=output, arms=plan.arms or None)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 000 from a Kaggle run plan")
    parser.add_argument("--plan", default=".kaggle-run")
    parser.add_argument("--output-root", default="/kaggle/working/results")
    args = parser.parse_args(argv)
    output = execute(load_plan(args.plan), args.output_root)
    print(f"results archive: {shutil.make_archive(str(output), 'gztar', output)}", flush=True)


if __name__ == "__main__":
    main()
