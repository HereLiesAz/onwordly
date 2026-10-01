from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Iterable

from onwordly.experiments.formal_logic import LOGIC_REGIMES, run_logic_experiment
from onwordly.experiments.logic_manifest import LogicExperimentManifest
from onwordly.models.base import ModelAdapter


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "runs": len(values),
        "mean": fmean(values),
        "stddev": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_logic_suite(
    manifest: LogicExperimentManifest,
    *,
    seeds: Iterable[int],
    output_dir: str | Path,
    create_adapter_for_seed: Callable[[int], ModelAdapter] | None = None,
) -> dict[str, object]:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seed_list:
        seeded = replace(manifest, training_seed=seed)
        factory = None
        if create_adapter_for_seed is not None:
            factory = lambda seed=seed: create_adapter_for_seed(seed)
        runs.append(
            run_logic_experiment(
                seeded,
                output_dir=output / f"seed-{seed}",
                create_adapter=factory,
            )
        )

    aggregate = {"runs": len(runs), "regime_order": list(LOGIC_REGIMES), "regimes": {}}
    for regime_name in LOGIC_REGIMES:
        heldout = []
        deeper = []
        tokens = []
        examples = []
        seconds = []
        for run in runs:
            regime = run["regimes"][regime_name]
            heldout.append(float(regime["evaluation"]["heldout"]["accuracy"]))
            deeper.append(float(regime["evaluation"]["deeper_formulas"]["accuracy"]))
            tokens.append(float(regime["training"]["training_tokens"]))
            examples.append(float(regime["training"]["examples_trained"]))
            seconds.append(float(regime["measurement_overhead"]["regime_wall_seconds"]))
        aggregate["regimes"][regime_name] = {
            "accuracy": {"heldout": _summary(heldout), "deeper_formulas": _summary(deeper)},
            "training_tokens": _summary(tokens),
            "examples_trained": _summary(examples),
            "regime_wall_seconds": _summary(seconds),
        }

    payload = {"seeds": seed_list, "regime_order": list(LOGIC_REGIMES), "aggregate": aggregate}
    (output / "aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated formal logic experiments")
    parser.add_argument("--manifest", default="experiments/006-formal-logic/manifest.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3303, 4404, 5505])
    parser.add_argument("--output", default="results/006-formal-logic-suite")
    args = parser.parse_args()
    run_logic_suite(
        LogicExperimentManifest.from_json(args.manifest),
        seeds=args.seeds,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
