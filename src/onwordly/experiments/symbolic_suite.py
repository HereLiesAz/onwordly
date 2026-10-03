from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Iterable

from onwordly.experiments.symbolic import SYMBOLIC_REGIMES, run_symbolic_experiment
from onwordly.experiments.symbolic_manifest import SymbolicExperimentManifest
from onwordly.models.base import ModelAdapter


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "runs": len(values),
        "mean": fmean(values),
        "stddev": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_symbolic_suite(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        raise ValueError("suite has no runs")
    result: dict[str, object] = {
        "runs": len(runs),
        "regime_order": list(SYMBOLIC_REGIMES),
        "regimes": {},
    }
    for regime_name in SYMBOLIC_REGIMES:
        split_values = {
            "heldout": [],
            "longer_sequences": [],
            "composition": [],
        }
        train_tokens: list[float] = []
        examples: list[float] = []
        core_seconds: list[float] = []
        wall_seconds: list[float] = []
        peak_memory: list[float] = []
        generation_seconds: list[float] = []
        unused_budget: list[float] = []
        repeated: list[float] = []
        for run in runs:
            regime = run["regimes"][regime_name]
            for split in split_values:
                split_values[split].append(
                    float(regime["evaluation"][split]["accuracy"])
                )
            training = regime["training"]
            train_tokens.append(float(training["training_tokens"]))
            examples.append(float(training["examples_trained"]))
            core_seconds.append(float(training["training_core_seconds"]))
            generation_seconds.append(float(training.get("generation_seconds", 0.0)))
            unused_budget.append(float(training["token_budget"] - training["training_tokens"]))
            repeated.append(float(training.get("repeated_examples", 0)))
            overhead = regime["measurement_overhead"]
            wall_seconds.append(float(overhead["regime_wall_seconds"]))
            peak = regime["model"].get("peak_memory_bytes")
            if peak is not None:
                peak_memory.append(float(peak))

        result["regimes"][regime_name] = {
            "accuracy": {
                split: _summary(values)
                for split, values in split_values.items()
            },
            "training_tokens": _summary(train_tokens),
            "examples_trained": _summary(examples),
            "training_core_seconds": _summary(core_seconds),
            "generation_seconds": _summary(generation_seconds),
            "unused_token_budget": _summary(unused_budget),
            "repeated_examples": _summary(repeated),
            "regime_wall_seconds": _summary(wall_seconds),
            "peak_memory_bytes": _summary(peak_memory) if peak_memory else None,
        }
    return result


def run_symbolic_suite(
    manifest: SymbolicExperimentManifest,
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
    runs: list[dict[str, object]] = []

    for seed in seed_list:
        seeded = replace(manifest, training_seed=seed)
        factory = None
        if create_adapter_for_seed is not None:
            factory = lambda seed=seed: create_adapter_for_seed(seed)
        runs.append(
            run_symbolic_experiment(
                seeded,
                output_dir=output / f"seed-{seed}",
                create_adapter=factory,
            )
        )

    aggregate = aggregate_symbolic_suite(runs)
    payload = {
        "seeds": seed_list,
        "regime_order": list(SYMBOLIC_REGIMES),
        "aggregate": aggregate,
    }
    (output / "aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated symbolic experiments")
    parser.add_argument(
        "--manifest",
        default="experiments/003-symbolic-transformations/manifest.json",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3303, 4404, 5505])
    parser.add_argument(
        "--output",
        default="results/003-symbolic-transformations-suite",
    )
    args = parser.parse_args()
    run_symbolic_suite(
        SymbolicExperimentManifest.from_json(args.manifest),
        seeds=args.seeds,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
