from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Iterable

from onwordly.experiments.arithmetic import DEFAULT_REGIMES, run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import ModelAdapter


def _number_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "runs": len(values),
        "mean": fmean(values),
        "stddev": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_suite(run_results: list[dict[str, object]]) -> dict[str, object]:
    if not run_results:
        raise ValueError("suite has no runs")

    regime_names = tuple(run_results[0].get("regime_order", DEFAULT_REGIMES))
    if any(tuple(run.get("regime_order", regime_names)) != regime_names for run in run_results):
        raise ValueError("suite runs do not share the same regime order")

    aggregate: dict[str, object] = {
        "runs": len(run_results),
        "regime_order": list(regime_names),
        "regimes": {},
    }

    for regime_name in regime_names:
        split_accuracies: dict[str, list[float]] = {
            "heldout": [],
            "withheld_prompts": [],
            "out_of_range": [],
        }
        training_tokens: list[float] = []
        examples_trained: list[float] = []
        total_generation_calls: list[float] = []
        training_seconds: list[float] = []
        threshold_values: dict[str, list[float]] = {
            "0.70": [],
            "0.80": [],
            "0.90": [],
            "0.95": [],
        }

        for run in run_results:
            regime = run["regimes"][regime_name]
            evaluation = regime["evaluation"]
            for split_name in split_accuracies:
                split_accuracies[split_name].append(float(evaluation[split_name]["accuracy"]))

            training = regime["training"]
            training_tokens.append(float(training["training_tokens"]))
            examples_trained.append(float(training["examples_trained"]))
            training_seconds.append(float(training["training_core_seconds"]))

            overhead = regime["measurement_overhead"]
            total_generation_calls.append(
                float(overhead["total_generation_calls_including_evaluation"])
            )

            tokens_to_threshold = regime["tokens_to_threshold"]
            for threshold, value in tokens_to_threshold.items():
                if value is not None:
                    threshold_values[threshold].append(float(value))

        threshold_summary: dict[str, object] = {}
        for threshold, values in threshold_values.items():
            threshold_summary[threshold] = {
                "reached_runs": len(values),
                "total_runs": len(run_results),
                "tokens_when_reached": _number_summary(values) if values else None,
            }

        aggregate["regimes"][regime_name] = {
            "accuracy": {
                split_name: _number_summary(values)
                for split_name, values in split_accuracies.items()
            },
            "training_tokens": _number_summary(training_tokens),
            "examples_trained": _number_summary(examples_trained),
            "training_core_seconds": _number_summary(training_seconds),
            "total_generation_calls_including_evaluation": _number_summary(
                total_generation_calls
            ),
            "tokens_to_threshold": threshold_summary,
        }

    return aggregate


def write_checkpoint_csv(
    run_results: list[dict[str, object]],
    seeds: list[int],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "regime", "scheduled_tokens", "actual_tokens", "accuracy"),
        )
        writer.writeheader()
        for seed, run in zip(seeds, run_results, strict=True):
            for regime_name, regime in run["regimes"].items():
                for checkpoint in regime["training"]["checkpoints"]:
                    writer.writerow(
                        {
                            "seed": seed,
                            "regime": regime_name,
                            "scheduled_tokens": checkpoint["scheduled_tokens"],
                            "actual_tokens": checkpoint["actual_tokens"],
                            "accuracy": checkpoint["evaluation"]["accuracy"],
                        }
                    )
    return destination


def run_suite(
    manifest: ArithmeticExperimentManifest,
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
    run_results: list[dict[str, object]] = []

    for seed in seed_list:
        seeded_manifest = replace(manifest, training_seed=seed)
        create_adapter = None
        if create_adapter_for_seed is not None:
            create_adapter = lambda seed=seed: create_adapter_for_seed(seed)

        run_results.append(
            run_experiment(
                seeded_manifest,
                output_dir=output / f"seed-{seed}",
                create_adapter=create_adapter,
            )
        )

    aggregate = aggregate_suite(run_results)
    suite_result = {
        "seeds": seed_list,
        "regime_order": list(DEFAULT_REGIMES),
        "aggregate": aggregate,
    }

    with (output / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(suite_result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    write_checkpoint_csv(run_results, seed_list, output / "checkpoints.csv")
    return suite_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated Onwordly arithmetic experiments")
    parser.add_argument(
        "--manifest",
        default="experiments/001-arithmetic-curriculum/manifest.json",
        help="Path to experiment manifest",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3303, 4404, 5505])
    parser.add_argument(
        "--output",
        default="results/001-arithmetic-curriculum-suite",
        help="Directory for repeated runs and aggregate results",
    )
    args = parser.parse_args()

    manifest = ArithmeticExperimentManifest.from_json(args.manifest)
    run_suite(manifest, seeds=args.seeds, output_dir=args.output)


if __name__ == "__main__":
    main()
