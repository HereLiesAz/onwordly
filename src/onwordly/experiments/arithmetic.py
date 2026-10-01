from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from onwordly.curricula.adaptive import AdaptiveArithmeticCurriculum
from onwordly.datasets.arithmetic import build_static_arithmetic_dataset, write_arithmetic_jsonl
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import ModelAdapter
from onwordly.models.huggingface import HuggingFaceCausalLMAdapter
from onwordly.training.evaluation import evaluate_arithmetic
from onwordly.training.harness import run_equal_token_training
from onwordly.training.sources import (
    AdaptiveArithmeticSource,
    ErrorFocusedArithmeticSource,
    StaticArithmeticSource,
)


def _adapter_factory(manifest: ArithmeticExperimentManifest) -> Callable[[], ModelAdapter]:
    def create() -> ModelAdapter:
        return HuggingFaceCausalLMAdapter(
            manifest.model_name,
            learning_rate=manifest.learning_rate,
            max_new_tokens=manifest.max_new_tokens,
        )

    return create


def _adaptive_curriculum(manifest: ArithmeticExperimentManifest) -> AdaptiveArithmeticCurriculum:
    return AdaptiveArithmeticCurriculum(
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
    )


def run_experiment(
    manifest: ArithmeticExperimentManifest,
    *,
    output_dir: str | Path,
    create_adapter: Callable[[], ModelAdapter] | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    static_tasks = build_static_arithmetic_dataset(
        seed=manifest.dataset_seed,
        size=manifest.static_dataset_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
    )
    evaluation_tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed,
        size=manifest.evaluation_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
    )
    write_arithmetic_jsonl(static_tasks, output / "static-train.jsonl")
    write_arithmetic_jsonl(evaluation_tasks, output / "evaluation.jsonl")

    factory = create_adapter or _adapter_factory(manifest)
    regimes = (
        ("static", lambda: StaticArithmeticSource(static_tasks)),
        ("adaptive", lambda: AdaptiveArithmeticSource(_adaptive_curriculum(manifest))),
        (
            "error-focused",
            lambda: ErrorFocusedArithmeticSource(
                _adaptive_curriculum(manifest),
                variants_per_failure=manifest.variants_per_failure,
            ),
        ),
    )

    results: dict[str, object] = {
        "manifest": asdict(manifest),
        "regimes": {},
    }

    for regime_name, source_factory in regimes:
        adapter = factory()
        training = run_equal_token_training(
            regime=regime_name,
            adapter=adapter,
            source=source_factory(),
            token_budget=manifest.token_budget,
            seed=manifest.training_seed,
        )
        evaluation = evaluate_arithmetic(adapter, evaluation_tasks)
        regime_result = {
            "training": training.to_dict(),
            "evaluation": evaluation.to_dict(),
        }
        results["regimes"][regime_name] = regime_result
        with (output / f"{regime_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(regime_result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        del adapter

    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Onwordly arithmetic curriculum experiment")
    parser.add_argument(
        "--manifest",
        default="experiments/001-arithmetic-curriculum/manifest.json",
        help="Path to experiment manifest",
    )
    parser.add_argument(
        "--output",
        default="results/001-arithmetic-curriculum",
        help="Directory for frozen datasets and result JSON",
    )
    args = parser.parse_args()

    manifest = ArithmeticExperimentManifest.from_json(args.manifest)
    run_experiment(manifest, output_dir=args.output)


if __name__ == "__main__":
    main()
