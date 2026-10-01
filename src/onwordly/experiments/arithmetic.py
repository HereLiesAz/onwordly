from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable

from onwordly.curricula.adaptive import AdaptiveArithmeticCurriculum
from onwordly.datasets.arithmetic import build_static_arithmetic_dataset, write_arithmetic_jsonl
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import ModelAdapter
from onwordly.models.huggingface import HuggingFaceCausalLMAdapter
from onwordly.training.evaluation import EvaluationResult, evaluate_arithmetic
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
            seed=manifest.training_seed,
        )

    return create


def _adaptive_curriculum(manifest: ArithmeticExperimentManifest) -> AdaptiveArithmeticCurriculum:
    return AdaptiveArithmeticCurriculum(
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
    )


def _tokens_to_threshold(
    checkpoints: tuple[dict[str, object], ...],
) -> dict[str, int | None]:
    thresholds = (0.70, 0.80, 0.90, 0.95)
    result: dict[str, int | None] = {f"{threshold:.2f}": None for threshold in thresholds}

    for checkpoint in checkpoints:
        evaluation = checkpoint.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        accuracy = evaluation.get("accuracy")
        if not isinstance(accuracy, (int, float)):
            continue
        actual_tokens = checkpoint.get("actual_tokens")
        if not isinstance(actual_tokens, int):
            continue
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            if result[key] is None and accuracy >= threshold:
                result[key] = actual_tokens
    return result


def _serialize_evaluations(
    heldout: EvaluationResult,
    withheld_prompts: EvaluationResult,
    out_of_range: EvaluationResult,
) -> dict[str, object]:
    return {
        "heldout": heldout.to_dict(),
        "withheld_prompts": withheld_prompts.to_dict(),
        "out_of_range": out_of_range.to_dict(),
    }


def run_experiment(
    manifest: ArithmeticExperimentManifest,
    *,
    output_dir: str | Path,
    create_adapter: Callable[[], ModelAdapter] | None = None,
) -> dict[str, object]:
    manifest.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    static_tasks = build_static_arithmetic_dataset(
        seed=manifest.dataset_seed,
        size=manifest.static_dataset_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
    )
    heldout_tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed,
        size=manifest.evaluation_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    withheld_prompt_tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed + 1,
        size=manifest.generalization_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        prompt_styles=manifest.withheld_prompt_styles,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    out_of_range_tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed + 2,
        size=manifest.generalization_size,
        operations=manifest.operations,
        digit_levels=manifest.out_of_range_digit_levels,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    checkpoint_tasks = heldout_tasks[: manifest.checkpoint_evaluation_size]

    write_arithmetic_jsonl(static_tasks, output / "static-train.jsonl")
    write_arithmetic_jsonl(heldout_tasks, output / "evaluation-heldout.jsonl")
    write_arithmetic_jsonl(withheld_prompt_tasks, output / "evaluation-withheld-prompts.jsonl")
    write_arithmetic_jsonl(out_of_range_tasks, output / "evaluation-out-of-range.jsonl")

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
        checkpoint_generation_calls = 0
        checkpoint_seconds = 0.0

        def checkpoint_callback(
            scheduled_tokens: int,
            actual_tokens: int,
            *,
            _adapter: ModelAdapter = adapter,
        ) -> dict[str, object]:
            nonlocal checkpoint_generation_calls, checkpoint_seconds
            del scheduled_tokens, actual_tokens
            started = perf_counter()
            evaluation = evaluate_arithmetic(_adapter, checkpoint_tasks)
            checkpoint_seconds += perf_counter() - started
            checkpoint_generation_calls += len(checkpoint_tasks)
            return {"evaluation": evaluation.to_dict()}

        training = run_equal_token_training(
            regime=regime_name,
            adapter=adapter,
            source=source_factory(),
            token_budget=manifest.token_budget,
            seed=manifest.training_seed,
            checkpoint_interval_tokens=manifest.checkpoint_interval_tokens,
            checkpoint_callback=checkpoint_callback,
        )

        started = perf_counter()
        heldout = evaluate_arithmetic(adapter, heldout_tasks)
        withheld = evaluate_arithmetic(adapter, withheld_prompt_tasks)
        out_of_range = evaluate_arithmetic(adapter, out_of_range_tasks)
        final_evaluation_seconds = perf_counter() - started
        final_evaluation_calls = (
            len(heldout_tasks) + len(withheld_prompt_tasks) + len(out_of_range_tasks)
        )
        total_evaluation_calls = checkpoint_generation_calls + final_evaluation_calls

        regime_result = {
            "model": {
                "parameter_count": getattr(adapter, "parameter_count", None),
                "device": getattr(adapter, "device_name", None),
            },
            "training": training.to_dict(),
            "tokens_to_threshold": _tokens_to_threshold(training.checkpoints),
            "evaluation": _serialize_evaluations(heldout, withheld, out_of_range),
            "measurement_overhead": {
                "checkpoint_generation_calls": checkpoint_generation_calls,
                "final_evaluation_generation_calls": final_evaluation_calls,
                "total_evaluation_generation_calls": total_evaluation_calls,
                "checkpoint_seconds": checkpoint_seconds,
                "final_evaluation_seconds": final_evaluation_seconds,
                "total_evaluation_seconds": checkpoint_seconds + final_evaluation_seconds,
                "total_generation_calls_including_evaluation": (
                    training.generation_calls + total_evaluation_calls
                ),
            },
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
