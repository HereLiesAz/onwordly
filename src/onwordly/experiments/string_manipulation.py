from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from onwordly.curricula.string_manipulation import AdaptiveStringCurriculum
from onwordly.datasets.string_manipulation import (
    build_static_string_dataset,
    write_string_jsonl,
)
from onwordly.experiments.string_manifest import StringExperimentManifest
from onwordly.models.base import ModelAdapter
from onwordly.models.huggingface import HuggingFaceCausalLMAdapter
from onwordly.training.harness import run_equal_token_training
from onwordly.training.string_evaluation import evaluate_string_tasks
from onwordly.training.string_sources import (
    ErrorFocusedStringSource,
    OnlineStringSource,
    StaticStringSource,
)
from onwordly.verifiers.string_manipulation import verify_string_answer

STRING_REGIMES: tuple[str, ...] = ("static", "adaptive", "error-focused")


def _factory(manifest: StringExperimentManifest) -> Callable[[], ModelAdapter]:
    def create() -> ModelAdapter:
        return HuggingFaceCausalLMAdapter(
            manifest.model_name,
            learning_rate=manifest.learning_rate,
            max_new_tokens=manifest.max_new_tokens,
            seed=manifest.training_seed,
        )
    return create


def _source(regime: str, manifest: StringExperimentManifest, static_tasks):
    if regime == "static":
        return StaticStringSource(static_tasks)
    curriculum = AdaptiveStringCurriculum(
        operations=manifest.operations,
        lengths=manifest.lengths,
        alphabet=manifest.alphabet,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
    )
    if regime == "adaptive":
        return OnlineStringSource(curriculum)
    if regime == "error-focused":
        return ErrorFocusedStringSource(
            curriculum,
            variants_per_failure=manifest.variants_per_failure,
        )
    raise ValueError(f"unknown string regime: {regime}")


def _sync(adapter: ModelAdapter) -> None:
    method = getattr(adapter, "synchronize", None)
    if callable(method):
        method()


def _close(adapter: ModelAdapter) -> None:
    method = getattr(adapter, "close", None)
    if callable(method):
        method()


def run_string_experiment(
    manifest: StringExperimentManifest,
    *,
    output_dir: str | Path,
    create_adapter: Callable[[], ModelAdapter] | None = None,
    regimes: Sequence[str] = STRING_REGIMES,
) -> dict[str, object]:
    manifest.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train = build_static_string_dataset(
        seed=manifest.dataset_seed,
        size=manifest.static_dataset_size,
        operations=manifest.operations,
        lengths=manifest.lengths,
        alphabet=manifest.alphabet,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
    )
    heldout = build_static_string_dataset(
        seed=manifest.evaluation_seed,
        size=manifest.evaluation_size,
        operations=manifest.operations,
        lengths=manifest.lengths,
        alphabet=manifest.alphabet,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    longer = build_static_string_dataset(
        seed=manifest.evaluation_seed + 1,
        size=manifest.evaluation_size,
        operations=manifest.operations,
        lengths=manifest.out_of_range_lengths,
        alphabet=manifest.alphabet,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    checkpoints = heldout[: manifest.checkpoint_evaluation_size]

    write_string_jsonl(train, output / "static-train.jsonl")
    write_string_jsonl(heldout, output / "evaluation-heldout.jsonl")
    write_string_jsonl(longer, output / "evaluation-longer-strings.jsonl")

    factory = create_adapter or _factory(manifest)
    result: dict[str, object] = {
        "manifest": asdict(manifest),
        "regime_order": list(regimes),
        "regimes": {},
    }

    for regime in regimes:
        wall_started = perf_counter()
        adapter = factory()
        checkpoint_calls = 0
        checkpoint_seconds = 0.0

        def checkpoint_callback(scheduled_tokens: int, actual_tokens: int):
            nonlocal checkpoint_calls, checkpoint_seconds
            del scheduled_tokens, actual_tokens
            _sync(adapter)
            started = perf_counter()
            evaluation = evaluate_string_tasks(adapter, checkpoints)
            _sync(adapter)
            checkpoint_seconds += perf_counter() - started
            checkpoint_calls += len(checkpoints)
            return {"evaluation": evaluation.to_dict()}

        training = run_equal_token_training(
            regime=regime,
            adapter=adapter,
            source=_source(regime, manifest, train),
            token_budget=manifest.token_budget,
            seed=manifest.training_seed,
            verifier=verify_string_answer,
            checkpoint_interval_tokens=manifest.checkpoint_interval_tokens,
            checkpoint_callback=checkpoint_callback,
        )
        _sync(adapter)
        eval_started = perf_counter()
        heldout_result = evaluate_string_tasks(adapter, heldout)
        longer_result = evaluate_string_tasks(adapter, longer)
        _sync(adapter)
        eval_seconds = perf_counter() - eval_started
        peak_reader = getattr(adapter, "peak_memory_bytes", None)
        peak_memory = peak_reader() if callable(peak_reader) else None

        regime_result = {
            "model": {
                "parameter_count": getattr(adapter, "parameter_count", None),
                "device": getattr(adapter, "device_name", None),
                "peak_memory_bytes": peak_memory,
            },
            "training": training.to_dict(),
            "evaluation": {
                "heldout": heldout_result.to_dict(),
                "longer_strings": longer_result.to_dict(),
            },
            "measurement_overhead": {
                "checkpoint_generation_calls": checkpoint_calls,
                "final_evaluation_generation_calls": len(heldout) + len(longer),
                "checkpoint_seconds": checkpoint_seconds,
                "final_evaluation_seconds": eval_seconds,
                "regime_wall_seconds": perf_counter() - wall_started,
            },
        }
        result["regimes"][regime] = regime_result
        (output / f"{regime}.json").write_text(
            json.dumps(regime_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _close(adapter)

    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained string experiment")
    parser.add_argument(
        "--manifest",
        default="experiments/004-string-manipulation/manifest.json",
    )
    parser.add_argument(
        "--output",
        default="results/004-string-manipulation",
    )
    args = parser.parse_args()
    run_string_experiment(
        StringExperimentManifest.from_json(args.manifest),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
