from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from onwordly.curricula.program_execution import AdaptiveProgramCurriculum
from onwordly.datasets.program_execution import (
    build_program_transition_dataset,
    build_static_program_dataset,
    write_program_jsonl,
)
from onwordly.experiments.program_manifest import ProgramExperimentManifest
from onwordly.models.base import ModelAdapter
from onwordly.models.huggingface import HuggingFaceCausalLMAdapter
from onwordly.training.harness import run_equal_token_training
from onwordly.training.program_evaluation import evaluate_programs
from onwordly.training.program_sources import (
    ErrorFocusedProgramSource,
    OnlineProgramSource,
    StaticProgramSource,
)
from onwordly.verifiers.program_execution import verify_program_answer

PROGRAM_REGIMES: tuple[str, ...] = ("static", "adaptive", "error-focused")


def _factory(manifest: ProgramExperimentManifest) -> Callable[[], ModelAdapter]:
    def create() -> ModelAdapter:
        return HuggingFaceCausalLMAdapter(
            manifest.model_name,
            learning_rate=manifest.learning_rate,
            max_new_tokens=manifest.max_new_tokens,
            seed=manifest.training_seed,
        )
    return create


def _source(regime: str, manifest: ProgramExperimentManifest, static_tasks):
    if regime == "static":
        return StaticProgramSource(static_tasks)
    curriculum = AdaptiveProgramCurriculum(
        lengths=manifest.lengths,
        operations=manifest.operations,
        argument_min=manifest.argument_min,
        argument_max=manifest.argument_max,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
        forbidden_transitions=(manifest.withheld_transition,),
    )
    if regime == "adaptive":
        return OnlineProgramSource(curriculum)
    if regime == "error-focused":
        return ErrorFocusedProgramSource(
            curriculum,
            variants_per_failure=manifest.variants_per_failure,
        )
    raise ValueError(f"unknown program regime: {regime}")


def _sync(adapter: ModelAdapter) -> None:
    method = getattr(adapter, "synchronize", None)
    if callable(method):
        method()


def _close(adapter: ModelAdapter) -> None:
    method = getattr(adapter, "close", None)
    if callable(method):
        method()


def run_program_experiment(
    manifest: ProgramExperimentManifest,
    *,
    output_dir: str | Path,
    create_adapter: Callable[[], ModelAdapter] | None = None,
    regimes: Sequence[str] = PROGRAM_REGIMES,
) -> dict[str, object]:
    manifest.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    common = {
        "operations": manifest.operations,
        "argument_min": manifest.argument_min,
        "argument_max": manifest.argument_max,
        "partition_modulus": manifest.holdout_modulus,
    }
    train = build_static_program_dataset(
        seed=manifest.dataset_seed,
        size=manifest.static_dataset_size,
        lengths=manifest.lengths,
        partition="train",
        **common,
    )
    heldout = build_static_program_dataset(
        seed=manifest.evaluation_seed,
        size=manifest.evaluation_size,
        lengths=manifest.lengths,
        partition="eval",
        forbidden_transitions=(manifest.withheld_transition,),
        **common,
    )
    longer = build_static_program_dataset(
        seed=manifest.evaluation_seed + 1,
        size=manifest.evaluation_size,
        lengths=manifest.out_of_range_lengths,
        partition="eval",
        forbidden_transitions=(manifest.withheld_transition,),
        **common,
    )
    transition_holdout = build_program_transition_dataset(
        seed=manifest.evaluation_seed + 2,
        size=manifest.evaluation_size,
        transition=manifest.withheld_transition,
        lengths=manifest.lengths,
        **common,
    )
    checkpoints = heldout[: manifest.checkpoint_evaluation_size]

    write_program_jsonl(train, output / "static-train.jsonl")
    write_program_jsonl(heldout, output / "evaluation-heldout.jsonl")
    write_program_jsonl(longer, output / "evaluation-longer-programs.jsonl")
    write_program_jsonl(
        transition_holdout,
        output / "evaluation-withheld-transition.jsonl",
    )

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
            evaluation = evaluate_programs(adapter, checkpoints)
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
            verifier=verify_program_answer,
            checkpoint_interval_tokens=manifest.checkpoint_interval_tokens,
            checkpoint_callback=checkpoint_callback,
        )
        _sync(adapter)
        eval_started = perf_counter()
        heldout_result = evaluate_programs(adapter, heldout)
        longer_result = evaluate_programs(adapter, longer)
        transition_result = evaluate_programs(adapter, transition_holdout)
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
                "longer_programs": longer_result.to_dict(),
                "withheld_transition": transition_result.to_dict(),
            },
            "measurement_overhead": {
                "checkpoint_generation_calls": checkpoint_calls,
                "final_evaluation_generation_calls": (
                    len(heldout) + len(longer) + len(transition_holdout)
                ),
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
    parser = argparse.ArgumentParser(description="Run simple program execution experiment")
    parser.add_argument("--manifest", default="experiments/005-program-execution/manifest.json")
    parser.add_argument("--output", default="results/005-program-execution")
    args = parser.parse_args()
    run_program_experiment(
        ProgramExperimentManifest.from_json(args.manifest),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
