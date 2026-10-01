from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Callable, Literal

from onwordly.datasets.program_execution import build_static_program_dataset
from onwordly.experiments.program_manifest import ProgramExperimentManifest
from onwordly.models.base import ModelAdapter
from onwordly.models.huggingface import HuggingFaceCausalLMAdapter
from onwordly.tasks.program_execution import (
    ProgramSupervisionTask,
    ProgramTask,
    make_program_supervision_task,
)
from onwordly.training.harness import run_equal_token_training
from onwordly.verifiers.program_execution import (
    verify_supervised_program_final,
    verify_supervised_program_trace,
)

SUPERVISION_REGIMES: tuple[str, ...] = ("outcome-only", "trace-supervised")


class StaticSupervisionSource:
    def __init__(
        self,
        programs: tuple[ProgramTask, ...],
        supervision: Literal["outcome", "trace"],
    ) -> None:
        if not programs:
            raise ValueError("program dataset cannot be empty")
        self.tasks = tuple(
            make_program_supervision_task(
                program.instructions,
                supervision=supervision,
            )
            for program in programs
        )
        self.index = 0

    def next_task(self, rng: Random) -> ProgramSupervisionTask:
        del rng
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task

    def observe(self, task: ProgramSupervisionTask, correct: bool) -> None:
        del task, correct


def _factory(manifest: ProgramExperimentManifest) -> Callable[[], ModelAdapter]:
    def create() -> ModelAdapter:
        return HuggingFaceCausalLMAdapter(
            manifest.model_name,
            learning_rate=manifest.learning_rate,
            max_new_tokens=max(manifest.max_new_tokens, 64),
            seed=manifest.training_seed,
        )
    return create


def _evaluate_final(
    adapter: ModelAdapter,
    programs: tuple[ProgramTask, ...],
    *,
    supervision: Literal["outcome", "trace"],
) -> dict[str, float | int]:
    tasks = tuple(
        make_program_supervision_task(program.instructions, supervision=supervision)
        for program in programs
    )
    correct = sum(
        verify_supervised_program_final(task, adapter.generate(task.prompt))
        for task in tasks
    )
    return {
        "examples": len(tasks),
        "correct": correct,
        "accuracy": correct / len(tasks),
    }


def _evaluate_trace(
    adapter: ModelAdapter,
    programs: tuple[ProgramTask, ...],
) -> dict[str, float | int]:
    tasks = tuple(
        make_program_supervision_task(program.instructions, supervision="trace")
        for program in programs
    )
    correct = sum(
        verify_supervised_program_trace(task, adapter.generate(task.prompt))
        for task in tasks
    )
    return {
        "examples": len(tasks),
        "correct": correct,
        "accuracy": correct / len(tasks),
    }


def run_process_supervision_experiment(
    manifest: ProgramExperimentManifest,
    *,
    output_dir: str | Path,
    create_adapter: Callable[[], ModelAdapter] | None = None,
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
    train_programs = build_static_program_dataset(
        seed=manifest.dataset_seed,
        size=manifest.static_dataset_size,
        lengths=manifest.lengths,
        partition="train",
        **common,
    )
    heldout_programs = build_static_program_dataset(
        seed=manifest.evaluation_seed,
        size=manifest.evaluation_size,
        lengths=manifest.lengths,
        partition="eval",
        **common,
    )

    factory = create_adapter or _factory(manifest)
    result: dict[str, object] = {
        "manifest": {
            "model_name": manifest.model_name,
            "token_budget": manifest.token_budget,
            "training_seed": manifest.training_seed,
        },
        "regime_order": list(SUPERVISION_REGIMES),
        "regimes": {},
    }

    for regime in SUPERVISION_REGIMES:
        supervision: Literal["outcome", "trace"] = (
            "outcome" if regime == "outcome-only" else "trace"
        )
        adapter = factory()
        started = perf_counter()
        training = run_equal_token_training(
            regime=regime,
            adapter=adapter,
            source=StaticSupervisionSource(train_programs, supervision),
            token_budget=manifest.token_budget,
            seed=manifest.training_seed,
            verifier=verify_supervised_program_final,
        )
        final_eval = _evaluate_final(
            adapter,
            heldout_programs,
            supervision=supervision,
        )
        trace_eval = _evaluate_trace(adapter, heldout_programs)
        regime_result = {
            "training": training.to_dict(),
            "evaluation": {
                "final_answer": final_eval,
                "exact_trace": trace_eval,
            },
            "measurement_overhead": {
                "regime_wall_seconds": perf_counter() - started,
            },
        }
        result["regimes"][regime] = regime_result
        (output / f"{regime}.json").write_text(
            json.dumps(regime_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact process-supervision ablation for program execution"
    )
    parser.add_argument(
        "--manifest",
        default="experiments/007-program-process-supervision/manifest.json",
    )
    parser.add_argument(
        "--output",
        default="results/007-program-process-supervision",
    )
    args = parser.parse_args()
    run_process_supervision_experiment(
        ProgramExperimentManifest.from_json(args.manifest),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
