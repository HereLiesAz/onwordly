from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from onwordly.curricula.adaptive import AdaptiveArithmeticCurriculum
from onwordly.curricula.uniform import UniformArithmeticCurriculum
from onwordly.datasets.arithmetic import (
    arithmetic_dataset_stats,
    build_static_arithmetic_dataset,
    write_arithmetic_jsonl,
)
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import ModelAdapter
from onwordly.models.huggingface import HuggingFaceCausalLMAdapter
from onwordly.training.evaluation import EvaluationResult, evaluate_arithmetic
from onwordly.training.harness import run_equal_token_training
from onwordly.training.sources import (
    AdaptiveArithmeticSource,
    ArithmeticTaskSource,
    ErrorFocusedArithmeticSource,
    StaticArithmeticSource,
)
from onwordly.tasks.arithmetic import ArithmeticTask

DEFAULT_REGIMES: tuple[str, ...] = ("static", "adaptive", "error-focused")
ABLATION_REGIMES: tuple[str, ...] = (
    "static",
    "online-uniform",
    "adaptive",
    "error-focused-uniform",
    "error-focused-adaptive",
)


def _synchronize_adapter(adapter: ModelAdapter) -> None:
    synchronize = getattr(adapter, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _reset_peak_memory(adapter: ModelAdapter) -> None:
    reset = getattr(adapter, "reset_peak_memory_stats", None)
    if callable(reset):
        reset()


def _peak_memory_bytes(adapter: ModelAdapter) -> int | None:
    read = getattr(adapter, "peak_memory_bytes", None)
    if not callable(read):
        return None
    value = read()
    return int(value) if value is not None else None


def _close_adapter(adapter: ModelAdapter) -> None:
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _capability_gain_per_million_tokens(
    checkpoints: tuple[dict[str, object], ...],
    training_tokens: int,
) -> float | None:
    if training_tokens <= 0 or len(checkpoints) < 2:
        return None
    first = checkpoints[0].get("evaluation")
    last = checkpoints[-1].get("evaluation")
    if not isinstance(first, dict) or not isinstance(last, dict):
        return None
    first_accuracy = first.get("accuracy")
    last_accuracy = last.get("accuracy")
    if not isinstance(first_accuracy, (int, float)) or not isinstance(
        last_accuracy, (int, float)
    ):
        return None
    return (float(last_accuracy) - float(first_accuracy)) * 1_000_000 / training_tokens


def _code_digest() -> str:
    """SHA-256 over the installed onwordly sources, so code changes break resume."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_fingerprint(manifest: ArithmeticExperimentManifest) -> str:
    """Identity of a run for resume: manifest plus the code that executes it."""
    payload = json.dumps(
        {"manifest": asdict(manifest), "code": _code_digest()}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _adapter_factory(manifest: ArithmeticExperimentManifest) -> Callable[[], ModelAdapter]:
    def create() -> ModelAdapter:
        return HuggingFaceCausalLMAdapter(
            manifest.model_name,
            learning_rate=manifest.learning_rate,
            max_new_tokens=manifest.max_new_tokens,
            seed=manifest.training_seed,
            lora=manifest.lora,
        )

    return create


def _adaptive_curriculum(manifest: ArithmeticExperimentManifest) -> AdaptiveArithmeticCurriculum:
    return AdaptiveArithmeticCurriculum(
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
    )


def _uniform_curriculum(manifest: ArithmeticExperimentManifest) -> UniformArithmeticCurriculum:
    return UniformArithmeticCurriculum(
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
    )


def _source_for_regime(
    regime: str,
    manifest: ArithmeticExperimentManifest,
    static_tasks: Sequence[ArithmeticTask],
) -> ArithmeticTaskSource:
    if regime == "static":
        return StaticArithmeticSource(static_tasks)
    if regime == "online-uniform":
        return AdaptiveArithmeticSource(_uniform_curriculum(manifest))
    if regime == "adaptive":
        return AdaptiveArithmeticSource(_adaptive_curriculum(manifest))
    if regime in ("error-focused", "error-focused-adaptive"):
        return ErrorFocusedArithmeticSource(
            _adaptive_curriculum(manifest),
            variants_per_failure=manifest.variants_per_failure,
        )
    if regime == "error-focused-uniform":
        return ErrorFocusedArithmeticSource(
            _uniform_curriculum(manifest),
            variants_per_failure=manifest.variants_per_failure,
        )
    raise ValueError(f"unknown arithmetic regime: {regime}")


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
    prompt_transfer_only: EvaluationResult,
    withheld_prompts: EvaluationResult,
    out_of_range: EvaluationResult,
) -> dict[str, object]:
    return {
        "heldout": heldout.to_dict(),
        "prompt_transfer_only": prompt_transfer_only.to_dict(),
        "withheld_prompts": withheld_prompts.to_dict(),
        "out_of_range": out_of_range.to_dict(),
    }


def run_experiment(
    manifest: ArithmeticExperimentManifest,
    *,
    output_dir: str | Path,
    create_adapter: Callable[[], ModelAdapter] | None = None,
    regimes: Sequence[str] = DEFAULT_REGIMES,
) -> dict[str, object]:
    manifest.validate()
    regime_names = tuple(regimes)
    if not regime_names or len(set(regime_names)) != len(regime_names):
        raise ValueError("regimes must be a non-empty sequence of unique names")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(manifest)
    # Refuse before writing anything, so a rejected run cannot overwrite the
    # frozen datasets that existing regime results were produced from.
    for saved_path in sorted(output.glob("*.json")):
        if saved_path.name == "summary.json":
            continue
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if "manifest_fingerprint" in saved and saved["manifest_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"{saved_path} was produced by a different manifest or code version; "
                "move it aside or use a fresh output directory"
            )

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
    prompt_transfer_tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed + 1,
        size=manifest.generalization_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        prompt_styles=manifest.withheld_prompt_styles,
        partition="train",
        partition_modulus=manifest.holdout_modulus,
    )
    withheld_prompt_tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed + 2,
        size=manifest.generalization_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        prompt_styles=manifest.withheld_prompt_styles,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    out_of_range_tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed + 3,
        size=manifest.generalization_size,
        operations=manifest.operations,
        digit_levels=manifest.out_of_range_digit_levels,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    checkpoint_tasks = heldout_tasks[: manifest.checkpoint_evaluation_size]

    write_arithmetic_jsonl(static_tasks, output / "static-train.jsonl")
    write_arithmetic_jsonl(heldout_tasks, output / "evaluation-heldout.jsonl")
    write_arithmetic_jsonl(
        prompt_transfer_tasks,
        output / "evaluation-prompt-transfer-only.jsonl",
    )
    write_arithmetic_jsonl(withheld_prompt_tasks, output / "evaluation-withheld-prompts.jsonl")
    write_arithmetic_jsonl(out_of_range_tasks, output / "evaluation-out-of-range.jsonl")

    factory = create_adapter or _adapter_factory(manifest)
    results: dict[str, object] = {
        "manifest": asdict(manifest),
        "regime_order": list(regime_names),
        "datasets": {
            "static_train": arithmetic_dataset_stats(static_tasks),
            "heldout": arithmetic_dataset_stats(heldout_tasks),
            "prompt_transfer_only": arithmetic_dataset_stats(prompt_transfer_tasks),
            "withheld_prompts": arithmetic_dataset_stats(withheld_prompt_tasks),
            "out_of_range": arithmetic_dataset_stats(out_of_range_tasks),
        },
        "regimes": {},
    }

    # Resume: a regime whose result file exists for this exact manifest is loaded,
    # not re-run. Regimes are independent (fresh model each), so this is sound.
    results["manifest_fingerprint"] = fingerprint

    for regime_name in regime_names:
        saved_path = output / f"{regime_name}.json"
        if saved_path.exists():
            results["regimes"][regime_name] = json.loads(saved_path.read_text(encoding="utf-8"))
            continue
        regime_started = perf_counter()
        model_load_started = perf_counter()
        adapter = factory()
        _synchronize_adapter(adapter)
        model_load_seconds = perf_counter() - model_load_started
        _reset_peak_memory(adapter)
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
            _synchronize_adapter(_adapter)
            started = perf_counter()
            evaluation = evaluate_arithmetic(_adapter, checkpoint_tasks)
            _synchronize_adapter(_adapter)
            checkpoint_seconds += perf_counter() - started
            checkpoint_generation_calls += len(checkpoint_tasks)
            return {"evaluation": evaluation.to_dict()}

        training = run_equal_token_training(
            regime=regime_name,
            adapter=adapter,
            source=_source_for_regime(regime_name, manifest, static_tasks),
            token_budget=manifest.token_budget,
            # Warm-up comes from the pool's tail, which the static regime
            # reaches last, so it rarely duplicates static training items.
            warmup_tasks=tuple(reversed(static_tasks)),
            warmup_token_budget=manifest.format_warmup_tokens,
            seed=manifest.training_seed,
            checkpoint_interval_tokens=manifest.checkpoint_interval_tokens,
            checkpoint_callback=checkpoint_callback,
        )

        _synchronize_adapter(adapter)
        started = perf_counter()
        heldout = evaluate_arithmetic(adapter, heldout_tasks)
        prompt_transfer = evaluate_arithmetic(adapter, prompt_transfer_tasks)
        withheld = evaluate_arithmetic(adapter, withheld_prompt_tasks)
        out_of_range = evaluate_arithmetic(adapter, out_of_range_tasks)
        _synchronize_adapter(adapter)
        final_evaluation_seconds = perf_counter() - started
        final_evaluation_calls = (
            len(heldout_tasks)
            + len(prompt_transfer_tasks)
            + len(withheld_prompt_tasks)
            + len(out_of_range_tasks)
        )
        total_evaluation_calls = checkpoint_generation_calls + final_evaluation_calls

        peak_memory_bytes = _peak_memory_bytes(adapter)
        regime_wall_seconds = perf_counter() - regime_started
        capability_gain = _capability_gain_per_million_tokens(
            training.checkpoints,
            training.training_tokens,
        )

        regime_result = {
            "model": {
                "parameter_count": getattr(adapter, "parameter_count", None),
                "trainable_parameter_count": getattr(adapter, "trainable_parameter_count", None),
                "lora": manifest.lora,
                "device": getattr(adapter, "device_name", None),
                "peak_memory_bytes": peak_memory_bytes,
            },
            "training": training.to_dict(),
            "tokens_to_threshold": _tokens_to_threshold(training.checkpoints),
            "evaluation": _serialize_evaluations(
                heldout,
                prompt_transfer,
                withheld,
                out_of_range,
            ),
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
                "model_load_seconds": model_load_seconds,
                "regime_wall_seconds": regime_wall_seconds,
                "capability_gain_per_million_training_tokens": capability_gain,
            },
        }
        regime_result["manifest_fingerprint"] = fingerprint
        results["regimes"][regime_name] = regime_result
        # Write then rename so an interrupted write never leaves a partial file
        # that a resumed run would trust.
        partial = saved_path.with_suffix(".json.partial")
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(regime_result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        partial.replace(saved_path)
        _close_adapter(adapter)
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
