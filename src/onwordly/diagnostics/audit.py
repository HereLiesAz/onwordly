"""Full-scale dataset and budget audit with no model training.

Runs every experiment's full manifest through its real runner with an
accounting-only adapter: token counts come from the manifest model's real
tokenizer (same formatting as ``HuggingFaceCausalLMAdapter``), ``generate``
always returns an empty string, and ``train_example`` updates nothing.

That yields, at real scale and without a GPU:
- the frozen datasets each runner writes, audited for train/eval item overlap,
  duplicates and answer skew;
- examples bought per token budget, static pool cycling, repeated examples and
  unused budget per regime.

Because every response is wrong, error-focused regimes run in their worst case
(every task spawns failure variants). Accuracy fields are meaningless.

    python scripts/audit_datasets.py --out /tmp/onwordly-audit --report docs/dataset-audit.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from onwordly.models.base import TrainStepMetrics

MANIFESTS = {
    "001": "experiments/001-arithmetic-curriculum/manifest.json",
    "002": "experiments/002-adaptive-ablation/manifest.json",
    "003": "experiments/003-symbolic-transformations/manifest.json",
    "004": "experiments/004-string-manipulation/manifest.json",
    "005": "experiments/005-program-execution/manifest.json",
    "006": "experiments/006-formal-logic/manifest.json",
    "007": "experiments/007-program-process-supervision/manifest.json",
}
COMMUTATIVE = {"add", "multiply"}


class AccountingAdapter:
    """Token accounting identical to the HF adapter; no model."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def _ids(self, prompt: str, target: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=True)
        target_ids = self.tokenizer.encode(target.strip(), add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            target_ids = [*target_ids, self.tokenizer.eos_token_id]
        return prompt_ids, target_ids

    def count_training_tokens(self, prompt: str, target: str) -> int:
        prompt_ids, target_ids = self._ids(prompt, target)
        return len(prompt_ids) + len(target_ids)

    def generate(self, prompt: str) -> str:
        del prompt
        return ""

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        return TrainStepMetrics(loss=0.0, tokens=self.count_training_tokens(prompt, target))


def _item_key(experiment: str, row: dict) -> tuple:
    """Identity of the underlying problem, ignoring prompt wording."""
    if experiment in {"001", "002"}:
        left, right = row["left"], row["right"]
        if row["operation"] in COMMUTATIVE:
            left, right = sorted((left, right))
        return (row["operation"], left, right)
    # Composition rows name two operations instead of one.
    operation = row.get("operation") or (row.get("first_operation"), row.get("second_operation"))
    if experiment == "003":
        return (operation, row["symbols"])
    if experiment == "004":
        return (operation, row["text"])
    if experiment == "005":
        return (json.dumps(row["instructions"], sort_keys=True),)
    if experiment == "006":
        return (row["expression"],)
    raise KeyError(experiment)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _audit_files(experiment: str, output: Path) -> dict[str, object]:
    train_path = output / "static-train.jsonl"
    if not train_path.exists():
        return {}
    train = _read(train_path)
    train_keys = {_item_key(experiment, row) for row in train}
    train_prompts = {row["prompt"] for row in train}
    splits: dict[str, object] = {}
    for path in sorted(output.glob("evaluation-*.jsonl")):
        rows = _read(path)
        keys = [_item_key(experiment, row) for row in rows]
        answers = Counter(str(row["answer"]) for row in rows)
        top_answer, top_count = answers.most_common(1)[0]
        splits[path.stem.removeprefix("evaluation-")] = {
            "size": len(rows),
            "unique_items": len(set(keys)),
            "items_seen_in_static_train": sum(key in train_keys for key in set(keys)),
            "prompts_seen_in_static_train": sum(row["prompt"] in train_prompts for row in rows),
            "most_common_answer": top_answer,
            "most_common_answer_share": round(top_count / len(rows), 3),
        }
    train_answers = Counter(str(row["answer"]) for row in train)
    top_answer, top_count = train_answers.most_common(1)[0]
    return {
        "static_train": {
            "size": len(train),
            "unique_items": len(train_keys),
            "unique_prompts": len(train_prompts),
            "most_common_answer": top_answer,
            "most_common_answer_share": round(top_count / len(train), 3),
        },
        "evaluation": splits,
    }


def _runner(experiment: str) -> Callable[..., dict]:
    if experiment in {"001", "002"}:
        from onwordly.experiments.ablation import ABLATION_REGIMES
        from onwordly.experiments.arithmetic import run_experiment
        from onwordly.experiments.manifest import ArithmeticExperimentManifest

        def run(path: str, out: Path, adapter: Callable) -> dict:
            manifest = ArithmeticExperimentManifest.from_json(path)
            kwargs = {"regimes": ABLATION_REGIMES} if experiment == "002" else {}
            return run_experiment(manifest, output_dir=out, create_adapter=adapter, **kwargs)
        return run
    modules = {
        "003": ("symbolic", "symbolic_manifest", "SymbolicExperimentManifest", "run_symbolic_experiment"),
        "004": ("string_manipulation", "string_manifest", "StringExperimentManifest", "run_string_experiment"),
        "005": ("program_execution", "program_manifest", "ProgramExperimentManifest", "run_program_experiment"),
        "006": ("formal_logic", "logic_manifest", "LogicExperimentManifest", "run_logic_experiment"),
        "007": ("process_supervision", "program_manifest", "ProgramExperimentManifest", "run_process_supervision_experiment"),
    }
    runner_module, manifest_module, manifest_class, runner_name = modules[experiment]
    import importlib

    runner = getattr(importlib.import_module(f"onwordly.experiments.{runner_module}"), runner_name)
    manifest_type = getattr(importlib.import_module(f"onwordly.experiments.{manifest_module}"), manifest_class)

    def run(path: str, out: Path, adapter: Callable) -> dict:
        return runner(manifest_type.from_json(path), output_dir=out, create_adapter=adapter)
    return run


def _training_rows(summary: dict) -> dict[str, dict[str, object]]:
    rows = {}
    for name, regime in summary["regimes"].items():
        training = regime.get("training", regime)
        rows[name] = {
            key: training.get(key)
            for key in (
                "token_budget",
                "training_tokens",
                "examples_trained",
                "unique_examples",
                "repeated_examples",
                "unused_token_budget",
                "mean_training_tokens_per_example",
            )
        }
    return rows


def _markdown(results: dict[str, dict]) -> str:
    lines = [
        "# Dataset and budget audit",
        "",
        "Generated by `scripts/audit_datasets.py` from the full manifests with an",
        "accounting-only adapter (real tokenizer, no model). Every response is treated",
        "as wrong, so error-focused regimes show their worst-case variant flood.",
        "Re-run after any manifest, generator or tokenizer change.",
        "",
        "## Budget: examples per regime",
        "",
        "| Exp | Regime | Tokens used | Examples | Unique | Repeated | Unused | Tokens/example |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment, result in results.items():
        for regime, row in result["training"].items():
            mean = row["mean_training_tokens_per_example"]
            lines.append(
                f"| {experiment} | {regime} | {row['training_tokens']} | {row['examples_trained']} | "
                f"{row['unique_examples']} | {row['repeated_examples']} | {row['unused_token_budget']} | "
                f"{mean:.1f} |" if mean is not None else "| | | | | | | | |"
            )
    lines += [
        "",
        "## Frozen data: overlap with the static training set",
        "",
        "Item identity ignores prompt wording (commutative arithmetic operands are",
        "sorted). Prompt-transfer splits share items with training by design.",
        "",
        "| Exp | Split | Size | Unique items | Items in train | Prompts in train | Top answer share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment, result in results.items():
        files = result.get("files") or {}
        if not files:
            continue
        train = files["static_train"]
        lines.append(
            f"| {experiment} | static-train | {train['size']} | {train['unique_items']} | — | — | "
            f"{train['most_common_answer_share']} (`{train['most_common_answer']}`) |"
        )
        for split, row in files["evaluation"].items():
            lines.append(
                f"| {experiment} | {split} | {row['size']} | {row['unique_items']} | "
                f"{row['items_seen_in_static_train']} | {row['prompts_seen_in_static_train']} | "
                f"{row['most_common_answer_share']} (`{row['most_common_answer']}`) |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    parser.add_argument("--experiments", nargs="*", default=sorted(MANIFESTS))
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    root = Path(args.out)
    tokenizers: dict[str, object] = {}
    results: dict[str, dict] = {}
    for experiment in args.experiments:
        manifest_path = MANIFESTS[experiment]
        model_name = json.loads(Path(manifest_path).read_text())["model_name"]
        tokenizer = tokenizers.setdefault(model_name, AutoTokenizer.from_pretrained(model_name))
        output = root / experiment
        summary = _runner(experiment)(manifest_path, output, lambda: AccountingAdapter(tokenizer))
        results[experiment] = {
            "training": _training_rows(summary),
            "files": _audit_files(experiment, output),
        }
        print(experiment, "done", flush=True)

    (root / "audit.json").write_text(json.dumps(results, indent=2))
    if args.report:
        Path(args.report).write_text(_markdown(results))


if __name__ == "__main__":
    main()
