"""Verdict prior diagnostic: does the untrained model already lean to ``right``?

No training. On the first ``--n`` held-out arithmetic problems of Experiment
009 (same manifest, same eval partition, same near misses as 009's verdict
evaluation), score the log-probability of complete replies after the 009
verdict prompt:

- ``right``;
- ``wrong`` (the verdict word alone, not a valid 009 reply; isolates the word);
- ``wrong: <correct answer>`` (the full valid reply for a wrong proposal).

Margins are ``logp(right) - logp(wrong)`` and ``logp(right) - logp(wrong: <answer>)``;
positive means the model prefers ``right``. Conditions: the correct answer
shown, a synthetic near miss shown, and the model's own greedy answer shown
when that answer is parseable and wrong.

Discrimination asks whether the margin moves with the truth of the proposal:
AUROC = P(margin when the correct answer is shown > margin when a wrong one
is shown), ties counted half, over all pairs; and the paired rate on the same
problem. AUROC 0.5 means the prior ignores the proposal's truth; > 0.5 means
it already prefers ``right`` more often for right proposals.

    python -m onwordly.diagnostics.verdict_prior --n 200 --out results/verdict-prior \
        --model Qwen/Qwen2.5-0.5B --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Callable, Sequence

from onwordly.datasets.arithmetic import build_static_arithmetic_dataset
from onwordly.experiments.manifest import ArithmeticExperimentManifest
from onwordly.models.base import ScoringAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.tasks.corrective import synthetic_wrong_answer
from onwordly.tasks.verdict import make_verdict_task
from onwordly.verifiers.arithmetic import parse_integer_answer

DEFAULT_MANIFEST = "experiments/009-verdict-language-game/manifest.json"
CONDITIONS = ("correct_shown", "near_miss_shown", "own_wrong_shown")
MARGINS = ("margin_bare", "margin_full")


def heldout_tasks(manifest: ArithmeticExperimentManifest, n: int) -> tuple[ArithmeticTask, ...]:
    """009's held-out problems (eval partition, evaluation_seed), first ``n``."""
    tasks = build_static_arithmetic_dataset(
        seed=manifest.evaluation_seed,
        size=manifest.evaluation_size,
        operations=manifest.operations,
        digit_levels=manifest.digit_levels,
        partition="eval",
        partition_modulus=manifest.holdout_modulus,
    )
    if not 1 <= n <= len(tasks):
        raise ValueError(f"n must be in [1, {len(tasks)}]")
    return tasks[:n]


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """P(pos > neg) + 0.5 P(pos == neg) over all pairs; None if either is empty."""
    if not positives or not negatives:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in positives for q in negatives)
    return wins / (len(positives) * len(negatives))


def _score(adapter: ScoringAdapter, task: ArithmeticTask, shown: int) -> dict[str, float | int]:
    prompt = make_verdict_task(task, shown).prompt
    right = adapter.continuation_logprob(prompt, "right")
    bare = adapter.continuation_logprob(prompt, "wrong")
    full = adapter.continuation_logprob(prompt, f"wrong: {task.answer}")
    return {
        "shown": shown,
        "logp_right": right,
        "logp_wrong": bare,
        "logp_wrong_answer": full,
        "margin_bare": right - bare,
        "margin_full": right - full,
    }


def _summarise(rows: list[dict[str, float | int]]) -> dict[str, object]:
    out: dict[str, object] = {"n": len(rows)}
    for key in MARGINS:
        values = [float(row[key]) for row in rows]
        out[key] = {
            "prefers_right": (sum(v > 0 for v in values) / len(values)) if values else None,
            "mean": fmean(values) if values else None,
        }
    return out


def run_verdict_prior(adapter: ScoringAdapter, tasks: Sequence[ArithmeticTask], *, seed: int) -> dict[str, object]:
    """Score every condition on ``tasks``. ``seed`` drives the near misses
    exactly as ``evaluate_verdict`` does (one draw per task, in order)."""
    rng = Random(seed)
    rows: dict[str, list[dict[str, float | int]]] = {name: [] for name in CONDITIONS}
    pairs: dict[str, list[tuple[float, float, str]]] = {"near_miss_shown": [], "own_wrong_shown": []}
    greedy_correct = greedy_unparseable = 0
    for index, task in enumerate(tasks):
        correct = _score(adapter, task, task.answer)
        rows["correct_shown"].append({"index": index, **correct})
        near = _score(adapter, task, synthetic_wrong_answer(task, rng))
        rows["near_miss_shown"].append({"index": index, **near})
        pairs["near_miss_shown"].append(
            (
                float(correct["margin_bare"]) - float(near["margin_bare"]),
                float(correct["margin_full"]) - float(near["margin_full"]),
                "near",
            )
        )
        own = parse_integer_answer(adapter.generate(task.prompt))
        if own is None:
            greedy_unparseable += 1
        elif own == task.answer:
            greedy_correct += 1
        else:
            scored = _score(adapter, task, own)
            rows["own_wrong_shown"].append({"index": index, **scored})
            pairs["own_wrong_shown"].append(
                (
                    float(correct["margin_bare"]) - float(scored["margin_bare"]),
                    float(correct["margin_full"]) - float(scored["margin_full"]),
                    "own",
                )
            )
    conditions = {name: _summarise(rows[name]) for name in CONDITIONS}
    discrimination: dict[str, object] = {}
    for wrong in ("near_miss_shown", "own_wrong_shown"):
        block: dict[str, object] = {}
        for position, key in enumerate(MARGINS):
            positive = [float(row[key]) for row in rows["correct_shown"]]
            negative = [float(row[key]) for row in rows[wrong]]
            paired = [pair[position] for pair in pairs[wrong]]
            block[key] = {
                "auroc": auroc(positive, negative),
                "paired_correct_margin_higher": (sum(d > 0 for d in paired) / len(paired)) if paired else None,
                "mean_margin_difference": fmean(paired) if paired else None,
            }
        discrimination[f"correct_vs_{wrong}"] = block
    return {
        "examples": len(tasks),
        "greedy": {
            "correct": greedy_correct,
            "unparseable": greedy_unparseable,
            "wrong_scored": len(rows["own_wrong_shown"]),
        },
        "conditions": conditions,
        "discrimination": discrimination,
        "rows": rows,
        "scoring_calls": 3 * (2 * len(tasks) + len(rows["own_wrong_shown"])),
        "generation_calls": len(tasks),
    }


def _fmt(value: object, pct: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{100 * value:.1f}%" if pct else f"{value:.3f}"


def render_verdict_prior(results: dict[str, dict[str, object]]) -> str:
    lines = [
        "# Verdict prior (untrained models, 009 held-out problems)",
        "",
        "Margin = logp(right) − logp(wrong-reply); positive prefers `right`. "
        "*bare*: the word `wrong`; *full*: `wrong: <correct answer>`. See `docs/verdict-prior.md`.",
        "",
    ]
    for model, result in results.items():
        greedy = result["greedy"]
        lines += [
            f"## {model}",
            "",
            f"{result['examples']} problems; greedy answers correct {greedy['correct']}, "
            f"wrong {greedy['wrong_scored']}, unparseable {greedy['unparseable']}.",
            "",
            "| Condition | n | Prefers right (bare) | Mean margin (bare) | Prefers right (full) | Mean margin (full) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in CONDITIONS:
            c = result["conditions"][name]
            lines.append(
                f"| {name} | {c['n']} | {_fmt(c['margin_bare']['prefers_right'], True)} | {_fmt(c['margin_bare']['mean'])} | "
                f"{_fmt(c['margin_full']['prefers_right'], True)} | {_fmt(c['margin_full']['mean'])} |"
            )
        lines += [
            "",
            "| Discrimination | AUROC (bare) | Paired higher (bare) | AUROC (full) | Paired higher (full) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, block in result["discrimination"].items():
            lines.append(
                f"| {name} | {_fmt(block['margin_bare']['auroc'])} | {_fmt(block['margin_bare']['paired_correct_margin_higher'], True)} | "
                f"{_fmt(block['margin_full']['auroc'])} | {_fmt(block['margin_full']['paired_correct_margin_higher'], True)} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_models(
    models: Sequence[str],
    *,
    manifest_path: str | Path,
    n: int,
    output_dir: str | Path,
    create_adapter: Callable[[str], ScoringAdapter] | None = None,
    device: str = "auto",
) -> dict[str, dict[str, object]]:
    manifest = ArithmeticExperimentManifest.from_json(manifest_path)
    tasks = heldout_tasks(manifest, n)

    def default(model: str) -> ScoringAdapter:
        from onwordly.models.huggingface import HuggingFaceCausalLMAdapter

        return HuggingFaceCausalLMAdapter(
            model, max_new_tokens=manifest.max_new_tokens, device=device, seed=manifest.training_seed
        )

    factory = create_adapter or default
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    for model in models:
        adapter = factory(model)
        results[model] = run_verdict_prior(adapter, tasks, seed=manifest.evaluation_seed)
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    (output / "verdict-prior.json").write_text(
        json.dumps({"manifest": str(manifest_path), "n": n, "models": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(render_verdict_prior(results) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the untrained verdict prior on 009 held-out problems")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", action="append", help="repeat to score several models (default: manifest model)")
    parser.add_argument("--out", default="results/verdict-prior")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    models = args.model or [ArithmeticExperimentManifest.from_json(args.manifest).model_name]
    run_models(models, manifest_path=args.manifest, n=args.n, output_dir=args.out, device=args.device)


if __name__ == "__main__":
    main()
