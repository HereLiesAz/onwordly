from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _num(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):,.2f}"


def _regime_order(payload: dict[str, Any]) -> tuple[str, ...]:
    declared = payload.get("regime_order")
    if isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        return tuple(declared)
    regimes = payload.get("regimes", {})
    return tuple(regimes)


def _evaluation_split_order(evaluation: dict[str, Any]) -> tuple[str, ...]:
    preferred = (
        "heldout",
        "prompt_transfer_only",
        "withheld_prompts",
        "out_of_range",
    )
    return tuple(
        split
        for split in preferred
        if split in evaluation
    ) + tuple(
        split
        for split in evaluation
        if split not in preferred
    )


def _split_label(split: str) -> str:
    return {
        "heldout": "Held-out",
        "prompt_transfer_only": "Prompt transfer only",
        "withheld_prompts": "Held-out + prompt transfer",
        "out_of_range": "Out-of-range",
    }.get(split, split.replace("_", " ").title())


def render_single_run(summary: dict[str, Any]) -> str:
    title = "# Arithmetic experiment results"
    first_regime = summary["regimes"][_regime_order(summary)[0]]
    split_order = _evaluation_split_order(first_regime["evaluation"])
    headers = [
        "Regime",
        *(_split_label(split) for split in split_order),
        "Train tokens",
        "Examples",
        "Total generation calls",
        "Wall seconds",
        "Peak GiB",
        "Δ accuracy / 1M train tokens",
    ]
    aligns = [
        "---",
        *("---:" for _ in split_order),
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
    ]
    lines = [
        title,
        "",
        "## Final evaluation",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]

    for regime_name in _regime_order(summary):
        regime = summary["regimes"][regime_name]
        evaluation = regime["evaluation"]
        training = regime["training"]
        overhead = regime["measurement_overhead"]
        cells = [regime_name]
        cells.extend(_pct(evaluation[split]["accuracy"]) for split in split_order)
        peak_memory = regime.get("model", {}).get("peak_memory_bytes")
        peak_gib = None if peak_memory is None else float(peak_memory) / (1024**3)
        cells.extend(
            (
                _num(training["training_tokens"]),
                _num(training["examples_trained"]),
                _num(overhead["total_generation_calls_including_evaluation"]),
                _num(overhead.get("regime_wall_seconds")),
                _num(peak_gib),
                _num(overhead.get("capability_gain_per_million_training_tokens")),
            )
        )
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Against the untrained baseline",
            "",
            "Step 0 is the checkpoint evaluation before any training (the untrained",
            "baseline on the checkpoint subset). Lenient = answer anywhere in the",
            "response; diagnostic only, never used for training.",
            "",
            "| Regime | Warm-up tokens | Step-0 exact | Step-0 lenient | Final held-out exact | Final held-out lenient |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for regime_name in _regime_order(summary):
        regime = summary["regimes"][regime_name]
        checkpoints = regime["training"].get("checkpoints") or [{}]
        step0 = checkpoints[0].get("evaluation", {}) if checkpoints[0].get("actual_tokens") == 0 else {}
        heldout = regime["evaluation"]["heldout"]
        lines.append(
            "| "
            + " | ".join(
                (
                    regime_name,
                    _num(regime["training"].get("warmup_tokens", 0)),
                    _pct(step0.get("accuracy")),
                    _pct(step0.get("lenient_accuracy")),
                    _pct(heldout["accuracy"]),
                    _pct(heldout.get("lenient_accuracy")),
                )
            )
            + " |"
        )

    corrective_rows = [
        (name, summary["regimes"][name].get("corrective_evaluation"), summary["regimes"][name].get("corrective_tasks_queued"))
        for name in _regime_order(summary)
    ]
    if any(row[1] for row in corrective_rows):
        lines.extend(
            [
                "",
                "## Corrective language game (Experiment 008)",
                "",
                "Correction: shown a synthetic wrong answer. Confirmation: shown the right answer.",
                "Self-correction: answer, then see your own answer in a corrective prompt.",
                "",
                "| Regime | Correction | Confirmation | Self-correction pass 1 | Pass 2 | Fixed | Broken | Corrective / confirm tasks trained |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, row, queued in corrective_rows:
            if not row:
                continue
            self_correction = row["self_correction"]
            queued_text = "—" if not queued else f"{queued['correct']} / {queued['confirm']}"
            lines.append(
                f"| {name} | {_pct(row['correction_accuracy'])} | {_pct(row['confirmation_accuracy'])} | "
                f"{_pct(self_correction['first_pass_accuracy'])} | {_pct(self_correction['second_pass_accuracy'])} | "
                f"{self_correction['fixed']} | {self_correction['broken']} | {queued_text} |"
            )

    lines.extend(
        [
            "",
            "## Tokens to checkpoint threshold",
            "",
            "| Regime | 70% | 80% | 90% | 95% |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for regime_name in _regime_order(summary):
        thresholds = summary["regimes"][regime_name]["tokens_to_threshold"]
        lines.append(
            "| "
            + " | ".join(
                (
                    regime_name,
                    _num(thresholds["0.70"]),
                    _num(thresholds["0.80"]),
                    _num(thresholds["0.90"]),
                    _num(thresholds["0.95"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Do not infer a training advantage from this run alone. Repeat across seeds and compare both accuracy and total resource cost.",
            "",
        ]
    )
    return "\n".join(lines)


def render_suite(aggregate_payload: dict[str, Any]) -> str:
    aggregate = aggregate_payload["aggregate"]
    regime_order = tuple(
        aggregate_payload.get("regime_order")
        or aggregate.get("regime_order")
        or aggregate["regimes"].keys()
    )
    first_regime = aggregate["regimes"][regime_order[0]]
    split_order = _evaluation_split_order(first_regime["accuracy"])
    headers = ["Regime", *(
        f"{_split_label(split)} mean ± sd"
        for split in split_order
    )]
    lines = [
        "# Arithmetic repeated-seed results",
        "",
        f"Runs: {aggregate['runs']}",
        "",
        "## Final accuracy",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *("---:" for _ in split_order)]) + " |",
    ]

    for regime_name in regime_order:
        regime = aggregate["regimes"][regime_name]
        cells = []
        for split in split_order:
            stats = regime["accuracy"][split]
            cells.append(f"{_pct(stats['mean'])} ± {_pct(stats['stddev'])}")
        lines.append(f"| {regime_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Resource use",
            "",
            "| Regime | Mean train tokens | Mean examples | Mean total generation calls | Mean core seconds | Mean wall seconds | Mean peak GiB | Mean Δ accuracy / 1M tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for regime_name in regime_order:
        regime = aggregate["regimes"][regime_name]
        lines.append(
            "| "
            + " | ".join(
                (
                    regime_name,
                    _num(regime["training_tokens"]["mean"]),
                    _num(regime["examples_trained"]["mean"]),
                    _num(regime["total_generation_calls_including_evaluation"]["mean"]),
                    _num(regime["training_core_seconds"]["mean"]),
                    _num(
                        regime["regime_wall_seconds"]["mean"]
                        if regime.get("regime_wall_seconds")
                        else None
                    ),
                    _num(
                        regime["peak_memory_bytes"]["mean"] / (1024**3)
                        if regime.get("peak_memory_bytes")
                        else None
                    ),
                    _num(
                        regime["capability_gain_per_million_training_tokens"]["mean"]
                        if regime.get("capability_gain_per_million_training_tokens")
                        else None
                    ),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Treat repeated differences as evidence only after checking per-seed consistency, measurement overhead, and whether the effect survives additional seeds or ablations.",
            "",
        ]
    )
    return "\n".join(lines)

def render_result(path: str | Path) -> str:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "aggregate" in payload:
        return render_suite(payload)
    if "regimes" in payload:
        return render_single_run(payload)
    raise ValueError("unrecognized Onwordly result JSON")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Onwordly arithmetic result JSON as Markdown")
    parser.add_argument("input", help="summary.json or aggregate.json")
    parser.add_argument("--output", help="Markdown output path; defaults to stdout")
    args = parser.parse_args()

    rendered = render_result(args.input)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
