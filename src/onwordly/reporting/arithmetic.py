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


def render_single_run(summary: dict[str, Any]) -> str:
    title = "# Arithmetic experiment results"
    lines = [
        title,
        "",
        "## Final evaluation",
        "",
        "| Regime | Held-out | Withheld prompts | Out-of-range | Train tokens | Examples | Total generation calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for regime_name in _regime_order(summary):
        regime = summary["regimes"][regime_name]
        evaluation = regime["evaluation"]
        training = regime["training"]
        overhead = regime["measurement_overhead"]
        lines.append(
            "| "
            + " | ".join(
                (
                    regime_name,
                    _pct(evaluation["heldout"]["accuracy"]),
                    _pct(evaluation["withheld_prompts"]["accuracy"]),
                    _pct(evaluation["out_of_range"]["accuracy"]),
                    _num(training["training_tokens"]),
                    _num(training["examples_trained"]),
                    _num(overhead["total_generation_calls_including_evaluation"]),
                )
            )
            + " |"
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
    lines = [
        "# Arithmetic repeated-seed results",
        "",
        f"Runs: {aggregate['runs']}",
        "",
        "## Final accuracy",
        "",
        "| Regime | Held-out mean ± sd | Withheld prompts mean ± sd | Out-of-range mean ± sd |",
        "| --- | ---: | ---: | ---: |",
    ]

    for regime_name in regime_order:
        regime = aggregate["regimes"][regime_name]
        cells = []
        for split in ("heldout", "withheld_prompts", "out_of_range"):
            stats = regime["accuracy"][split]
            cells.append(f"{_pct(stats['mean'])} ± {_pct(stats['stddev'])}")
        lines.append(f"| {regime_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Resource use",
            "",
            "| Regime | Mean train tokens | Mean examples | Mean total generation calls | Mean core seconds |",
            "| --- | ---: | ---: | ---: | ---: |",
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
