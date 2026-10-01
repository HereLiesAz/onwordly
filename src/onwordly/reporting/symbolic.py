from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def _num(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.2f}"


def render_symbolic_result(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "aggregate" in payload:
        aggregate = payload["aggregate"]
        lines = [
            "# Symbolic transformation repeated-seed results",
            "",
            f"Runs: {aggregate['runs']}",
            "",
            "| Regime | Held-out mean ± sd | Longer-sequence mean ± sd | Mean core seconds | Mean wall seconds |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in payload.get("regime_order", aggregate["regime_order"]):
            regime = aggregate["regimes"][name]
            heldout = regime["accuracy"]["heldout"]
            longer = regime["accuracy"]["longer_sequences"]
            lines.append(
                f"| {name} | {_pct(heldout['mean'])} ± {_pct(heldout['stddev'])} | "
                f"{_pct(longer['mean'])} ± {_pct(longer['stddev'])} | "
                f"{_num(regime['training_core_seconds']['mean'])} | "
                f"{_num(regime['regime_wall_seconds']['mean'])} |"
            )
        return "\n".join(lines) + "\n"

    lines = [
        "# Symbolic transformation results",
        "",
        "| Regime | Held-out | Longer sequences | Train tokens | Examples | Wall seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["regime_order"]:
        regime = payload["regimes"][name]
        lines.append(
            f"| {name} | {_pct(regime['evaluation']['heldout']['accuracy'])} | "
            f"{_pct(regime['evaluation']['longer_sequences']['accuracy'])} | "
            f"{_num(regime['training']['training_tokens'])} | "
            f"{_num(regime['training']['examples_trained'])} | "
            f"{_num(regime['measurement_overhead']['regime_wall_seconds'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render symbolic result JSON")
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = render_symbolic_result(args.input)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
