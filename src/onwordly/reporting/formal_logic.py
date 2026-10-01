from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def render_logic_result(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "aggregate" in payload:
        aggregate = payload["aggregate"]
        lines = [
            "# Formal logic repeated-seed results",
            "",
            f"Runs: {aggregate['runs']}",
            "",
            "| Regime | Held-out mean ± sd | Deeper-formula mean ± sd | Withheld-composition mean ± sd | Mean wall seconds |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in payload["regime_order"]:
            regime = aggregate["regimes"][name]
            heldout = regime["accuracy"]["heldout"]
            deeper = regime["accuracy"]["deeper_formulas"]
            composition = regime["accuracy"]["withheld_composition"]
            lines.append(
                f"| {name} | {_pct(heldout['mean'])} ± {_pct(heldout['stddev'])} | "
                f"{_pct(deeper['mean'])} ± {_pct(deeper['stddev'])} | "
                f"{_pct(composition['mean'])} ± {_pct(composition['stddev'])} | "
                f"{regime['regime_wall_seconds']['mean']:.2f} |"
            )
        return "\n".join(lines) + "\n"

    lines = [
        "# Formal logic results",
        "",
        "| Regime | Held-out | Deeper formulas | Withheld composition | Train tokens | Examples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["regime_order"]:
        regime = payload["regimes"][name]
        lines.append(
            f"| {name} | {_pct(regime['evaluation']['heldout']['accuracy'])} | "
            f"{_pct(regime['evaluation']['deeper_formulas']['accuracy'])} | "
            f"{_pct(regime['evaluation']['withheld_composition']['accuracy'])} | "
            f"{regime['training']['training_tokens']} | "
            f"{regime['training']['examples_trained']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render formal-logic result JSON")
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = render_logic_result(args.input)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
