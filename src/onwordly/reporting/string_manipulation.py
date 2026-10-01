from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def render_string_result(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "aggregate" in payload:
        aggregate = payload["aggregate"]
        lines = [
            "# Constrained string repeated-seed results",
            "",
            f"Runs: {aggregate['runs']}",
            "",
            "| Regime | Held-out mean ± sd | Longer-string mean ± sd | Mean wall seconds |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name in payload["regime_order"]:
            regime = aggregate["regimes"][name]
            heldout = regime["accuracy"]["heldout"]
            longer = regime["accuracy"]["longer_strings"]
            lines.append(
                f"| {name} | {_pct(heldout['mean'])} ± {_pct(heldout['stddev'])} | "
                f"{_pct(longer['mean'])} ± {_pct(longer['stddev'])} | "
                f"{regime['regime_wall_seconds']['mean']:.2f} |"
            )
        return "\n".join(lines) + "\n"

    lines = [
        "# Constrained string results",
        "",
        "| Regime | Held-out | Longer strings | Train tokens | Examples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["regime_order"]:
        regime = payload["regimes"][name]
        lines.append(
            f"| {name} | {_pct(regime['evaluation']['heldout']['accuracy'])} | "
            f"{_pct(regime['evaluation']['longer_strings']['accuracy'])} | "
            f"{regime['training']['training_tokens']} | "
            f"{regime['training']['examples_trained']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render constrained-string result JSON")
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = render_string_result(args.input)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
