from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def render_process_supervision_result(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "aggregate" in payload:
        raise ValueError("process-supervision repeated-seed reporting is not enabled yet")

    lines = [
        "# Exact process-supervision results",
        "",
        "| Regime | Final answer | Exact trace | Train tokens | Examples | Mean tokens/example | First-pass tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["regime_order"]:
        regime = payload["regimes"][name]
        training = regime["training"]
        exposure = regime["training_exposure"]
        mean_tokens = training["mean_training_tokens_per_example"]
        lines.append(
            f"| {name} | {_pct(regime['evaluation']['final_answer']['accuracy'])} | "
            f"{_pct(regime['evaluation']['exact_trace']['accuracy'])} | "
            f"{training['training_tokens']} | {training['examples_trained']} | "
            f"{float(mean_tokens):.2f} | {exposure['first_pass_training_tokens']} |"
        )

    comparison = payload["comparison"]
    lines.extend(
        [
            "",
            "## Supervision-cost comparison",
            "",
            f"- Equal token budget: **{comparison['training_token_budget_equal']}**",
            f"- Trace/outcome examples trained: **{comparison['trace_to_outcome_example_ratio']:.4f}**",
            f"- Trace/outcome first-pass token cost: **{comparison['trace_to_outcome_first_pass_token_ratio']:.4f}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render exact process-supervision result JSON")
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = render_process_supervision_result(args.input)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
