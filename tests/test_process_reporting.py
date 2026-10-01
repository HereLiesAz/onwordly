import json

from onwordly.reporting.process_supervision import render_process_supervision_result


def test_render_process_supervision_result(tmp_path) -> None:
    payload = {
        "regime_order": ["outcome-only", "trace-supervised"],
        "regimes": {
            "outcome-only": {
                "training": {
                    "training_tokens": 100,
                    "examples_trained": 10,
                    "mean_training_tokens_per_example": 10.0,
                },
                "training_exposure": {"first_pass_training_tokens": 1000},
                "evaluation": {
                    "final_answer": {"accuracy": 0.5},
                    "exact_trace": {"accuracy": 0.1},
                },
            },
            "trace-supervised": {
                "training": {
                    "training_tokens": 100,
                    "examples_trained": 6,
                    "mean_training_tokens_per_example": 16.6667,
                },
                "training_exposure": {"first_pass_training_tokens": 1600},
                "evaluation": {
                    "final_answer": {"accuracy": 0.6},
                    "exact_trace": {"accuracy": 0.4},
                },
            },
        },
        "comparison": {
            "training_token_budget_equal": True,
            "trace_to_outcome_example_ratio": 0.6,
            "trace_to_outcome_first_pass_token_ratio": 1.6,
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rendered = render_process_supervision_result(path)
    assert "# Exact process-supervision results" in rendered
    assert "60.00%" in rendered
    assert "1.6000" in rendered
