import json

from onwordly.reporting.arithmetic import render_result


def test_render_single_run(tmp_path) -> None:
    payload = {
        "regimes": {
            name: {
                "evaluation": {
                    "heldout": {"accuracy": 0.5},
                    "withheld_prompts": {"accuracy": 0.4},
                    "out_of_range": {"accuracy": 0.3},
                },
                "training": {
                    "training_tokens": 100,
                    "examples_trained": 5,
                },
                "measurement_overhead": {
                    "total_generation_calls_including_evaluation": 50,
                },
                "tokens_to_threshold": {
                    "0.70": None,
                    "0.80": None,
                    "0.90": None,
                    "0.95": None,
                },
            }
            for name in ("static", "adaptive", "error-focused")
        }
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rendered = render_result(path)
    assert "# Experiment 001 results" in rendered
    assert "50.00%" in rendered
    assert "error-focused" in rendered


def test_render_suite(tmp_path) -> None:
    regime = {
        "accuracy": {
            split: {"mean": 0.5, "stddev": 0.1, "min": 0.4, "max": 0.6, "runs": 3}
            for split in ("heldout", "withheld_prompts", "out_of_range")
        },
        "training_tokens": {"mean": 100.0},
        "examples_trained": {"mean": 5.0},
        "total_generation_calls_including_evaluation": {"mean": 50.0},
        "training_core_seconds": {"mean": 2.0},
    }
    payload = {
        "aggregate": {
            "runs": 3,
            "regimes": {
                name: regime
                for name in ("static", "adaptive", "error-focused")
            },
        }
    }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rendered = render_result(path)
    assert "# Experiment 001 repeated-seed results" in rendered
    assert "50.00% ± 10.00%" in rendered
