import json

from onwordly.reporting.arithmetic import render_result


def test_render_single_run(tmp_path) -> None:
    payload = {
        "regime_order": ["static", "adaptive", "error-focused"],
        "regimes": {
            name: {
                "evaluation": {
                    "heldout": {"accuracy": 0.5},
                    "prompt_transfer_only": {"accuracy": 0.45},
                    "withheld_prompts": {"accuracy": 0.4},
                    "out_of_range": {"accuracy": 0.3},
                },
                "training": {
                    "training_tokens": 100,
                    "examples_trained": 5,
                },
                "measurement_overhead": {
                    "total_generation_calls_including_evaluation": 50,
                    "regime_wall_seconds": 3.5,
                    "capability_gain_per_million_training_tokens": 12.5,
                },
                "model": {"peak_memory_bytes": 1073741824},
                "tokens_to_threshold": {
                    "0.70": None,
                    "0.80": None,
                    "0.90": None,
                    "0.95": None,
                },
            }
            for name in ("static", "adaptive", "error-focused")
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rendered = render_result(path)
    assert "# Arithmetic experiment results" in rendered
    assert "50.00%" in rendered
    assert "Prompt transfer only" in rendered
    assert "error-focused" in rendered


def test_render_suite(tmp_path) -> None:
    regime = {
        "accuracy": {
            split: {"mean": 0.5, "stddev": 0.1, "min": 0.4, "max": 0.6, "runs": 3}
            for split in ("heldout", "prompt_transfer_only", "withheld_prompts", "out_of_range")
        },
        "training_tokens": {"mean": 100.0},
        "examples_trained": {"mean": 5.0},
        "total_generation_calls_including_evaluation": {"mean": 50.0},
        "training_core_seconds": {"mean": 2.0},
        "regime_wall_seconds": {"mean": 3.0},
        "peak_memory_bytes": {"mean": 1073741824.0},
        "capability_gain_per_million_training_tokens": {"mean": 12.5},
    }
    payload = {
        "regime_order": ["static", "adaptive", "error-focused"],
        "aggregate": {
            "runs": 3,
            "regime_order": ["static", "adaptive", "error-focused"],
            "regimes": {
                name: regime
                for name in ("static", "adaptive", "error-focused")
            },
        },
    }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rendered = render_result(path)
    assert "# Arithmetic repeated-seed results" in rendered
    assert "50.00% ± 10.00%" in rendered


def test_render_ablation_uses_declared_order(tmp_path) -> None:
    order = [
        "static",
        "online-uniform",
        "adaptive",
        "error-focused-uniform",
        "error-focused-adaptive",
    ]
    payload = {
        "regime_order": order,
        "regimes": {
            name: {
                "evaluation": {
                    "heldout": {"accuracy": 0.1},
                    "prompt_transfer_only": {"accuracy": 0.1},
                    "withheld_prompts": {"accuracy": 0.1},
                    "out_of_range": {"accuracy": 0.1},
                },
                "training": {"training_tokens": 10, "examples_trained": 1},
                "measurement_overhead": {
                    "total_generation_calls_including_evaluation": 5,
                },
                "tokens_to_threshold": {
                    "0.70": None,
                    "0.80": None,
                    "0.90": None,
                    "0.95": None,
                },
            }
            for name in order
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rendered = render_result(path)
    positions = [rendered.index(name) for name in order]
    assert positions == sorted(positions)
