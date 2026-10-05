"""Experiment 011: verdict on constrained strings (010's design, new domain).

Arms (identical model, data, seed and training-token budget):
- ``static``: SFT on constrained-string tasks only (control);
- ``verdict-synthetic``: the same stream plus balanced SFT verdict moves
  (``right`` / ``wrong: <witness>``) on synthetic rule violations;
- ``selfcheck-rl-binary`` / ``selfcheck-rl-graded``: the same stream plus
  on-policy two-turn self-check episodes (produce, judge own, keep or repair),
  rewarded binary or graded (``training/constrained_sources.py``);
- ``challenge-rl-graded``: the same stream plus on-policy fallible-challenge
  episodes (answer, then hold or change under a challenger that is wrong with
  probability ``challenge_error_rate``; ``training/challenge_episodes.py``).

Regimes are independent; each writes ``<regime>.json`` with a fingerprint of
manifest + code, and a rerun into the same directory resumes finished regimes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from onwordly.experiments.arithmetic import _code_digest
from onwordly.models.base import ModelAdapter
from onwordly.tasks.constrained_strings import ConstrainedStringTask, build_constrained_dataset
from onwordly.training.constrained_evaluation import (
    evaluate_constrained_generation,
    evaluate_constrained_verdict,
)
from onwordly.training.constrained_sources import (
    ConstrainedSelfCheckConfig,
    ConstrainedSelfCheckSource,
    ConstrainedVerdictSource,
    StaticConstrainedSource,
    verify_constrained,
)
from onwordly.training.challenge_episodes import (
    CONSTRAINED_CHALLENGE,
    ChallengeEpisodeConfig,
    ChallengeEpisodeSource,
    evaluate_challenge,
    render_challenge_lines,
)
from onwordly.training.harness import run_equal_token_training

CONSTRAINED_VERDICT_REGIMES: tuple[str, ...] = (
    "static",
    "verdict-synthetic",
    "selfcheck-rl-binary",
    "selfcheck-rl-graded",
    "challenge-rl-graded",
)


@dataclass(frozen=True, slots=True)
class ConstrainedExperimentManifest:
    model_name: str
    token_budget: int
    static_dataset_size: int
    evaluation_size: int
    longer_evaluation_size: int
    verdict_evaluation_size: int
    checkpoint_evaluation_size: int
    checkpoint_interval_tokens: int
    dataset_seed: int
    evaluation_seed: int
    training_seed: int
    learning_rate: float
    max_new_tokens: int
    lengths: tuple[int, ...]
    out_of_range_lengths: tuple[int, ...]
    alphabet: tuple[str, ...]
    optional_rules: int
    holdout_modulus: int
    verdict_rate: float = 0.3
    rl_samples: int = 4
    rl_temperature: float = 1.0
    challenge_error_rate: float = 0.3

    @classmethod
    def from_json(cls, path: str | Path) -> "ConstrainedExperimentManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("lengths", "out_of_range_lengths", "alphabet"):
            payload[key] = tuple(payload[key])
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.token_budget < 1 or self.static_dataset_size < 1:
            raise ValueError("token_budget and static_dataset_size must be positive")
        if self.evaluation_size < 1 or self.longer_evaluation_size < 1:
            raise ValueError("evaluation sizes must be positive")
        if not 1 <= self.verdict_evaluation_size <= self.evaluation_size:
            raise ValueError("verdict_evaluation_size must be in [1, evaluation_size]")
        if not 1 <= self.checkpoint_evaluation_size <= self.evaluation_size:
            raise ValueError("checkpoint_evaluation_size must be in [1, evaluation_size]")
        if self.checkpoint_interval_tokens < 1:
            raise ValueError("checkpoint_interval_tokens must be positive")
        if not self.lengths or min(self.lengths) < 2:
            raise ValueError("lengths must be at least 2")
        if not self.out_of_range_lengths or min(self.out_of_range_lengths) <= max(self.lengths):
            raise ValueError("out_of_range_lengths must be above the training range")
        if len(set(self.alphabet)) < 3 or any(len(c) != 1 or not c.isascii() or not c.isalnum() for c in self.alphabet):
            raise ValueError("alphabet needs at least three single ASCII alphanumeric characters")
        if not 0 <= self.optional_rules <= 5:
            raise ValueError("optional_rules must be in [0, 5]")
        if self.holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least 2")
        if not 0.0 < self.verdict_rate <= 1.0:
            raise ValueError("verdict_rate must be in (0, 1]")
        if self.rl_samples < 2 or self.rl_temperature <= 0.0:
            raise ValueError("rl_samples must be >= 2 and rl_temperature positive")
        if not 0.0 <= self.challenge_error_rate < 1.0:
            raise ValueError("challenge_error_rate must be in [0, 1)")


def run_fingerprint(manifest: ConstrainedExperimentManifest) -> str:
    payload = json.dumps({"manifest": asdict(manifest), "code": _code_digest()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_for_regime(regime: str, manifest: ConstrainedExperimentManifest, tasks: Sequence[ConstrainedStringTask]):
    if regime == "static":
        return StaticConstrainedSource(tasks)
    if regime == "verdict-synthetic":
        return ConstrainedVerdictSource(tasks, verdict_rate=manifest.verdict_rate, seed=manifest.training_seed)
    if regime in ("selfcheck-rl-binary", "selfcheck-rl-graded"):
        # Identical episodes; only the reward differs.
        return ConstrainedSelfCheckSource(
            tasks,
            config=ConstrainedSelfCheckConfig(
                reward="graded" if regime == "selfcheck-rl-graded" else "binary",
                samples=manifest.rl_samples,
                temperature=manifest.rl_temperature,
            ),
            verdict_rate=manifest.verdict_rate,
            seed=manifest.training_seed,
        )
    if regime == "challenge-rl-graded":
        return ChallengeEpisodeSource(
            StaticConstrainedSource(tasks),
            base_type=ConstrainedStringTask,
            config=ChallengeEpisodeConfig(
                domain=CONSTRAINED_CHALLENGE,
                error_rate=manifest.challenge_error_rate,
                samples=manifest.rl_samples,
                temperature=manifest.rl_temperature,
            ),
            rate=manifest.verdict_rate,
            seed=manifest.training_seed,
        )
    raise ValueError(f"unknown constrained-string regime: {regime}")


def _factory(manifest: ConstrainedExperimentManifest) -> Callable[[], ModelAdapter]:
    def create() -> ModelAdapter:
        from onwordly.models.huggingface import HuggingFaceCausalLMAdapter

        return HuggingFaceCausalLMAdapter(
            manifest.model_name,
            learning_rate=manifest.learning_rate,
            max_new_tokens=manifest.max_new_tokens,
            seed=manifest.training_seed,
        )

    return create


def _call(adapter: object, name: str):
    method = getattr(adapter, name, None)
    return method() if callable(method) else None


def _write_jsonl(tasks: Sequence[ConstrainedStringTask], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps({"prompt": task.prompt, "rules": task.rules, "answer": task.witness}) + "\n")


def run_constrained_experiment(
    manifest: ConstrainedExperimentManifest,
    *,
    output_dir: str | Path,
    create_adapter: Callable[[], ModelAdapter] | None = None,
    regimes: Sequence[str] = CONSTRAINED_VERDICT_REGIMES,
) -> dict[str, object]:
    manifest.validate()
    names = tuple(regimes)
    if not names or len(set(names)) != len(names):
        raise ValueError("regimes must be a non-empty sequence of unique names")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(manifest)
    for saved_path in sorted(output.glob("*.json")):
        if saved_path.name == "summary.json":
            continue
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved.get("manifest_fingerprint", fingerprint) != fingerprint:
            raise RuntimeError(f"{saved_path} was produced by a different manifest or code version")

    common = dict(alphabet=manifest.alphabet, optional_rules=manifest.optional_rules,
                  partition_modulus=manifest.holdout_modulus)
    train = build_constrained_dataset(seed=manifest.dataset_seed, size=manifest.static_dataset_size,
                                      lengths=manifest.lengths, partition="train", **common)
    heldout = build_constrained_dataset(seed=manifest.evaluation_seed, size=manifest.evaluation_size,
                                        lengths=manifest.lengths, partition="eval", **common)
    longer = build_constrained_dataset(seed=manifest.evaluation_seed + 1, size=manifest.longer_evaluation_size,
                                       lengths=manifest.out_of_range_lengths, partition="eval", **common)
    checkpoint_tasks = heldout[: manifest.checkpoint_evaluation_size]
    _write_jsonl(train, output / "static-train.jsonl")
    _write_jsonl(heldout, output / "evaluation-heldout.jsonl")
    _write_jsonl(longer, output / "evaluation-longer.jsonl")

    factory = create_adapter or _factory(manifest)
    results: dict[str, object] = {
        "manifest": asdict(manifest),
        "regime_order": list(names),
        "manifest_fingerprint": fingerprint,
        "regimes": {},
    }
    for regime in names:
        saved_path = output / f"{regime}.json"
        if saved_path.exists():
            results["regimes"][regime] = json.loads(saved_path.read_text(encoding="utf-8"))
            continue
        started_regime = perf_counter()
        adapter = factory()
        _call(adapter, "reset_peak_memory_stats")
        checkpoint_calls = 0
        checkpoint_seconds = 0.0

        def checkpoint_callback(scheduled: int, actual: int, *, _adapter: ModelAdapter = adapter) -> dict[str, object]:
            nonlocal checkpoint_calls, checkpoint_seconds
            del scheduled, actual
            started = perf_counter()
            evaluation = evaluate_constrained_generation(_adapter, checkpoint_tasks)
            _call(_adapter, "synchronize")
            checkpoint_seconds += perf_counter() - started
            checkpoint_calls += len(checkpoint_tasks)
            return {"evaluation": evaluation}

        source = source_for_regime(regime, manifest, train)
        training = run_equal_token_training(
            regime=regime,
            adapter=adapter,
            source=source,
            token_budget=manifest.token_budget,
            seed=manifest.training_seed,
            verifier=verify_constrained,
            checkpoint_interval_tokens=manifest.checkpoint_interval_tokens,
            checkpoint_callback=checkpoint_callback,
        )
        started = perf_counter()
        heldout_result = evaluate_constrained_generation(adapter, heldout)
        longer_result = evaluate_constrained_generation(adapter, longer)
        verdict = evaluate_constrained_verdict(
            adapter, heldout[: manifest.verdict_evaluation_size], seed=manifest.evaluation_seed
        )
        challenge = None
        if regime.startswith("challenge-"):
            challenge = evaluate_challenge(
                adapter, heldout[: manifest.verdict_evaluation_size],
                domain=CONSTRAINED_CHALLENGE, seed=manifest.evaluation_seed,
            )
        _call(adapter, "synchronize")
        final_seconds = perf_counter() - started
        final_calls = len(heldout) + len(longer) + int(verdict["generation_calls"])
        final_calls += int(challenge["generation_calls"]) if challenge else 0
        regime_result = {
            "model": {
                "parameter_count": getattr(adapter, "parameter_count", None),
                "device": getattr(adapter, "device_name", None),
                "peak_memory_bytes": _call(adapter, "peak_memory_bytes"),
            },
            "training": training.to_dict(),
            "tasks_queued": getattr(source, "queued", None),
            "evaluation": {"heldout": heldout_result, "longer": longer_result},
            "verdict_evaluation": verdict,
            **({"challenge_evaluation": challenge} if challenge else {}),
            "measurement_overhead": {
                "checkpoint_generation_calls": checkpoint_calls,
                "final_evaluation_generation_calls": final_calls,
                "checkpoint_seconds": checkpoint_seconds,
                "final_evaluation_seconds": final_seconds,
                "total_generation_calls_including_evaluation": training.generation_calls + checkpoint_calls + final_calls,
                "regime_wall_seconds": perf_counter() - started_regime,
            },
            "manifest_fingerprint": fingerprint,
        }
        results["regimes"][regime] = regime_result
        partial = saved_path.with_suffix(".json.partial")
        partial.write_text(json.dumps(regime_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        partial.replace(saved_path)
        _call(adapter, "close")
        del adapter

    (output / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def _pct(value: object) -> str:
    return f"{100 * float(value):.1f}%" if isinstance(value, (int, float)) else "—"


def render_constrained_result(summary_path: str | Path) -> str:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    lines = [
        "# Experiment 011 — verdict on constrained strings",
        "",
        "| Regime | Tokens | Held-out valid | Held-out mean s | Longer valid | Right shown kept | Wrong shown caught | Wrong shown repaired | Balanced verdict | Self-check first → final | Mean s(final) | Fixed / broken |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in summary["regime_order"]:
        r = summary["regimes"][name]
        v = r["verdict_evaluation"]
        sc = v["self_check"]
        lines.append(
            f"| {name} | {r['training']['training_tokens']} | {_pct(r['evaluation']['heldout']['accuracy'])} | "
            f"{r['evaluation']['heldout']['mean_satisfaction']:.3f} | {_pct(r['evaluation']['longer']['accuracy'])} | "
            f"{_pct(v['right_shown_accuracy'])} | {_pct(v['wrong_shown_judgement_accuracy'])} | "
            f"{_pct(v['wrong_shown_repair_accuracy'])} | {_pct(v['balanced_verdict_accuracy'])} | "
            f"{_pct(sc['first_pass_accuracy'])} → {_pct(sc['final_accuracy'])} | "
            f"{sc['mean_final_satisfaction']:.3f} | {sc['fixed']} / {sc['broken']} |"
        )
    lines += ["", "## Per-class counts", ""]
    for name in summary["regime_order"]:
        v = summary["regimes"][name]["verdict_evaluation"]
        lines.append(f"- **{name}** shown: {json.dumps(v['class_counts'])}; self-check: {json.dumps(v['self_check']['class_counts'])}")
    on_policy = [(n, summary["regimes"][n]["training"].get("on_policy")) for n in summary["regime_order"]]
    on_policy = [(n, s) for n, s in on_policy if s]
    if on_policy:
        lines += ["", "## On-policy episode groups", "",
                  "| Regime | Groups | Trained | Skipped (equal rewards) | Sample calls | Reward tiers |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"]
        for name, stats in on_policy:
            lines.append(
                f"| {name} | {stats['groups']} | {stats['groups_trained']} | {stats['groups_skipped_equal_rewards']} | "
                f"{stats['sample_generation_calls']} | {json.dumps(stats['reward_tiers'], sort_keys=True)} |"
            )
    lines += render_challenge_lines(
        [(n, summary["regimes"][n]["challenge_evaluation"]) for n in summary["regime_order"]
         if summary["regimes"][n].get("challenge_evaluation")]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 011 (verdict on constrained strings)")
    parser.add_argument("--manifest", default="experiments/011-verdict-constrained-strings/manifest.json")
    parser.add_argument("--output", default="results/011-verdict-constrained-strings")
    parser.add_argument("--regimes", nargs="+", default=list(CONSTRAINED_VERDICT_REGIMES))
    args = parser.parse_args(argv)
    run_constrained_experiment(
        ConstrainedExperimentManifest.from_json(args.manifest), output_dir=args.output, regimes=args.regimes
    )


if __name__ == "__main__":
    main()
