"""Experiment 000 -- the Onwordly learner vs matched baselines.

``python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/manifest.json --output results/000``
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Sequence

import torch

from onwordly.learner.manifest import LearnerManifest
from onwordly.learner.model import Encoding, OnwordlyLearner, matched_plain_hidden, parameter_count
from onwordly.learner.task import Corrector, build_dataset
from onwordly.learner.train import (
    VARIANTS,
    evaluate_challenges,
    handcoded_decider,
    learner_decider,
    probe_sample,
    probe_second_visit,
    probe_training_frames,
    solve_learner,
    solve_metrics,
    solve_plain,
    train_learner,
    train_plain,
)


def _code_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for package in ("learner", "memory"):
        for path in sorted((root / package).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_learner_experiment(
    manifest: LearnerManifest,
    *,
    output_dir: str | Path,
    device: torch.device | None = None,
    arms: Sequence[str] | None = None,
) -> dict[str, object]:
    manifest.validate()
    device = device or default_device()
    arms = tuple(arms or manifest.arms)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_tasks = build_dataset(
        seed=manifest.dataset_seed, size=manifest.train_size, lengths=manifest.lengths, alphabet=manifest.alphabet,
        optional_rules=manifest.optional_rules, partition="train", modulus=manifest.holdout_modulus,
    )
    eval_tasks = build_dataset(
        seed=manifest.evaluation_seed, size=manifest.eval_size, lengths=manifest.lengths, alphabet=manifest.alphabet,
        optional_rules=manifest.optional_rules, partition="eval", modulus=manifest.holdout_modulus,
    )
    train_rules = {t.rules for t in train_tasks}
    if any(t.rules in train_rules for t in eval_tasks):
        raise RuntimeError("held-out rule sets leaked into training")
    encoding = Encoding(manifest.alphabet, manifest.slots)
    learner_params = parameter_count(
        OnwordlyLearner(encoding, d_model=manifest.d_model, layers=manifest.layers, heads=manifest.heads)
    )
    plain_hidden = matched_plain_hidden(encoding, learner_params)
    probe_tasks = probe_sample(manifest, train_tasks)
    conditions = [(Corrector(n, r), True) for n, r in manifest.correctors] + [(Corrector(*manifest.unseen_corrector), False)]
    summary: dict[str, object] = {
        "experiment": "000-onwordly-learner",
        "manifest": asdict(manifest),
        "fingerprint": hashlib.sha256(
            json.dumps({"manifest": asdict(manifest), "code": _code_digest()}, sort_keys=True).encode()
        ).hexdigest(),
        "device": str(device),
        "matched_budget": {
            "optimizer_steps": manifest.train_steps,
            "training_problems": manifest.train_steps * manifest.batch_size,
            "problem_order": "Random(training_seed), identical for every arm",
            "learner_parameters": learner_params,
            "plain_hidden": plain_hidden,
        },
        "datasets": {"train": len(train_tasks), "eval": len(eval_tasks), "train_rule_sets": len(train_rules)},
        "arms": {},
    }
    for arm in arms:
        if arm == "handcoded":
            continue  # produced together with "plain"
        started = perf_counter()
        if arm in VARIANTS:
            variant = VARIANTS[arm]
            model, memory, log = train_learner(manifest, variant, train_tasks, device=device, regime=arm)
            answers, confs = solve_learner(
                model, memory.eval_view("eval:solve"), variant, eval_tasks, manifest.revision_steps, device
            )
            drafts = [row[-1] for row in answers]
            challenge = evaluate_challenges(
                eval_tasks, drafts, decide=learner_decider(model, variant, manifest.revision_steps, device),
                memory=memory, conditions=conditions, seed=manifest.evaluation_seed,
            )
            diagnostics = None
            if variant.use_memory:
                steps = manifest.revision_steps
                diagnostics = {
                    "training_frames": probe_training_frames(model, memory, variant, probe_tasks, steps, device),
                    "second_visit": probe_second_visit(model, memory, variant, eval_tasks, steps, device),
                }
            summary["arms"][arm] = {
                "parameters": parameter_count(model),
                "training": log.to_dict(),
                "memory": memory.report(),
                "memory_diagnostics": diagnostics,
                "solve": solve_metrics(eval_tasks, answers, confs),
                "challenge": challenge,
                "wall_seconds": perf_counter() - started,
            }
        elif arm == "plain":
            model, memory, log = train_plain(manifest, train_tasks, hidden=plain_hidden, device=device)
            answers, confs = solve_plain(model, eval_tasks, device)
            solve = solve_metrics(eval_tasks, answers, confs)
            summary["arms"]["plain"] = {
                "parameters": parameter_count(model),
                "training": log.to_dict(),
                "solve": solve,
                "wall_seconds": perf_counter() - started,
            }
            if "handcoded" in arms:
                drafts = [row[-1] for row in answers]
                summary["arms"]["handcoded"] = {
                    "parameters": parameter_count(model),
                    "note": "plain network's answers; hold/change by the fixed rule from the training ledger",
                    "training": log.to_dict(),
                    "memory": memory.report(),
                    "solve": solve,
                    "challenge": evaluate_challenges(
                        eval_tasks, drafts,
                        decide=handcoded_decider(manifest.handcoded_threshold, manifest.handcoded_strikes),
                        memory=memory, conditions=conditions, seed=manifest.evaluation_seed,
                    ),
                }
        else:
            raise ValueError(f"unknown arm: {arm}")
        (output / f"{arm}.json").write_text(json.dumps(summary["arms"][arm], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(render_result(summary), encoding="utf-8")
    return summary


def _pct(value: object) -> str:
    return f"{100 * float(value):.1f}%" if isinstance(value, (int, float)) else "—"


def _num(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "—"


def render_result(summary: dict) -> str:
    budget = summary["matched_budget"]
    lines = [
        "# Experiment 000 — Onwordly learner (provisional, one seed)",
        "",
        f"Matched budget: {budget['optimizer_steps']} optimizer steps, {budget['training_problems']} training problems "
        f"(same order every arm). Learner parameters {budget['learner_parameters']}; plain hidden width {budget['plain_hidden']}.",
        "",
        "## Solve (held-out rule sets, exact)",
        "",
        "| Arm | Params | Accuracy | Per-step accuracy | Fixes / breaks (steps) | Mean s | Conf ECE | Forward steps |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, r in summary["arms"].items():
        s = r["solve"]
        lines.append(
            f"| {arm} | {r['parameters']} | {_pct(s['accuracy'])} | {' → '.join(_pct(a) for a in s['per_step_accuracy'])} | "
            f"{s['step_fixes']} / {s['step_breaks']} | {_num(s['mean_satisfaction'])} | "
            f"{_num(s['confidence_calibration']['ece'])} | {r['training']['forward_steps']} |"
        )
    lines += [
        "",
        "## Fallible challenge (held out, exact)",
        "",
        "| Arm | Corrector (error rate, seen) | Ledger reliability | Hold right vs wrong challenge | Change wrong under correct (to correct) | Hold when told wrong | Discrimination | Final | Decision ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, r in summary["arms"].items():
        if "challenge" not in r:
            continue
        for name, c in r["challenge"]["by_corrector"].items():
            lines.append(
                f"| {arm} | {name} ({c['error_rate']:g}, {'seen' if c['seen_in_training'] else 'unseen'}) | {_num(c['ledger_reliability'])} | "
                f"{_pct(c['hold_rate_right_under_wrong_challenge'])} | {_pct(c['change_rate_wrong_under_correct_challenge'])} "
                f"({_pct(c['change_to_correct_rate_wrong_under_correct_challenge'])}) | {_pct(c['hold_rate_when_said_wrong'])} | "
                f"{_num(c['hold_discrimination'])} | {_pct(c['final_accuracy'])} | {_num(c['decision_calibration']['ece'])} |"
            )
    diag = [(arm, r) for arm, r in summary["arms"].items() if r.get("memory_diagnostics")]
    if diag:
        lines += [
            "",
            "## Memory diagnostics (register as answer channel?)",
            "",
            "Training reads: solve-pass register reads during training (before that visit's writes). "
            "Probe: fixed sample of training problems after training, eval mode, read-only. "
            "Second visit: held-out frames, attempt 2 reads only the model's own attempt-1 answer.",
            "",
            "| Arm | Train reads non-empty | Train top other = target | Dropped | Probe as-is | Probe emptied | Drop | Probe non-empty | Held-out visit 1 | Visit 2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm, r in diag:
            t, d = r["training"]["register_diagnostics"], r["memory_diagnostics"]
            p, v = d["training_frames"], d["second_visit"]
            lines.append(
                f"| {arm} | {_pct(t['nonempty_fraction'])} | {_pct(t['top_other_is_target_fraction'])} | {_pct(t['dropped_fraction'])} | "
                f"{_pct(p['accuracy_register_as_is'])} | {_pct(p['accuracy_register_emptied'])} | {_pct(p['drop_on_emptying'])} | "
                f"{_pct(p['register_nonempty_fraction'])} | {_pct(v['accuracy_first_visit'])} | {_pct(v['accuracy_second_visit'])} |"
            )
    lines += ["", "Hold tracks reliability (hold-when-told-wrong, unreliable minus reliable corrector):", ""]
    for arm, r in summary["arms"].items():
        if "challenge" in r:
            lines.append(f"- **{arm}**: {_num(r['challenge']['hold_tracks_reliability'])}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment 000: the Onwordly learner")
    parser.add_argument("--manifest", default="experiments/000-onwordly-learner/manifest.json")
    parser.add_argument("--output", default="results/000-onwordly-learner")
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    run_learner_experiment(
        LearnerManifest.from_json(args.manifest), output_dir=args.output,
        device=torch.device(args.device) if args.device else None, arms=args.arms or None,
    )


if __name__ == "__main__":
    main()
