"""Recurring-frame evaluation and the learner-first-visit arm (CPU, tiny)."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from onwordly.learner.manifest import LearnerManifest
from onwordly.learner.recurring import _chunks, recurring_frames, recurring_metrics, recurring_schedule, run_recurring
from onwordly.learner.task import Corrector, build_dataset
from onwordly.learner.train import VARIANTS, learner_decider, solve_learner, train_learner
from onwordly.memory import frame_for

CPU = torch.device("cpu")
CONDITIONS = [(Corrector("A", 0.1), True), (Corrector("B", 0.5), True), (Corrector("C", 0.3), False)]


def _setup(**overrides):
    manifest = LearnerManifest.from_json("experiments/000-onwordly-learner/recurring-smoke-manifest.json")
    manifest = replace(manifest, **{"train_steps": 4, "train_size": 12, "eval_size": 12, **overrides})
    kw = dict(lengths=manifest.lengths, alphabet=manifest.alphabet, optional_rules=manifest.optional_rules,
              modulus=manifest.holdout_modulus)
    train = build_dataset(seed=manifest.dataset_seed, size=manifest.train_size, partition="train", **kw)
    held = build_dataset(seed=manifest.evaluation_seed, size=manifest.eval_size, partition="eval", **kw)
    return manifest, train, held


def test_schedule_deterministic_with_gaps() -> None:
    for frames, visits, gap in ((10, 4, 5), (500, 4, 25), (7, 3, 1)):
        a = recurring_schedule(frames, visits, seed="s", min_gap=gap)
        assert a == recurring_schedule(frames, visits, seed="s", min_gap=gap)
        assert a != recurring_schedule(frames, visits, seed="t", min_gap=gap)
        assert len(a) == frames * visits
        last: dict[int, tuple[int, int]] = {}
        for position, (frame, visit) in enumerate(a):
            if frame in last:
                prev_pos, prev_visit = last[frame]
                assert visit == prev_visit + 1
                assert position - prev_pos - 1 >= gap  # at least `gap` other visits in between
            else:
                assert visit == 1
            last[frame] = (position, visit)
        assert all(v == visits for _, v in last.values())
        for chunk in _chunks(a):
            assert len({a[p][0] for p in chunk}) == len(chunk)
    with pytest.raises(ValueError):
        recurring_schedule(4, 2, seed="s", min_gap=3)


def test_eval_writes_never_reach_training_memory_and_visits_see_history() -> None:
    manifest, train, held = _setup()
    variant = VARIANTS["learner"]
    model, memory, _ = train_learner(manifest, variant, train, device=CPU, regime="r")
    before = (len(memory.store), memory.store.head, {f: len(memory.register.entries(f)) for f in memory.register.frames()},
              memory.step, memory.ledger.snapshot())
    tasks = recurring_frames(held, 6)
    stream = recurring_schedule(len(tasks), 3, seed="x", min_gap=2)
    seen: list[tuple[int, int, int, set]] = []  # (frame, visit, readable entries, readable sources)

    position = {"i": 0}

    def solve(batch, view):
        for task in batch:
            frame, visit = stream[position["i"]]
            position["i"] += 1
            summary = view.register.summary(frame_for(task), exclude_sources=view._excluded(task, "all"), cap=100)
            seen.append((frame, visit, summary.total, {s for _, s, _ in summary.entries}))
        return [row[-1] for row in solve_learner(model, view, variant, batch, manifest.revision_steps, CPU)[0]]

    records, view = run_recurring(
        tasks, stream, solve=solve, decide=learner_decider(model, variant, manifest.revision_steps, CPU), memory=memory,
        correctors=CONDITIONS, seed=1,
    )
    after = (len(memory.store), memory.store.head, {f: len(memory.register.entries(f)) for f in memory.register.frames()},
             memory.step, memory.ledger.snapshot())
    assert before == after
    assert not any(frame_for(t) in memory.register.frames() for t in tasks)
    assert view.store.verify() and len(view.store) > 0
    # Visit k reads every earlier visit's draft (+ proposal); visit 1 reads nothing; verifier never readable.
    for frame, visit, total, sources in seen:
        if visit == 1:
            assert total == 0
        else:
            assert total >= visit - 1 and "self" in sources and "verifier" not in sources
    for task in tasks:
        stored = {e.source for e in view.register.entries(frame_for(task))}
        assert "verifier" in stored  # stored as in training, not read
    assert [r["register_nonempty"] for r in records] == [v > 1 for _, v in stream]
    assert {r["corrector"] for r in records} <= {"A", "B", "C"}


def test_metric_arithmetic() -> None:
    def rec(frame, visit, draft, final, kind, action, corrector="A", outcome=None):
        return {"frame": frame, "visit": visit, "register_nonempty": visit > 1, "draft_right": draft, "final_right": final,
                "corrector": corrector, "kind": kind, "action": action, "outcome": outcome}

    records = [
        rec(0, 1, False, True, "wrong_corrected", "change", outcome="changed_correct"),
        rec(1, 1, True, True, "right_challenged", "hold", corrector="B"),
        rec(2, 1, False, False, "wrong_confirmed", "hold"),
        rec(3, 1, True, False, "right_challenged", "change", corrector="B", outcome="caved"),
        rec(0, 2, True, True, "right_confirmed", "hold"),
        rec(1, 2, True, True, "right_challenged", "hold", corrector="B"),
        rec(2, 2, False, True, "wrong_corrected", "change", outcome="changed_correct"),
        rec(3, 2, True, True, "right_challenged", "hold", corrector="B"),
    ]
    m = recurring_metrics(records, 2)
    v1, v2 = m["per_visit"]["1"], m["per_visit"]["2"]
    assert v1["solve_accuracy"] == 0.5 and v1["final_accuracy"] == 0.5
    assert v2["solve_accuracy"] == 0.75 and v2["final_accuracy"] == 1.0
    assert v1["hold_rate_right_under_wrong_challenge"] == 0.5 and v2["hold_rate_right_under_wrong_challenge"] == 1.0
    assert v1["change_rate_wrong_under_correct_challenge"] == 1.0
    assert v1["hold_rate_when_said_wrong"] == pytest.approx(1 / 3)
    assert v1["register_nonempty_fraction"] == 0.0 and v2["register_nonempty_fraction"] == 1.0
    assert v1["by_corrector"]["B"]["final_accuracy"] == 0.5 and v1["by_corrector"]["A"]["solve_accuracy"] == 0.0
    assert m["learning_curve"] == {"solve_last_minus_first": 0.25, "final_last_minus_first": 0.5}
    assert m["first_wrong_frames"] == 2
    assert m["first_wrong_right_at_visit"]["1"] == {"draft_right": 0.0, "final_right": 0.5}
    assert m["first_wrong_right_at_visit"]["2"] == {"draft_right": 0.5, "final_right": 1.0}


def test_first_visit_arm_counts_revisits_and_keeps_budget() -> None:
    manifest, train, _ = _setup(train_steps=6, train_size=6)
    _, _, log = train_learner(manifest, VARIANTS["learner-first-visit"], train, device=CPU, regime="f")
    _, _, base = train_learner(manifest, VARIANTS["learner"], train, device=CPU, regime="b")
    diag = log.to_dict()["register_diagnostics"]
    assert log.optimizer_steps == base.optimizer_steps and log.training_problems == base.training_problems
    assert diag["revisits_decision_only"] > 0  # 6 frames, 48 draws: most are revisits
    assert diag["revisits_decision_only"] <= log.register_nonempty
    assert log.forward_steps == base.forward_steps + manifest.revision_steps * diag["revisits_decision_only"]
    assert base.revisits_decision_only == 0
