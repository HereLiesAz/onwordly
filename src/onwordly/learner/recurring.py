"""Recurring-frame evaluation for Experiment 000 (README, "Recurring frames").

Held-out frames never seen in training, each visited ``recurring_visits``
times in a deterministic shuffled stream, with at least ``recurring_min_gap``
other visits between two visits of the same frame. On every visit the model
attempts (reading the frame's *eval* register), a fallible corrector
challenges (A / B / C with the same error rates as the challenge evaluation,
assigned per visit), the model holds or changes, and the exact verifier grades
the outcome. Everything is written into one ``eval_view`` (forked register,
scratch store, frozen ledger) with the training write rules: draft and
proposal into the register, a changed final answer as a ``self`` filler, the
verifier filler stored. Reads on later visits see ``self`` and corrector
fillers; the verifier filler is stored but never read, exactly as in training.
The ledger stays frozen (no trust update from evaluation outcomes), so any gain
across visits comes from the frame's register, not from trust.

Visits are processed in stream order in chunks that never contain the same
frame twice, so every visit reads all earlier visits of its frame.
"""
from __future__ import annotations

from random import Random
from typing import Callable, Sequence

from onwordly.learner.task import Challenge, ConstrainedTask, Corrector, decision_correct, draw_challenge, grade_reply, is_right
from onwordly.memory import frame_for

# solve(tasks, memory) -> drafts ; decide as train.Decider (or None: no challenge, e.g. plain)
Solver = Callable[[Sequence[ConstrainedTask], object], list[str]]

MAX_CHUNK = 256


def recurring_frames(tasks: Sequence[ConstrainedTask], count: int) -> list[ConstrainedTask]:
    """The first ``count`` held-out tasks with distinct frames (fixed for every arm)."""
    out, seen = [], set()
    for task in tasks:
        frame = frame_for(task)
        if frame not in seen:
            seen.add(frame)
            out.append(task)
        if len(out) == count:
            break
    return out


def recurring_schedule(frames: int, visits: int, *, seed: str, min_gap: int) -> list[tuple[int, int]]:
    """Stream of (frame index, visit index 1..visits). Each round is a seeded
    shuffle of all frames; frames among the last ``min_gap`` of the previous
    round are moved (in shuffled order) to the end of the next round, so two
    visits of a frame are always more than ``min_gap`` positions apart
    (requires 2 * min_gap <= frames)."""
    if 2 * min_gap > frames:
        raise ValueError("need 2 * min_gap <= frames")
    stream: list[tuple[int, int]] = []
    tail: set[int] = set()
    for visit in range(1, visits + 1):
        order = list(range(frames))
        Random(f"{seed}:round:{visit}").shuffle(order)
        order = [f for f in order if f not in tail] + [f for f in order if f in tail]
        stream += [(f, visit) for f in order]
        tail = set(order[-min_gap:]) if min_gap else set()
    return stream


def _chunks(stream: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Consecutive position chunks with no repeated frame."""
    chunks, current, frames = [], [], set()
    for position, (frame, _) in enumerate(stream):
        if frame in frames or len(current) >= MAX_CHUNK:
            chunks.append(current)
            current, frames = [], set()
        current.append(position)
        frames.add(frame)
    if current:
        chunks.append(current)
    return chunks


def run_recurring(
    tasks: Sequence[ConstrainedTask],
    stream: Sequence[tuple[int, int]],
    *,
    solve: Solver,
    decide,
    memory,
    correctors: Sequence[tuple[Corrector, bool]],
    seed: int,
    sources: str = "all",
) -> tuple[list[dict[str, object]], object]:
    """Run the stream on one eval view; return per-visit records and the view."""
    view = memory.eval_view("eval:recurring", register_verifier=True)
    rng = Random(f"{seed}:recurring")
    assigned = [correctors[rng.randrange(len(correctors))][0] for _ in stream]
    records: list[dict[str, object]] = []
    for chunk in _chunks(stream):
        batch = [tasks[stream[p][0]] for p in chunk]
        readable = [view.register_stats(t, sources)[0] for t in batch]
        drafts = solve(batch, view)
        if decide is None:
            for p, task, draft, nonempty in zip(chunk, batch, drafts, readable):
                right = is_right(task, draft)
                records.append(_record(stream[p], None, nonempty, right, right, None, None))
            continue
        challenges: list[Challenge] = []
        links = []
        for p, task, draft in zip(chunk, batch, drafts):
            challenge = draw_challenge(task, draft, assigned[p], rng)
            challenges.append(challenge)
            links.append(view.write_challenge(task, draft, challenge))
        decisions = decide(batch, challenges, view)
        for p, task, challenge, link, nonempty, (action, value, conf) in zip(chunk, batch, challenges, links, readable, decisions):
            outcome, reward, final = grade_reply(task, challenge, action, value)
            ok = decision_correct(challenge, outcome)
            view.write_outcome(
                task, challenge, link, action=action, value=value, outcome=outcome, reward=reward, final=final,
                decision_ok=ok, confidence=conf, inputs={"eval": True, "recurring": True},
            )
            records.append(_record(stream[p], challenge, nonempty, challenge.first_right, is_right(task, final), action, outcome))
    return records, view


def _record(slot, challenge, nonempty, draft_right, final_right, action, outcome) -> dict[str, object]:
    frame, visit = slot
    return {
        "frame": frame, "visit": visit, "register_nonempty": bool(nonempty), "draft_right": bool(draft_right),
        "final_right": bool(final_right), "corrector": challenge.corrector if challenge else None,
        "kind": challenge.kind if challenge else None, "action": action, "outcome": outcome,
    }


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _block(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    challenged = [r for r in rows if r["kind"] is not None]
    right_chal = [r for r in challenged if r["kind"] == "right_challenged"]
    wrong_corr = [r for r in challenged if r["kind"] == "wrong_corrected"]
    said_wrong = right_chal + wrong_corr
    return {
        "n": len(rows),
        "solve_accuracy": _rate(sum(r["draft_right"] for r in rows), len(rows)),
        "final_accuracy": _rate(sum(r["final_right"] for r in challenged), len(challenged)) if challenged else None,
        "hold_rate_right_under_wrong_challenge": _rate(sum(r["action"] == "hold" for r in right_chal), len(right_chal)),
        "change_rate_wrong_under_correct_challenge": _rate(sum(r["action"] == "change" for r in wrong_corr), len(wrong_corr)),
        "change_to_correct_rate_wrong_under_correct_challenge": _rate(
            sum(r["outcome"] == "changed_correct" for r in wrong_corr), len(wrong_corr)
        ),
        "hold_rate_when_said_wrong": _rate(sum(r["action"] == "hold" for r in said_wrong), len(said_wrong)),
        "register_nonempty_fraction": _rate(sum(r["register_nonempty"] for r in rows), len(rows)),
    }


def recurring_metrics(records: Sequence[dict[str, object]], visits: int) -> dict[str, object]:
    """Per visit index: solve (before challenge) and final (after) accuracy,
    hold-right / change-wrong rates, per corrector; learning curve (visit k -
    visit 1); and, for frames whose visit-1 draft was wrong, the fraction
    right at each visit (draft and final)."""
    by_visit = {v: [r for r in records if r["visit"] == v] for v in range(1, visits + 1)}
    per_visit = {}
    for v, rows in by_visit.items():
        block = _block(rows)
        names = sorted({r["corrector"] for r in rows if r["corrector"] is not None})
        block["by_corrector"] = {name: _block([r for r in rows if r["corrector"] == name]) for name in names}
        per_visit[str(v)] = block
    first_wrong = {r["frame"] for r in by_visit[1] if not r["draft_right"]}
    fixed = {}
    for v, rows in by_visit.items():
        sub = [r for r in rows if r["frame"] in first_wrong]
        fixed[str(v)] = {
            "draft_right": _rate(sum(r["draft_right"] for r in sub), len(sub)),
            "final_right": _rate(sum(r["final_right"] for r in sub), len(sub)) if any(r["kind"] for r in sub) else None,
        }

    def delta(key: str) -> float | None:
        a, b = per_visit["1"][key], per_visit[str(visits)][key]
        return None if a is None or b is None else b - a

    return {
        "visits": visits,
        "frames": len(by_visit[1]),
        "per_visit": per_visit,
        "learning_curve": {"solve_last_minus_first": delta("solve_accuracy"), "final_last_minus_first": delta("final_accuracy")},
        "first_wrong_frames": len(first_wrong),
        "first_wrong_right_at_visit": fixed,
    }
