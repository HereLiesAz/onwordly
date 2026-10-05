from random import Random

from onwordly.tasks.arithmetic import make_arithmetic_task
from onwordly.tasks.verdict import make_verdict_task, parse_verdict, verify_task
from onwordly.training.verdict_evaluation import evaluate_verdict
from onwordly.training.verdict_sources import VerdictArithmeticSource


def test_verdict_targets_and_exact_verification() -> None:
    base = make_arithmetic_task(47, 6, "multiply")
    right = make_verdict_task(base, 282)
    wrong = make_verdict_task(base, 272)
    assert right.target_text == "right" and wrong.target_text == "wrong: 282"
    assert verify_task(right, " right ") and not verify_task(right, "Right")
    assert verify_task(wrong, "wrong: 282") and not verify_task(wrong, "wrong: 272")
    assert not verify_task(wrong, "right") and not verify_task(wrong, "282")
    assert parse_verdict("wrong: 007") is None
    assert verify_task(base, "282") and not verify_task(base, "right")


def _drain(source, rng):
    moves = []
    while source.pending:
        moves.append(source.next_task(rng))
    return moves


def test_verdict_source_is_balanced_and_independent_of_correctness() -> None:
    tasks = [make_arithmetic_task(a, b, "add") for a in range(1, 9) for b in range(1, 9)]
    source = VerdictArithmeticSource(tasks, wrong_source="synthetic", verdict_rate=1.0, seed=0)
    rng = Random(0)
    for index in range(400):
        task = source.base.next_task(rng)
        source.observe_response(task, str(task.answer), index % 2 == 0)
    counts = source.queued
    assert counts["wrong_own"] == 0
    assert 160 < counts["right"] < 240 and counts["right"] + counts["wrong_synthetic"] == 400
    assert all(not move.shown_is_right for move in _drain(source, rng) if move.target_text != "right")


def test_mixed_source_uses_own_wrong_answer() -> None:
    task = make_arithmetic_task(2, 3, "add")
    source = VerdictArithmeticSource([task], wrong_source="mixed", verdict_rate=1.0, seed=3)
    rng = Random(0)
    shown = set()
    for _ in range(40):
        source.observe_response(task, "9", False)
    for move in _drain(source, rng):
        shown.add(move.shown)
    assert 9 in shown and 5 in shown and source.queued["wrong_own"] > 0


class AlwaysRight:
    def generate(self, prompt: str) -> str:
        if "proposed answer" in prompt:
            return "right"
        left, rest = prompt.split("Compute ")[1].split(" + ")
        return str(int(left) + int(rest.split(".")[0]))


def test_always_right_scores_half_on_balanced_verdict() -> None:
    heldout = [make_arithmetic_task(a, b, "add") for a, b in [(1, 2), (3, 4), (5, 6), (7, 8)]]
    result = evaluate_verdict(AlwaysRight(), heldout, seed=0)
    assert result["right_shown_accuracy"] == 1.0
    assert result["wrong_shown_judgement_accuracy"] == 0.0
    assert result["balanced_verdict_accuracy"] == 0.5
    assert result["self_check"]["final_accuracy"] == 1.0 and result["self_check"]["broken"] == 0
