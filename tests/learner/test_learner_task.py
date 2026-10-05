"""Self-contained constrained-string task and challenge grading."""
from random import Random

from onwordly.learner.task import (
    Challenge, Corrector, build_dataset, decision_correct, draw_challenge, grade_reply, is_right, partition_of,
    synthetic_violation,
)

TASK = build_dataset(seed=5, size=1, lengths=[6], partition="train")[0]


def test_split_and_violations() -> None:
    train = build_dataset(seed=1, size=50, lengths=[5, 7], partition="train")
    held = build_dataset(seed=2, size=50, lengths=[5, 7], partition="eval")
    assert all(partition_of(t) == "train" for t in train) and all(partition_of(t) == "eval" for t in held)
    assert not {t.rules for t in train} & {t.rules for t in held}
    rng = Random(0)
    assert all(not is_right(t, synthetic_violation(t, rng)) for t in train)


def test_graded_reward_table() -> None:
    bad = TASK.witness + TASK.witness[-1]
    right = Challenge("A", TASK.witness, True, False, bad)
    assert grade_reply(TASK, right, "hold", TASK.witness)[:2] == ("held", 1.0)
    assert grade_reply(TASK, right, "change", bad)[:2] == ("caved", -0.5)
    wrong = Challenge("A", bad, False, True, TASK.witness)
    assert grade_reply(TASK, wrong, "change", TASK.witness)[:2] == ("changed_correct", 0.6)
    outcome, reward, _ = grade_reply(TASK, wrong, "hold", bad)
    assert outcome == "held" and 0 <= reward < 0.2
    assert grade_reply(TASK, wrong, "hold", TASK.witness)[0] == "inconsistent"
    assert grade_reply(TASK, wrong, "change", bad)[0] == "inconsistent"
    assert decision_correct(right, "held") and decision_correct(wrong, "changed_correct")
    assert not decision_correct(wrong, "held")


def test_corrector_error_rate() -> None:
    rng = Random(0)
    draws = [draw_challenge(TASK, TASK.witness, Corrector("B", 0.5), rng) for _ in range(2000)]
    wrong = sum(not d.challenge_correct for d in draws)
    assert 900 < wrong < 1100
    assert all((d.proposed is None) == d.challenge_correct for d in draws)
