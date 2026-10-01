from random import Random

from onwordly.curricula.adaptive import AdaptiveArithmeticCurriculum, Bucket
from onwordly.tasks.arithmetic import ArithmeticTask


def _task(operation: str, digits: int) -> ArithmeticTask:
    return ArithmeticTask(
        prompt="",
        answer=0,
        operation=operation,
        digits=digits,
        left=0,
        right=0,
    )


def test_weaker_bucket_gets_higher_weight() -> None:
    curriculum = AdaptiveArithmeticCurriculum(
        operations=("add",),
        digit_levels=(1, 2),
        prior_attempts=0,
        prior_correct=0,
    )

    for _ in range(10):
        curriculum.update(_task("add", 1), True)
        curriculum.update(_task("add", 2), False)

    easy = Bucket("add", 1)
    hard = Bucket("add", 2)

    assert curriculum.weight(hard) > curriculum.weight(easy)


def test_sampling_favors_weaker_bucket() -> None:
    curriculum = AdaptiveArithmeticCurriculum(
        operations=("add",),
        digit_levels=(1, 2),
        minimum_weight=0.01,
        prior_attempts=0,
        prior_correct=0,
    )

    for _ in range(20):
        curriculum.update(_task("add", 1), True)
        curriculum.update(_task("add", 2), False)

    rng = Random(19)
    counts = {1: 0, 2: 0}
    for _ in range(500):
        counts[curriculum.choose_bucket(rng).digits] += 1

    assert counts[2] > counts[1]
