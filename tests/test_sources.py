from random import Random

from onwordly.curricula.adaptive import AdaptiveArithmeticCurriculum, Bucket
from onwordly.tasks.arithmetic import make_arithmetic_task
from onwordly.training.sources import ErrorFocusedArithmeticSource, StaticArithmeticSource


def test_static_source_cycles_frozen_tasks() -> None:
    tasks = (
        make_arithmetic_task(2, 1, "add"),
        make_arithmetic_task(3, 1, "add"),
    )
    source = StaticArithmeticSource(tasks)
    rng = Random(1)
    assert source.next_task(rng) == tasks[0]
    assert source.next_task(rng) == tasks[1]
    assert source.next_task(rng) == tasks[0]


def test_error_focused_source_queues_nearby_failures() -> None:
    curriculum = AdaptiveArithmeticCurriculum(operations=("add",), digit_levels=(2,))
    source = ErrorFocusedArithmeticSource(curriculum, variants_per_failure=3)
    failed = make_arithmetic_task(20, 30, "add")
    source.observe(failed, False)

    variants = [source.next_task(Random(1)) for _ in range(3)]
    assert all(task.operation == failed.operation for task in variants)
    assert all(task.digits == failed.digits for task in variants)
    assert all((task.left, task.right) != (failed.left, failed.right) for task in variants)
    assert curriculum.stats[Bucket("add", 2)].attempts == 1
