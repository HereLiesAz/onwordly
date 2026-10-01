from random import Random

from onwordly.curricula.uniform import UniformArithmeticCurriculum
from onwordly.tasks.arithmetic import arithmetic_partition


def test_uniform_curriculum_respects_train_partition() -> None:
    curriculum = UniformArithmeticCurriculum(
        operations=("add", "multiply"),
        digit_levels=(1, 2),
        partition="train",
    )
    rng = Random(17)
    tasks = [curriculum.generate(rng) for _ in range(80)]
    assert all(
        arithmetic_partition(task.left, task.right, task.operation) == "train"
        for task in tasks
    )


def test_uniform_curriculum_does_not_adapt_after_observation() -> None:
    curriculum = UniformArithmeticCurriculum(operations=("add",), digit_levels=(1,))
    rng = Random(4)
    task = curriculum.generate(rng)
    curriculum.update(task, False)
    assert len(curriculum.buckets) == 1
