from random import Random

from onwordly.curricula.symbolic import (
    AdaptiveSymbolicCurriculum,
    SymbolicBucket,
    UniformSymbolicCurriculum,
)
from onwordly.tasks.symbolic import make_symbolic_task, symbolic_partition
from onwordly.training.symbolic_sources import ErrorFocusedSymbolicSource


def test_symbolic_adaptive_weights_weaker_bucket_higher() -> None:
    curriculum = AdaptiveSymbolicCurriculum(
        operations=("reverse",),
        lengths=(4, 8),
        prior_attempts=0,
        prior_correct=0,
    )
    for _ in range(10):
        curriculum.update(make_symbolic_task("ABCD", "reverse"), True)
        curriculum.update(make_symbolic_task("ABCDEFGH", "reverse"), False)
    assert curriculum.weight(SymbolicBucket("reverse", 8)) > curriculum.weight(
        SymbolicBucket("reverse", 4)
    )


def test_symbolic_uniform_respects_partition() -> None:
    curriculum = UniformSymbolicCurriculum(
        operations=("sort",),
        lengths=(6,),
        partition="train",
    )
    rng = Random(8)
    tasks = [curriculum.generate(rng) for _ in range(40)]
    assert all(symbolic_partition(task.symbols, task.operation) == "train" for task in tasks)


def test_symbolic_error_focus_queues_partition_safe_variants() -> None:
    curriculum = AdaptiveSymbolicCurriculum(
        operations=("reverse",),
        lengths=(6,),
        partition="train",
    )
    source = ErrorFocusedSymbolicSource(curriculum, variants_per_failure=3)
    rng = Random(11)
    task = curriculum.generate(rng)
    source.observe(task, False)
    variants = [source.next_task(rng) for _ in range(3)]
    assert all(variant.symbols != task.symbols for variant in variants)
    assert all(symbolic_partition(variant.symbols, variant.operation) == "train" for variant in variants)
