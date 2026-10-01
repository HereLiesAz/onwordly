from random import Random

from onwordly.curricula.string_manipulation import (
    AdaptiveStringCurriculum,
    StringBucket,
    UniformStringCurriculum,
)
from onwordly.tasks.string_manipulation import make_string_task, string_partition
from onwordly.training.string_sources import ErrorFocusedStringSource


def test_string_adaptive_weights_weaker_bucket_higher() -> None:
    curriculum = AdaptiveStringCurriculum(
        operations=("take_even_indices",),
        lengths=(4, 8),
        prior_attempts=0,
        prior_correct=0,
    )
    for _ in range(10):
        curriculum.update(make_string_task("AB12", "take_even_indices"), True)
        curriculum.update(make_string_task("ABCD1234", "take_even_indices"), False)
    assert curriculum.weight(StringBucket("take_even_indices", 8)) > curriculum.weight(
        StringBucket("take_even_indices", 4)
    )


def test_string_uniform_respects_partition() -> None:
    curriculum = UniformStringCurriculum(
        operations=("reverse_pairs",),
        lengths=(6,),
        partition="train",
    )
    rng = Random(5)
    tasks = [curriculum.generate(rng) for _ in range(40)]
    assert all(string_partition(task) == "train" for task in tasks)


def test_string_error_focus_queues_nearby_variants() -> None:
    curriculum = AdaptiveStringCurriculum(
        operations=("duplicate_each",),
        lengths=(6,),
        partition="train",
    )
    source = ErrorFocusedStringSource(curriculum, variants_per_failure=3)
    rng = Random(7)
    task = curriculum.generate(rng)
    source.observe(task, False)
    variants = [source.next_task(rng) for _ in range(3)]
    assert all(variant.text != task.text for variant in variants)
    assert all(string_partition(variant) == "train" for variant in variants)
