from random import Random

from onwordly.curricula.program_execution import (
    AdaptiveProgramCurriculum,
    ProgramBucket,
    UniformProgramCurriculum,
)
from onwordly.tasks.program_execution import Instruction, make_program_task, program_partition
from onwordly.training.program_sources import ErrorFocusedProgramSource


def test_program_adaptive_weights_weaker_length_higher() -> None:
    curriculum = AdaptiveProgramCurriculum(
        lengths=(3, 5),
        prior_attempts=0,
        prior_correct=0,
    )
    easy = make_program_task((Instruction("SET", 1), Instruction("ADD", 1), Instruction("MUL", 2)))
    hard = make_program_task((Instruction("SET", 1), Instruction("ADD", 1), Instruction("MUL", 2), Instruction("SUB", 1), Instruction("NEG")))
    for _ in range(10):
        curriculum.update(easy, True)
        curriculum.update(hard, False)
    assert curriculum.weight(ProgramBucket(5)) > curriculum.weight(ProgramBucket(3))


def test_program_uniform_respects_partition() -> None:
    curriculum = UniformProgramCurriculum(lengths=(4,), partition="train")
    rng = Random(8)
    tasks = [curriculum.generate(rng) for _ in range(40)]
    assert all(program_partition(task) == "train" for task in tasks)


def test_program_error_focus_queues_nearby_variants() -> None:
    curriculum = AdaptiveProgramCurriculum(lengths=(4,), partition="train")
    source = ErrorFocusedProgramSource(curriculum, variants_per_failure=2)
    rng = Random(9)
    task = curriculum.generate(rng)
    source.observe(task, False)
    variants = [source.next_task(rng) for _ in range(2)]
    assert all(variant.program_text != task.program_text for variant in variants)
    assert all(program_partition(variant) == "train" for variant in variants)
