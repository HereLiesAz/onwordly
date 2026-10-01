from random import Random

from onwordly.curricula.formal_logic import (
    AdaptiveLogicCurriculum,
    LogicBucket,
    UniformLogicCurriculum,
)
from onwordly.tasks.formal_logic import LogicExpr, make_logic_task, logic_partition
from onwordly.training.logic_sources import ErrorFocusedLogicSource


def test_logic_adaptive_weights_weaker_depth_higher() -> None:
    curriculum = AdaptiveLogicCurriculum(
        depths=(1, 2),
        prior_attempts=0,
        prior_correct=0,
    )
    easy = make_logic_task(
        LogicExpr("not", left=LogicExpr("var", value="A")),
        {"A": True, "B": False, "C": True, "D": False},
    )
    hard_expr = LogicExpr(
        "and",
        left=LogicExpr("not", left=LogicExpr("var", value="A")),
        right=LogicExpr(
            "or",
            left=LogicExpr("var", value="B"),
            right=LogicExpr("var", value="C"),
        ),
    )
    hard = make_logic_task(
        hard_expr,
        {"A": True, "B": False, "C": True, "D": False},
    )
    for _ in range(10):
        curriculum.update(easy, True)
        curriculum.update(hard, False)
    assert curriculum.weight(LogicBucket(2)) > curriculum.weight(LogicBucket(1))


def test_logic_uniform_respects_partition() -> None:
    curriculum = UniformLogicCurriculum(depths=(2,), partition="train")
    rng = Random(10)
    tasks = [curriculum.generate(rng) for _ in range(40)]
    assert all(logic_partition(task) == "train" for task in tasks)


def test_logic_error_focus_flips_assignments() -> None:
    curriculum = AdaptiveLogicCurriculum(depths=(2,), partition="train")
    source = ErrorFocusedLogicSource(curriculum, variants_per_failure=2)
    rng = Random(11)
    task = curriculum.generate(rng)
    source.observe(task, False)
    variants = [source.next_task(rng) for _ in range(2)]
    assert all(variant.assignment != task.assignment for variant in variants)
    assert all(logic_partition(variant) == "train" for variant in variants)
