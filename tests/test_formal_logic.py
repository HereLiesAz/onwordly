from collections import Counter
from random import Random

from onwordly.datasets.formal_logic import build_static_logic_dataset
from onwordly.tasks.formal_logic import (
    LogicExpr,
    evaluate_logic,
    generate_logic_task,
    logic_partition,
    make_logic_task,
)
from onwordly.verifiers.formal_logic import parse_logic_answer, verify_logic_answer


def test_logic_evaluator_handles_all_operators() -> None:
    assignment = {"A": True, "B": False}
    assert evaluate_logic(LogicExpr("and", left=LogicExpr("var", value="A"), right=LogicExpr("var", value="B")), assignment) is False
    assert evaluate_logic(LogicExpr("or", left=LogicExpr("var", value="A"), right=LogicExpr("var", value="B")), assignment) is True
    assert evaluate_logic(LogicExpr("xor", left=LogicExpr("var", value="A"), right=LogicExpr("var", value="B")), assignment) is True
    assert evaluate_logic(LogicExpr("not", left=LogicExpr("var", value="B")), assignment) is True


def test_generated_logic_tasks_verify() -> None:
    rng = Random(15)
    for depth in (1, 2, 3):
        task = generate_logic_task(rng, depth)
        assert verify_logic_answer(task, task.answer)
        assert not verify_logic_answer(task, f"The answer is {task.answer}.")


def test_logic_verifier_is_strict_but_case_insensitive() -> None:
    task = make_logic_task(LogicExpr("var", value="A"), {"A": True})
    assert verify_logic_answer(task, "TRUE")
    assert parse_logic_answer(" yes ") is None


def test_logic_partition_separates_datasets() -> None:
    train = build_static_logic_dataset(seed=1, size=40, depths=(2,), partition="train")
    evaluation = build_static_logic_dataset(seed=2, size=40, depths=(2,), partition="eval")
    assert all(logic_partition(task) == "train" for task in train)
    assert all(logic_partition(task) == "eval" for task in evaluation)


def test_logic_dataset_balances_depths() -> None:
    tasks = build_static_logic_dataset(seed=3, size=30, depths=(1, 2, 3))
    counts = Counter(task.depth for task in tasks)
    assert max(counts.values()) - min(counts.values()) <= 1
