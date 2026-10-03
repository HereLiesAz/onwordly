"""Regressions for the repository audit (2026-10-03)."""
from random import Random

from onwordly.datasets.formal_logic import (
    build_logic_composition_dataset,
    build_static_logic_dataset,
)
from onwordly.tasks.formal_logic import logic_partition, make_logic_task
from onwordly.verifiers.arithmetic import parse_integer_answer
from onwordly.verifiers.program_execution import (
    parse_program_trace,
    parse_supervised_program_final,
)


def test_integer_parser_rejects_non_canonical_forms() -> None:
    for text in ("٥", "+5", "007", "-0", "1_0", "５"):
        assert parse_integer_answer(text) is None, text
    assert parse_integer_answer(" -12 ") == -12
    assert parse_integer_answer("0") == 0


def test_trace_parser_rejects_non_canonical_states() -> None:
    assert parse_program_trace("1_0,2") is None
    assert parse_program_trace("01,2") is None
    assert parse_program_trace("10,-2") == (10, -2)


def test_final_parser_rejects_prefix_junk() -> None:
    assert parse_supervised_program_final("FINAL=3") == 3
    assert parse_supervised_program_final("TRACE=1,3;FINAL=3") == 3
    assert parse_supervised_program_final("garbage FINAL=3") is None
    assert parse_supervised_program_final("TRACE=x;FINAL=3") is None


def test_static_logic_dataset_builds_with_manifest_depths() -> None:
    tasks = build_static_logic_dataset(seed=1, size=9, depths=(1, 2, 3))
    assert {task.depth for task in tasks} <= {1, 2, 3}


def test_logic_composition_skips_ineligible_depths() -> None:
    tasks = build_logic_composition_dataset(
        seed=1, size=4, composition=("and", "or"), depths=(1, 2, 3)
    )
    assert all(task.depth >= 2 for task in tasks)


def test_logic_partition_holds_out_whole_formulas() -> None:
    task = build_static_logic_dataset(seed=3, size=1, depths=(2,))[0]
    rng = Random(0)
    for _ in range(8):
        assignment = {name: rng.random() < 0.5 for name, _ in task.assignment}
        variant = make_logic_task(task.expression, assignment)
        assert logic_partition(variant) == logic_partition(task)
