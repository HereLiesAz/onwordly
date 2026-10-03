from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from random import Random
from typing import Literal, Mapping, Sequence

LogicKind = Literal["var", "not", "and", "or", "xor"]
LogicPartition = Literal["train", "eval"]


@dataclass(frozen=True, slots=True)
class LogicExpr:
    kind: LogicKind
    value: str | None = None
    left: "LogicExpr | None" = None
    right: "LogicExpr | None" = None

    def render(self) -> str:
        if self.kind == "var":
            if self.value is None:
                raise ValueError("variable expression is missing its name")
            return self.value
        if self.kind == "not":
            if self.left is None:
                raise ValueError("NOT expression is missing its operand")
            return f"(NOT {self.left.render()})"
        if self.left is None or self.right is None:
            raise ValueError(f"{self.kind.upper()} expression is missing an operand")
        operator = self.kind.upper()
        return f"({self.left.render()} {operator} {self.right.render()})"


@dataclass(frozen=True, slots=True)
class LogicTask:
    prompt: str
    answer: str
    expression: LogicExpr
    assignment: tuple[tuple[str, bool], ...]
    depth: int

    @property
    def target_text(self) -> str:
        return self.answer

    @property
    def bucket_key(self) -> str:
        return f"logic:{self.depth}"


def evaluate_logic(expr: LogicExpr, assignment: Mapping[str, bool]) -> bool:
    if expr.kind == "var":
        if expr.value is None or expr.value not in assignment:
            raise ValueError("unbound logic variable")
        return bool(assignment[expr.value])
    if expr.kind == "not":
        if expr.left is None:
            raise ValueError("NOT expression is missing its operand")
        return not evaluate_logic(expr.left, assignment)
    if expr.left is None or expr.right is None:
        raise ValueError(f"{expr.kind.upper()} expression is missing an operand")
    left = evaluate_logic(expr.left, assignment)
    right = evaluate_logic(expr.right, assignment)
    if expr.kind == "and":
        return left and right
    if expr.kind == "or":
        return left or right
    if expr.kind == "xor":
        return left != right
    raise ValueError(f"unsupported logic kind: {expr.kind}")


def logic_contains_operator_composition(
    expr: LogicExpr,
    composition: tuple[LogicKind, LogicKind],
) -> bool:
    """Return whether a parent operator directly contains the requested child operator."""
    parent_kind, child_kind = composition
    if parent_kind == "var" or child_kind == "var":
        raise ValueError("operator composition cannot contain var")
    if expr.kind == parent_kind:
        children = tuple(
            child
            for child in (expr.left, expr.right)
            if child is not None
        )
        if any(child.kind == child_kind for child in children):
            return True
    return any(
        logic_contains_operator_composition(child, composition)
        for child in (expr.left, expr.right)
        if child is not None
    )


def expression_depth(expr: LogicExpr) -> int:
    if expr.kind == "var":
        return 0
    if expr.kind == "not":
        if expr.left is None:
            raise ValueError("NOT expression is missing its operand")
        return 1 + expression_depth(expr.left)
    if expr.left is None or expr.right is None:
        raise ValueError("binary expression is missing an operand")
    return 1 + max(expression_depth(expr.left), expression_depth(expr.right))


def make_logic_task(
    expression: LogicExpr,
    assignment: Mapping[str, bool],
) -> LogicTask:
    normalized_assignment = tuple(sorted((name, bool(value)) for name, value in assignment.items()))
    result = evaluate_logic(expression, dict(normalized_assignment))
    assignment_text = ", ".join(
        f"{name}={'true' if value else 'false'}"
        for name, value in normalized_assignment
    )
    prompt = (
        f"Evaluate the propositional formula {expression.render()} under the assignment "
        f"{assignment_text}. Use ordinary Boolean AND, OR, XOR, and NOT. "
        "Return only true or false."
    )
    return LogicTask(
        prompt=prompt,
        answer="true" if result else "false",
        expression=expression,
        assignment=normalized_assignment,
        depth=expression_depth(expression),
    )


def generate_logic_expr(
    rng: Random,
    depth: int,
    *,
    variables: Sequence[str] = ("A", "B", "C", "D"),
) -> LogicExpr:
    if depth < 0:
        raise ValueError("depth cannot be negative")
    names = tuple(name.upper() for name in variables)
    if not names or any(len(name) != 1 or not name.isalpha() or not name.isascii() for name in names):
        raise ValueError("variables must be single ASCII letters")
    if depth == 0:
        return LogicExpr("var", value=rng.choice(names))
    if rng.random() < 0.25:
        return LogicExpr(
            "not",
            left=generate_logic_expr(rng, depth - 1, variables=names),
        )
    kind = rng.choice(("and", "or", "xor"))
    return LogicExpr(
        kind,
        left=generate_logic_expr(rng, depth - 1, variables=names),
        right=generate_logic_expr(rng, depth - 1, variables=names),
    )


def generate_logic_task(
    rng: Random,
    depth: int,
    *,
    variables: Sequence[str] = ("A", "B", "C", "D"),
) -> LogicTask:
    expression = generate_logic_expr(rng, depth, variables=variables)
    assignment = {name.upper(): bool(rng.getrandbits(1)) for name in variables}
    return make_logic_task(expression, assignment)


def logic_partition(task: LogicTask, *, modulus: int = 5) -> LogicPartition:
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    # Partition by formula alone: every assignment of a held-out formula is held
    # out, so "held-out" means an unseen formula, not an unseen assignment.
    key = task.expression.render().encode("utf-8")
    residue = int.from_bytes(blake2b(key, digest_size=8).digest(), "big") % modulus
    return "eval" if residue == 0 else "train"
