"""Constrained string generation and its verdict move (Experiment 011).

A task lists independent rules over a short string (allowed alphabet, exact
length, first character, last character, a required character, a forbidden
character, no equal neighbours). Many strings satisfy a task. Every rule is
checked exactly and separately, so ``satisfaction(task, s)`` (the fraction of
rules ``s`` satisfies) gives graded credit, and judging a proposal needs only
local checks, not producing an answer first. That is the deliberate contrast
with Experiment 010's arithmetic, where judging ``47 * 6 = 272`` needs the
product. Generation-verification gaps are established prior work (see
``docs/novelty-ledger.md``); 011 uses one, it does not claim it.

Each task is built from a random witness string that satisfies every rule, so
it is satisfiable by construction; the witness is the SFT target and the
repair target. Any string that satisfies every rule is accepted as correct.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import blake2b
from random import Random
from typing import Literal, Sequence

ConstrainedPartition = Literal["train", "eval"]
DEFAULT_ALPHABET: tuple[str, ...] = tuple("ABCDE12345")

# Rule kinds, checked in this order. Values: allowed (alphabet string),
# length (int), starts/ends/contains/excludes (one character), no_repeat (None).
RULE_KINDS: tuple[str, ...] = ("allowed", "length", "starts", "ends", "contains", "excludes", "no_repeat")
Rule = tuple[str, "str | int | None"]

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class ConstrainedStringTask:
    prompt: str
    rules: tuple[Rule, ...]
    witness: str

    @property
    def answer(self) -> str:
        return self.witness

    @property
    def target_text(self) -> str:
        return self.witness

    @property
    def length(self) -> int:
        return int(dict(self.rules)["length"])  # type: ignore[arg-type]

    @property
    def bucket_key(self) -> str:
        return f"constrained:{len(self.rules)}:{self.length}"

    def verify(self, response: str) -> bool:
        candidate = parse_string_answer(response)
        return candidate is not None and satisfies_all(self, candidate)


def parse_string_answer(response: str) -> str | None:
    """The stripped reply if it is a single non-empty token, else ``None``."""
    text = response.strip()
    return text if _TOKEN.fullmatch(text) else None


def check_rule(rule: Rule, candidate: str) -> bool:
    kind, value = rule
    if kind == "allowed":
        return all(char in str(value) for char in candidate)
    if kind == "length":
        return len(candidate) == value
    if kind == "starts":
        return candidate.startswith(str(value))
    if kind == "ends":
        return candidate.endswith(str(value))
    if kind == "contains":
        return str(value) in candidate
    if kind == "excludes":
        return str(value) not in candidate
    if kind == "no_repeat":
        return all(a != b for a, b in zip(candidate, candidate[1:]))
    raise ValueError(f"unknown rule kind: {kind}")


def satisfaction(task: ConstrainedStringTask, candidate: str | None) -> float:
    """Fraction of the task's rules ``candidate`` satisfies; 0.0 for ``None``."""
    if candidate is None:
        return 0.0
    return sum(check_rule(rule, candidate) for rule in task.rules) / len(task.rules)


def satisfies_all(task: ConstrainedStringTask, candidate: str) -> bool:
    return all(check_rule(rule, candidate) for rule in task.rules)


def _describe(rule: Rule) -> str:
    kind, value = rule
    return {
        "allowed": lambda: f"use only the characters {' '.join(str(value))}",
        "length": lambda: f"have exactly {value} characters",
        "starts": lambda: f"start with {value}",
        "ends": lambda: f"end with {value}",
        "contains": lambda: f"contain {value}",
        "excludes": lambda: f"not contain {value}",
        "no_repeat": lambda: "never have the same character twice in a row",
    }[kind]()


def make_constrained_task(rules: Sequence[Rule], witness: str) -> ConstrainedStringTask:
    ordered = tuple(sorted(rules, key=lambda rule: RULE_KINDS.index(rule[0])))
    kinds = [kind for kind, _ in ordered]
    if len(set(kinds)) != len(kinds) or "allowed" not in kinds or "length" not in kinds:
        raise ValueError("rules need one each of 'allowed' and 'length' and no duplicate kinds")
    task_prompt = (
        "Write one string that satisfies every rule. It must "
        + "; ".join(_describe(rule) for rule in ordered)
        + ". Return only the string with no spaces or explanation."
    )
    task = ConstrainedStringTask(prompt=task_prompt, rules=ordered, witness=witness)
    if not satisfies_all(task, witness):
        raise ValueError("witness does not satisfy the rules")
    return task


def generate_constrained_task(
    rng: Random,
    length: int,
    *,
    alphabet: Sequence[str] = DEFAULT_ALPHABET,
    optional_rules: int = 3,
) -> ConstrainedStringTask:
    """Random witness of ``length``, then ``allowed`` + ``length`` + ``optional_rules``
    rules drawn from starts/ends/contains/excludes/no_repeat that it satisfies."""
    symbols = "".join(alphabet)
    if length < 2 or len(set(symbols)) < 3:
        raise ValueError("need length >= 2 and at least three distinct symbols")
    if not 0 <= optional_rules <= 5:
        raise ValueError("optional_rules must be in [0, 5]")
    no_repeat = rng.random() < 0.5
    chars: list[str] = []
    for _ in range(length):
        options = [c for c in symbols if not (no_repeat and chars and c == chars[-1])]
        chars.append(rng.choice(options))
    witness = "".join(chars)
    absent = [c for c in symbols if c not in witness]
    candidates: list[Rule] = [("starts", witness[0]), ("ends", witness[-1]), ("contains", rng.choice(witness))]
    if absent:
        candidates.append(("excludes", rng.choice(absent)))
    if no_repeat:
        candidates.append(("no_repeat", None))
    chosen = rng.sample(candidates, min(optional_rules, len(candidates)))
    return make_constrained_task([("allowed", symbols), ("length", length), *chosen], witness)


def constrained_partition(task: ConstrainedStringTask, *, modulus: int = 5) -> ConstrainedPartition:
    """Stable split keyed on the rule set alone (never on the witness)."""
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    key = repr(task.rules).encode("utf-8")
    residue = int.from_bytes(blake2b(key, digest_size=8).digest(), "big") % modulus
    return "eval" if residue == 0 else "train"


def build_constrained_dataset(
    *,
    seed: int,
    size: int,
    lengths: Sequence[int],
    alphabet: Sequence[str] = DEFAULT_ALPHABET,
    optional_rules: int = 3,
    partition: ConstrainedPartition | None = None,
    partition_modulus: int = 5,
) -> tuple[ConstrainedStringTask, ...]:
    if size < 1 or not lengths:
        raise ValueError("size must be positive and lengths non-empty")
    rng = Random(seed)
    tasks: list[ConstrainedStringTask] = []
    for index in range(size):
        for _ in range(10_000):
            task = generate_constrained_task(
                rng, lengths[index % len(lengths)], alphabet=alphabet, optional_rules=optional_rules
            )
            if partition is None or constrained_partition(task, modulus=partition_modulus) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError("could not generate a constrained task in the requested partition")
    rng.shuffle(tasks)
    return tuple(tasks)


def synthetic_violation(task: ConstrainedStringTask, rng: Random) -> str:
    """A near miss: the witness with one edit that breaks at least one rule."""
    witness = task.witness
    symbols = str(dict(task.rules)["allowed"])
    for _ in range(64):
        edit = rng.randrange(4)
        position = rng.randrange(len(witness))
        if edit == 0:  # substitute one character
            candidate = witness[:position] + rng.choice(symbols) + witness[position + 1 :]
        elif edit == 1:  # delete one character
            candidate = witness[:position] + witness[position + 1 :]
        elif edit == 2:  # insert one character
            candidate = witness[:position] + rng.choice(symbols) + witness[position:]
        else:  # duplicate a neighbour
            candidate = witness[: position + 1] + witness[position:]
        if candidate and not satisfies_all(task, candidate):
            return candidate
    return witness + witness[-1]  # wrong length: always violates 'length'


# --- verdict move -----------------------------------------------------------

_WRONG = re.compile(r"wrong:\s*(\S+)")


def parse_string_verdict(response: str) -> tuple[bool, str | None] | None:
    """``(True, None)`` for ``right``; ``(False, s)`` for ``wrong: s``; else ``None``."""
    text = response.strip()
    if text == "right":
        return True, None
    match = _WRONG.fullmatch(text)
    return None if match is None else (False, match.group(1))


@dataclass(frozen=True, slots=True)
class ConstrainedVerdictTask:
    prompt: str
    shown: str
    base: ConstrainedStringTask

    @property
    def shown_is_right(self) -> bool:
        return satisfies_all(self.base, self.shown)

    @property
    def target_text(self) -> str:
        return "right" if self.shown_is_right else f"wrong: {self.base.witness}"

    @property
    def bucket_key(self) -> str:
        return f"verdict-{'right' if self.shown_is_right else 'wrong'}:{self.base.bucket_key}"

    def verify(self, response: str) -> bool:
        verdict = parse_string_verdict(response)
        if verdict is None:
            return False
        if self.shown_is_right:
            return verdict == (True, None)
        return not verdict[0] and satisfies_all(self.base, str(verdict[1]))


def make_constrained_verdict_task(base: ConstrainedStringTask, shown: str) -> ConstrainedVerdictTask:
    prompt = (
        f"{base.prompt.rstrip()} A proposed string is {shown}. "
        "If it satisfies every rule, reply exactly: right. "
        "If it breaks any rule, reply exactly: wrong: <a string that satisfies every rule>."
    )
    return ConstrainedVerdictTask(prompt=prompt, shown=shown, base=base)
