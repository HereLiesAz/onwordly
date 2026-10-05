"""Frames, fillers, the variant register and divergence markers.

Frame = what an item is about; filler = what it says (docs/model-design.md,
"Similarity conflated frame with filler"). For arithmetic the frame is the
canonical problem signature (operation, normalised operands, digit level) and
the filler the integer answer; for constrained strings the frame is a hash of
the rule set and the filler the string.

The ``VariantRegister`` keeps, per frame, an add-only list of fillers, each
tagged with record time (training step), context (regime/phase), subject
(task family/bucket), source (``self``, a corrector id, or ``verifier``) and a
running occurrence count. Nothing in it is ranked as correct. Adding a filler
whose canonical form differs from one already in the frame yields a
``Divergence`` marker: "same question, different answer" -- a flag, never a
verdict.

Prior art: frame semantics (Fillmore; FrameNet); entity canonicalisation /
alias tables; minimal-pair alignment for contrast detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Callable, Iterable

Canonicalizer = Callable[[Any], str]


def canonical_filler(value: Any) -> str:
    """Default canonical form: integers as plain decimal, strings stripped."""
    if isinstance(value, bool):
        raise TypeError("booleans are not fillers")
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def arithmetic_frame(operation: str, left: int, right: int, digits: int) -> str:
    """Commutative operations normalise operand order (2+3 and 3+2 share a frame)."""
    if operation in ("add", "multiply") and right < left:
        left, right = right, left
    return f"arith:{operation}:{left}:{right}:d{digits}"


def rules_frame(rules: Iterable[tuple[str, Any]]) -> str:
    key = repr(tuple(sorted(tuple(rule) for rule in rules))).encode("utf-8")
    return "rules:" + blake2b(key, digest_size=8).hexdigest()


def frame_for(task: Any) -> str:
    """Frame of an ``ArithmeticTask`` or ``ConstrainedStringTask``."""
    if hasattr(task, "rules"):
        return rules_frame(task.rules)
    if hasattr(task, "operation"):
        return arithmetic_frame(task.operation, task.left, task.right, task.digits)
    raise TypeError(f"no frame for {type(task).__name__}")


@dataclass(frozen=True, slots=True)
class VariantEntry:
    filler: str
    record_time: int
    context: str
    subject: str
    source: str
    count: int  # occurrences of (filler, source) in this frame, this one included


@dataclass(frozen=True, slots=True)
class Divergence:
    frame: str
    existing: str
    new: str
    record_time: int
    source: str


@dataclass(frozen=True, slots=True)
class RegisterSummary:
    """What a reader gets: distinct (filler, source) pairs with counts, in
    first-seen order (never ranked), capped to the most recent ``cap``."""

    frame: str
    entries: tuple[tuple[str, str, int], ...]
    distinct_fillers: int
    divergent: bool
    total: int


class VariantRegister:
    def __init__(self, canonicalize: Canonicalizer = canonical_filler) -> None:
        self._canonicalize = canonicalize
        self._entries: dict[str, list[VariantEntry]] = {}
        self._divergences: list[Divergence] = []

    def add(
        self,
        frame: str,
        filler: Any,
        *,
        record_time: int,
        context: str,
        subject: str,
        source: str,
    ) -> tuple[VariantEntry, tuple[Divergence, ...]]:
        value = self._canonicalize(filler)
        entries = self._entries.setdefault(frame, [])
        seen = {entry.filler for entry in entries}
        count = 1 + sum(1 for entry in entries if entry.filler == value and entry.source == source)
        entry = VariantEntry(value, record_time, context, subject, source, count)
        markers = tuple(
            Divergence(frame, existing, value, record_time, source)
            for existing in sorted(seen)
            if existing != value and value not in seen
        )
        entries.append(entry)
        self._divergences.extend(markers)
        return entry, markers

    def entries(self, frame: str) -> tuple[VariantEntry, ...]:
        return tuple(self._entries.get(frame, ()))

    def divergences(self, frame: str | None = None) -> tuple[Divergence, ...]:
        if frame is None:
            return tuple(self._divergences)
        return tuple(marker for marker in self._divergences if marker.frame == frame)

    def frames(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def summary(self, frame: str, *, exclude_sources: Iterable[str] = (), cap: int = 8) -> RegisterSummary:
        excluded = set(exclude_sources)
        counts: dict[tuple[str, str], int] = {}
        first_seen: dict[tuple[str, str], int] = {}
        last_seen: dict[tuple[str, str], int] = {}
        for position, entry in enumerate(self._entries.get(frame, ())):
            if entry.source in excluded:
                continue
            key = (entry.filler, entry.source)
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, position)
            last_seen[key] = position
        recent = sorted(counts, key=last_seen.__getitem__)[-cap:] if cap > 0 else []
        keys = sorted(recent, key=first_seen.__getitem__)
        fillers = {filler for filler, _ in counts}
        return RegisterSummary(
            frame=frame,
            entries=tuple((filler, source, counts[(filler, source)]) for filler, source in keys),
            distinct_fillers=len(fillers),
            divergent=len(fillers) > 1,
            total=sum(counts.values()),
        )

    def fork(self) -> "VariantRegister":
        """An independent copy. Evaluation writes into a fork, so nothing it
        adds can reach the training register."""
        clone = VariantRegister(self._canonicalize)
        clone._entries = {frame: list(entries) for frame, entries in self._entries.items()}
        clone._divergences = list(self._divergences)
        return clone

    def __deepcopy__(self, memo: dict) -> "VariantRegister":
        return self.fork()


__all__ = [
    "Divergence",
    "RegisterSummary",
    "VariantEntry",
    "VariantRegister",
    "arithmetic_frame",
    "canonical_filler",
    "frame_for",
    "rules_frame",
]
