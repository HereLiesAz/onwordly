"""Add-only episode store (docs/model-design.md, "Episode store").

Every attempt, challenge, correction, verifier result, deliberation and trust
update is one immutable record. Records are hash-chained: each carries the
SHA-256 of its own canonical content plus the previous record's hash, so any
edit to history is detectable by ``verify()``. There is no update or delete
API; a correction is a new record linked to the attempt it answers.

Prior art (not ours): event sourcing / append-only logs; hash chains as in
tamper-evident logs; immutable databases (Datomic†, XTDB†).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

RECORD_KINDS: tuple[str, ...] = (
    "attempt",
    "challenge",
    "correction",
    "verifier_result",
    "deliberation",
    "trust_update",
)
GENESIS_HASH = "0" * 64


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(index: int, kind: str, step: int, payload: str, links: tuple[int, ...], prev_hash: str) -> str:
    body = json.dumps([index, kind, step, payload, list(links), prev_hash], separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Record:
    """One immutable store entry. ``payload`` is canonical JSON (a string, so
    the record cannot be mutated through a nested dict); read it via ``data``."""

    index: int
    kind: str
    step: int
    payload: str
    links: tuple[int, ...]
    prev_hash: str
    hash: str

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self.payload)

    def to_json(self) -> str:
        return json.dumps(
            {
                "index": self.index,
                "kind": self.kind,
                "step": self.step,
                "payload": self.payload,
                "links": list(self.links),
                "prev_hash": self.prev_hash,
                "hash": self.hash,
            },
            sort_keys=True,
        )


class EpisodeStore:
    """Append-only, hash-chained log. Supports ``append``, reads and
    ``verify``; item assignment and deletion raise ``TypeError``."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: list[Record] = []

    def append(
        self,
        kind: str,
        data: Mapping[str, Any],
        *,
        step: int,
        links: Sequence[int] = (),
    ) -> Record:
        if kind not in RECORD_KINDS:
            raise ValueError(f"unknown record kind: {kind}")
        if step < 0:
            raise ValueError("step must be non-negative")
        index = len(self._records)
        link_tuple = tuple(int(link) for link in links)
        if any(not 0 <= link < index for link in link_tuple):
            raise ValueError("links must point to earlier records")
        payload = canonical_json(data)
        prev = self._records[-1].hash if self._records else GENESIS_HASH
        record = Record(index, kind, step, payload, link_tuple, prev, _digest(index, kind, step, payload, link_tuple, prev))
        self._records.append(record)
        return record

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(tuple(self._records))

    def __getitem__(self, index: int) -> Record:
        return self._records[index]

    @property
    def records(self) -> tuple[Record, ...]:
        return tuple(self._records)

    @property
    def head(self) -> str:
        return self._records[-1].hash if self._records else GENESIS_HASH

    def of_kind(self, kind: str) -> tuple[Record, ...]:
        return tuple(record for record in self._records if record.kind == kind)

    def counts(self) -> dict[str, int]:
        counts = {kind: 0 for kind in RECORD_KINDS}
        for record in self._records:
            counts[record.kind] += 1
        return counts

    def verify(self) -> bool:
        return verify_chain(self._records)

    def dump_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(record.to_json() + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "EpisodeStore":
        """Rebuild a store from a dump; raises if the chain does not verify."""
        records = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                records.append(
                    Record(raw["index"], raw["kind"], raw["step"], raw["payload"], tuple(raw["links"]), raw["prev_hash"], raw["hash"])
                )
        if not verify_chain(records):
            raise ValueError("episode store hash chain does not verify")
        store = cls()
        store._records.extend(records)
        return store


def verify_chain(records: Sequence[Record]) -> bool:
    prev = GENESIS_HASH
    for position, record in enumerate(records):
        if record.index != position or record.prev_hash != prev:
            return False
        if record.hash != _digest(record.index, record.kind, record.step, record.payload, record.links, record.prev_hash):
            return False
        prev = record.hash
    return True
