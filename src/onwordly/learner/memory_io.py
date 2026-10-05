"""The learner's external, non-parametric memory: writes and reads.

Writes go to the add-only ``EpisodeStore`` (attempt, challenge, correction,
deliberation, verifier_result, trust_update), the ``VariantRegister`` (draft,
proposal, final and verifier fillers per frame) and the ``TrustLedger``
(training outcomes only). Reads turn a frame's register summary and the ledger
estimates into input features.

Register reads exclude ``verifier`` fillers (``sources="self"`` additionally
excludes corrector proposals; ``sources="correctors"`` additionally excludes
the model's own ``self`` fillers -- only what it was told): on a revisited training frame
they would hand the network the answer, a shortcut that cannot exist on
held-out frames. The verifier filler is still stored.

Evaluation uses ``eval_view()``: same frozen ledger, a forked register and a
fresh scratch store, so nothing evaluation writes can reach training memory,
and the ledger raises on any update. ``eval_view(..., register_verifier=True)``
(the recurring-frame evaluation) also stores the verifier filler in the forked
register, exactly as training does; reads still exclude it.
"""
from __future__ import annotations

import math

from onwordly.learner.model import Encoding, MEMORY_GLOBAL_DIM, TRUST_DIM
from onwordly.learner.task import Challenge, ConstrainedTask, is_right
from onwordly.memory import EpisodeStore, TrustLedger, VariantRegister, frame_for


class LearnerMemory:
    def __init__(
        self,
        encoding: Encoding,
        *,
        context: str,
        prior: tuple[float, float] = (1.0, 1.0),
        half_life: float | None = None,
        max_evidence: float | None = None,
        register_cap: int = 6,
        partition: str = "train",
    ) -> None:
        self.encoding = encoding
        self.context = context
        self.partition = partition
        self.register_cap = register_cap
        self.store = EpisodeStore()
        self.register = VariantRegister()
        self.ledger = TrustLedger(self.store, prior=prior, half_life=half_life, max_evidence=max_evidence)
        self.step = 0
        self.divergences = 0
        self.register_verifier = partition == "train"

    def eval_view(self, context: str, *, register_verifier: bool = False) -> "LearnerMemory":
        self.ledger.freeze()
        view = LearnerMemory.__new__(LearnerMemory)
        view.encoding = self.encoding
        view.context = context
        view.partition = "eval"
        view.register_cap = self.register_cap
        view.store = EpisodeStore()
        view.register = self.register.fork()
        view.ledger = self.ledger
        view.step = self.step
        view.divergences = 0
        view.register_verifier = register_verifier
        return view

    # --- reads -----------------------------------------------------------------

    def _excluded(self, task: ConstrainedTask, sources: str) -> tuple[str, ...]:
        if sources == "all":
            return ("verifier",)
        if sources == "self":
            # Own fillers only: drop corrector proposals and the verifier.
            return tuple({e.source for e in self.register.entries(frame_for(task))} - {"self"}) or ("verifier",)
        if sources == "correctors":
            # What it was told only: drop its own fillers and the verifier.
            return ("self", "verifier")
        raise ValueError(f"unknown register sources: {sources}")

    def empty_read(self) -> tuple[list[float], list[list[float]]]:
        """Exactly what ``read`` returns for a frame with no readable fillers."""
        enc = self.encoding
        return [0.0] * MEMORY_GLOBAL_DIM, [[0.0] * (2 * enc.vocab) for _ in range(enc.slots)]

    def register_stats(self, task: ConstrainedTask, sources: str = "all") -> tuple[bool, str | None]:
        """(register readable for this frame is non-empty, top non-self filler by count or None).
        Diagnostic only; ties go to the most recent entry."""
        summary = self.register.summary(frame_for(task), exclude_sources=self._excluded(task, sources), cap=self.register_cap)
        others = [(count, i, filler) for i, (filler, source, count) in enumerate(summary.entries) if source != "self"]
        return summary.total > 0, (max(others)[2] if others else None)

    def write_self_filler(self, task: ConstrainedTask, filler: str) -> None:
        """Register-only write of the model's own answer (eval views only:
        used by the held-out second-visit probe)."""
        if self.partition != "eval":
            raise RuntimeError("write_self_filler is for evaluation views only")
        self._add(task, filler, "self")

    def read(self, task: ConstrainedTask, sources: str = "all") -> tuple[list[float], list[list[float]]]:
        """(global [log-count, distinct fillers, divergent], per-slot [self hist | others hist]).
        ``sources="self"`` reads only the model's own fillers."""
        enc = self.encoding
        summary = self.register.summary(frame_for(task), exclude_sources=self._excluded(task, sources), cap=self.register_cap)
        mine = [[0.0] * enc.vocab for _ in range(enc.slots)]
        others = [[0.0] * enc.vocab for _ in range(enc.slots)]
        weight = {"self": 0, "other": 0}
        for filler, source, count in summary.entries:
            target = mine if source == "self" else others
            weight["self" if source == "self" else "other"] += count
            for slot, idx in enumerate(enc.slots_of(filler)):
                target[slot][idx] += count
        for table, key in ((mine, "self"), (others, "other")):
            if weight[key]:
                for row in table:
                    for j in range(len(row)):
                        row[j] /= weight[key]
        glob = [math.log1p(summary.total) / 3.0, summary.distinct_fillers / 8.0, float(summary.divergent)]
        assert len(glob) == MEMORY_GLOBAL_DIM
        return glob, [m + o for m, o in zip(mine, others)]

    def trust(self, task: ConstrainedTask, corrector: str | None) -> list[float]:
        out = [
            self.ledger.self_trust(task.domain),
            self.ledger.corrector_reliability(corrector) if corrector else 0.0,
            1.0 if corrector else 0.0,
        ]
        assert len(out) == TRUST_DIM
        return out

    # --- writes ----------------------------------------------------------------

    def _add(self, task: ConstrainedTask, filler: str, source: str) -> None:
        _, markers = self.register.add(
            frame_for(task), filler, record_time=self.step, context=self.context, subject=task.domain, source=source
        )
        self.divergences += len(markers)

    def write_challenge(self, task: ConstrainedTask, draft: str, challenge: Challenge) -> tuple[int, int]:
        """Attempt + challenge (+ correction) records; draft and proposal into
        the register. Returns (attempt index, challenge index)."""
        frame = frame_for(task)
        attempt = self.store.append(
            "attempt", {"frame": frame, "domain": task.domain, "answer": draft, "partition": self.partition}, step=self.step
        )
        self._add(task, draft, "self")
        claim = self.store.append(
            "challenge",
            {"frame": frame, "corrector": challenge.corrector, "claim": "wrong" if challenge.says_wrong else "right",
             "proposed": challenge.proposed, "partition": self.partition},
            step=self.step,
            links=(attempt.index,),
        )
        if challenge.proposed is not None:
            self.store.append(
                "correction",
                {"frame": frame, "corrector": challenge.corrector, "proposed": challenge.proposed, "partition": self.partition},
                step=self.step,
                links=(attempt.index, claim.index),
            )
            self._add(task, challenge.proposed, challenge.corrector)
        return attempt.index, claim.index

    def write_outcome(
        self,
        task: ConstrainedTask,
        challenge: Challenge,
        links: tuple[int, int],
        *,
        action: str,
        value: str | None,
        outcome: str,
        reward: float,
        final: str | None,
        decision_ok: bool,
        confidence: float,
        inputs: dict[str, object],
    ) -> None:
        """Deliberation + verifier result; register fillers; ledger updates
        (training only). ``inputs`` names what the reasoner read."""
        frame = frame_for(task)
        deliberation = self.store.append(
            "deliberation",
            {"frame": frame, "action": action, "value": value, "confidence": round(confidence, 6),
             "considered": inputs, "partition": self.partition},
            step=self.step,
            links=links,
        )
        verdict = self.store.append(
            "verifier_result",
            {"frame": frame, "first_right": challenge.first_right, "challenge_correct": challenge.challenge_correct,
             "outcome": outcome, "reward": reward, "final_right": is_right(task, final),
             "decision_correct": decision_ok, "partition": self.partition},
            step=self.step,
            links=(*links, deliberation.index),
        )
        if final is not None and final != challenge.first:
            self._add(task, final, "self")
        if self.register_verifier:
            self._add(task, task.witness, "verifier")  # stored, never read (see module docstring)
        if self.partition == "train":
            self.ledger.record("self", task.domain, decision_ok, step=self.step, partition="train", links=(verdict.index,))
            self.ledger.record(
                "corrector", challenge.corrector, challenge.challenge_correct, step=self.step, partition="train",
                links=(verdict.index,),
            )
        self.step += 1

    def report(self) -> dict[str, object]:
        return {
            "records": len(self.store),
            "record_counts": self.store.counts(),
            "head_hash": self.store.head,
            "chain_verified": self.store.verify(),
            "frames": len(self.register.frames()),
            "divergence_markers": self.divergences,
            "ledger": self.ledger.snapshot(),
            "ledger_steps": self.step,
        }
