"""Trust ledger: per-domain self-trust and per-corrector reliability.

Each tracked quantity is a Beta(alpha, beta) belief over a success
probability -- Beta-Bernoulli reputation in the style of Josang's beta
reputation system† (established; not an Onwordly invention). Corrector
reliability is the simplest one-coin form of annotator-reliability models
(Dawid & Skene 1979†); truth discovery with source dependence (Dong et al.
2009†) is the fuller form, not implemented here.

- Self-trust (key = domain): success = the hold/change decision was right
  against the exact verifier (held a right answer, or changed a wrong one).
- Corrector reliability (key = corrector id): success = the corrector's claim
  was right.

Saturating: the prior never leaves the posterior and ``max_evidence`` caps the
accumulated evidence mass, so the mean approaches but never reaches 1 (or 0).
Decay: between updates, evidence relaxes toward the prior with half-life
``half_life`` (in ledger steps), so trust must be re-earned. Raw counts and a
consecutive-failure streak are kept alongside, undecayed, for the hand-coded
baseline (success rate + three-strike breaker).

Every update is appended to the ``EpisodeStore`` as a ``trust_update`` record
and only training outcomes may update it: ``record`` requires
``partition="train"`` and raises once the ledger is frozen for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from onwordly.memory.store import EpisodeStore

LEDGER_KINDS: tuple[str, ...] = ("self", "corrector")


@dataclass(frozen=True, slots=True)
class BetaTrust:
    alpha: float
    beta: float
    prior_alpha: float
    prior_beta: float
    step: int = 0
    successes: int = 0
    trials: int = 0
    failure_streak: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence(self) -> float:
        return self.alpha + self.beta - self.prior_alpha - self.prior_beta

    @property
    def success_rate(self) -> float | None:
        return None if self.trials == 0 else self.successes / self.trials

    def decayed(self, step: int, half_life: float | None) -> "BetaTrust":
        if half_life is None or step <= self.step:
            return self
        factor = 0.5 ** ((step - self.step) / half_life)
        return replace(
            self,
            alpha=self.prior_alpha + (self.alpha - self.prior_alpha) * factor,
            beta=self.prior_beta + (self.beta - self.prior_beta) * factor,
            step=step,
        )

    def updated(
        self,
        success: bool,
        *,
        step: int,
        weight: float = 1.0,
        half_life: float | None = None,
        max_evidence: float | None = None,
    ) -> "BetaTrust":
        if weight < 0:
            raise ValueError("weight must be non-negative")
        base = self.decayed(step, half_life)
        alpha = base.alpha + (weight if success else 0.0)
        beta = base.beta + (0.0 if success else weight)
        evidence = alpha + beta - base.prior_alpha - base.prior_beta
        if max_evidence is not None and evidence > max_evidence:
            scale = max_evidence / evidence
            alpha = base.prior_alpha + (alpha - base.prior_alpha) * scale
            beta = base.prior_beta + (beta - base.prior_beta) * scale
        return replace(
            base,
            alpha=alpha,
            beta=beta,
            step=max(step, base.step),
            successes=base.successes + int(success),
            trials=base.trials + 1,
            failure_streak=0 if success else base.failure_streak + 1,
        )


class TrustLedger:
    def __init__(
        self,
        store: EpisodeStore,
        *,
        prior: Sequence[float] = (1.0, 1.0),
        half_life: float | None = None,
        max_evidence: float | None = None,
    ) -> None:
        if len(prior) != 2 or min(prior) <= 0:
            raise ValueError("prior must be two positive numbers")
        if half_life is not None and half_life <= 0:
            raise ValueError("half_life must be positive")
        if max_evidence is not None and max_evidence <= 0:
            raise ValueError("max_evidence must be positive")
        self.store = store
        self.prior = (float(prior[0]), float(prior[1]))
        self.half_life = half_life
        self.max_evidence = max_evidence
        self._entries: dict[tuple[str, str], BetaTrust] = {}
        self._frozen = False
        self.step = 0

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """End of training: no further updates; reads keep the final state."""
        self._frozen = True

    def entry(self, kind: str, key: str, step: int | None = None) -> BetaTrust:
        if kind not in LEDGER_KINDS:
            raise ValueError(f"unknown ledger kind: {kind}")
        current = self._entries.get((kind, key))
        if current is None:
            return BetaTrust(*self.prior, *self.prior)
        return current.decayed(self.step if step is None else step, self.half_life)

    def self_trust(self, domain: str, step: int | None = None) -> float:
        return self.entry("self", domain, step).mean

    def corrector_reliability(self, corrector: str, step: int | None = None) -> float:
        return self.entry("corrector", corrector, step).mean

    def record(
        self,
        kind: str,
        key: str,
        success: bool,
        *,
        step: int,
        partition: str,
        weight: float = 1.0,
        links: Sequence[int] = (),
    ) -> BetaTrust:
        if partition != "train":
            raise ValueError("the trust ledger is updated from training outcomes only")
        if self._frozen:
            raise RuntimeError("the trust ledger is frozen (evaluation)")
        if step < self.step:
            raise ValueError("ledger steps must not go backwards")
        if kind not in LEDGER_KINDS:
            raise ValueError(f"unknown ledger kind: {kind}")
        current = self._entries.get((kind, key)) or BetaTrust(*self.prior, *self.prior, step=step)
        new = current.updated(
            bool(success), step=step, weight=weight, half_life=self.half_life, max_evidence=self.max_evidence
        )
        self._entries[(kind, key)] = new
        self.step = step
        self.store.append(
            "trust_update",
            {
                "ledger": kind,
                "key": key,
                "success": bool(success),
                "weight": weight,
                "alpha": new.alpha,
                "beta": new.beta,
                "mean": new.mean,
                "partition": partition,
            },
            step=step,
            links=links,
        )
        return new

    def snapshot(self) -> dict[str, dict[str, dict[str, float | int | None]]]:
        out: dict[str, dict[str, dict[str, float | int | None]]] = {kind: {} for kind in LEDGER_KINDS}
        for (kind, key) in sorted(self._entries):
            entry = self.entry(kind, key)
            out[kind][key] = {
                "mean": entry.mean,
                "alpha": entry.alpha,
                "beta": entry.beta,
                "success_rate": entry.success_rate,
                "trials": entry.trials,
                "failure_streak": entry.failure_streak,
            }
        return out

    @classmethod
    def replay(
        cls,
        store: EpisodeStore,
        *,
        prior: Sequence[float] = (1.0, 1.0),
        half_life: float | None = None,
        max_evidence: float | None = None,
    ) -> "TrustLedger":
        """Re-derive a ledger from a store's ``trust_update`` records (the
        ledger is a projection of the log, as in event sourcing)."""
        ledger = cls(EpisodeStore(), prior=prior, half_life=half_life, max_evidence=max_evidence)
        for record in store.of_kind("trust_update"):
            data = record.data
            ledger.record(
                data["ledger"], data["key"], data["success"], step=record.step, partition=data["partition"], weight=data["weight"]
            )
        return ledger
