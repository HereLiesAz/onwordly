"""Experiment 000 manifest."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ARMS: tuple[str, ...] = (
    "learner",  # full Onwordly learner: memory + trust inputs, confidence x surprise weighting
    "learner-flat",  # baseline 2: same, flat update weighting
    "learner-no-memory",  # baseline 3: memory read zeroed
    "learner-no-trust",  # baseline 4: trust read zeroed
    "plain",  # baseline 1: one-pass MLP, standard supervised loss, matched parameters
    "handcoded",  # baseline 5: aive-style hold/change rule on top of "plain"
)


@dataclass(frozen=True, slots=True)
class LearnerManifest:
    lengths: tuple[int, ...]
    alphabet: tuple[str, ...]
    optional_rules: int
    holdout_modulus: int
    train_size: int
    eval_size: int
    dataset_seed: int
    evaluation_seed: int
    training_seed: int
    # Matched budget (every arm): train_steps optimizer steps x batch_size
    # training problems, drawn in the same order. Revision steps and challenge
    # passes are extra forward compute, recorded separately.
    batch_size: int
    train_steps: int
    revision_steps: int
    d_model: int
    layers: int
    heads: int
    learning_rate: float
    correctors: tuple[tuple[str, float], ...] = (("A", 0.1), ("B", 0.5))
    unseen_corrector: tuple[str, float] = ("C", 0.3)
    trust_prior: tuple[float, float] = (1.0, 1.0)
    trust_half_life: float | None = 20000.0
    trust_max_evidence: float | None = 500.0
    update_gain: float = 2.0
    update_max_weight: float = 3.0
    register_cap: int = 6
    handcoded_threshold: float = 0.7
    handcoded_strikes: int = 3
    arms: tuple[str, ...] = field(default=ARMS)

    @property
    def slots(self) -> int:
        # One spare slot so one-edit insertions (proposals) fit the workspace.
        return max(self.lengths) + 1

    @classmethod
    def from_json(cls, path: str | Path) -> "LearnerManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("lengths", "alphabet", "trust_prior", "arms"):
            if key in payload:
                payload[key] = tuple(payload[key])
        if "correctors" in payload:
            payload["correctors"] = tuple((str(n), float(r)) for n, r in payload["correctors"])
        if "unseen_corrector" in payload:
            payload["unseen_corrector"] = (str(payload["unseen_corrector"][0]), float(payload["unseen_corrector"][1]))
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.lengths or min(self.lengths) < 2:
            raise ValueError("lengths must be at least 2")
        if len(set(self.alphabet)) < 3 or any(len(c) != 1 for c in self.alphabet):
            raise ValueError("alphabet needs at least three single characters")
        if self.train_size < 1 or self.eval_size < 1 or self.batch_size < 2 or self.train_steps < 1:
            raise ValueError("sizes, batch_size (>= 2) and train_steps must be positive")
        if self.revision_steps < 1:
            raise ValueError("revision_steps must be at least 1")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if len(self.correctors) < 2:
            raise ValueError("at least two training correctors are needed for reliability to be learnable")
        names = [name for name, _ in self.correctors]
        if len(set(names)) != len(names) or self.unseen_corrector[0] in names:
            raise ValueError("corrector names must be unique and the unseen corrector must be new")
        for _, rate in (*self.correctors, self.unseen_corrector):
            if not 0.0 <= rate < 1.0:
                raise ValueError("corrector error rates must be in [0, 1)")
        if set(self.arms) - set(ARMS) or not self.arms:
            raise ValueError(f"arms must be chosen from {ARMS}")
        if "handcoded" in self.arms and "plain" not in self.arms:
            raise ValueError("the handcoded arm runs on top of the plain arm")
