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
    "learner-self-memory",  # diagnostic: register read limited to source="self" fillers
    "learner-memory-dropout",  # diagnostic: training register read replaced by empty with prob memory_dropout
    "learner-first-visit",  # remedy (broken by design, run 3: decision untrained without memory); code kept, not in manifests
    "learner-child",  # childhood: memory-dropout with no self-trust input, flat weighting, reads only corrector fillers
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
    # Diagnostics (README, "Memory diagnostics"): probability that a training
    # register read is replaced by the empty-register encoding
    # (learner-memory-dropout only), and the number of training problems in
    # the post-training register probe.
    memory_dropout: float = 0.5
    probe_size: int = 512
    # Recurring-frame evaluation (README, "Recurring frames"): this many
    # held-out frames, each visited recurring_visits times in a deterministic
    # shuffled stream with at least recurring_min_gap other visits between two
    # visits of the same frame. recurring_visits = 0 disables it.
    recurring_frames: int = 500
    recurring_visits: int = 4
    recurring_min_gap: int = 25
    # Eval-time memory-source ablation on the recurring stream (no retraining):
    # for each listed arm, the stream is rerun with eval-register reads limited
    # to each of these sources ("self", "correctors", "all") other than the
    # arm's own, reported as rows "<arm>[<sources>]".
    recurring_source_ablation: tuple[str, ...] = ()
    recurring_ablation_sources: tuple[str, ...] = ("self", "correctors", "all")
    arms: tuple[str, ...] = field(default=ARMS)

    @property
    def slots(self) -> int:
        # One spare slot so one-edit insertions (proposals) fit the workspace.
        return max(self.lengths) + 1

    @classmethod
    def from_json(cls, path: str | Path) -> "LearnerManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("lengths", "alphabet", "trust_prior", "arms", "recurring_source_ablation", "recurring_ablation_sources"):
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
        if not 0.0 <= self.memory_dropout <= 1.0:
            raise ValueError("memory_dropout must be in [0, 1]")
        if self.probe_size < 1:
            raise ValueError("probe_size must be positive")
        if self.recurring_visits:
            if self.recurring_visits < 2 or self.recurring_frames < 1 or self.recurring_min_gap < 1:
                raise ValueError("recurring_visits must be 0 or >= 2; recurring_frames and recurring_min_gap positive")
            if self.recurring_frames > self.eval_size or 2 * self.recurring_min_gap > self.recurring_frames:
                raise ValueError("need recurring_frames <= eval_size and 2 * recurring_min_gap <= recurring_frames")
        if set(self.recurring_source_ablation) - set(self.arms):
            raise ValueError("recurring_source_ablation arms must be in arms")
        if set(self.recurring_ablation_sources) - {"self", "correctors", "all"}:
            raise ValueError("recurring_ablation_sources must be chosen from self, correctors, all")
        if set(self.arms) - set(ARMS) or not self.arms:
            raise ValueError(f"arms must be chosen from {ARMS}")
        if "handcoded" in self.arms and "plain" not in self.arms:
            raise ValueError("the handcoded arm runs on top of the plain arm")
