"""The Onwordly learner and its one-pass baseline (Experiment 000).

Representation: an editable answer workspace -- one slot per output position,
each holding logits over the alphabet plus PAD -- next to a global token
carrying the problem encoding, the memory read, the trust read and the
challenge. The answer is an object the network revises, not a token stream it
committed to.

Computation: ``steps`` revision steps with shared weights. Each step reads the
whole workspace (softmax of its logits), the global token and per-slot extras,
and adds a delta to *every* slot's logits, so any slot can change at any step.
Every step emits the current answer, a confidence and a hold logit. There is no
step embedding and no halting unit: the same network can be run for any number
of steps; the experiment fixes and records the budget.

Prior art (not ours): iterative refinement / recurrent-depth transformers
(Universal Transformers, Dehghani et al. 2019†), non-autoregressive iterative
decoding (Mask-Predict, Ghazvininejad et al. 2019†), deep supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from onwordly.learner.task import ConstrainedTask, RULE_KINDS

MEMORY_GLOBAL_DIM = 3
TRUST_DIM = 3
CHALLENGE_GLOBAL_DIM = 2


@dataclass(frozen=True, slots=True)
class Encoding:
    alphabet: tuple[str, ...]
    slots: int

    @property
    def vocab(self) -> int:
        return len(self.alphabet) + 1  # + PAD

    @property
    def pad(self) -> int:
        return len(self.alphabet)

    @property
    def problem_dim(self) -> int:
        a = len(self.alphabet)
        return a + (self.slots + 1) + 4 * (a + 1) + 1

    @property
    def global_dim(self) -> int:
        return self.problem_dim + MEMORY_GLOBAL_DIM + TRUST_DIM + CHALLENGE_GLOBAL_DIM

    @property
    def extra_dim(self) -> int:
        return 3 * self.vocab  # memory: self + others histograms; challenge: proposal

    def index(self, char: str) -> int:
        return self.alphabet.index(char) if char in self.alphabet else self.pad

    def problem(self, task: ConstrainedTask) -> list[float]:
        a = len(self.alphabet)
        rules = dict(task.rules)
        allowed = [1.0 if c in str(rules["allowed"]) else 0.0 for c in self.alphabet]
        length = [0.0] * (self.slots + 1)
        length[min(task.length, self.slots)] = 1.0
        out = allowed + length
        for kind in ("starts", "ends", "contains", "excludes"):
            onehot = [0.0] * (a + 1)
            value = rules.get(kind)
            onehot[self.index(str(value)) if value is not None else a] = 1.0
            out += onehot
        out.append(1.0 if "no_repeat" in rules else 0.0)
        assert len(out) == self.problem_dim and set(rules) <= set(RULE_KINDS)
        return out

    def slots_of(self, text: str) -> list[int]:
        ids = [self.index(c) for c in text[: self.slots]]
        return ids + [self.pad] * (self.slots - len(ids))

    def onehot(self, text: str) -> list[list[float]]:
        rows = []
        for idx in self.slots_of(text):
            row = [0.0] * self.vocab
            row[idx] = 1.0
            rows.append(row)
        return rows

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.alphabet[i] for i in ids if i != self.pad)


@dataclass(frozen=True, slots=True)
class StepOutput:
    logits: torch.Tensor  # (batch, slots, vocab)
    confidence_logit: torch.Tensor  # (batch,)
    hold_logit: torch.Tensor  # (batch,)


class OnwordlyLearner(nn.Module):
    def __init__(self, encoding: Encoding, *, d_model: int, layers: int, heads: int) -> None:
        super().__init__()
        self.encoding = encoding
        self.global_in = nn.Sequential(nn.Linear(encoding.global_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.workspace_in = nn.Linear(encoding.vocab, d_model, bias=False)
        self.extra_in = nn.Linear(encoding.extra_dim, d_model)
        self.position = nn.Parameter(torch.randn(encoding.slots, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.revise = nn.Linear(d_model, encoding.vocab)
        self.confidence = nn.Linear(d_model, 1)
        self.hold = nn.Linear(d_model, 1)

    def forward(self, global_features: torch.Tensor, extras: torch.Tensor, init_logits: torch.Tensor, steps: int) -> list[StepOutput]:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        g = self.global_in(global_features).unsqueeze(1)
        fixed = self.extra_in(extras) + self.position
        logits = init_logits
        outputs: list[StepOutput] = []
        for _ in range(steps):
            workspace = self.workspace_in(torch.softmax(logits, dim=-1)) + fixed
            h = self.norm(self.core(torch.cat([g, workspace], dim=1)))
            logits = logits + self.revise(h[:, 1:])
            outputs.append(StepOutput(logits, self.confidence(h[:, 0]).squeeze(-1), self.hold(h[:, 0]).squeeze(-1)))
        return outputs


class PlainNet(nn.Module):
    """Baseline 1: one-pass MLP from the problem encoding to every slot."""

    def __init__(self, encoding: Encoding, *, hidden: int) -> None:
        super().__init__()
        self.encoding = encoding
        self.net = nn.Sequential(
            nn.Linear(encoding.problem_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, encoding.slots * encoding.vocab),
        )

    def forward(self, problem: torch.Tensor) -> torch.Tensor:
        return self.net(problem).view(-1, self.encoding.slots, self.encoding.vocab)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def matched_plain_hidden(encoding: Encoding, target: int) -> int:
    """Hidden width whose PlainNet parameter count is closest to ``target``."""
    best, best_gap = 8, None
    p, o = encoding.problem_dim, encoding.slots * encoding.vocab
    for hidden in range(8, 4097):
        count = p * hidden + hidden + 2 * (hidden * hidden + hidden) + hidden * o + o
        gap = abs(count - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = hidden, gap
        if count > target:
            break
    return best
