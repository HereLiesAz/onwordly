"""Training and exact held-out evaluation for Experiment 000.

Matched budget (all arms): ``train_steps`` optimizer steps, each on
``batch_size`` training problems drawn from the training pool by
``Random(training_seed)`` -- the same problems, in the same order, for every
arm. The learner additionally runs ``revision_steps`` forward steps per pass
and a second (challenge) pass per problem; that extra compute is recorded as
forward-step counts, never hidden.

Learner loss per episode (all terms per problem, then scaled by the episode's
update weight and averaged):

- solve: cross-entropy of every revision step's workspace vs the witness
  (deep supervision), plus BCE of each step's confidence vs exact correctness;
- challenge: REINFORCE on the sampled hold/change decision with the graded
  reward of the 010/011 challenge game and a batch-mean baseline, plus
  cross-entropy of the challenge pass's workspace vs the witness (so a
  ``change`` has something right to change to).

Update weight: ``update_weight(confidence, surprise)`` with confidence = the
probability the network gave the move it made and surprise = 1 when the final
answer is wrong (``flat`` for the ablation), normalised to batch mean 1.

``learner-child`` (README, "Childhood arm"): learner-memory-dropout with no
self-trust anywhere -- the ledger's self-trust input is zeroed, the update
weight is ``flat`` (the model's own confidence never scales an update; flat
rather than corrector-reliability weighting, which would be a new component
needing its own ablation), and register reads use ``sources="correctors"``
(own ``self`` fillers excluded; training targets still teach via the loss).

``learner-first-visit`` (README, "Recurring frames"): a problem whose frame
already has readable register entries when it is drawn is a *revisit*. For a
revisit the memory-reading passes are trained only through the hold/change
decision (the REINFORCE term): its solve and confidence losses are computed
on an extra pass of the same problem with the empty-register encoding (extra
forward compute, recorded), and its challenge-pass workspace cross-entropy is
dropped. Memory therefore cannot be used to shortcut the solve or the revised
answer; it can only inform the decision. First visits train exactly as
``learner``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from time import perf_counter
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from onwordly.learner.manifest import LearnerManifest
from onwordly.learner.memory_io import LearnerMemory
from onwordly.learner.model import CHALLENGE_GLOBAL_DIM, MEMORY_GLOBAL_DIM, TRUST_DIM, Encoding, OnwordlyLearner, PlainNet
from onwordly.learner.task import (
    CHALLENGE_TYPES,
    Challenge,
    ConstrainedTask,
    Corrector,
    decision_correct,
    draw_challenge,
    grade_reply,
    is_right,
    satisfaction,
)
from onwordly.memory import frame_for, normalize_weights, update_weight

DRAFT_LOGIT_SCALE = 4.0  # challenge pass starts from the draft as confident one-hot logits
EVAL_BATCH = 256


@dataclass(frozen=True, slots=True)
class LearnerVariant:
    weighting: str = "confidence_surprise"
    use_memory: bool = True
    use_trust: bool = True
    register_sources: str = "all"  # "self": own fillers only; "correctors": corrector proposals only
    self_trust: bool = True  # False: the ledger's self-trust feature is zeroed in the input
    memory_dropout: bool = False  # training reads emptied with prob manifest.memory_dropout
    first_visit_loss: bool = False  # revisits: memory-reading passes trained via the decision only


VARIANTS: dict[str, LearnerVariant] = {
    "learner": LearnerVariant(),
    "learner-flat": LearnerVariant(weighting="flat"),
    "learner-no-memory": LearnerVariant(use_memory=False),
    "learner-no-trust": LearnerVariant(use_trust=False),
    "learner-self-memory": LearnerVariant(register_sources="self"),
    "learner-memory-dropout": LearnerVariant(memory_dropout=True),
    "learner-first-visit": LearnerVariant(first_visit_loss=True),  # broken by design (run 3); kept, not run
    # Childhood (docs/model-design.md, mindset): no self-trust anywhere. Built on
    # learner-memory-dropout; self-trust input zeroed, flat update weighting (no
    # own-confidence weighting), register reads only what it was told.
    "learner-child": LearnerVariant(
        memory_dropout=True, self_trust=False, weighting="flat", register_sources="correctors"
    ),
}


@dataclass(slots=True)
class TrainingLog:
    optimizer_steps: int = 0
    training_problems: int = 0
    forward_steps: int = 0  # revision steps x passes x problems (extra compute, reported)
    seconds: float = 0.0
    mean_loss: float | None = None
    mean_weight_wrong: float | None = None
    mean_weight_right: float | None = None
    challenge_outcomes: dict[str, int] = field(default_factory=dict)
    # Register diagnostics, solve-pass reads (before this visit's own writes).
    register_reads: int = 0
    register_nonempty: int = 0
    register_top_other_is_target: int = 0
    register_dropped: int = 0
    revisits_decision_only: int = 0  # learner-first-visit: problems trained via the decision only

    def to_dict(self) -> dict[str, object]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "training_problems": self.training_problems,
            "forward_steps": self.forward_steps,
            "seconds": self.seconds,
            "mean_loss": self.mean_loss,
            "mean_weight_final_wrong": self.mean_weight_wrong,
            "mean_weight_final_right": self.mean_weight_right,
            "challenge_outcomes": dict(sorted(self.challenge_outcomes.items())),
            "register_diagnostics": {
                "solve_reads": self.register_reads,
                "nonempty_fraction": _rate(self.register_nonempty, self.register_reads),
                "top_other_is_target_fraction": _rate(self.register_top_other_is_target, self.register_reads),
                "dropped": self.register_dropped,
                "dropped_fraction": _rate(self.register_dropped, self.register_reads),
                "revisits_decision_only": self.revisits_decision_only,
                "revisits_decision_only_fraction": _rate(self.revisits_decision_only, self.register_reads),
            },
        }


def make_memory(manifest: LearnerManifest, encoding: Encoding, context: str) -> LearnerMemory:
    return LearnerMemory(
        encoding,
        context=context,
        prior=manifest.trust_prior,
        half_life=manifest.trust_half_life,
        max_evidence=manifest.trust_max_evidence,
        register_cap=manifest.register_cap,
    )


def _inputs(
    encoding: Encoding,
    tasks: Sequence[ConstrainedTask],
    memory: LearnerMemory,
    variant: LearnerVariant,
    challenges: Sequence[Challenge] | None,
    device: torch.device,
    *,
    empty: Sequence[bool] | bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``empty``: per-task (or global) override replacing the register read by
    the empty-register encoding (dropout and the emptied-register probe)."""
    globals_, extras = [], []
    for index, task in enumerate(tasks):
        challenge = challenges[index] if challenges is not None else None
        drop = empty if isinstance(empty, bool) else empty[index]
        if variant.use_memory and drop:
            mem_g, mem_s = memory.empty_read()
        elif variant.use_memory:
            mem_g, mem_s = memory.read(task, variant.register_sources)
        else:
            mem_g, mem_s = [0.0] * MEMORY_GLOBAL_DIM, [[0.0] * (2 * encoding.vocab)] * encoding.slots
        trust = memory.trust(task, challenge.corrector if challenge else None) if variant.use_trust else [0.0] * TRUST_DIM
        if not variant.self_trust:
            trust = [0.0, *trust[1:]]  # index 0 = ledger self-trust
        if challenge is None:
            chal_g, proposal = [0.0] * CHALLENGE_GLOBAL_DIM, [[0.0] * encoding.vocab] * encoding.slots
        else:
            chal_g = [float(challenge.says_wrong), float(not challenge.says_wrong)]
            proposal = encoding.onehot(challenge.proposed) if challenge.proposed is not None else [[0.0] * encoding.vocab] * encoding.slots
        globals_.append(encoding.problem(task) + mem_g + trust + chal_g)
        extras.append([m + p for m, p in zip(mem_s, proposal)])
    return torch.tensor(globals_, device=device), torch.tensor(extras, device=device)


def _decode_all(encoding: Encoding, logits: torch.Tensor) -> list[str]:
    return [encoding.decode(row) for row in logits.argmax(-1).tolist()]


def _targets(encoding: Encoding, tasks: Sequence[ConstrainedTask], device: torch.device) -> torch.Tensor:
    return torch.tensor([encoding.slots_of(t.witness) for t in tasks], device=device)


def _ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-problem mean slot cross-entropy, shape (batch,)."""
    return F.cross_entropy(logits.transpose(1, 2), target, reduction="none").mean(-1)


def train_learner(
    manifest: LearnerManifest,
    variant: LearnerVariant,
    train_tasks: Sequence[ConstrainedTask],
    *,
    device: torch.device,
    regime: str,
) -> tuple[OnwordlyLearner, LearnerMemory, TrainingLog]:
    torch.manual_seed(manifest.training_seed)
    encoding = Encoding(manifest.alphabet, manifest.slots)
    model = OnwordlyLearner(encoding, d_model=manifest.d_model, layers=manifest.layers, heads=manifest.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=manifest.learning_rate)
    memory = make_memory(manifest, encoding, f"train:{regime}")
    correctors = [Corrector(name, rate) for name, rate in manifest.correctors]
    task_rng, challenge_rng = Random(manifest.training_seed), Random(manifest.training_seed + 1)
    dropout_rng = Random(manifest.training_seed + 2)  # separate stream: task/challenge order unchanged
    log = TrainingLog()
    losses: list[float] = []
    weights_wrong: list[float] = []
    weights_right: list[float] = []
    steps = manifest.revision_steps
    started = perf_counter()
    model.train()
    for _ in range(manifest.train_steps):
        tasks = [task_rng.choice(train_tasks) for _ in range(manifest.batch_size)]
        target = _targets(encoding, tasks, device)
        dropped = [False] * len(tasks)
        revisit = [False] * len(tasks)
        if variant.use_memory:
            if variant.memory_dropout:
                dropped = [dropout_rng.random() < manifest.memory_dropout for _ in tasks]
            for index, task in enumerate(tasks):
                nonempty, top_other = memory.register_stats(task, variant.register_sources)
                log.register_reads += 1
                log.register_nonempty += nonempty
                log.register_top_other_is_target += top_other is not None and is_right(task, top_other)
                revisit[index] = variant.first_visit_loss and nonempty
            log.register_dropped += sum(dropped)
            log.revisits_decision_only += sum(revisit)
        g, x = _inputs(encoding, tasks, memory, variant, None, device, empty=dropped)
        zeros = torch.zeros(len(tasks), encoding.slots, encoding.vocab, device=device)
        solve = model(g, x, zeros, steps)
        solve_ce = torch.stack([_ce(out.logits, target) for out in solve]).mean(0)
        conf_terms = []
        for out in solve:
            right = torch.tensor([float(is_right(t, s)) for t, s in zip(tasks, _decode_all(encoding, out.logits))], device=device)
            conf_terms.append(F.binary_cross_entropy_with_logits(out.confidence_logit, right, reduction="none"))
        conf_loss = torch.stack(conf_terms).mean(0)
        drafts = _decode_all(encoding, solve[-1].logits)
        revisit_t = torch.tensor(revisit, device=device)
        if any(revisit):
            # Revisits: solve/confidence loss from an empty-register pass instead.
            sub = [t for t, r in zip(tasks, revisit) if r]
            g0, x0 = _inputs(encoding, sub, memory, variant, None, device, empty=True)
            anchor = model(g0, x0, zeros[revisit_t], steps)
            anchor_ce = torch.stack([_ce(out.logits, target[revisit_t]) for out in anchor]).mean(0)
            anchor_conf = []
            for out in anchor:
                right = torch.tensor([float(is_right(t, a)) for t, a in zip(sub, _decode_all(encoding, out.logits))], device=device)
                anchor_conf.append(F.binary_cross_entropy_with_logits(out.confidence_logit, right, reduction="none"))
            solve_ce = solve_ce.masked_scatter(revisit_t, anchor_ce)
            conf_loss = conf_loss.masked_scatter(revisit_t, torch.stack(anchor_conf).mean(0))
            log.forward_steps += steps * len(sub)

        challenges, links = [], []
        for task, draft in zip(tasks, drafts):
            challenge = draw_challenge(task, draft, challenge_rng.choice(correctors), challenge_rng)
            challenges.append(challenge)
            links.append(memory.write_challenge(task, draft, challenge))
        g2, x2 = _inputs(encoding, tasks, memory, variant, challenges, device, empty=dropped)
        init = torch.tensor([encoding.onehot(d) for d in drafts], device=device) * DRAFT_LOGIT_SCALE
        reason = model(g2, x2, init, steps)
        hold_p = torch.sigmoid(reason[-1].hold_logit).clamp(1e-6, 1 - 1e-6)
        hold = torch.bernoulli(hold_p.detach()).bool()
        revised = _decode_all(encoding, reason[-1].logits)
        rewards, finals_right, outcome_ok = [], [], []
        for i, (task, challenge) in enumerate(zip(tasks, challenges)):
            action = "hold" if hold[i] else "change"
            value = challenge.first if action == "hold" else revised[i]
            outcome, reward, final = grade_reply(task, challenge, action, value)
            ok = decision_correct(challenge, outcome)
            rewards.append(reward)
            finals_right.append(is_right(task, final))
            outcome_ok.append((action, value, outcome, reward, final, ok))
            log.challenge_outcomes[f"{challenge.corrector}:{challenge.kind}:{outcome}"] = (
                log.challenge_outcomes.get(f"{challenge.corrector}:{challenge.kind}:{outcome}", 0) + 1
            )
        reward_t = torch.tensor(rewards, device=device)
        logp = torch.where(hold, hold_p.log(), (1 - hold_p).log())
        policy = -(reward_t - reward_t.mean()) * logp
        revise_ce = torch.stack([_ce(out.logits, target) for out in reason]).mean(0)
        revise_ce = torch.where(revisit_t, torch.zeros_like(revise_ce), revise_ce)

        confidence = torch.where(hold, hold_p, 1 - hold_p).detach().tolist()
        raw = [
            update_weight(c, 0.0 if right else 1.0, rule=variant.weighting, gain=manifest.update_gain, max_weight=manifest.update_max_weight)
            for c, right in zip(confidence, finals_right)
        ]
        weights = normalize_weights(raw)
        for w, right in zip(weights, finals_right):
            (weights_right if right else weights_wrong).append(w)
        loss = (torch.tensor(weights, device=device) * (solve_ce + conf_loss + policy + revise_ce)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        # Ledger and store learn from training outcomes only, after the update.
        for task, challenge, link, conf, (action, value, outcome, reward, final, ok) in zip(
            tasks, challenges, links, confidence, outcome_ok
        ):
            memory.write_outcome(
                task, challenge, link, action=action, value=value, outcome=outcome, reward=reward, final=final,
                decision_ok=ok, confidence=conf,
                inputs={"draft": True, "challenge": True, "memory": variant.use_memory, "trust": variant.use_trust},
            )
        log.optimizer_steps += 1
        log.training_problems += len(tasks)
        log.forward_steps += 2 * steps * len(tasks)
    log.seconds = perf_counter() - started
    log.mean_loss = sum(losses) / len(losses)
    log.mean_weight_wrong = sum(weights_wrong) / len(weights_wrong) if weights_wrong else None
    log.mean_weight_right = sum(weights_right) / len(weights_right) if weights_right else None
    return model, memory, log


def handcoded_decision(memory: LearnerMemory, task: ConstrainedTask, challenge: Challenge, *, threshold: float, strikes: int) -> str:
    """aive-style rule: hold under confirmation; under "wrong", change to the
    proposal only if the corrector's raw success rate clears ``threshold``,
    beats the self success rate for this domain, and its circuit breaker
    (``strikes`` consecutive wrong claims) is closed."""
    if not challenge.says_wrong:
        return "hold"
    corrector = memory.ledger.entry("corrector", challenge.corrector)
    if corrector.failure_streak >= strikes:
        return "hold"
    own = memory.ledger.entry("self", task.domain).success_rate
    rate = corrector.success_rate
    rate = 0.5 if rate is None else rate
    own = 0.5 if own is None else own
    return "change" if rate >= threshold and rate > own else "hold"


def train_plain(
    manifest: LearnerManifest,
    train_tasks: Sequence[ConstrainedTask],
    *,
    hidden: int,
    device: torch.device,
) -> tuple[PlainNet, LearnerMemory, TrainingLog]:
    """Baseline 1 (supervised CE, one pass) on the identical problem stream.
    Its pre-update drafts are challenged and the hand-coded rule's decisions
    graded, building the ledger baseline 5 reads; nothing of this trains the net."""
    torch.manual_seed(manifest.training_seed)
    encoding = Encoding(manifest.alphabet, manifest.slots)
    model = PlainNet(encoding, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=manifest.learning_rate)
    memory = make_memory(manifest, encoding, "train:handcoded")
    correctors = [Corrector(name, rate) for name, rate in manifest.correctors]
    task_rng, challenge_rng = Random(manifest.training_seed), Random(manifest.training_seed + 1)
    log = TrainingLog()
    losses = []
    started = perf_counter()
    model.train()
    for _ in range(manifest.train_steps):
        tasks = [task_rng.choice(train_tasks) for _ in range(manifest.batch_size)]
        problem = torch.tensor([encoding.problem(t) for t in tasks], device=device)
        logits = model(problem)
        loss = _ce(logits, _targets(encoding, tasks, device)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        for task, draft in zip(tasks, _decode_all(encoding, logits.detach())):
            challenge = draw_challenge(task, draft, challenge_rng.choice(correctors), challenge_rng)
            link = memory.write_challenge(task, draft, challenge)
            action = handcoded_decision(memory, task, challenge, threshold=manifest.handcoded_threshold, strikes=manifest.handcoded_strikes)
            value = draft if action == "hold" else challenge.proposed
            outcome, reward, final = grade_reply(task, challenge, action, value)
            key = f"{challenge.corrector}:{challenge.kind}:{outcome}"
            log.challenge_outcomes[key] = log.challenge_outcomes.get(key, 0) + 1
            memory.write_outcome(
                task, challenge, link, action=action, value=value, outcome=outcome, reward=reward, final=final,
                decision_ok=decision_correct(challenge, outcome), confidence=1.0, inputs={"rule": "handcoded"},
            )
        log.optimizer_steps += 1
        log.training_problems += len(tasks)
        log.forward_steps += len(tasks)
    log.seconds = perf_counter() - started
    log.mean_loss = sum(losses) / len(losses)
    return model, memory, log


# --- evaluation ----------------------------------------------------------------------


def calibration(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> dict[str, float | int | None]:
    """Expected calibration error (equal-width bins) and Brier score."""
    if not pairs:
        return {"n": 0, "ece": None, "brier": None, "mean_confidence": None, "accuracy": None}
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, ok in pairs:
        buckets[min(bins - 1, int(conf * bins))].append((conf, ok))
    n = len(pairs)
    ece = sum(
        len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in buckets if b
    )
    return {
        "n": n,
        "ece": ece,
        "brier": sum((c - float(o)) ** 2 for c, o in pairs) / n,
        "mean_confidence": sum(c for c, _ in pairs) / n,
        "accuracy": sum(o for _, o in pairs) / n,
    }


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


@torch.no_grad()
def solve_learner(
    model: OnwordlyLearner, memory: LearnerMemory, variant: LearnerVariant, tasks: Sequence[ConstrainedTask], steps: int,
    device: torch.device, *, empty_register: bool = False,
):
    """Per-task list of per-step answers and the final step's confidence.
    Reads only; ``empty_register`` feeds the empty-register encoding."""
    model.eval()
    encoding = model.encoding
    answers: list[list[str]] = []
    confidences: list[float] = []
    for start in range(0, len(tasks), EVAL_BATCH):
        chunk = tasks[start : start + EVAL_BATCH]
        g, x = _inputs(encoding, chunk, memory, variant, None, device, empty=empty_register)
        outs = model(g, x, torch.zeros(len(chunk), encoding.slots, encoding.vocab, device=device), steps)
        per_step = [_decode_all(encoding, out.logits) for out in outs]
        answers += [list(row) for row in zip(*per_step)]
        confidences += torch.sigmoid(outs[-1].confidence_logit).tolist()
    return answers, confidences


@torch.no_grad()
def solve_plain(model: PlainNet, tasks: Sequence[ConstrainedTask], device: torch.device):
    model.eval()
    encoding = model.encoding
    answers, confidences = [], []
    for start in range(0, len(tasks), EVAL_BATCH):
        chunk = tasks[start : start + EVAL_BATCH]
        logits = model(torch.tensor([encoding.problem(t) for t in chunk], device=device))
        answers += [[s] for s in _decode_all(encoding, logits)]
        # Confidence of a one-pass net: mean over slots of the top probability.
        confidences += torch.softmax(logits, -1).max(-1).values.mean(-1).tolist()
    return answers, confidences


def solve_metrics(tasks: Sequence[ConstrainedTask], answers: Sequence[Sequence[str]], confidences: Sequence[float]) -> dict[str, object]:
    steps = len(answers[0])
    right = [[is_right(t, a) for a in row] for t, row in zip(tasks, answers)]
    fixes = sum(not r[i] and r[i + 1] for r in right for i in range(steps - 1))
    breaks = sum(r[i] and not r[i + 1] for r in right for i in range(steps - 1))
    return {
        "examples": len(tasks),
        "revision_steps": steps,
        "accuracy": sum(r[-1] for r in right) / len(tasks),
        "per_step_accuracy": [sum(r[i] for r in right) / len(tasks) for i in range(steps)],
        "mean_satisfaction": sum(satisfaction(t, row[-1]) for t, row in zip(tasks, answers)) / len(tasks),
        "step_fixes": fixes,
        "step_breaks": breaks,
        "first_to_last_fixes": sum(not r[0] and r[-1] for r in right),
        "first_to_last_breaks": sum(r[0] and not r[-1] for r in right),
        "confidence_calibration": calibration([(c, r[-1]) for c, r in zip(confidences, right)]),
    }


# --- memory diagnostics -----------------------------------------------------------


def probe_sample(manifest: LearnerManifest, train_tasks: Sequence[ConstrainedTask]) -> list[ConstrainedTask]:
    """Fixed sample of training problems for the register probe (same for every arm)."""
    rng = Random(f"{manifest.evaluation_seed}:register-probe")
    return rng.sample(list(train_tasks), min(manifest.probe_size, len(train_tasks)))


def _accuracy(tasks: Sequence[ConstrainedTask], answers: Sequence[Sequence[str]]) -> float:
    return sum(is_right(t, row[-1]) for t, row in zip(tasks, answers)) / len(tasks)


def probe_training_frames(
    model: OnwordlyLearner, memory: LearnerMemory, variant: LearnerVariant, tasks: Sequence[ConstrainedTask],
    steps: int, device: torch.device,
) -> dict[str, object]:
    """Diagnostic 1: training frames, after training, eval mode, read-only on
    an eval view. Accuracy with the register as built vs emptied."""
    view = memory.eval_view("probe:train-frames")
    stats = [view.register_stats(t, variant.register_sources) for t in tasks]
    as_is, _ = solve_learner(model, view, variant, tasks, steps, device)
    emptied, _ = solve_learner(model, view, variant, tasks, steps, device, empty_register=True)
    acc, acc_empty = _accuracy(tasks, as_is), _accuracy(tasks, emptied)
    return {
        "examples": len(tasks),
        "accuracy_register_as_is": acc,
        "accuracy_register_emptied": acc_empty,
        "drop_on_emptying": acc - acc_empty,
        "register_nonempty_fraction": sum(n for n, _ in stats) / len(tasks),
        "top_other_is_target_fraction": sum(o is not None and is_right(t, o) for t, (_, o) in zip(tasks, stats)) / len(tasks),
        "view_records_written": len(view.store),
    }


def probe_second_visit(
    model: OnwordlyLearner, memory: LearnerMemory, variant: LearnerVariant, tasks: Sequence[ConstrainedTask],
    steps: int, device: torch.device,
) -> dict[str, object]:
    """Diagnostic 2: held-out frames visited twice. Attempt 1 reads the
    (empty) register; only the model's own attempt-1 answer is written into a
    forked eval register; attempt 2 reads it."""
    view = memory.eval_view("probe:second-visit")
    empty_first = sum(not view.register_stats(t, "all")[0] for t in tasks)
    first, _ = solve_learner(model, view, variant, tasks, steps, device)
    for task, row in zip(tasks, first):
        view.write_self_filler(task, row[-1])
    sources = sorted({e.source for t in tasks for e in view.register.entries(frame_for(t))})
    second, _ = solve_learner(model, view, variant, tasks, steps, device)
    a1, a2 = _accuracy(tasks, first), _accuracy(tasks, second)
    return {
        "examples": len(tasks),
        "first_visit_empty_fraction": empty_first / len(tasks),
        "accuracy_first_visit": a1,
        "accuracy_second_visit": a2,
        "second_minus_first": a2 - a1,
        "second_visit_register_sources": sources,
        "changed_answers": sum(f[-1] != s[-1] for f, s in zip(first, second)),
    }


# decide(tasks, challenges, memory) -> list of (action, value, confidence)
Decider = Callable[[Sequence[ConstrainedTask], Sequence[Challenge], LearnerMemory], list[tuple[str, str | None, float]]]


def learner_decider(model: OnwordlyLearner, variant: LearnerVariant, steps: int, device: torch.device) -> Decider:
    @torch.no_grad()
    def decide(tasks, challenges, memory):
        encoding = model.encoding
        out: list[tuple[str, str | None, float]] = []
        for start in range(0, len(tasks), EVAL_BATCH):
            chunk, chal = tasks[start : start + EVAL_BATCH], challenges[start : start + EVAL_BATCH]
            g, x = _inputs(encoding, chunk, memory, variant, chal, device)
            init = torch.tensor([encoding.onehot(c.first) for c in chal], device=device) * DRAFT_LOGIT_SCALE
            final = model(g, x, init, steps)[-1]
            hold_p = torch.sigmoid(final.hold_logit).tolist()
            revised = _decode_all(encoding, final.logits)
            for c, p, r in zip(chal, hold_p, revised):
                out.append(("hold", c.first, p) if p >= 0.5 else ("change", r, 1 - p))
        return out

    return decide


def handcoded_decider(threshold: float, strikes: int) -> Decider:
    def decide(tasks, challenges, memory):
        out = []
        for task, challenge in zip(tasks, challenges):
            action = handcoded_decision(memory, task, challenge, threshold=threshold, strikes=strikes)
            out.append((action, challenge.first if action == "hold" else challenge.proposed, 1.0))
        return out

    return decide


def evaluate_challenges(
    tasks: Sequence[ConstrainedTask],
    drafts: Sequence[str],
    *,
    decide: Decider,
    memory: LearnerMemory,
    conditions: Sequence[tuple[Corrector, bool]],
    seed: int,
) -> dict[str, object]:
    """Held-out fallible-challenge evaluation per corrector condition.

    Each condition gets a fresh ``eval_view`` (frozen training ledger, forked
    register), so evaluation never feeds training memory. Drafts are the
    final solve answers, shared by every condition."""
    by_condition: dict[str, object] = {}
    hold_said_wrong: dict[str, float | None] = {}
    for corrector, seen in conditions:
        view = memory.eval_view(f"eval:{corrector.name}")
        rng = Random(f"{seed}:{corrector.name}:{corrector.error_rate}")
        challenges, links = [], []
        for task, draft in zip(tasks, drafts):
            challenge = draw_challenge(task, draft, corrector, rng)
            challenges.append(challenge)
            links.append(view.write_challenge(task, draft, challenge))
        decisions = decide(tasks, challenges, view)
        counts = {kind: {"held": 0, "changed_correct": 0, "changed_wrong": 0, "invalid": 0} for kind in CHALLENGE_TYPES}
        final_right = 0
        decision_pairs = []
        for task, challenge, link, (action, value, conf) in zip(tasks, challenges, links, decisions):
            outcome, reward, final = grade_reply(task, challenge, action, value)
            bucket = {"held": "held", "changed_correct": "changed_correct", "inconsistent": "invalid"}.get(outcome, "changed_wrong")
            counts[challenge.kind][bucket] += 1
            final_right += is_right(task, final)
            ok = decision_correct(challenge, outcome)
            decision_pairs.append((conf, ok))
            view.write_outcome(
                task, challenge, link, action=action, value=value, outcome=outcome, reward=reward, final=final,
                decision_ok=ok, confidence=conf, inputs={"eval": True},
            )

        def held(*kinds: str) -> int:
            return sum(counts[k]["held"] for k in kinds)

        def n(*kinds: str) -> int:
            return sum(sum(counts[k].values()) for k in kinds)

        b = counts["wrong_corrected"]
        hr = _rate(held("right_challenged", "right_confirmed"), n("right_challenged", "right_confirmed"))
        hw = _rate(held("wrong_corrected", "wrong_confirmed"), n("wrong_corrected", "wrong_confirmed"))
        said_wrong = _rate(held("right_challenged", "wrong_corrected"), n("right_challenged", "wrong_corrected"))
        hold_said_wrong[corrector.name] = said_wrong
        by_condition[corrector.name] = {
            "corrector": corrector.name,
            "error_rate": corrector.error_rate,
            "seen_in_training": seen,
            "ledger_reliability": memory.ledger.corrector_reliability(corrector.name),
            "class_counts": counts,
            "hold_rate_right_under_wrong_challenge": _rate(held("right_challenged"), n("right_challenged")),
            "change_rate_wrong_under_correct_challenge": _rate(b["changed_correct"] + b["changed_wrong"], n("wrong_corrected")),
            "change_to_correct_rate_wrong_under_correct_challenge": _rate(b["changed_correct"], n("wrong_corrected")),
            "hold_rate_when_said_wrong": said_wrong,
            "hold_discrimination": None if hr is None or hw is None else hr - hw,
            "final_accuracy": final_right / len(tasks),
            "decision_calibration": calibration(decision_pairs),
            "eval_store_chain_verified": view.store.verify(),
        }
    names = [c.name for c, seen in conditions if seen]
    rates = {c.name: c.error_rate for c, _ in conditions}
    tracked = None
    if len(names) >= 2:
        reliable, unreliable = min(names, key=rates.get), max(names, key=rates.get)
        a, b = hold_said_wrong[reliable], hold_said_wrong[unreliable]
        tracked = None if a is None or b is None else b - a
    return {
        "by_corrector": by_condition,
        # Positive: holds more against the less reliable training corrector.
        "hold_tracks_reliability": tracked,
    }
