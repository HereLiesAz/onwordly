"""Training sources and self-check episodes for Experiment 011 (constrained strings).

Mirrors Experiment 010's ``VerdictArithmeticSource`` / ``SelfCheckEpisodeSource``
and ``SelfCheckEpisodeConfig`` on the constrained-string task, and plugs into
the same generic on-policy harness update (REINFORCE with a group-mean
baseline; see ``harness.OnPolicyConfig``). On-policy episode tasks skip the
greedy pre-update attempt, exactly as in 010.

Graded episode reward (``selfcheck-rl-graded``), with s(x) the fraction of
rules x satisfies (``satisfaction``):

| episode | reward |
| --- | --- |
| turn 1 valid, said right | ``RIGHT_KEPT_REWARD`` = 1.0 |
| turn 1 valid, said wrong | ``RIGHT_REJECTED_REWARD`` = -0.5 |
| turn 1 invalid, said wrong, repair valid | ``WRONG_REPAIRED_REWARD`` = 0.6 |
| turn 1 invalid, said wrong, repair invalid | ``PARTIAL_REPAIR_BASE`` + ``PARTIAL_REPAIR_SCALE`` * s(repair) (0.3 to < 0.4) |
| turn 1 invalid, said right | ``WRONG_ACCEPTED_SCALE`` * s(turn 1) (0 to < 0.2) |
| unparseable turn 1 or verdict | ``UNPARSEABLE_REWARD`` = 0.0 |

An invalid string has s < 1, so a partial repair stays below 0.4 (010's close
repair) and an accepted invalid string below 0.2 (010's close accepted). The
binary reward (``selfcheck-rl-binary``) is 1 if the final string satisfies
every rule, else 0, on identical episodes. Reward shaping is established
(SCoRe stage II, arXiv:2409.12917); not an Onwordly invention.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Literal, Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.constrained_strings import (
    ConstrainedStringTask,
    ConstrainedVerdictTask,
    make_constrained_verdict_task,
    parse_string_answer,
    parse_string_verdict,
    satisfaction,
    satisfies_all,
    synthetic_violation,
)
from onwordly.training.harness import Episode, Rollout, Verifier

RIGHT_KEPT_REWARD = 1.0
RIGHT_REJECTED_REWARD = -0.5
WRONG_REPAIRED_REWARD = 0.6
PARTIAL_REPAIR_BASE = 0.3
PARTIAL_REPAIR_SCALE = 0.1
WRONG_ACCEPTED_SCALE = 0.2
UNPARSEABLE_REWARD = 0.0

ConstrainedReward = Literal["graded", "binary"]


def classify_constrained_episode(
    task: ConstrainedStringTask, turn1: str, turn2: str | None
) -> tuple[str, str | None]:
    """``(tier, final_string)``; final is turn 1 if judged right, else the repair."""
    first = parse_string_answer(turn1)
    if first is None:
        return "unparseable_answer", None
    verdict = parse_string_verdict(turn2 or "")
    if verdict is None:
        return "unparseable_verdict", None
    first_ok = satisfies_all(task, first)
    if verdict[0]:
        return ("right_kept" if first_ok else "wrong_accepted"), first
    repair = verdict[1]
    if first_ok:
        return "right_rejected", repair
    return ("wrong_repaired" if satisfies_all(task, str(repair)) else "wrong_caught_partial_repair"), repair


def constrained_episode_reward(
    kind: ConstrainedReward, task: ConstrainedStringTask, tier: str, turn1: str, final: str | None
) -> float:
    if kind == "binary":
        return float(final is not None and satisfies_all(task, final))
    if tier == "right_kept":
        return RIGHT_KEPT_REWARD
    if tier == "right_rejected":
        return RIGHT_REJECTED_REWARD
    if tier == "wrong_repaired":
        return WRONG_REPAIRED_REWARD
    if tier == "wrong_caught_partial_repair":
        return PARTIAL_REPAIR_BASE + PARTIAL_REPAIR_SCALE * satisfaction(task, final)
    if tier == "wrong_accepted":
        return WRONG_ACCEPTED_SCALE * satisfaction(task, parse_string_answer(turn1))
    return UNPARSEABLE_REWARD


@dataclass(frozen=True, slots=True)
class ConstrainedSelfCheckEpisode:
    """Queued marker: run a group of self-check episodes on ``base``."""

    base: ConstrainedStringTask

    @property
    def prompt(self) -> str:
        return self.base.prompt

    @property
    def target_text(self) -> str:
        return self.base.target_text

    @property
    def bucket_key(self) -> str:
        return f"selfcheck:{self.base.bucket_key}"

    def verify(self, response: str) -> bool:
        return self.base.verify(response)


@dataclass(frozen=True, slots=True)
class ConstrainedSelfCheckConfig:
    """Group of ``samples`` two-turn episodes (produce, then judge own and keep or
    repair) at ``temperature``; ``reward`` picks graded or binary."""

    reward: ConstrainedReward
    samples: int = 4
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.reward not in ("graded", "binary"):
            raise ValueError("reward must be 'graded' or 'binary'")
        if self.samples < 2:
            raise ValueError("on-policy groups need at least two samples")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    def rollout(self, adapter: ModelAdapter, task: object, verifier: Verifier) -> Rollout:
        del verifier  # rewards come from the exact rule checker
        if not isinstance(task, ConstrainedSelfCheckEpisode):
            raise TypeError("constrained self-check rollouts need a ConstrainedSelfCheckEpisode")
        base = task.base
        answers = list(adapter.sample(base.prompt, self.samples, self.temperature))
        if len(answers) != self.samples:
            raise RuntimeError("adapter returned the wrong number of samples")
        calls = len(answers)
        episodes = []
        for turn1 in answers:
            first = parse_string_answer(turn1)
            turns: tuple[tuple[str, str], ...] = ((base.prompt, turn1),)
            turn2 = None
            if first is not None:
                verdict_prompt = make_constrained_verdict_task(base, first).prompt
                turn2 = adapter.sample(verdict_prompt, 1, self.temperature)[0]
                calls += 1
                turns += ((verdict_prompt, turn2),)
            tier, final = classify_constrained_episode(base, turn1, turn2)
            episodes.append(
                Episode(
                    turns=turns,
                    reward=constrained_episode_reward(self.reward, base, tier, turn1, final),
                    tier=tier,
                    correct=final is not None and satisfies_all(base, final),
                )
            )
        return Rollout(tuple(episodes), generation_calls=calls, verifier_calls=len(episodes))


class StaticConstrainedSource:
    def __init__(self, tasks: Sequence[ConstrainedStringTask]) -> None:
        if not tasks:
            raise ValueError("static dataset cannot be empty")
        self.tasks = tuple(tasks)
        self.index = 0

    def next_task(self, rng: Random) -> ConstrainedStringTask:
        del rng
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task

    def observe(self, task: object, correct: bool) -> None:
        del task, correct


class ConstrainedVerdictSource:
    """Static string stream with balanced SFT verdict moves interleaved.

    After the pre-update attempt on each string task, with probability
    ``verdict_rate`` a verdict move on it is queued: the shown string is the
    witness with p = 0.5 (target ``right``), else a synthetic violation
    (target ``wrong: <witness>``). The trigger ignores the attempt's outcome.
    """

    def __init__(self, tasks: Sequence[ConstrainedStringTask], *, verdict_rate: float, seed: int) -> None:
        if not 0.0 < verdict_rate <= 1.0:
            raise ValueError("verdict_rate must be in (0, 1]")
        self.base = StaticConstrainedSource(tasks)
        self.verdict_rate = verdict_rate
        self.rng = Random(seed)
        self.pending: deque[ConstrainedVerdictTask] = deque()
        self.queued = {"right": 0, "wrong_synthetic": 0}

    def next_task(self, rng: Random) -> ConstrainedStringTask | ConstrainedVerdictTask:
        return self.pending.popleft() if self.pending else self.base.next_task(rng)

    def observe(self, task: object, correct: bool) -> None:
        del task, correct

    def observe_response(self, task: object, response: str, correct: bool) -> None:
        del response, correct
        if not isinstance(task, ConstrainedStringTask) or self.rng.random() >= self.verdict_rate:
            return
        if self.rng.random() < 0.5:
            self.pending.append(make_constrained_verdict_task(task, task.witness))
            self.queued["right"] += 1
        else:
            self.pending.append(make_constrained_verdict_task(task, synthetic_violation(task, self.rng)))
            self.queued["wrong_synthetic"] += 1


class ConstrainedSelfCheckSource:
    """Static string stream with on-policy self-check episode groups queued
    with probability ``verdict_rate`` after each string task (which stays SFT)."""

    def __init__(
        self,
        tasks: Sequence[ConstrainedStringTask],
        *,
        config: ConstrainedSelfCheckConfig,
        verdict_rate: float,
        seed: int,
    ) -> None:
        if not 0.0 < verdict_rate <= 1.0:
            raise ValueError("verdict_rate must be in (0, 1]")
        self.base = StaticConstrainedSource(tasks)
        self.config = config
        self.verdict_rate = verdict_rate
        self.rng = Random(seed)
        self.pending: deque[ConstrainedSelfCheckEpisode] = deque()
        self.queued = {"episode_groups": 0}

    def on_policy_config(self, task: object) -> ConstrainedSelfCheckConfig | None:
        return self.config if isinstance(task, ConstrainedSelfCheckEpisode) else None

    def next_task(self, rng: Random) -> ConstrainedStringTask | ConstrainedSelfCheckEpisode:
        return self.pending.popleft() if self.pending else self.base.next_task(rng)

    def observe(self, task: object, correct: bool) -> None:
        del task, correct

    def observe_response(self, task: object, response: str, correct: bool) -> None:
        del response, correct
        if isinstance(task, ConstrainedStringTask) and self.rng.random() < self.verdict_rate:
            self.pending.append(ConstrainedSelfCheckEpisode(task))
            self.queued["episode_groups"] += 1


def verify_constrained(task: object, response: str) -> bool:
    """Exact harness verifier for every 011 task type."""
    return bool(task.verify(response))  # type: ignore[attr-defined]
