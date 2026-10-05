"""Two-turn self-check episodes for on-policy training (Experiment 010).

Arms ``verdict-rl-graded`` and ``selfcheck-rl-binary`` train the same episode:

1. turn 1 samples an answer to an arithmetic training prompt;
2. turn 2 samples a verdict on ``make_verdict_task(task, parsed_turn1)``
   (``right`` or ``wrong: <n>``), i.e. the model judges its own answer.

If turn 1 is unparseable the episode stops there (reward 0.0, only turn 1 is
trained). The two arms differ only in the episode reward, so graded-vs-binary
is isolated on identical episodes.

Graded rewards for an improvement-bonus style signal (turn-1 right and kept
scores most; a wrong answer caught and exactly repaired scores 0.6; rejecting
a right answer is penalised) are reward shaping, as in SCoRe stage II
(arXiv:2409.12917); established, not an Onwordly invention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from onwordly.models.base import ModelAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.tasks.verdict import make_verdict_task, parse_verdict
from onwordly.training.harness import Episode, Rollout, Verifier
from onwordly.verifiers.arithmetic import parse_integer_answer

# A repair is "close" when |repair - answer| <= max(CLOSE_REPAIR_MIN_ABS,
# CLOSE_REPAIR_REL * |answer|): within 1, or within 5% of the answer if larger.
CLOSE_REPAIR_MIN_ABS = 1
CLOSE_REPAIR_REL = 0.05

# Episode tier -> graded reward (verdict-rl-graded).
GRADED_EPISODE_REWARDS: dict[str, float] = {
    "right_kept": 1.0,
    "wrong_repaired": 0.6,
    "wrong_caught_close_repair": 0.4,
    "wrong_caught_far_repair": 0.3,
    "wrong_accepted": 0.0,
    "right_rejected": -0.5,
    "unparseable_verdict": 0.0,
    "unparseable_answer": 0.0,
}

EpisodeReward = Literal["graded", "binary"]


def is_close_repair(repair: int, answer: int) -> bool:
    return abs(repair - answer) <= max(CLOSE_REPAIR_MIN_ABS, CLOSE_REPAIR_REL * abs(answer))


def classify_episode(task: ArithmeticTask, turn1: str, turn2: str | None) -> tuple[str, int | None]:
    """``(tier, final_answer)`` for a self-check episode.

    The final answer is turn 1 if judged right, the repair if judged wrong,
    and ``None`` (no final answer) when either turn is unparseable.
    """
    first = parse_integer_answer(turn1)
    if first is None:
        return "unparseable_answer", None
    verdict = parse_verdict(turn2 or "")
    if verdict is None:
        return "unparseable_verdict", None
    first_ok = first == task.answer
    if verdict[0]:
        return ("right_kept" if first_ok else "wrong_accepted"), first
    repair = verdict[1]
    assert repair is not None
    if first_ok:
        return "right_rejected", repair
    if repair == task.answer:
        return "wrong_repaired", repair
    return ("wrong_caught_close_repair" if is_close_repair(repair, task.answer) else "wrong_caught_far_repair"), repair


def episode_reward(kind: EpisodeReward, tier: str, final: int | None, answer: int) -> float:
    if kind == "graded":
        return GRADED_EPISODE_REWARDS[tier]
    # Binary: 1 if the final answer is correct; no final answer scores 0.
    return float(final is not None and final == answer)


@dataclass(frozen=True, slots=True)
class SelfCheckEpisode:
    """Queued marker: run a group of self-check episodes on ``base``."""

    base: ArithmeticTask

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
        return parse_integer_answer(response) == self.base.answer


@dataclass(frozen=True, slots=True)
class SelfCheckEpisodeConfig:
    """Group of ``samples`` two-turn episodes at ``temperature``; ``reward`` picks
    the graded or binary episode reward. Plugs into the harness's on-policy
    update (REINFORCE with a group-mean baseline; see ``OnPolicyConfig``)."""

    reward: EpisodeReward
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
        del verifier  # rewards are computed by the exact episode classifier
        if not isinstance(task, SelfCheckEpisode):
            raise TypeError("self-check rollouts need a SelfCheckEpisode task")
        base = task.base
        answers = list(adapter.sample(base.prompt, self.samples, self.temperature))
        if len(answers) != self.samples:
            raise RuntimeError("adapter returned the wrong number of samples")
        calls = len(answers)
        episodes = []
        for turn1 in answers:
            first = parse_integer_answer(turn1)
            turns: tuple[tuple[str, str], ...] = ((base.prompt, turn1),)
            turn2 = None
            if first is not None:
                verdict_prompt = make_verdict_task(base, first).prompt
                turn2 = adapter.sample(verdict_prompt, 1, self.temperature)[0]
                calls += 1
                turns += ((verdict_prompt, turn2),)
            tier, final = classify_episode(base, turn1, turn2)
            episodes.append(
                Episode(
                    turns=turns,
                    reward=episode_reward(self.reward, tier, final, base.answer),
                    tier=tier,
                    correct=final is not None and final == base.answer,
                )
            )
        return Rollout(tuple(episodes), generation_calls=calls, verifier_calls=len(episodes))
