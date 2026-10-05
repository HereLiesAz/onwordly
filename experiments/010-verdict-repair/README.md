# Experiment 010 — Verdict repair

**Status:** prepared, CPU-tested with fake adapters; no real-model run.

## Question

Experiment 009 (one seed, Qwen2.5-0.5B, SFT, 100k training tokens) collapsed
to acceptance: `verdict-mixed` said `right` to every proposal (balanced 50.0%);
`verdict-synthetic` caught 13.4% of wrong proposals (balanced 51.9%) and
repaired every catch. Solving was not the bottleneck. Rejecting was.

009 named three untested explanations. 010 gives each one an arm and changes
only one thing. For the first three arms that is relative to `verdict-synthetic`:

| Explanation | Arm | The one change |
| --- | --- | --- |
| Too few verdict examples | `verdict-dense` | `verdict_rate` = `dense_verdict_rate` (1.0 vs 0.3): about 3x more verdict moves inside the same token budget |
| The format asks for the verdict before the answer | `solve-judge-synthetic` | reply `<correct integer>; right` or `<correct integer>; wrong`. The model states the answer first and then judges |
| SFT cannot install conditional rejection here | `verdict-rl` | verdict moves trained on-policy with the exact checker as reward. Arithmetic tasks stay SFT |
| A 0/1 reward on the model's own answers gives too little signal | `verdict-rl-graded` | on-policy two-turn self-check episodes (answer, then judge that answer) with a graded reward |
| (ablation of the row above) | `selfcheck-rl-binary` | the same episodes, with a binary reward on the final answer |

Controls: `static`, and `verdict-synthetic` (009's arm, rerun so seeds and
code match).

## Arms in detail

All arms use the same model, data, seed, 100k-token budget and frozen
evaluation as 009. Verdict proposals are right with p = 0.5, independent of
the model's own attempt. Wrong proposals are synthetic near misses.

- **verdict-dense** uses the 009 verdict move (`right` / `wrong: <n>`). A move
  is queued after every arithmetic attempt instead of 30% of them. It sees
  more verdict moves and less arithmetic. That trade is part of the arm.
- **solve-judge-synthetic** is the same stream as `verdict-synthetic`, but
  uses the answer-then-judge move (`src/onwordly/tasks/solve_judge.py`). It is
  graded exactly: the canonical integer, `;`, then `right` or `wrong`.
- **verdict-rl** is the same stream as `verdict-synthetic`. For each verdict
  move the harness samples `rl_samples` (4) completions at `rl_temperature`
  (1.0) and rewards each 1 or 0 with the exact verifier. Each sample gets
  advantage = reward − group mean. Each sampled completion then gets one
  update on advantage × its negative log-likelihood. If all rewards in a
  group are equal, the update is skipped. This is **REINFORCE with a
  group-mean baseline**: GRPO-style group advantages without ratio clipping,
  a KL penalty or std normalisation. It is an established method, not an
  Onwordly one. Every trained (prompt, completion) pair counts toward the same
  100k-token budget. Sampling is counted as generation calls and time, and is
  also reported separately (`training.on_policy`).
- **verdict-rl-graded** and **selfcheck-rl-binary** run identical two-turn
  self-check episodes (`src/onwordly/training/selfcheck_episodes.py`). With
  probability `verdict_rate`, an arithmetic training task gets a group of
  `rl_samples` episodes at `rl_temperature`:
  - turn 1 samples an answer;
  - turn 2 samples a verdict (`right` / `wrong: <n>`) on that same answer;
  - if turn 1 is unparseable, the episode stops there and only turn 1 is trained.

  Both turns are trained with the episode's advantage (reward − group mean).
  A group whose rewards are all equal is skipped, as in `verdict-rl`.

  On-policy tasks in all three RL arms go straight to sampling. They get no
  greedy pre-update attempt and no source observation. They are therefore
  absent from `pre_update_attempts`, `correct_before_train` and
  `bucket_stats`, which cover SFT tasks only.

  Episode rewards. Turn 1 is classed right, close, far or unparseable, using
  the close tolerance below; "wrong" means close or far.

  | Episode | graded | binary (final answer correct) |
  | --- | ---: | ---: |
  | turn 1 right, said right | 1.0 | 1 |
  | turn 1 right, said wrong | −0.5 | 1 if the "repair" equals the answer, else 0 |
  | turn 1 wrong, said wrong, exact repair | 0.6 | 1 |
  | turn 1 wrong, said wrong, close repair | 0.4 | 0 |
  | turn 1 wrong, said wrong, far repair | 0.3 | 0 |
  | turn 1 close, said right | 0.2 (`CLOSE_ACCEPTED_REWARD`) | 0 |
  | turn 1 far, said right | 0.0 | 0 |
  | unparseable verdict or turn 1 | 0.0 | 0 |

  A repair is "close" when |repair − answer| ≤ max(1, 0.05·|answer|)
  (`CLOSE_REPAIR_MIN_ABS`, `CLOSE_REPAIR_REL`). Tier counts are recorded in
  `training.on_policy.reward_tiers` for both arms, so graded vs binary is
  isolated on identical episodes. Graded rewards and improvement bonuses are
  established reward shaping (SCoRe stage II, arXiv:2409.12917), not an
  Onwordly idea. The graded-vs-binary comparison is the ablation.

## Evaluation (first 500 held-out problems, identical for every arm)

009's verdict measures plus per-class counts for every arm:

- shown a right proposal: kept / rejected / unparseable;
- shown a wrong proposal: caught + repaired / caught + misrepaired / accepted / unparseable;
- self-check on the model's own first answers: own-right kept / rejected,
  own-wrong caught + repaired / caught + misrepaired / accepted, unparseable
  verdict, unparseable first pass.

Each arm is evaluated in the format it was trained on, and the result
records which (`verdict_evaluation.format`). For `solve-judge-synthetic`:

- right-shown accuracy counts the verdict alone, so it compares with 009;
- `right_shown_with_answer_accuracy` also requires the correct number;
- repair means saying `wrong` with the correct number;
- in the self-check, `right` keeps the first answer and `wrong` takes the stated number.

The other arms and `static` are evaluated in the 009 format. 009's 001
arithmetic splits are reported for every arm.

## Pre-registered reading

An arm **works** if two things hold:

- its balanced verdict accuracy is clearly above 50%;
- its right-shown accuracy does not collapse. Trading always-accept for
  always-reject is not a repair.

With one seed, every reading is provisional. A claim needs the repeated-seed
suite, which stays gated until a single run has been inspected.

| Outcome | Reading |
| --- | --- |
| `verdict-dense` works, others do not | 009 was under-trained on verdicts. Budget is the lever. Next, check how much arithmetic it costs. |
| `solve-judge-synthetic` works | Ordering matters: the model can judge by comparing once it has written its own answer. Judging before solving was the obstacle. |
| `verdict-rl-graded` beats `selfcheck-rl-binary` | The shaped credit for rejecting and repairing (and the penalty for rejecting right answers) helps beyond the final-answer reward. Read the per-tier counts before crediting the shaping. |
| `selfcheck-rl-binary` ≈ `verdict-rl-graded`, both above `verdict-rl` | Training on the model's own answers matters; the reward shape does not. |
| `verdict-rl` works, SFT arms do not | Consistent with SCoRe and related work: on-policy reward installs rejection where imitation does not. Next, check sample cost per point gained. |
| Several work | Compare their cost: tokens are matched, so compare generation calls and arithmetic accuracy lost. |
| None works | At 0.5B and 100k tokens, conditional rejection is not learned by any of the three levers. That is a useful null. Do not add machinery just to rescue it. |

Watch the arithmetic splits too. 009's verdict arms lost held-out and
out-of-range accuracy relative to `static`.

## Claim discipline

Self-verification, answer-before-verdict generative verification and
REINFORCE/GRPO-style RL are prior work (see `docs/novelty-ledger.md`). 010's
contribution, if any, is the controlled lever comparison under exact
verification and matched training tokens.

## Run

Use the Kaggle notebook plan `experiment: 010` / `mode: single` (no
`manifest:` line), or batch `jobs: 010`. That is seven regimes over two T4s.
Expect roughly 4–4.5 hours. The three RL arms each add about 15–25 minutes
of sampling to the roughly 42-minute SFT regime:
- `verdict-rl` makes one 4-sample call per verdict move;
- the self-check arms make one 4-sample call plus up to 4 single verdict
  calls per episode group, with about 900 groups at `verdict_rate` 0.3.

The estimate is from 009's timings and has not been measured. Smoke manifest:
`smoke-manifest.json`.
