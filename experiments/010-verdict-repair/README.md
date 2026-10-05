# Experiment 010 — Verdict repair

**Status:** prepared, CPU-tested with fake adapters; no real-model run.

## Question

Experiment 009 (one seed, Qwen2.5-0.5B, SFT, 100k training tokens) collapsed
to acceptance: `verdict-mixed` said `right` to every proposal (balanced 50.0%);
`verdict-synthetic` caught 13.4% of wrong proposals (balanced 51.9%) and
repaired every catch. Solving was not the bottleneck. Rejecting was.

009 named three untested explanations. 010 gives each one an arm and changes
only that one thing relative to `verdict-synthetic`:

| Explanation | Arm | The one change |
| --- | --- | --- |
| Too few verdict examples | `verdict-dense` | `verdict_rate` = `dense_verdict_rate` (1.0 vs 0.3): about 3x more verdict moves inside the same token budget |
| The format asks for the verdict before the answer | `solve-judge-synthetic` | reply `<correct integer>; right` or `<correct integer>; wrong`. The model states the answer first and then judges |
| SFT cannot install conditional rejection here | `verdict-rl` | verdict moves trained on-policy with the exact checker as reward. Arithmetic tasks stay SFT |

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
`manifest:` line), or batch `jobs: 010`. That is five regimes over two T4s.
Expect roughly 3–3.5 hours. `verdict-rl` is slower because it generates 4
extra samples per verdict move. Smoke manifest: `smoke-manifest.json`.
