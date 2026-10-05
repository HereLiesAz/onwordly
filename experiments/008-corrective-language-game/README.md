# Experiment 008 — Corrective arithmetic language game

**Status:** one real-model seed run; see `RESULTS.md`.

## Why

Experiment 001 (run 16, one seed) taught the arithmetic language game: static and
adaptive reached ~80% held-out, but error-focused training was worse everywhere.
Error-focused training only adds more moves of the same game (neighbouring
problems after a failure); the model never sees its own wrong answer and never
plays the game of correcting one. Experiment 008 asks whether teaching that
corrective game is what was missing.

## The corrective game

~~~text
Compute 47 * 6. Return only the integer answer. A previous answer was 272.
If it is wrong, return the correct integer; if it is right, return it unchanged.
→ 282
~~~

The target is always the true answer, so the exact integer verifier applies
unchanged. Confirm moves show the right answer and expect it repeated, so the
model cannot learn that a previous answer is always wrong.

## Regimes (matched 100k-token budget, same model, seeds and data as 001)

| Regime | Arithmetic stream | After a wrong, parseable attempt | After a right attempt |
| --- | --- | --- | --- |
| static | frozen pool | — | — |
| corrective-own | frozen pool | corrective move showing the model's own wrong answer | confirm move with p = 0.25 |
| corrective-synthetic | frozen pool | corrective move showing a synthetic plausible error | confirm move with p = 0.25 |
| error-focused | adaptive (as in 001) | neighbouring arithmetic problems | — |

- `corrective-own` vs `corrective-synthetic` share the trigger and the target;
  only the shown answer differs. That isolates learning from one's *own* errors.
- `corrective-*` vs `static` share the arithmetic stream; corrective moves are
  paid for out of the same token budget.
- The model's own answer comes from the pre-update attempt every regime already
  makes; no extra generation is spent in training.
- Corrective moves are built only from training-partition problems.

## Evaluation

All of 001's splits, plus on the first 500 held-out problems (identical for
every regime, including static):

- **correction** — shown a synthetic wrong answer, return the right one;
- **confirmation** — shown the right answer, return it unchanged;
- **self-correction** — answer, then see your own answer in a corrective prompt;
  report pass 1, pass 2, and how many answers were fixed or broken.

## Claim discipline

Self-correction training is prior work (self-refinement, learning from
mistakes, RL-based self-correction). The Onwordly contribution, if any, is the
controlled comparison under exact verification and matched budgets, and the
language-game framing — not the technique.

## Run

Kaggle plan `experiment: 008` (single) or batch `jobs: 008`; four regimes run
two per GPU on T4 x2, roughly 2 hours.
