# Experiment 009 — Verdict arithmetic language game

**Status:** prepared, not run.

## Why

Experiment 008 (one seed) taught a corrective move whose target never depended
on judging the shown answer. The model learned surface rules instead of
checking: with its own (usually far-off) errors it learned to *ignore* the
shown answer; with synthetic near misses it learned to *copy* it (correction
4.6%, confirmation 88.6%). Neither arm improved in two-pass self-correction.

009 makes judgement the move itself.

## The verdict game

~~~text
Compute 47 * 6. Return only the integer answer. A proposed answer is 272.
If it is right, reply exactly: right. If it is wrong, reply exactly: wrong: <the correct integer>.
→ wrong: 282
~~~

- Proposals are right with probability 0.5, by construction, independent of
  whether the model's own attempt was right. Always `right` scores 50%;
  always `wrong` cannot name the right number without computing it.
- Copying scores 50%; ignoring cannot produce `right`/`wrong` at all. Checking
  is the only strategy that wins.
- Exact verification: `right` or `wrong: <canonical integer>`, nothing else.

## Regimes (matched 100k-token budget; same model, seed and data as 001/008)

| Regime | Arithmetic stream | Verdict moves (p = 0.3 after each attempt) | Wrong proposals |
| --- | --- | --- | --- |
| static | frozen pool | — | — |
| verdict-synthetic | frozen pool | balanced right/wrong | synthetic near misses |
| verdict-mixed | frozen pool | balanced right/wrong | the model's own wrong answer when it has one, else synthetic |
| corrective-synthetic | frozen pool | 008's corrective moves | synthetic (the copying arm, as a control) |

## Evaluation (first 500 held-out problems, identical for every regime)

- **right shown** — reply `right`;
- **wrong shown** — reply `wrong: <answer>`; *judgement* counts the verdict,
  *repair* also the number;
- **balanced verdict** — mean of the two judgement rates (chance 50%);
- **self-check** — answer, judge your own answer, keep it or take the repair;
  pass 1 vs final, fixed and broken.

Plus all of 001's arithmetic splits.

## Claim discipline

Self-verification and critique-then-revise are prior work. 009's contribution,
if any, is the controlled comparison under exact verification and matched
budgets, and the language-game framing.

## Run

Kaggle notebook plan `experiment: 009` / `mode: single` (no `manifest:` line).
Four regimes, two per T4, roughly 2.5 hours.
