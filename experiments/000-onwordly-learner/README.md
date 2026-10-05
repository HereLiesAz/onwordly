# Experiment 000 — the Onwordly learner

**Status:** prepared; CPU-tested (tiny end-to-end run in `tests/learner/`); no full run.

Code: `src/onwordly/learner/` (model, training, evaluation, Kaggle entry) and
`src/onwordly/memory/` (episode store, variant register, trust ledger,
update-weight rule). Self-contained: nothing imports the archived LLM harness.

## Question

Does a small network built from scratch around the design in
`docs/model-design.md` — an answer it keeps revising, add-only external memory,
and trust earned per domain and per corrector — (1) solve held-out
constrained-string problems at least as well as a plain network of the same
size on the same training examples, and (2) learn to hold a right answer
against a wrong corrector and change a wrong one when the corrector is right,
with its hold rate following how reliable each corrector has proved?

## Task

The constrained-string task from archived 011 (copied into
`learner/task.py`): five exact rules over a string from `ABCDE12345`
(alphabet, length 5/7/9, plus three of starts/ends/contains/excludes/no-repeat).
Any string that satisfies every rule is correct, and checking a string is
cheaper than producing one. Train and held-out problems are split by a hash of
the rule set, so no held-out rule set is ever trained on. Arithmetic, the
second domain, is **not built yet** (frames exist in `memory/register.py`; the
encoder does not).

## The model (`learner/model.py`)

- **Workspace:** one slot per output position (max length + 1), each holding
  logits over the alphabet plus PAD, next to one global token carrying the
  problem encoding, the memory read, the trust read and the challenge.
- **Revision:** `revision_steps` passes of a shared transformer over
  [global, slots]. Each pass adds a delta to every slot's logits, so any slot
  can change at any step, and emits an answer, a confidence and a hold logit.
  There is no step embedding and no halting unit; the budget is fixed by the
  manifest and recorded.
- **Memory (external, non-parametric, add-only):** before each pass the
  network reads the frame's variant-register summary: per-slot character
  histograms of its own earlier fillers and of corrector fillers, the number
  of entries, the number of distinct fillers and a divergence flag. Verifier
  fillers are stored but never read back, because on a revisited training frame
  they would give away the answer.
- **Reasoner:** when a challenge arrives, a second pass starts from the draft
  and reads the challenge (says right / says wrong, plus the proposal per slot),
  the register (which now holds draft and proposal), and the ledger's
  self-trust for this domain and reliability for this corrector. It outputs
  hold/change and a revised answer. A deliberation record that lists what it
  read is written to the store.
- **Learning:** cross-entropy on every revision step against the witness
  (deep supervision); BCE on confidence; REINFORCE with a batch-mean baseline
  on the hold/change decision, using the graded reward from the 010/011
  challenge game (right held 1.0, caved −0.5, wrong changed to right 0.6,
  changed to wrong 0.3 + 0.1·s, wrong held 0.2·s, inconsistent 0); and
  cross-entropy of the challenge pass against the witness. The episode loss is
  scaled by `update_weight(confidence, surprise)`. Here confidence is the
  probability the network gave the move it made, surprise is 1 if the final
  answer is wrong, and weights are normalised to a batch mean of 1. This
  inverts Kalman-gain behaviour on purpose; that is the hypothesis, not an
  established result.
- **Ledger:** Beta(α, β) self-trust per domain (success = the hold/change
  decision was right) and per-corrector reliability (success = the claim was
  right). Prior Beta(1, 1), half-life 20k episodes, evidence cap 500. It is
  updated after each optimizer step from **training** outcomes only. It is
  frozen before evaluation and raises on any later update.

## Correctors

Training: A (wrong 10% of the time) and B (wrong 50%), one drawn per problem.
Evaluation: A, B, and an unseen C (30%). C has no ledger history, so it reads
the prior (0.5). Correctors are identified to the network **only** through
the ledger's reliability estimate. That is what makes the no-trust ablation
clean: without it, the network cannot tell correctors apart.

## Arms and matched budget

Budget: `train_steps` (4000) optimizer steps × `batch_size` (64) training
problems, drawn in the same order (`Random(training_seed)`) for every arm.
Parameters are matched: the learner has 1.84M; the plain MLP's width is chosen
to land within 2% of that. Extra compute is recorded as `forward_steps`
(revision steps × passes × problems), not hidden: the learner does 8 forward
passes per problem, the plain net 1.

| Arm | What changes |
| --- | --- |
| `learner` | full model |
| `learner-flat` | update weight = 1 (ablates confidence × surprise) |
| `learner-no-memory` | register read zeroed (store still written) |
| `learner-no-trust` | ledger read zeroed |
| `plain` | one-pass 3-hidden-layer MLP, supervised cross-entropy, same problems and steps |
| `handcoded` | `plain`'s answers + an aive-style rule. Under "you are wrong", change to the proposal only if the corrector's raw success rate is ≥ 0.7, beats the domain's self success rate, and its three-strike breaker is closed. Otherwise hold. The ledger is built during `plain`'s training from challenges to its pre-update drafts. Reference point, reported separately. |

## Evaluation (held-out rule sets, exact)

- Solve: final accuracy, per-step accuracy, fixes and breaks between steps and
  from first to last step, mean rule satisfaction, confidence calibration
  (ECE, Brier).
- Challenge, per corrector A / B / C: 010's four types × held / changed to
  correct / changed to wrong / invalid. Also: hold rate when right under a
  wrong challenge; change rate when wrong under a correct challenge (and
  change-to-correct rate); hold rate when told wrong; hold discrimination;
  final accuracy; decision calibration (confidence of the move taken vs
  whether the decision was right).
- `hold_tracks_reliability`: hold-when-told-wrong against B minus against A.
- Each corrector condition gets a fresh eval view: the frozen ledger, a forked
  register and a scratch store. No evaluation outcome reaches the training
  ledger, register or store (enforced and tested).

## Pre-registered reading (one seed is provisional)

1. **Solve:** `learner` within 2 points of `plain`, or better. If it is
   clearly worse, the revising architecture costs capability at this size,
   and the challenge results must be read with that in mind.
2. **Earned trust:** `learner` hold-right-under-wrong > 70% *and*
   change-wrong-under-correct > 50% against A, with hold discrimination
   > 0.3. If it holds everything or changes everything, it found a shortcut.
3. **Reliability tracking:** `hold_tracks_reliability` > 0.1 for `learner`
   and ≈ 0 for `learner-no-trust`. If both are > 0.1, the trust input is not
   what drives it.
4. **Update rule:** `learner` vs `learner-flat`. Attribute nothing to
   confidence × surprise unless the gap shows across ≥ 3 seeds.
5. **Memory:** `learner` vs `learner-no-memory` on held-out frames, where the
   register holds only the draft and the proposal. A gap here comes from
   within-episode divergence features, not from recall.
6. **Hand-coded:** if `handcoded` matches `learner` on the challenge metrics,
   the learned reasoner adds nothing over a fixed rule.

A null result is a result.

## Honest risks

- Witness supervision: cross-entropy targets one witness, but any valid string
  is accepted. This penalises valid alternatives. It is identical for every
  arm, but it caps solve accuracy.
- The draft enters the challenge pass as strong one-hot logits. "change" to
  the same string counts as inconsistent (reward 0), which may push the
  network toward hold.
- Self-trust has one key per (rule count, length), so it is coarse, and
  corrector reliability converges within a few hundred episodes. In practice
  the trust input may act as a corrector ID, which is still what the no-trust
  ablation is meant to test.
- The hand-coded breaker's state at the end of training is noisy for B
  (three consecutive errors are common at 50%), so the baseline can flip
  between always holding and following B.
- REINFORCE with batch 64 is noisy. The decision may never leave its initial
  hold rate in 4000 steps.
- Held-out frames are never revisited, so register recall cannot help on them
  by construction. A recall test would need a revisit split.
- The model and its components are established (iterative refinement /
  recurrent-depth transformers, Beta reputation, Dawid–Skene-style
  reliability, event sourcing). Only the combination is the hypothesis.

## Runtime

CPU probe (full-size model, 1.84M params): about 0.6 s per learner step, and
about 0.25 s per plain step including the hand-coded ledger loop. On one T4,
the Python memory and feature loop dominates. The estimate is about 1–2 h for
all six arms (4 learner trainings + 1 plain). That is unmeasured. Run the
smoke manifest first.

## How to run

```
python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/smoke-manifest.json --output results/000-smoke
python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/manifest.json --output results/000-onwordly-learner
```

Kaggle: `.kaggle-run` with `experiment: 000` (optional `manifest:`, `arms:`),
then `python -m onwordly.learner.kaggle --plan .kaggle-run`. This entry point
is standalone (the archived LLM runner is gone); the `onwordly-kaggle` console script used by central dispatch points to it.
Suite mode is gated until a single run has been inspected.
