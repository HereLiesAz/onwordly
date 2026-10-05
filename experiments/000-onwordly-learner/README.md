# Experiment 000 — the Onwordly learner

**Status:** two full runs (one seed, `RESULTS.md`); recurring-frame evaluation and `learner-first-visit` prepared and CPU-tested, not run.

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
| `learner-self-memory` | diagnostic: register read uses only `source="self"` fillers (no corrector proposals, no verifier), in training and evaluation |
| `learner-memory-dropout` | diagnostic: in training, with probability `memory_dropout` (default 0.5) per problem, the register read (both passes) is replaced by the empty-register encoding; separate RNG stream, so problem and challenge order are unchanged; evaluation unchanged |
| `learner-first-visit` | remedy (see "Recurring frames"): on a training revisit (the frame's readable register is non-empty when drawn), the memory-reading passes are trained only through the hold/change decision; that problem's solve and confidence losses come from an extra empty-register pass, and its challenge-pass workspace loss is dropped |
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

## Memory diagnostics

Added after the first full run (`RESULTS.md` on the run branch): every arm that
reads the register collapsed (≤ 0.7% held-out solve) while
`learner-no-memory` reached 75.3%. Hypothesis: the register is a train-time
answer channel. Training rule sets are revisited ~13×, so the register holds
that frame's earlier fillers, including corrector proposals (A is right 90%);
held-out frames arrive with an empty register, a condition seen on ~1 in 13
training visits. Existing arms' behaviour is unchanged by the diagnostics.

Reported for every memory-reading arm (`memory_diagnostics` in the arm JSON,
"Memory diagnostics" table in `RESULTS.md`):

- **Training reads** (`training.register_diagnostics`): fraction of
  solve-pass training reads (before that visit's writes) where the readable
  register was non-empty, and where its top non-self filler (by count) is a
  valid answer; dropped fraction for the dropout arm.
- **(1) Training-frame probe:** `probe_size` (default 512) training problems,
  a fixed sample (`Random("<evaluation_seed>:register-probe")`), after
  training, eval mode, read-only on an eval view. Accuracy with the register
  as built vs emptied (empty-register encoding), plus the same non-empty /
  top-other-is-target fractions.
- **(2) Held-out second visit:** attempt 1 on held-out frames (empty
  register); only the model's own attempt-1 answer is written into a forked
  eval register; attempt 2 reads it. Accuracy for both visits.
- **(3) `learner-self-memory`** and **(4) `learner-memory-dropout`**: arms
  above, same matched budget (optimizer steps × problems, same order).

Manifest fields: `memory_dropout` (default 0.5, in [0, 1]), `probe_size`
(default 512; smoke 32).

**Pre-registered reading.** If (1) shows a large drop on emptying (as-is ≫
emptied) for `learner` and (3) and/or (4) recover held-out solve accuracy
toward `learner-no-memory`, the answer-channel hypothesis holds. If emptying
costs little and neither arm recovers, the collapse has another cause (look
at the memory input path itself). (2) says whether the model's own earlier
answer helps or hurts on a revisit; it is descriptive, not a fix. One seed is
provisional.

## Recurring frames

Added after run 2 (`RESULTS.md`): the register is keyed by exact frame and
held-out frames are new, so in the standard evaluation memory is always empty
and can only teach a train-time shortcut. The recurring-frame evaluation gives
memory a chance to matter: memory of one's own attempts, corrections and
contrasts on a frame that comes back. Code: `learner/recurring.py`.

**Stream.** The first `recurring_frames` (default 500) held-out tasks with
distinct frames, each visited `recurring_visits` (default 4) times. Each round
is a shuffle of all frames seeded by `"<evaluation_seed>:recurring:round:<k>"`;
frames among the last `recurring_min_gap` (default 25) of a round move to the
end of the next round, so at least `recurring_min_gap` other visits separate
two visits of a frame. Identical for every arm. `recurring_visits: 0` disables
the evaluation.

**Each visit.** (1) attempt: the solve pass reads the frame's register in the
eval view; (2) a corrector challenges: A / B / C with the challenge-evaluation
error rates, drawn per visit from `Random("<evaluation_seed>:recurring")`
(the assignment is the same for every arm); (3) the model (or the hand-coded
rule) holds or changes; (4) the exact verifier grades it. Everything is written
with the training write rules into one `eval_view(register_verifier=True)`:
attempt, challenge, correction, deliberation and verifier records into a
scratch store; draft, proposal, changed final answer and verifier filler into
a forked register. **Visible on later visits:** the model's own fillers
(`self`) and corrector proposals, as in training. **Not visible:** the verifier
filler: training reads exclude it, so this evaluation does too. It is stored,
never read. The **ledger stays frozen** (no trust update from evaluation, as in
the challenge evaluation), so a gain across visits comes from the frame's
register, not from trust. Nothing reaches training memory (tested). Visits run
in stream order, in chunks with no repeated frame, so visit k reads every
earlier visit of its frame.

**Arms.** `learner-no-memory` (reads nothing; its visits differ only by
corrector draw), `learner-memory-dropout`, `learner-first-visit`, `plain` (no
memory, no challenge: solve only) and `handcoded` (plain's answers + the fixed
rule on the frozen ledger). Manifest: `recurring-manifest.json` (full size,
those five arms); `recurring-smoke-manifest.json` (tiny).

**Training-side remedy (`learner-first-visit`).** The rule chosen, of the
options considered: training reads stay as in `learner` (self + corrector
fillers of the frame; the verifier filler is excluded, because reading it
would hand the decision the answer). A problem is a *revisit* when its
frame's readable register is non-empty at the moment it is drawn (checked
before the batch's writes). For a revisit, gradients reach the memory-reading
passes only through the REINFORCE hold/change term; its solve and confidence
losses are computed on an extra pass of the same problem with the
empty-register encoding (extra forward compute, recorded in
`forward_steps`), and its challenge-pass workspace cross-entropy is dropped.
So memory cannot shortcut the solve or the revised answer; it can only inform
whether to hold or change. First visits train exactly as `learner`. Same
matched budget (optimizer steps × problems, same order). The `min_gap`-delayed
read variant was not built. `training.register_diagnostics.revisits_decision_only`
counts revisits.

**Metrics** (`recurring` in each arm's JSON; "Recurring frames" tables in
`RESULTS.md`), per visit index 1..k and per corrector: solve accuracy (draft,
before the challenge), final accuracy (after), hold rate when right under a
wrong challenge, change rate when wrong under a correct challenge (and to
correct), hold rate when told wrong, and the fraction of visits whose readable
register was non-empty. Learning curve: visit k − visit 1 for solve and final.
"vs no-mem": final minus `learner-no-memory`'s final at the same visit. For
frames whose visit-1 draft was wrong: fraction right at each visit (draft and
final).

**Pre-registered reading (one seed is provisional).** Memory *works* on
recurring frames if a memory arm's final accuracy rises across visits (Δ final)
by more than `learner-no-memory`'s does (whose Δ is corrector-draw noise) and
more than `plain`'s (Δ = 0 by construction), **with challenge behaviour
intact**: at the last visit, hold-right-under-wrong and
change-wrong-under-correct no worse than its own visit 1 by more than 5
points, and hold-when-told-wrong still higher against B than against A. A rise
in final accuracy with collapsing hold-right (it changes everything once
memory is non-empty) is copying corrector proposals, not memory working; check
the per-corrector table (following B's wrong proposals would show as falling
final accuracy for B). If no memory arm beats `learner-no-memory`'s Δ, memory
of one's own past attempts on a frame does not help this model; the next step
is similarity recall (RESULTS run-2 reading, direction 2). If
`learner-first-visit` has a clearly lower visit-1 solve than
`learner-no-memory`, the remedy costs capability and its curve must be read
with that in mind.

**Runtime.** First full runs: ~8 min training per learner arm on a T4;
`learner-first-visit` adds one 4-step solve pass on revisits (~90% of
problems), ~+50% forward compute, so ~12 min. Standard evaluations and probes
a few minutes per arm; the recurring stream (2000 visits) under a minute per
arm. Estimate for `recurring-manifest.json` (3 learner arms + plain +
handcoded): ~40–50 min on one T4, unmeasured.

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
- Held-out frames are never revisited in the standard evaluation, so register
  recall cannot help there by construction; the recurring-frame evaluation is
  the revisit split.
- Recurring frames: the register holds no verifier filler a reader can see,
  so a later visit learns only from its own earlier drafts and the
  correctors' proposals (A's are right 90% of the time). Copying proposals is
  a legitimate use of remembered corrections, but it is not the same as
  remembering what was verified.
- The model and its components are established (iterative refinement /
  recurrent-depth transformers, Beta reputation, Dawid–Skene-style
  reliability, event sourcing). Only the combination is the hypothesis.

## Runtime

CPU probe (full-size model, 1.84M params): about 0.6 s per learner step, and
about 0.25 s per plain step including the hand-coded ledger loop. On one T4,
the Python memory and feature loop dominates. The estimate is about 1–2 h for
all six arms (4 learner trainings + 1 plain). That is unmeasured. Run the
smoke manifest first.

With the two diagnostic arms (6 learner trainings + 1 plain): at ~0.6 s per
learner step, each learner arm is ~40 min for 4000 steps, so add ~1.5 h; the
probes add a few minutes per memory arm (512 + 2 × eval_size solves).
Estimate ~2.5–3.5 h on one T4, unmeasured. To run only the diagnostics, set
`arms: learner,learner-no-memory,learner-self-memory,learner-memory-dropout`
in `.kaggle-run` (~2–2.5 h).

## How to run

```
python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/recurring-smoke-manifest.json --output results/000-recurring-smoke
python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/recurring-manifest.json --output results/000-recurring
python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/smoke-manifest.json --output results/000-smoke
python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/manifest.json --output results/000-onwordly-learner
```

Kaggle: `.kaggle-run` with `experiment: 000` (optional `manifest:`, `arms:`),
then `python -m onwordly.learner.kaggle --plan .kaggle-run`. This entry point
is standalone (the archived LLM runner is gone); the `onwordly-kaggle` console script used by central dispatch points to it.
Suite mode is gated until a single run has been inspected.
