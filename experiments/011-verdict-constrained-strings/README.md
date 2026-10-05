# Experiment 011 — Verdict on constrained strings

**Status:** prepared, CPU-tested with fake adapters and a tiny-gpt2 smoke; no real-model run.

## Question

Experiments 009 and 010 train self-verification on arithmetic. In arithmetic,
judging `47 * 6 = 272` requires computing 282, so checking costs as much as
producing. There is no verification asymmetry.

**Hypothesis:** self-correction is learnable where checking is cheaper than
producing (a generation-verification gap), and is not learnable where it isn't.
011 ports 010's self-check design to a domain where checking is cheap.

That a generation-verification gap exists, and that it matters for
self-improvement, is established prior work; see `docs/novelty-ledger.md`. 011
tests it under exact verification and matched tokens at 0.5B. It does not claim it.

## Domain

Constrained string generation (`src/onwordly/tasks/constrained_strings.py`). This
is new; it is not Experiment 004's deterministic transformations. 004 has a unique
answer that must be computed, so it has the same lack of asymmetry as arithmetic.

Each task lists 5 independent rules over a short string from `ABCDE12345`:

- always: "use only these characters" and "exactly L characters";
- plus 3 drawn from: start with X, end with Y, contain Z, not contain W, no equal neighbours.

Tasks are built from a random witness string that satisfies every rule, so they
are satisfiable by construction. The witness is the SFT target and the repair
target. **Any** string that satisfies every rule counts as correct. Each rule is a
local exact check, so judging a proposal never requires producing one.

s(x) is the fraction of rules x satisfies. It gives graded credit.

Train/eval split: hashed on the rule set alone, never on the witness, so no
held-out rule set appears in training. Held-out lengths are 5/7/9; the "longer"
split uses length 12.

## Arms

All arms use the same model, seed, static pool and 100k-token budget. RL settings
match 010 (`verdict_rate` 0.3, 4 samples, T = 1.0).

| Arm | What it trains |
| --- | --- |
| `static` | SFT on string tasks only (control). |
| `verdict-synthetic` | Same stream. After 30% of attempts, an SFT verdict move. The proposal is the witness with p = 0.5 (target `right`); otherwise a synthetic violation, the witness with one substitution, deletion, insertion or duplication that breaks at least one rule (target `wrong: <witness>`). |
| `selfcheck-rl-binary` | Same stream. After 30% of attempts, a group of 4 on-policy two-turn episodes: produce a string, then judge your own string (`right` / `wrong: <string>`). Reward 1 if the final string (turn 1 if kept, else the repair) satisfies every rule, else 0. |
| `selfcheck-rl-graded` | Identical episodes, with the graded reward below. |
| `challenge-rl-graded` | Same stream. After 30% of attempts, a group of 4 on-policy fallible-challenge episodes (answer, then `hold` / `change` against another player who is wrong 30% of the time); graded reward. See the challenge section below. |

The episodes use 010's generic on-policy harness update. This is REINFORCE with a
group-mean baseline (GRPO-style advantages without clipping or KL); established,
not Onwordly's. Both turns are trained with the episode advantage, and groups with
equal rewards are skipped. On-policy tasks get no greedy pre-update attempt.
String tasks stay SFT.

Graded reward (`src/onwordly/training/constrained_sources.py`, named constants):

| Episode | graded | binary |
| --- | ---: | ---: |
| turn 1 valid, said right | 1.0 `RIGHT_KEPT_REWARD` | 1 |
| turn 1 valid, said wrong | −0.5 `RIGHT_REJECTED_REWARD` | 1 if the repair is valid |
| turn 1 invalid, caught, repair valid | 0.6 `WRONG_REPAIRED_REWARD` | 1 |
| turn 1 invalid, caught, repair invalid | 0.3 + 0.1·s(repair), in [0.3, 0.4) | 0 |
| turn 1 invalid, said right | 0.2·s(turn 1), in [0, 0.2) | 0 |
| unparseable turn 1 or verdict | 0.0 | 0 |

An invalid string has s < 1, so a partial repair stays below 0.4 (010's "close
repair") and an accepted invalid string stays below 0.2 (010's "close accepted").
s plays the role 010's closeness tolerance played. Reward shaping is established
(SCoRe stage II, arXiv:2409.12917); graded vs binary is the ablation.

## Evaluation (identical for every arm)

- held-out and longer: fraction valid, and mean s;
- the first 500 held-out tasks, as in 010:
  - shown the witness: kept / rejected / unparseable;
  - shown a synthetic violation: caught + repaired (repair valid) / caught + misrepaired / accepted / unparseable;
  - self-check on the model's own first string, with own-right and own-wrong classes. It reports first → final accuracy, fixed/broken, mean s(first) and **mean s(final)**.

Verification is exact and programmatic throughout.

## Fallible-challenge arm (earned self-trust)

`challenge-rl-graded` runs 010's fallible-challenge game on constrained strings
(`src/onwordly/training/challenge_episodes.py`, `CONSTRAINED_CHALLENGE`; no new
architecture, generic on-policy harness, no greedy pre-update attempt for
episode tasks). Turn 1 the model writes a string; turn 2 another player either
says it is wrong and proposes Y, or says it is right. The challenger is wrong
with probability `challenge_error_rate` (manifest, 0.3): a wrong challenge to a
valid string proposes a synthetic one-edit violation of the witness; a wrong
confirmation endorses an invalid string. The model replies `hold: <string>` or
`change: <string>`. "Right" means every rule holds (exact checker).

Graded reward, as 010's challenge table with 011's satisfaction scaling:
right & held 1.0; right & changed to an invalid string (caved) −0.5; right &
changed to another valid string 0.6; wrong & changed to a valid string 0.6;
wrong & changed to an invalid string 0.3 + 0.1·s; wrong & held 0.2·s(turn 1);
unparseable or inconsistent 0.0.

Evaluation (`challenge_evaluation`, this arm only, first `verdict_evaluation_size`
held-out tasks): per-class counts (4 types × held / changed to valid / changed
to invalid / unparseable), hold rate when right under a wrong challenge
(anti-sycophancy), change rate when wrong under a correct challenge
(corrigibility), discrimination (hold rate when turn 1 is valid minus when
invalid), and final accuracy, each at eval error rates 0, 0.3 and 0.5. The
reading is 010's challenge table; in addition, compare with 010: if checking
is cheaper than producing here, discrimination should be higher in 011 than
in 010's `challenge-rl-graded`. There is no SFT challenge control in 011.

Prior work: sycophancy under challenge (FlipFlop, arXiv:2311.08596; Sharma et
al., arXiv:2310.13548) is established; see `docs/novelty-ledger.md`.

## Pre-registered reading

As in 010, an arm **works** if two things hold:

- its balanced verdict accuracy is clearly above 50%, or, for the self-check arms, the self-check is net-positive (fixed > broken, with mean s(final) > mean s(first));
- right-shown accuracy does not collapse.

| Outcome | Reading |
| --- | --- |
| A 011 arm works and the matching 010 arm does not | Consistent with the hypothesis: verification asymmetry is the variable that makes self-correction learnable here. It is one domain and one seed, so check with seeds and a second asymmetric domain before saying more. |
| 011 fails like 010 (accept collapse, no fixes) | Asymmetry alone is not enough at 0.5B / 100k tokens. The bottleneck is elsewhere (model scale, budget, the prior; see `docs/verdict-prior.md`). |
| 011 and 010 both work | The asymmetry is not needed. Whatever worked in 010 is the lever. |
| `verdict-synthetic` works, RL arms do not | Imitating local checks is enough when checks are local. |
| graded beats binary | Partial credit (s) helps. Read the per-tier counts first: graded can win by accepting nearly valid strings (wrong_accepted tier). |
| Plain held-out validity drops vs static | Verification is bought with generation. Report it alongside the gain. |

One seed is provisional. The suite is gated until a single run is inspected.

## Run

Kaggle notebook plan:

```
experiment: 011
mode: single
```

Five regimes run one worker process each, queued over both T4s (`run_units`), then
the summary and `RESULTS.md` are written. Smoke: add
`manifest: experiments/011-verdict-constrained-strings/smoke-manifest.json`.

**Runtime estimate (unmeasured):**

- 011 prompts are about 70–90 tokens against about 30 for arithmetic, so 100k tokens is about 1,000 SFT examples, fewer than in 010. Each greedy attempt generates up to 24 tokens.
- That puts the SFT arms at about 30–40 minutes each, and each RL arm at about 15–25 minutes more for sampling (100–300 episode groups, fewer when trained pairs use up the budget, each with one 4-sample call and up to 4 verdict calls).
- Four regimes on two T4s: roughly 1.5–2 hours, including evaluation (about 1,000 + 300 + 3 × 500 generations per arm).
- `challenge-rl-graded` adds one regime: about 30–40 minutes of SFT plus 20–30 minutes of sampling (challenge prompts are longer), plus about 2,000 challenge evaluation generations (about 5–10 minutes). Five regimes on two T4s: roughly 2–2.5 hours (unmeasured).
