# Positioning

Where Onwordly stands relative to prior work, where it leads, and what comes next. Citations and verification status live in `docs/novelty-ledger.md` and `reports/Self correction and verifier training.md`.

## What is not new

Every component is borrowed: corrective SFT (LeMa, Pair-SFT), joint verification (GenRM), synthetic error injection (Physics of LMs 2.2), curricula, hard-example mining, LoRA. The claim that correction constitutes a rule is Wittgenstein's, then Kripke's, then Brandom's. None of it is ours.

## What is different

The comparison, not the parts.

- **Exact grading.** Every verdict, correction and answer is checked programmatically. No neural judge, nothing to reward-hack.
- **Matched budgets.** Regimes compete at equal training tokens (100k); inference and verifier work are recorded separately.
- **Balanced proposals.** Right and wrong proposals at 50/50, so always-accept and always-reject earn nothing and keep/flip rates are both measurable.
- **Error source as a variable.** Own errors vs synthetic errors, crossed rather than chosen.
- **Scale.** A pretrained 0.5B model. The nearest verifier studies use ≥7B models or toy transformers trained from scratch.
- **The language-game heuristic.** Moves (confirm, reject, repair) are chosen by asking what a competent player of the game must be able to do — including correct someone else. A design heuristic, never a finding.

## What 008 already says

SCoRe reported that correction SFT collapses to "no edits". Experiment 008 shows the collapse has a direction, and the error source picks it:

| Training errors | Behaviour | Correction | Confirmation |
| --- | --- | --- | --- |
| Synthetic | copies the shown answer | 4.6 | 88.6 |
| Own | ignores the shown answer | 74.8 | 74.6 |

Neither uses the proposal conditionally. The untrained model, meanwhile, broke 114 of 500 right answers when shown a wrong one — sycophancy, measured exactly. The model learns to correct nobody; it learns which shortcut its errors pay for.

## Where it leads

Experiment 009 asks whether an explicit verdict move makes the shown answer *informative*: a correction–confirmation gap in the right direction under both error sources. If yes, the claim worth making is narrow and testable — at small scale, under exact verification and matched budgets, judging must be trained before repairing, and the wrong-answer distribution decides what gets learned. If no, the collapse is deeper than the training format, and on-policy RL becomes the next lever.

The longer arc: exact-verification domains (003–007) as cheap laboratories for questions usually asked of frontier models with neural judges. Small, controlled, falsifiable.

## What comes next

1. **Per-class keep/flip rates** for every arm: confirmation-keep, correction-fix, right→wrong breaks, wrong→wrong copies.
2. **Factorial cross:** own vs synthetic wrong proposals × balanced vs natural right/wrong ratio, matched tokens.
3. **Error-free control:** same tokens spent on plain correct solutions (Ye et al.), separating "learning from mistakes" from "more arithmetic".
4. **Pacing control:** same data-growth schedule, random order (Wu et al.), before crediting any adaptive arm.
5. **SCoRe-style RL stage** from the best SFT arm, exact checker as reward, compute recorded separately.
6. **Verification → generation:** plain no-proposal accuracy after verdict-only training vs a generation-only arm (tests arXiv:2602.07594 at sub-1B).

Nothing above is claimed until 009 reports.
