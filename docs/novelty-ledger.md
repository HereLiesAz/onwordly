# Novelty ledger

This file exists to prevent accidental reinvention from acquiring a fake moustache and filing a patent.

## Established families

The following are **not claimed as Onwordly inventions**:

| Technique | Role here |
| --- | --- |
| Active learning | Select informative examples rather than sampling uniformly. |
| Hard-example mining | Concentrate training on examples the model currently fails. |
| Self-correction / learning from mistakes | Train a model to revise a shown or self-generated answer (self-refinement, RL-based self-correction). Experiment 008 applies it under exact verification; not an Onwordly invention. |
| Self-verification / critique-then-revise | Judge a proposed answer before repairing it (generative verifiers, critique-and-revise). Experiment 009 applies it under exact verification; not an Onwordly invention. |
| Curriculum learning | Change task difficulty or distribution over training. |
| Counterexample-guided refinement | Use failures/counterexamples to improve a learner. |
| Self-play | Generate learning pressure through interaction among agents or policies. |
| Synthetic data generation | Programmatically produce additional training examples. |
| Process supervision | Evaluate intermediate reasoning or execution states. |
| Outcome supervision | Evaluate final task success. |
| GRPO/PPO-family RL | Optimize policies from relative or scalar rewards. Experiment 010's `verdict-rl` uses GRPO-style group-relative advantages (Shao et al., DeepSeekMath, arXiv:2402.03300) without clipping or KL; not an Onwordly invention. |
| REINFORCE with a baseline | Policy gradient weighting sampled completions by reward minus a baseline (Williams 1992). 010's `verdict-rl` uses the group mean as baseline with an exact 0/1 reward; not an Onwordly invention. |
| Graded / shaped rewards, improvement bonuses | Partial credit and bonuses for improving an answer (reward shaping, Ng et al. 1999; SCoRe stage II, arXiv:2409.12917). 010's `verdict-rl-graded` uses a graded self-check episode reward; not an Onwordly invention. Its comparison against `selfcheck-rl-binary` is the ablation. |
| Answer-before-verdict verification | The verifier solves before judging (generative verifiers with CoT, e.g. GenRM arXiv:2408.15240). 010's `solve-judge-synthetic` is a minimal exact form; not an Onwordly invention. |
| Generation-verification gap / verification asymmetry | Checking a candidate can be easier than producing one, and self-improvement by self-verification depends on that gap (Song et al., "Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models", arXiv:2412.02674, verified 2026-10-05; for constraint problems, Stechly, Marquez & Kambhampati, "GPT-4 Doesn't Know It's Wrong", arXiv:2310.12397, verified 2026-10-05, on graph colouring with self-critique vs an external verifier). Experiment 011 uses a constrained-string domain with the gap and contrasts it with 010's arithmetic, which lacks it; not an Onwordly invention. The asymmetry framing also echoes the general NP-style "easy to check, hard to solve" intuition, which is folklore, not a citation. |
| Sycophancy / robustness to user challenge | Models flip correct answers when a user pushes back, and this can be measured and trained against (FlipFlop, Laban et al., arXiv:2311.08596; Sharma et al., "Towards Understanding Sycophancy in Language Models", arXiv:2310.13548). Experiments 010/011's `challenge-rl-graded` / `challenge-sft` arms use a fallible challenger (wrong with a set probability) and a hold/change reply; not an Onwordly invention. Onwordly's part, if any, is only the framing as graded earned self-trust (hold when right, change when wrong, credit tied to the exact checker) under exact verification and matched training tokens, with discrimination (hold rate when right minus when wrong) as the measure. |
| MCTS/search | Explore candidate trajectories before selecting or distilling them. |
| Knowledge/trajectory distillation | Transfer expensive teacher/search behavior into a cheaper model. |
| LoRA/parameter-efficient tuning | Adapt models without updating all parameters. |

## Onwordly hypotheses

These are research questions, not claims of invention:

1. A compact executable curriculum can outperform a much larger static dataset under an equal token budget.
2. Curriculum generators that respond to measured competence can improve capability-per-token for small models.
3. Exact verifiers plus adaptive task generation can postpone or reduce the need for learned reward models.
4. Expensive search can be concentrated at training time and distilled sufficiently well that inference remains small and cheap.
5. Interactive language games may supply useful grounding and transfer with less static text than conventional pretraining/fine-tuning pipelines.

## Nearest prior work for Experiments 008–009

Literature review: `reports/Self correction and verifier training.md` (notes in `research_notes/`). Citations marked † were not re-verified against the source and must be checked before publication.

| Area | Nearest work | Finding | Onwordly difference |
| --- | --- | --- | --- |
| SFT correction collapse | Kumar et al., SCoRe, arXiv:2409.12917 | SFT on correction traces collapses to "no edits"; off-policy errors cause train/test mismatch. | 008 shows error source sets the collapse direction: synthetic → copy (4.6 vs 88.6), own → ignore (74.8 vs 74.6). |
| Intrinsic self-correction | Huang et al., arXiv:2310.01798 (ICLR 2024) | Without external feedback, self-correction can lower accuracy. | Exact external verifier throughout. |
| Deference to shown answers | Sharma et al., arXiv:2310.13548; Laban et al. FlipFlop, arXiv:2311.08596 | Assistants are sycophantic; "Are you sure?" flips 46% of answers. | Measured with exact checks at 0.5B (untrained model broke 114/500). |
| Self-correction survey | Kamoi et al., TACL 2024, arXiv:2406.01297 | Works with reliable feedback or large-scale fine-tuning, not prompting alone. | — |
| Joint verify + generate | Zhang et al., GenRM, arXiv:2408.15240 | One model trained to verify and solve beats discriminative verifiers. | 009 trains a verdict move, balanced 50/50, at matched tokens. |
| Verification improves generation | arXiv:2602.07594 (ICML 2026) | Learning to self-verify improves generation; not the reverse. | Untested at sub-1B with exact grading — candidate follow-up. |
| Small models need verifiers | arXiv:2404.17140 | Small (≤13B) models self-correct only with a strong verifier. | — |
| Critique training | Critique Fine-Tuning, arXiv:2501.17703 | Critiquing noisy responses beats SFT by 4–10%. | — |
| Errors-then-corrections data | Ye et al., Physics of LMs 2.2, arXiv:2408.16293 (ICLR 2025) | Beats same amount of error-free data on synthetic math. | Error-free matched-token control not yet run. |
| Tiny-model self-verification | Yu et al., arXiv:2510.12157 (NeurIPS 2025) | Few-million-parameter transformers self-verify on multiplication/Sudoku; RL fits shallow patterns. | — |
| Curricula under budgets | Wu, Dyer & Neyshabur, arXiv:2012.03107 (ICLR 2021); Elgaar & Amiri, arXiv:2601.21698 | Curricula help mainly under tight budgets / ≤160M params. | Random-order pacing control not yet run. |
| Verifier foundations† | Cobbe et al. 2110.14168; Lightman et al. 2305.20050; Math-Shepherd 2312.08935; CriticGPT 2407.00215 | Outcome/process verifiers; critic models. | — |
| Wittgenstein and LLMs | Pérez-Escobar & Sarikaya, *Philosophy & Technology* 37(3), 2024; Molino & Tagliabue, arXiv:2302.01570 | Philosophical treatments; none ties §§143–145/185/202/258 to verifier training. | Framing is a design heuristic, not a claim. |

No reviewed paper combines: a sub-1B model, a balanced 50/50 verdict move, exact programmatic grading, matched training tokens, and own-vs-synthetic wrong proposals. That combination — a controlled comparison of known techniques — is the defensible contribution. See `docs/positioning.md`.

## Claim discipline

Any future mechanism proposed as novel must be accompanied by:

- a related-work search;
- the nearest known methods;
- a precise statement of the difference;
- an ablation demonstrating that difference matters;
- language describing it as a hypothesis until evidence exists.
