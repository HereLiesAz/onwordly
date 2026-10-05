# Novelty ledger

This file exists to prevent accidental reinvention from acquiring a fake moustache and filing a patent.

## Established families

The following are **not claimed as Onwordly inventions**:

| Technique | Role here |
| --- | --- |
| Active learning | Select informative examples rather than sampling uniformly. |
| Hard-example mining | Concentrate training on examples the model currently fails. |
| Self-correction / learning from mistakes | Train a model to revise a shown or self-generated answer (self-refinement, RL-based self-correction). Archived 008 applied it; not an Onwordly invention. |
| Self-verification / critique-then-revise | Judge a proposed answer before repairing it (generative verifiers, critique-and-revise). Archived 009 applied it; not an Onwordly invention. |
| Curriculum learning | Change task difficulty or distribution over training. |
| Counterexample-guided refinement | Use failures/counterexamples to improve a learner. |
| Self-play | Generate learning pressure through interaction among agents or policies. |
| Model-design components | Iterative refinement / recurrent-depth networks, Beta reputation, Dawid–Skene reliability, event sourcing, precision-weighted error: see `docs/model-design.md` (Borrow map, Prior art). |
| Synthetic data generation | Programmatically produce additional training examples. |
| Process supervision | Evaluate intermediate reasoning or execution states. |
| Outcome supervision | Evaluate final task success. |
| GRPO/PPO-family RL | Optimize policies from relative or scalar rewards (GRPO: Shao et al., arXiv:2402.03300). |
| REINFORCE with a baseline | Policy gradient weighting sampled completions by reward minus a baseline (Williams 1992). Experiment 000 trains its hold/change decision this way with a batch-mean baseline; not an Onwordly invention. |
| Graded / shaped rewards, improvement bonuses | Partial credit and bonuses for improving an answer (reward shaping, Ng et al. 1999; SCoRe stage II, arXiv:2409.12917). Experiment 000 uses the graded challenge reward carried over from archived 010/011; not an Onwordly invention. |
| Answer-before-verdict verification | The verifier solves before judging (generative verifiers with CoT, e.g. GenRM arXiv:2408.15240). Not an Onwordly invention. |
| Generation-verification gap / verification asymmetry | Checking a candidate can be easier than producing one, and self-improvement by self-verification depends on that gap (Song et al., "Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models", arXiv:2412.02674, verified 2026-10-05; for constraint problems, Stechly, Marquez & Kambhampati, "GPT-4 Doesn't Know It's Wrong", arXiv:2310.12397, verified 2026-10-05, on graph colouring with self-critique vs an external verifier). Experiment 000 uses the constrained-string task, which has the gap; not an Onwordly invention. The asymmetry framing also echoes the general NP-style "easy to check, hard to solve" intuition, which is folklore, not a citation. |
| Sycophancy / robustness to user challenge | Models flip correct answers when a user pushes back, and this can be measured and trained against (FlipFlop, Laban et al., arXiv:2311.08596; Sharma et al., "Towards Understanding Sycophancy in Language Models", arXiv:2310.13548). Experiment 000's fallible correctors and hold/change reply follow this; not an Onwordly invention. Onwordly's part, if any, is only the framing as earned trust under exact verification, with hold discrimination as the measure. |
| MCTS/search | Explore candidate trajectories before selecting or distilling them. |
| Knowledge/trajectory distillation | Transfer expensive teacher/search behavior into a cheaper model. |

## Onwordly hypotheses

Research questions, not claims of invention (details in `experiments/000-onwordly-learner/README.md`):

1. A small network with a revising workspace, add-only memory and earned trust solves held-out exact problems at least as well as a parameter-matched plain network.
2. It learns to hold right answers against wrong correctors and change wrong ones, with hold rate tracking each corrector's earned reliability.
3. Scaling updates by confidence × surprise helps (ablated against flat updates).

## Archived findings

Nearest prior work for the archived LLM experiments (SCoRe, sycophancy, GenRM, verification asymmetry, intrinsic self-correction and others) and what 008/009 found against it: `docs/findings-llm-phase.md`, and the full table and review on branch `archive/llm-phase` (`docs/novelty-ledger.md`, `reports/Self correction and verifier training.md`).

## Claim discipline

Any future mechanism proposed as novel must be accompanied by:

- a related-work search;
- the nearest known methods;
- a precise statement of the difference;
- an ablation demonstrating that difference matters;
- language describing it as a hypothesis until evidence exists.
