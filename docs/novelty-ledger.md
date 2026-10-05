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
| GRPO/PPO-family RL | Optimize policies from relative or scalar rewards. |
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

## Claim discipline

Any future mechanism proposed as novel must be accompanied by:

- a related-work search;
- the nearest known methods;
- a precise statement of the difference;
- an ablation demonstrating that difference matters;
- language describing it as a hypothesis until evidence exists.
