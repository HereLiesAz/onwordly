# Experiment matrix

This is the compact map of what each Onwordly experiment changes, what stays
fixed, and what evidence it is allowed to support.

| ID | Domain / question | Primary regimes | Exact evaluation | Execution state |
| --- | --- | --- | --- | --- |
| 001 | Arithmetic curriculum efficiency | static / adaptive / error-focused | held-out operands; prompt transfer only; joint operand+prompt transfer; longer digits | suite run-plan revision 14 launched 2026-10-01; outcome unrecorded; Actions since disabled |
| 002 | Arithmetic mechanism ablation | static / online-uniform / adaptive / error-focused-uniform / error-focused-adaptive | same arithmetic controls | single ablation run-plan revision 15 launched 2026-10-01 ahead of its 001 gate; outcome unrecorded; repeated suite gated |
| 003 | Symbolic transformations | static / adaptive / error-focused | held-out sequences; longer sequences; unseen two-step composition | runner, smoke/full manifests, suite and Kaggle path prepared |
| 004 | Constrained strings | static / adaptive / error-focused | held-out strings; longer strings; unseen two-operation composition | runner, smoke/full manifests, suite and Kaggle path prepared |
| 005 | Stateful program execution | static / adaptive / error-focused | held-out programs; longer programs; withheld adjacent instruction transition | runner, smoke/full manifests, suite and Kaggle path prepared |
| 006 | Formal logic | static / adaptive / error-focused | held-out formulas (partitioned by formula); deeper formulas; withheld parent→child operator composition | runner, smoke/full manifests, suite and Kaggle path prepared; dataset builders fixed 2026-10-03 |
| 007 | Exact process supervision | outcome-only / trace-supervised | final answer; exact machine-state trace | smoke/full ablation scaffold prepared; no real-model run; example-count confound unresolved |

## Cross-experiment invariants

For curriculum-efficiency experiments, keep these fixed unless the experiment
explicitly names the change:

- model family and starting checkpoint;
- optimizer and learning rate;
- exact training-token ceiling;
- random-seed set for repeated comparisons;
- frozen evaluation data;
- verifier semantics;
- checkpoint schedule;
- reporting of extra generation/verifier/evaluation cost.

Adaptive training never receives held-out outcomes.

## Evidence ladder

1. **Deterministic unit tests** establish generator/verifier mechanics.
2. **Real-model smoke** establishes that the adapter and serialization path work.
3. **One full seed** detects catastrophic runtime or measurement problems.
4. **Repeated seeds** are the minimum basis for discussing a stable training effect.
5. **Ablation** is required before attributing an effect to a particular component.
6. **Cross-domain replication** is required before describing a finding as a general
   executable-curriculum effect rather than a domain-specific result.

A null result is useful. Do not add expensive machinery merely to manufacture a
positive result.

## Compute gates

- Experiment 001 and Experiment 002 currently own the active real-model budget.
- Experiments 003–006 should run smoke-first after current code passes centralized CI. No smoke result is recorded yet.
- Experiment 007 should not receive full repeated-seed compute until its token-cost
  accounting and common-prompt comparison survive smoke validation.
- Search, learned process reward, self-play, and multi-agent language games remain
  later phases; none is justified merely because an earlier experiment fails.
