# Experiment 006 — Formal logic

**Status: deterministic substrate and equal-token training comparison implemented; real-model run not yet launched.**

This completes the initial Phase 0 list with a mechanically decidable formal
logic domain.

Tasks contain:

- a variable assignment;
- a generated propositional formula;
- Boolean AND, OR, XOR, and NOT;
- a required exact `true` or `false` answer.

Expression depth controls difficulty. The generator produces an AST, the local
evaluator computes the target, and a stable hash partitions formula/assignment
pairs into train and evaluation sets.

## Implemented

- recursive propositional-expression generator;
- exact Boolean evaluator;
- canonical formula rendering;
- strict answer verifier;
- stable train/eval partitioning;
- balanced frozen datasets;
- tests across operators, depth, verification, and partitioning.

## Implemented beyond the substrate

- adaptive and error-focused depth curricula;
- equal-token static/adaptive/error-focused runner;
- held-out and deeper-formula evaluation;
- full and smoke manifests;
- repeated-seed suite;
- Kaggle single/suite execution support.

## Next

1. add withheld operator-composition evaluation;
2. smoke test the runner;
3. run one full seed, then repeated seeds only if the path is clean.

No learned judge or chain-of-thought target is needed at this stage.
