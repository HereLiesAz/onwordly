# Experiment 006 — Formal logic

**Status: deterministic substrate implemented; training integration pending.**

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

## Next

1. integrate the generic equal-token curriculum harness;
2. evaluate deeper formulas and withheld operator compositions;
3. add repeated-seed execution only after smoke validation.

No learned judge or chain-of-thought target is needed at this stage.
