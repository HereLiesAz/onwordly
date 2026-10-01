# Experiment 005 — Simple program execution

**Status: deterministic substrate and equal-token training comparison implemented; real-model run not yet launched.**

This Phase 0 domain tests a different kind of exact technique: executing a tiny
stateful program rather than transforming one expression or string.

The DSL has one integer accumulator and five instructions:

- `SET n`
- `ADD n`
- `SUB n`
- `MUL n`
- `NEG`

Programs are interpreted by Onwordly itself, so targets and evaluation require no
judge model. Program length controls difficulty, and a stable hash partitions
whole programs into train/eval sets.

## Implemented

- deterministic program generator;
- exact interpreter;
- strict integer verifier;
- balanced frozen datasets;
- stable train/eval partitioning;
- unit tests for execution, generation, partitioning, and dataset balance.

## Implemented beyond the substrate

- adaptive and error-focused curricula over program length;
- equal-token static/adaptive/error-focused runner;
- held-out and longer-program evaluation;
- full and smoke manifests;
- repeated-seed suite;
- Kaggle single/suite execution support.

## Next

1. add unseen instruction-composition evaluation;
2. smoke test the runner;
3. run one full seed before spending on repeated seeds.
