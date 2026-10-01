# Experiment 005 — Simple program execution

**Status: deterministic substrate implemented; training integration pending.**

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

## Next

1. add adaptive and error-focused curricula over instruction-family/program-length buckets;
2. add longer-program and unseen instruction-composition evaluation;
3. reuse the generic equal-token harness;
4. smoke test before any full model run.
