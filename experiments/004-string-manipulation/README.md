# Experiment 004 — Constrained string manipulation

**Status: deterministic substrate and equal-token training comparison implemented; real-model run not yet launched.**

This is the third exact-verification domain in the Phase 0 substrate. It changes
the skill family again while retaining unlimited generation and mechanical
verification.

Initial operations:

1. remove vowels;
2. keep even-index characters;
3. reverse characters within each consecutive pair;
4. duplicate every character.

The generator supports stable train/eval partitioning, balanced difficulty
buckets, deterministic generation, and strict exact-answer verification.

## Implemented beyond the substrate

- adaptive and error-focused curricula;
- equal-token static/adaptive/error-focused runner;
- held-out and longer-string evaluation;
- full and smoke manifests;
- repeated-seed suite;
- Kaggle single/suite execution support.

## Next

1. add operation-composition evaluation;
2. run the smoke manifest;
3. run one full seed if the smoke path is clean;
4. only then launch repeated seeds.

No novelty claim is attached to these transformations or curricula.
