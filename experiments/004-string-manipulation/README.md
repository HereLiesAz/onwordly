# Experiment 004 — Constrained string manipulation

**Status: deterministic substrate implemented; training comparison not yet integrated.**

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

## Next

1. add adaptive/uniform/error-focused curricula using the generic task contract;
2. add longer-string and operation-composition evaluation;
3. reuse the equal-token model runner;
4. repeat across seeds only after smoke validation.

No novelty claim is attached to these transformations or curricula.
