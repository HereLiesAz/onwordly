# Experiment 003 — Symbolic transformations

**Status: deterministic substrate implemented; training comparison not yet launched.**

## Purpose

Arithmetic is only one mechanically verifiable domain. This experiment begins the
next Phase 0 substrate so Onwordly can test whether any curriculum-efficiency
effect transfers beyond numerical calculation.

The first symbolic task family uses exact finite-string transformations:

1. reverse a symbol sequence;
2. sort symbols alphabetically;
3. collapse adjacent duplicate runs;
4. rotate the sequence left by one position.

Generation and verification are deterministic and require no judge model.

## Why this domain

It preserves the useful properties of arithmetic—unlimited synthetic tasks,
exact verification, controllable difficulty, and cheap failure analysis—while
changing the underlying technique being learned.

A result that only exists for arithmetic is substantially less interesting than
a result that survives a second exact domain.

## Implemented

- deterministic task generator;
- exact transformation function;
- strict answer verifier;
- balanced frozen dataset generation;
- JSONL round-trip support;
- unit tests across all operations and difficulty lengths.

## Next dependency chain

1. generalize the equal-token harness from arithmetic-specific task metadata to a
   minimal verifiable-task protocol;
2. add static, online-uniform, adaptive, and error-focused symbolic sources;
3. define train/eval sequence partitions that prevent exact-sequence leakage;
4. add in-range, longer-sequence, and composition evaluation splits;
5. run the same repeated-seed discipline used for arithmetic.

No symbolic result should be interpreted until those controls exist.
