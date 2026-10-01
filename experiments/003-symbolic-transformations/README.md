# Experiment 003 — Symbolic transformations

**Status: deterministic substrate and equal-token training comparison implemented; real-model run not yet launched.**

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
- exact transformation function and strict verifier;
- stable train/eval sequence partitioning;
- balanced frozen dataset generation and JSONL round-trip support;
- generalized equal-token training contract shared with arithmetic;
- adaptive and error-focused symbolic curricula;
- held-out and longer-sequence evaluation;
- full and smoke manifests;
- unit tests across generation, partitioning, curricula, sources, and experiment serialization.

## Run

```bash
pip install -e '.[train,test]'
pytest
onwordly-symbolic \
  --manifest experiments/003-symbolic-transformations/smoke-manifest.json \
  --output results/003-symbolic-transformations-smoke
```

## Next dependency chain

1. add a repeated-seed symbolic suite and report renderer;
2. add a composition split that chains two transformations;
3. run the smoke path, then one full seed;
4. only if sane, run repeated seeds and compare transfer of the arithmetic finding.

No symbolic result should be interpreted until repeated-seed controls exist.
