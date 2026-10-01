# Experiment 001 — Static vs adaptive arithmetic curricula

## Question

Can an adaptive executable curriculum teach a small language model arithmetic techniques with fewer training tokens than a fixed supervised dataset?

## Why arithmetic

Arithmetic is intentionally boring.

That gives us:

- unlimited generated tasks;
- exact answers;
- cheap verification;
- controllable difficulty;
- clean held-out distributions;
- no judge-model ambiguity.

If the training idea cannot earn its keep here, adding philosophical upholstery will not rescue it.

## Regimes

All regimes use the same base model, optimizer family, total training-token budget, evaluation prompts, decoding settings, and random-seed set.

### A. Static SFT

Generate a fixed dataset before training and sample from it normally.

### B. Adaptive curriculum

Generate tasks online. Sampling weight rises for operation/difficulty buckets where recent accuracy is low.

### C. Error-focused adaptive training

Start with regime B, then generate targeted variants around observed failures. This regime intentionally uses established hard-example/counterexample ideas; it is an experimental condition, not a novelty claim.

## Initial task space

Operations:

- addition;
- subtraction;
- multiplication.

Difficulty is parameterized by operand digit count.

## Primary metrics

- exact-answer held-out accuracy;
- accuracy by operation/difficulty bucket;
- training tokens to reach 70%, 80%, 90%, and 95% held-out accuracy;
- held-out accuracy at fixed token budgets;
- examples consumed;
- verifier calls.

## Generalization splits

Evaluation should include:

1. unseen operand combinations within trained digit ranges;
2. larger digit counts than training;
3. mixed-operation prompts;
4. prompt phrasings withheld from training.

## Success criterion

The adaptive regime is interesting only if it produces a reproducible improvement in capability per training token over static SFT. A larger final score obtained by quietly spending more compute does not count.

## Current implementation

Implemented:

- deterministic arithmetic task generation;
- exact final-answer verification;
- adaptive bucket statistics;
- weighted sampling toward weaker buckets.

Next:

- frozen static-dataset generator;
- model adapter;
- equal-token training harness;
- evaluation runner;
- experiment manifest and result serialization.
