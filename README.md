# Onwordly

**Onwordly** is a research workspace for making language-model training more data-efficient and compute-efficient, especially for small language models.

The central question is simple:

> How much static training data can be replaced by compact, executable teaching environments that generate tasks, test behavior, expose failures, and adapt what they teach?

The repository deliberately separates **established techniques** from **new hypotheses and combinations**. Active learning, hard-example mining, curriculum learning, self-play, process supervision, GRPO, search, and distillation are prior art. Onwordly is not a renaming ceremony for other people's inventions.

## Research direction

Onwordly explores:

- executable curricula instead of fixed corpora;
- automatic task generation with exact or programmatic verification;
- adaptive sampling concentrated on unresolved skills;
- counterexample generation and repair;
- language games and multi-agent training;
- outcome and process supervision;
- search-assisted trajectory generation;
- distillation of expensive training-time search into small models;
- measurements of capability gained per training token, example, FLOP, and parameter.

## First experiment

The first controlled experiment uses arithmetic because generation is unlimited and verification is exact.

Three regimes will be compared under equal training-token budgets:

1. static supervised fine-tuning;
2. adaptive curriculum training;
3. adaptive error/counterexample-focused training.

The point is not to invent arithmetic. The point is to measure whether generated, responsive curricula can buy more capability per token than static examples.

See [experiments/001-arithmetic-curriculum](experiments/001-arithmetic-curriculum/README.md).

## Repository map

- `docs/research-program.md` — research thesis, principles, and phases.
- `docs/novelty-ledger.md` — what is established, what is merely combined here, and what is actually being tested.
- `experiments/` — controlled experiments and results.
- `src/onwordly/` — reusable task generators, verifiers, curricula, and training components.
- `tests/` — deterministic tests for the machinery.

## Status

The repository has been initialized. Experiment 001 now has a working arithmetic task generator, exact verifier, and adaptive curriculum scheduler. Model training integration is next.
