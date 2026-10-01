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

Experiment 001 compares three arithmetic-training regimes under the same model-token budget:

1. frozen static supervised fine-tuning;
2. adaptive curriculum training;
3. adaptive error-focused training.

Arithmetic is deliberately unglamorous: generation is unlimited and correctness is exact. That lets the training method fail without hiring a neural judge to explain away the corpse.

See [experiments/001-arithmetic-curriculum](experiments/001-arithmetic-curriculum/README.md).

## Run it

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[train,test]'
pytest
onwordly-arithmetic
```

The default manifest uses `Qwen/Qwen2.5-0.5B`. Edit `experiments/001-arithmetic-curriculum/manifest.json` to change the model, token budget, task distribution, or seeds.

Generated datasets and results are written beneath `results/` and intentionally ignored by git.

## Repository map

- `docs/research-program.md` — research thesis, principles, and phases.
- `docs/novelty-ledger.md` — prior art versus actual Onwordly hypotheses.
- `experiments/` — experiment specifications and manifests.
- `src/onwordly/datasets/` — frozen baseline dataset generation.
- `src/onwordly/tasks/` — task generators.
- `src/onwordly/verifiers/` — exact/programmatic verification.
- `src/onwordly/curricula/` — adaptive curriculum logic.
- `src/onwordly/training/` — task sources, equal-token harness, and evaluation.
- `src/onwordly/models/` — model-independent adapter interface and backends.
- `tests/` — deterministic tests for the experimental machinery.

## Status

Experiment 001 now has a complete first-pass pipeline: frozen baseline generation, adaptive and error-focused task sources, exact verification, a model adapter, strict token-budget accounting, evaluation, manifests, and JSON result serialization.

The next work is empirical: run the controlled experiment, inspect failure modes, and only then complicate the machinery.
