# Onwordly

**Onwordly** is a research workspace for making language-model training more data-efficient and compute-efficient, especially for small language models.

The central question is simple:

> How much static training data can be replaced by compact, executable teaching environments that generate tasks, test behavior, expose failures, and adapt what they teach?

The repository deliberately separates **established techniques** from **new hypotheses and combinations**. Active learning, hard-example mining, curriculum learning, self-play, process supervision, GRPO, search, and distillation are prior art. Onwordly is not a renaming ceremony for other people's inventions.

## Experiment 001

The first controlled experiment compares frozen static SFT, adaptive curriculum training, and adaptive error-focused training under the same training-token budget.

It includes deterministic train/eval partitioning, periodic checkpoints, a prompt-transfer-only split, a joint heldout+prompt split, out-of-range digit tests, duplicate-rate diagnostics, synchronized resource accounting, peak-memory capture, and repeated-seed aggregation.

The first real-model run is currently executing on Kaggle GPU through the central workflow controller.

See [experiments/001-arithmetic-curriculum](experiments/001-arithmetic-curriculum/README.md).

## Experiment 003 — symbolic substrate

The next exact-verification domain now has deterministic generation, exact
verification, stable train/eval partitioning, adaptive/error-focused curricula,
and an equal-token real-model runner. No empirical result is claimed yet.

See [experiments/003-symbolic-transformations](experiments/003-symbolic-transformations/README.md).

## Experiment 004 — constrained string substrate

A third exact-verification domain now has deterministic generation, stable
train/eval partitioning, strict verification, adaptive/error-focused curricula,
an equal-token runner, repeated-seed aggregation, and a Kaggle execution path.

See [experiments/004-string-manipulation](experiments/004-string-manipulation/README.md).

## Experiment 002 — prepared ablation

The next experiment is already scaffolded so we do not have to invent an explanation after seeing Experiment 001.

It separates:

1. frozen static data;
2. fresh online data sampled uniformly;
3. competence-responsive adaptive sampling;
4. failure-neighborhood examples without adaptive bucket weighting;
5. failure-neighborhood examples combined with adaptive weighting.

Experiment 002 is deliberately **gated** until Experiment 001 has a sane repeated-seed result worth explaining.

See [experiments/002-adaptive-ablation](experiments/002-adaptive-ablation/README.md).

## Local commands

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[train,test]'
pytest

# Experiment 001, one seed
onwordly-arithmetic

# Experiment 001, repeated seeds
onwordly-arithmetic-suite --seeds 3303 4404 5505

# Experiment 002 ablation
onwordly-arithmetic-ablation

# Experiment 003 symbolic transformations
onwordly-symbolic
```

## Kaggle execution

`.kaggle-run` is the repository-owned run plan consumed by the centralized Kaggle executor.

Supported prepared modes:

- Experiment 001 + `single`
- Experiment 001 + `suite`
- Experiment 002 + `single`

Experiment 002 + `suite` intentionally remains blocked until Experiment 001 justifies spending that compute.

## Read first in a new session

Read `AGENTS.md`, then `docs/session-handoff.md`.

## Repository map

- `AGENTS.md` — rules for future sessions and contributors.
- `docs/research-program.md` — research thesis and phases.
- `docs/novelty-ledger.md` — prior art versus actual hypotheses.
- `docs/session-handoff.md` — current state and next actions.
- `experiments/` — experiment specifications and manifests.
- `src/onwordly/datasets/` — frozen dataset generation.
- `src/onwordly/tasks/` — arithmetic and symbolic task generators and split logic.
- `src/onwordly/verifiers/` — exact/programmatic arithmetic and symbolic verification.
- `src/onwordly/curricula/` — uniform and adaptive curricula.
- `src/onwordly/training/` — sources, equal-token harness, evaluation.
- `src/onwordly/models/` — model adapters.
- `src/onwordly/experiments/` — single-run, suite, ablation, and Kaggle orchestration.
- `tests/` — deterministic tests.
