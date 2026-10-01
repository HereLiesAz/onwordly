# Onwordly

**Onwordly** is a research workspace for making language-model training more data-efficient and compute-efficient, especially for small language models.

The central question is simple:

> How much static training data can be replaced by compact, executable teaching environments that generate tasks, test behavior, expose failures, and adapt what they teach?

The repository deliberately separates **established techniques** from **new hypotheses and combinations**. Active learning, hard-example mining, curriculum learning, self-play, process supervision, GRPO, search, and distillation are prior art. Onwordly is not a renaming ceremony for other people's inventions.

## Experiment 001

The first controlled experiment compares frozen static SFT, adaptive curriculum training, and adaptive error-focused training under the same training-token budget.

It includes deterministic train/eval partitioning, periodic checkpoints, a prompt-transfer-only split, a joint heldout+prompt split, out-of-range digit tests, duplicate-rate diagnostics, synchronized resource accounting, peak-memory capture, and repeated-seed aggregation.

The Kaggle execution path now distinguishes a confirmed Kaggle submission from a GitHub-side watcher. Experiment 001 run 9 and Experiment 002 run 10 are queued for fresh dispatch; neither is considered live until Kaggle confirms a successful kernel push and reports QUEUED or RUNNING.

See [experiments/001-arithmetic-curriculum](experiments/001-arithmetic-curriculum/README.md).

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

## Experiment 005 — simple program execution substrate

A tiny accumulator-machine DSL now provides another exact-verification domain,
with deterministic execution, adaptive/error-focused curricula, an equal-token
runner, longer-program evaluation, repeated-seed aggregation, and Kaggle support.

See [experiments/005-program-execution](experiments/005-program-execution/README.md).

## Experiment 006 — formal logic substrate

The initial deterministic Phase 0 domain list is now complete. Formal logic now
also has adaptive/error-focused curricula, an equal-token runner, deeper-formula
evaluation, repeated-seed aggregation, and Kaggle execution support.

See [experiments/006-formal-logic](experiments/006-formal-logic/README.md).

## Experiment 007 — exact process-supervision substrate

The program-execution domain now exposes exact accumulator traces for every
instruction, creating a mechanically verified process-supervision testbed. The
comparison is prepared as a research direction, not yet an empirical result.

See [experiments/007-program-process-supervision](experiments/007-program-process-supervision/README.md).

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

# Experiment 004 constrained strings
onwordly-string

# Experiment 005 program execution
onwordly-program

# Experiment 006 formal logic
onwordly-logic
```

## Kaggle execution

`.kaggle-run` is the repository-owned run plan consumed by the centralized Kaggle executor.

Supported prepared modes:

- Experiment 001 + `single` / `suite`
- Experiment 002 + `single`
- Experiment 003 + `single` / `suite`
- Experiment 004 + `single` / `suite`
- Experiment 005 + `single` / `suite`
- Experiment 006 + `single` / `suite`
- Experiment 007 + `single` only

Experiment 002 + `suite` remains gated until Experiment 001 is interpreted. Experiment 007 + `suite` remains gated until the single-run supervision-cost accounting is inspected.

## Read first in a new session

Read `AGENTS.md`, then `docs/session-handoff.md`.

## Repository map

- `AGENTS.md` — rules for future sessions and contributors.
- `docs/research-program.md` — research thesis and phases.
- `docs/novelty-ledger.md` — prior art versus actual hypotheses.
- `docs/experiment-matrix.md` — experiment states, invariants, evidence ladder, and compute gates.
- `docs/session-handoff.md` — current state and next actions.
- `experiments/` — experiment specifications and manifests.
- `src/onwordly/datasets/` — frozen dataset generation.
- `src/onwordly/tasks/` — arithmetic, symbolic, string, program, and logic task generators and split logic.
- `src/onwordly/verifiers/` — exact/programmatic verification across every Phase 0 domain.
- `src/onwordly/curricula/` — uniform and adaptive curricula.
- `src/onwordly/training/` — sources, equal-token harness, evaluation.
- `src/onwordly/models/` — model adapters.
- `src/onwordly/experiments/` — single-run, suite, ablation, and Kaggle orchestration.
- `tests/` — deterministic tests.
