# Onwordly

**Onwordly** is a research workspace for making language-model training more data-efficient and compute-efficient, especially for small language models.

The central question is simple:

> How much static training data can be replaced by compact, executable teaching environments that generate tasks, test behavior, expose failures, and adapt what they teach?

The repository deliberately separates **established techniques** from **new hypotheses and combinations**. Active learning, hard-example mining, curriculum learning, self-play, process supervision, GRPO, search, and distillation are prior art. Onwordly is not a renaming ceremony for other people's inventions.

## Research direction

Onwordly explores executable curricula, exact/programmatic verification, adaptive sampling, counterexample repair, language games, process supervision, search-assisted trajectories, and distillation into small models.

## Experiment 001

The first controlled experiment compares:

1. frozen static supervised fine-tuning;
2. adaptive curriculum training;
3. adaptive error-focused training.

It now includes deterministic train/eval partitioning, periodic accuracy checkpoints, withheld prompt-form tests, out-of-range digit tests, measurement-overhead accounting, and repeated-seed aggregation.

See [experiments/001-arithmetic-curriculum](experiments/001-arithmetic-curriculum/README.md).

## Run one experiment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[train,test]'
pytest
onwordly-arithmetic
```

## Run the repeated-seed suite

```bash
onwordly-arithmetic-suite --seeds 3303 4404 5505
```

The suite writes one directory per seed plus `aggregate.json` and `checkpoints.csv`.

## Read first in a new session

Read `AGENTS.md`, then `docs/session-handoff.md`. Those documents define the research constraints, current state, blocker, and exact next actions.

## Repository map

- `AGENTS.md` — rules for future sessions and contributors.
- `docs/research-program.md` — research thesis and phases.
- `docs/novelty-ledger.md` — prior art versus actual hypotheses.
- `docs/session-handoff.md` — current state and next actions.
- `experiments/` — experiment specifications and manifests.
- `src/onwordly/datasets/` — frozen dataset generation.
- `src/onwordly/tasks/` — task generators and split logic.
- `src/onwordly/verifiers/` — exact/programmatic verification.
- `src/onwordly/curricula/` — adaptive curriculum logic.
- `src/onwordly/training/` — sources, equal-token harness, evaluation.
- `src/onwordly/models/` — model adapters.
- `src/onwordly/experiments/` — single-run and repeated-run orchestration.
- `tests/` — deterministic tests.

## Current blocker

The real Qwen 0.5B run still requires an available compute path. Connected Hugging Face Jobs currently returns HTTP 402 for both CPU and GPU jobs. The experiment code should not be redesigned merely to appease a billing page.
