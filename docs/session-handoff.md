# Session handoff

This is the continuity document for a new session or contributor.

## Objective

Onwordly studies whether small language models can acquire useful techniques with substantially less static training data by replacing fixed corpora with executable curricula: generators that create tasks, observe behavior, verify outcomes, and adapt future training.

The project does **not** claim that active learning, hard-example mining, curriculum learning, counterexample refinement, process supervision, self-play, search, GRPO, LoRA, or distillation are new.

## Experiment 001

Experiment 001 compares:

1. **static** — frozen supervised examples;
2. **adaptive** — online generation weighted toward weaker operation/difficulty buckets;
3. **error-focused** — adaptive training plus nearby variants after failures.

All regimes receive the same training-token budget and the same pre-update generation/check step.

Leakage control, periodic held-out checkpoints, withheld-prompt evaluation, out-of-range evaluation, token-threshold measurements, and resource accounting are implemented.

## Kaggle execution

Experiment execution is centralized through `.github/workflows/kaggle-experiment.yml`.

The central workflow:

- uses the repository-level `KAGGLE_TOKEN`;
- submits a private Kaggle script;
- enables Internet access;
- requests `NvidiaTeslaT4`;
- polls the kernel to completion;
- downloads Kaggle outputs;
- preserves them as a GitHub Actions artifact.

The repository now owns the actual run intent in `.kaggle-run`. The central purpose profile calls `onwordly-kaggle`, so changing from a single run to the repeated-seed suite no longer requires editing the central controller.

Prepared run-plan combinations:

- Experiment 001 / single;
- Experiment 001 / suite;
- Experiment 002 / single.

Experiment 002 / suite is intentionally gated.

## Current live run

The original Kaggle run `36852362318` was cancelled after a later development push superseded it.

The current canonical Experiment 001 single run is:

- source commit: `4cc7a681354aa4d3817c6cc2e32867d8a975aa47`;
- central Kaggle run: `36853381019`;
- central Kaggle job: `110340375478`;
- accelerator: `NvidiaTeslaT4`;
- authentication, package build, and Kaggle submission: **successful**;
- current state: waiting for the Kaggle kernel.

The old GitHub-hosted CPU experiment is superseded.

## Experiment 001 repeated-seed suite

The suite is ready for Kaggle. To launch it after checking the first result, change `.kaggle-run` to:

```text
experiment: 001
mode: suite
backend: kaggle
accelerator: NvidiaTeslaT4
model: Qwen/Qwen2.5-0.5B
manifest: experiments/001-arithmetic-curriculum/manifest.json
seeds: 3303,4404,5505
```

The Kaggle runner will write:

- one result directory per seed;
- `aggregate.json`;
- `checkpoints.csv`;
- rendered `RESULTS.md`.

## Experiment 002 prepared

`experiments/002-adaptive-ablation/` is scaffolded but not yet promoted to a repeated real-model experiment.

Its five regimes isolate:

1. frozen static data;
2. online-uniform generation;
3. competence-responsive adaptive sampling;
4. error-focused variants with uniform sampling;
5. error-focused variants with adaptive sampling.

This separates the contribution of fresh online data, competence response, and local hard-example generation.

The underlying mechanisms remain established prior art; the experiment measures their contribution inside Onwordly rather than renaming them.

## Current queued/running experiments

Two real-model Kaggle jobs have now been launched instead of waiting serially:

- **Experiment 001 repeated-seed suite** — target commit `408f441600348607cf852adc45cce5642a6dba7f`, central run `36857400628`; seeds 3303, 4404, 5505.
- **Experiment 002 five-regime ablation** — target commit `7a382a4ffb9fab5ee4fe20842f7b862e7d366c85`, central run `36857267229`; Kaggle submission succeeded and the kernel is running.

The central Kaggle executor now gives each Kaggle revision a unique kernel slug and uses target-SHA concurrency isolation, so long experiment revisions do not cancel each other.

## Exact next actions

1. Let Experiment 001 suite and Experiment 002 ablation run independently.
2. Inspect Experiment 001 `aggregate.json`, `checkpoints.csv`, and each seed result.
3. Inspect Experiment 002 component differences without treating one seed as settled evidence.
4. If Experiment 001 is stable, repeat Experiment 002 across seeds.
5. Only then decide whether process supervision, search, or multi-agent language games deserve the next compute budget.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
