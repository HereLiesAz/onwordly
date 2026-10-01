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

## Work continuing while compute runs

Development is no longer waiting on the live Kaggle jobs.

Since the suite/ablation launches, the codebase has also gained:

- a prompt-transfer-only arithmetic evaluation split, separate from the harder joint heldout+prompt split;
- logical duplicate-rate diagnostics for generated arithmetic datasets;
- accelerator-synchronized timing, end-to-end regime wall time, peak CUDA memory, explicit adapter cleanup, and capability-gain-per-million-token measurement;
- a cheap Experiment 001 real-model smoke manifest;
- a generic trainable-task contract so the equal-token harness is no longer arithmetic-only;
- deterministic symbolic transformation generation and exact verification;
- stable symbolic train/eval partitioning;
- adaptive, uniform, and error-focused symbolic curricula;
- Experiment 003 full and smoke manifests;
- held-out, longer-sequence, and unseen two-step composition evaluation for Experiment 003;
- repeated-seed symbolic aggregation, Markdown reporting, and Kaggle single/suite execution support.

These changes are on commits from `cb4d4f38` through `f9e0e5e9`. Their centralized CI runs are queued behind the long-running Kaggle jobs, so they must not be described as CI-validated until those runs actually execute.

## Exact next actions

1. Keep Experiment 001 suite and Experiment 002 ablation running independently; do not gate development on either result.
2. Let the queued centralized CI runs validate the new measurement and symbolic-domain code.
3. If CI finds defects, fix those without altering the scientific comparison.
4. Run the Experiment 003 smoke path, then one full symbolic seed.
5. If the symbolic run is sane, launch the 3303/4404/5505 symbolic suite.
6. Add the next deterministic Phase 0 domain: constrained string manipulation, then simple program execution, then formal logic.
7. Only after cross-domain evidence exists decide whether process supervision, search, or multi-agent language games deserve additional compute.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
