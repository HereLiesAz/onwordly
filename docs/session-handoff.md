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
- Experiment 002 / single;
- Experiment 003 / single;
- Experiment 003 / suite;
- Experiment 004 / single;
- Experiment 004 / suite;
- Experiment 005 / single;
- Experiment 005 / suite;
- Experiment 006 / single;
- Experiment 006 / suite.

Experiment 002 / suite is intentionally gated.

## Current live runs

The old GitHub-hosted CPU experiment and earlier single-seed Kaggle revisions are superseded.

The current canonical arithmetic runs are the independently isolated Kaggle jobs listed below:

- Experiment 001 repeated-seed suite — central run `36857400628`;
- Experiment 002 five-regime ablation — central run `36857267229`.

Both are running on pinned target revisions, so continued development on `main` does not alter their code or cancel them.

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

Main now includes:

- arithmetic prompt-transfer isolation, duplicate diagnostics, synchronized timing, peak-memory capture, explicit cleanup, capability-gain-per-million-token reporting, and a smoke manifest;
- a generic trainable-task contract shared by exact domains;
- Experiment 003 symbolic transformations with stable partitions, adaptive/error-focused curricula, held-out, longer-sequence, and unseen two-step composition evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- Experiment 004 constrained string manipulation with exact verification, adaptive/error-focused curricula, held-out/longer-string evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- Experiment 005 simple program execution with an exact accumulator DSL interpreter, adaptive/error-focused curricula, held-out/longer-program evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- Experiment 006 formal logic with an exact propositional AST evaluator, adaptive/error-focused curricula, held-out/deeper-formula evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- experiment-specific Kaggle manifest defaults and report CLIs for all prepared domains.

The newer development commits are still queued for centralized CI behind the long-running experiment workload. Do not describe them as CI-validated until the latest queued validation completes.

## Exact next actions

1. Keep Experiment 001 suite and Experiment 002 ablation running independently; never rewrite their pinned target revisions.
2. Inspect and fix the newest centralized CI result as soon as it executes.
3. Smoke-test Experiments 003–006 before any full new-domain spend.
4. Add composition-specific evaluation to Experiments 004–006 where it measures a genuinely distinct generalization axis.
5. Use the exact program interpreter as the first process-supervision substrate: record intermediate accumulator states and compare outcome-only training against exact trace supervision without claiming process supervision as novel.
6. Only after cross-domain results exist decide whether search, distillation, or multi-agent language games deserve the next compute budget.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
