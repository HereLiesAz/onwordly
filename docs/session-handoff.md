# Session handoff

This is the continuity document for a new session or contributor.

## Objective

Onwordly studies whether small language models can acquire useful techniques with substantially less static training data by replacing fixed corpora with executable curricula: generators that create tasks, observe behavior, verify outcomes, and adapt future training.

The project does **not** claim that active learning, hard-example mining, curriculum learning, counterexample refinement, process supervision, self-play, search, GRPO, LoRA, or distillation are new.

## Current experiment: 001

Experiment 001 compares three arithmetic regimes:

1. **static** — frozen supervised examples;
2. **adaptive** — online generation weighted toward weaker operation/difficulty buckets;
3. **error-focused** — adaptive training plus nearby variants after failures.

All three receive the same training-token budget and the same pre-update generation/check step.

## Implemented safeguards and measurements

### Leakage control

Operand combinations are deterministically partitioned into train/eval sets with a stable hash. Addition and multiplication normalize operand order **only for the partition key**, not for the generated prompt, so reversed commutative forms cannot cross the split without collapsing the training distribution. Adaptive generation and error-focused variants are constrained to the training partition.

### Evaluation

Each run records baseline and periodic held-out checkpoints, final held-out accuracy, withheld-prompt accuracy, out-of-range-digit accuracy, and first observed token count reaching 70%, 80%, 90%, and 95% checkpoint accuracy.

Checkpoint outcomes are never passed to the adaptive curriculum.

### Resource accounting

Each regime records exact training tokens, examples trained, training-time generation calls, verifier calls, core training seconds, evaluation generation calls, evaluation seconds, total generation calls, model parameter count, and device when exposed by the adapter.

### Repeated runs

`onwordly-arithmetic-suite` repeats the complete experiment across seeds and writes one result directory per seed, `aggregate.json`, and `checkpoints.csv`.

### Reporting

`onwordly-arithmetic-report` renders either a single-run `summary.json` or suite `aggregate.json` into a compact Markdown report. The canonical report target is:

`experiments/001-arithmetic-curriculum/RESULTS.md`

That file currently contains only instructions because no real model result exists yet.

## Current execution

The repository is registered with the central workflow controller. Normal CI is bound to shared `ci-validation.yml` and has passed on Python 3.10, 3.11, and 3.12.

Experiment 001 now runs through a curated `.github/workflows/kaggle-experiment.yml` binding. The central executor uses the existing repository-level `KAGGLE_TOKEN` secret, submits a private Kaggle script with Internet enabled, requests `NvidiaTeslaT4` GPU acceleration, polls the Kaggle kernel to completion, downloads its outputs, and preserves them as a GitHub Actions artifact.

The first Kaggle GPU run is **in progress**:

- trigger commit: `fde7b80723be0eb78992a2e217a804a76ca6c415`
- target tracker run: `36852342199`
- central Kaggle run: `36852362318`
- central Kaggle job: `110336796322`
- accelerator: `NvidiaTeslaT4`
- Kaggle submission step: **succeeded**
- current central step: waiting for the Kaggle kernel
- expected output: `results/001-arithmetic-curriculum/**`, including `summary.json` and rendered `RESULTS.md`

The earlier GitHub-hosted CPU experiment run `36830779628` is superseded. It is not the primary Experiment 001 result path.

Hugging Face Jobs remains unavailable because the connected account returned HTTP 402.

## Exact next actions

1. Wait for central Kaggle run `36852362318` to finish.
2. Inspect the downloaded Kaggle artifact and `summary.json`.
3. Render/check the single-run report in `RESULTS.md`.
4. If sane, move the repeated-seed suite to the same Kaggle GPU path and run seeds 3303, 4404, and 5505.
5. Inspect `checkpoints.csv` and every seed before interpreting the aggregate.
6. If an adaptive regime shows a repeatable advantage, design an ablation before adding PRMs, MCTS, or multi-agent language games.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
