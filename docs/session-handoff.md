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

## Current blocker

The repository is now registered with the central workflow controller and `.github/workflows/ci.yml` is actively bound to the shared `ci-validation.yml` executor. The bootstrap Python CI run passed before the controller replaced the source with its tracker.

A real model run has not yet been completed. Attempts to launch both CPU and GPU Hugging Face Jobs from the connected account returned HTTP **402 Payment Required**. The connected environment therefore cannot currently execute the Qwen experiment.

## Exact next actions

1. Confirm the centralized tracker successfully follows a shared `ci-validation` run on a normal push.
2. Obtain compute for `Qwen/Qwen2.5-0.5B`.
3. Run one seed with `onwordly-arithmetic`.
4. Render and inspect the single-run report.
5. If sane, run `onwordly-arithmetic-suite --seeds 3303 4404 5505`.
6. Render the suite report into `experiments/001-arithmetic-curriculum/RESULTS.md`.
7. Inspect `checkpoints.csv` and every seed before interpreting the aggregate.
8. If an adaptive regime shows a repeatable advantage, design an ablation before adding PRMs, MCTS, or multi-agent language games.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
