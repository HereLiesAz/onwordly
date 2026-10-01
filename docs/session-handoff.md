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

Operand combinations are deterministically partitioned into train/eval sets with a stable hash. Addition and multiplication normalize operand order before partitioning, so reversed commutative forms cannot cross the split. Adaptive generation and error-focused variants are constrained to the training partition.

### Evaluation

Each run records:

- a baseline checkpoint at 0 training tokens;
- periodic held-out checkpoints;
- a final held-out evaluation;
- a final withheld-prompt-form evaluation;
- a final out-of-range-digit evaluation;
- first observed token count reaching 70%, 80%, 90%, and 95% checkpoint accuracy.

Checkpoint outcomes are never passed to the adaptive curriculum.

### Resource accounting

Each regime records:

- exact training tokens;
- examples trained;
- training-time generation calls;
- verifier calls;
- core training seconds;
- checkpoint and final evaluation generation calls;
- checkpoint and final evaluation seconds;
- total generation calls including measurement overhead;
- model parameter count and device when exposed by the adapter.

### Repeated runs

`onwordly-arithmetic-suite` repeats the complete three-regime experiment across seeds and writes:

- one result directory per seed;
- `aggregate.json` with mean/stddev/min/max summaries;
- `checkpoints.csv` for accuracy-versus-training-token curves.

The default suite seeds are 3303, 4404, and 5505.

## Default configuration

See `experiments/001-arithmetic-curriculum/manifest.json`.

Current defaults:

- model: `Qwen/Qwen2.5-0.5B`;
- 100,000 training tokens per regime;
- 10,000 frozen static examples;
- 1,000 held-out examples;
- 600 examples per generalization split;
- checkpoint every 10,000 training tokens on 180 held-out examples;
- training digits: 1–3;
- out-of-range digits: 4;
- withheld prompt forms: question, words, expression;
- holdout partition: one stable residue out of five.

## Current blocker

A real model run has not yet been completed. Attempts to launch both CPU and GPU Hugging Face Jobs from the connected account returned HTTP **402 Payment Required**. The connected environment therefore cannot currently execute the Qwen experiment.

A local container test attempt also could not clone GitHub because that container has no outbound DNS. Do not report the new code as externally executed until CI or another compute environment actually runs it.

## Exact next actions

1. Let centralized `ci-validation` run once the workflow controller provisions it, or run `pytest` in any Python 3.10+ environment.
2. Obtain compute for `Qwen/Qwen2.5-0.5B`.
3. Run a single seed first:
   ```bash
   pip install -e '.[train,test]'
   pytest
   onwordly-arithmetic
   ```
4. Inspect `results/001-arithmetic-curriculum/summary.json` for implementation mistakes or pathological behavior.
5. If the single run is sane, run:
   ```bash
   onwordly-arithmetic-suite --seeds 3303 4404 5505
   ```
6. Compare regimes using both accuracy and resource cost. Do not select a conclusion from one seed.
7. If an adaptive regime shows a repeatable advantage, design the next ablation before adding PRMs, MCTS, or multi-agent language games.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
