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

### Leakage control

Operand combinations are deterministically partitioned into train/eval sets with a stable hash. For addition and multiplication, operand order is normalized before partitioning, so `2+3` cannot be held out while `3+2` leaks into training.

Adaptive generation and error-focused variants are constrained to the training partition.

### Evaluation

Experiment 001 now records:

- a baseline checkpoint at 0 training tokens;
- periodic held-out checkpoints every configured token interval;
- a final held-out evaluation;
- a final withheld-prompt-form evaluation;
- a final out-of-range-digit evaluation;
- first observed token count reaching 70%, 80%, 90%, and 95% checkpoint accuracy.

Checkpoint evaluation uses a smaller frozen subset to keep measurement overhead tolerable. Checkpoint outcomes are never passed to the adaptive curriculum.

### Default configuration

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

A real model run has not yet been completed. Attempts to launch both CPU and GPU Hugging Face Jobs from the connected account returned HTTP **402 Payment Required**.

This is a compute-access/billing blocker, not a research-design reason to change the experiment.

## Exact next actions

1. Run the repository tests in an environment with Python 3.10+.
2. Obtain an available compute path for the 0.5B model.
3. Run:
   ```bash
   pip install -e '.[train,test]'
   pytest
   onwordly-arithmetic
   ```
4. Preserve the generated `results/001-arithmetic-curriculum/summary.json`.
5. Compare the three regimes on:
   - held-out accuracy at matched training tokens;
   - tokens-to-threshold;
   - withheld-prompt accuracy;
   - out-of-range-digit accuracy;
   - examples trained;
   - generation/verifier overhead.
6. Repeat across multiple seeds before treating any difference as evidence.
7. Only after the repeated baseline exists should the project add more expensive mechanisms such as process reward models, search, or multi-agent language games.

## Interpretation rule

A higher final score is not enough. The research question is whether an adaptive executable curriculum purchases more capability per constrained resource. Any gain that disappears after accounting for training tokens, additional generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
