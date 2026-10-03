# Experiment 001 — Static vs adaptive arithmetic curricula

## Question

Can an adaptive executable curriculum teach a small language model arithmetic techniques with fewer training tokens than a fixed supervised dataset?

## Why arithmetic

Arithmetic gives us unlimited generated tasks, exact answers, cheap verification, controllable difficulty, and no judge-model ambiguity. If the idea cannot earn its keep here, adding philosophical upholstery will not rescue it.

## Regimes

All regimes use a fresh copy of the same base model, identical optimizer settings, the same total model-token budget, the same held-out evaluation sets, and the same training seed.

Every regime performs a pre-update generation and exact verification for each training example. Adaptive regimes use that observation to steer later sampling; the static regime discards it.

The harness records that work separately from training: `generation_seconds`,
`verifier_seconds`, `generated_characters`, and `training_core_seconds` (update
step only). It also records `repeated_examples`, since the static pool cycles while
online regimes draw fresh tasks, and `unused_token_budget`.

### A. Static SFT

A deterministic approximately balanced dataset is generated once, frozen, shuffled, and cycled.

### B. Adaptive curriculum

Tasks are generated online. Sampling weight rises for operation/difficulty buckets where recent pre-update accuracy is lower.

### C. Error-focused adaptive training

Regime B plus nearby arithmetic variants after failures. This is established hard-example/counterexample machinery used as an experimental condition, not a novelty claim.

## Leakage control

Training and evaluation operand combinations are assigned by a stable deterministic partition. One residue out of `holdout_modulus` is reserved for evaluation.

For commutative operations, operands are normalized before partitioning. Thus `2+3` and `3+2` always belong to the same side of the split.

The frozen static baseline, online adaptive generator, and error-focused variants are all constrained to the training partition.

## Budget accounting

The harness asks the active model adapter for the exact number of prompt + supervised-target tokens before each update. An example is trained only if the entire example fits inside the remaining budget. The adapter must report the same count after training or the run fails.

Checkpoint evaluation costs are recorded separately from the training-token budget and are identical in schedule and evaluation set across regimes. Accelerator timing is synchronized when the adapter supports it, peak CUDA memory is recorded when available, model-load and end-to-end regime wall time are separated, and checkpoint capability gain is normalized per million training tokens.

## Evaluation

The experiment uses four frozen evaluation splits:

1. **heldout** — unseen operand combinations, canonical training prompt form, trained digit range;
2. **prompt_transfer_only** — training-partition operand combinations rendered in prompt forms never used for training, isolating prompt-form transfer;
3. **withheld_prompts** — unseen operand combinations and unseen prompt forms together;
4. **out_of_range** — unseen operand combinations one digit range beyond training.

Generated dataset files also record logical-identity duplication statistics so small arithmetic buckets cannot silently masquerade as thousands of unique examples.

During training, a smaller frozen subset of the held-out split is evaluated at token checkpoints. This yields accuracy-vs-training-token curves and tokens-to-70/80/90/95%-accuracy measurements without exposing evaluation results to the curriculum.

## Default manifest

`manifest.json` currently specifies:

- model: `Qwen/Qwen2.5-0.5B`;
- 100,000 training tokens per regime;
- 10,000 frozen static examples;
- 1,000 held-out examples;
- 600 examples per generalization split;
- 180 checkpoint examples every 10,000 training tokens;
- training digits: 1–3;
- out-of-range digits: 4;
- withheld styles: question, words, expression;
- four nearby variants per observed failure;
- holdout modulus: 5.

## Run

```bash
pip install -e '.[train,test]'
pytest

# Cheap end-to-end adapter smoke test before spending a full run.
onwordly-arithmetic \
  --manifest experiments/001-arithmetic-curriculum/smoke-manifest.json \
  --output results/001-arithmetic-curriculum-smoke

# Full experiment.
onwordly-arithmetic
```

Outputs are written beneath `results/001-arithmetic-curriculum/`:

- `static-train.jsonl`;
- `evaluation-heldout.jsonl`;
- `evaluation-withheld-prompts.jsonl`;
- `evaluation-out-of-range.jsonl`;
- `static.json`;
- `adaptive.json`;
- `error-focused.json`;
- `summary.json`.

Each regime result contains training statistics, periodic checkpoints, tokens-to-threshold, and all three final evaluation splits.

## Next

The remaining critical path is empirical:

1. obtain compute;
2. run the real three-regime experiment;
3. repeat it across multiple seeds;
4. compare capability per training token **and** additional generation/verifier overhead;
5. only then decide whether a more elaborate mechanism deserves to exist.

For session continuity, see `docs/session-handoff.md`.
