# Experiment 001 — Static vs adaptive arithmetic curricula

## Question

Can an adaptive executable curriculum teach a small language model arithmetic techniques with fewer training tokens than a fixed supervised dataset?

## Why arithmetic

Arithmetic gives us unlimited generated tasks, exact answers, cheap verification, controllable difficulty, clean held-out distributions, and no judge-model ambiguity.

If the training idea cannot earn its keep here, adding philosophical upholstery will not rescue it.

## Regimes

All regimes use a fresh copy of the same base model, the same optimizer settings, the same total model-token budget, the same held-out evaluation set, and the same training seed.

Every regime also performs a pre-update generation and exact verification for every training example. Adaptive regimes use that observation to steer later sampling; the static regime discards it. This keeps observation overhead structurally comparable.

### A. Static SFT

A deterministic approximately balanced dataset is generated once, frozen, shuffled, and cycled for training.

### B. Adaptive curriculum

Tasks are generated online. Sampling weight rises for operation/difficulty buckets where recent pre-update accuracy is lower.

### C. Error-focused adaptive training

Regime B plus a queue of nearby arithmetic variants after failed attempts. This is intentionally an application of established hard-example/counterexample methods, not a novelty claim.

## Initial task space

Operations:

- addition;
- subtraction;
- multiplication.

Difficulty is parameterized by operand digit count.

## Budget accounting

The harness asks the active model adapter for the exact number of prompt + supervised-target tokens before each update. An example is trained only if the entire example fits within the remaining budget. The adapter must report the same count after training or the run fails.

This guarantees that no regime silently exceeds the configured training-token budget.

## Primary metrics

- exact-answer held-out accuracy;
- accuracy by operation/difficulty bucket;
- training tokens consumed;
- examples trained;
- pre-update accuracy during training;
- verifier and generation calls;
- mean training loss.

Threshold metrics such as tokens-to-80%-accuracy require periodic evaluation checkpoints and are deliberately deferred until the first end-to-end run proves the harness behaves sensibly.

## Generalization splits

The first manifest evaluates unseen operand combinations from the same configured buckets. Follow-up manifests will add:

1. larger digit counts than training;
2. mixed-operation prompts;
3. prompt phrasings withheld from training;
4. symbolic and algorithmic transfer tasks.

## Default manifest

`manifest.json` currently specifies:

- model: `Qwen/Qwen2.5-0.5B`;
- training budget: 100,000 model tokens per regime;
- frozen static dataset: 10,000 examples;
- held-out evaluation: 1,000 examples;
- difficulty: 1–3 digit operands;
- four nearby variants per observed failure in the error-focused regime.

## Run

From the repository root:

```bash
pip install -e '.[train,test]'
pytest
onwordly-arithmetic
```

Outputs are written to `results/001-arithmetic-curriculum/`:

- `static-train.jsonl`;
- `evaluation.jsonl`;
- `static.json`;
- `adaptive.json`;
- `error-focused.json`;
- `summary.json`.

## Implementation status

Implemented:

- deterministic arithmetic task generation;
- deterministic frozen static dataset generation and JSONL round-tripping;
- exact final-answer verification;
- adaptive bucket statistics and weakness-weighted sampling;
- error-focused nearby-variant queue;
- model-independent adapter protocol;
- Hugging Face causal-LM adapter;
- strict equal-token training harness;
- fixed held-out evaluation runner;
- experiment manifest;
- per-regime and summary JSON serialization;
- unit tests that exercise the pipeline without downloading a model.

Next:

- run the first real three-regime experiment;
- inspect accuracy and token-efficiency curves;
- add periodic evaluation checkpoints;
- add withheld prompt forms and out-of-range digit generalization;
- decide from evidence whether the adaptive mechanisms deserve another ounce of machinery.
