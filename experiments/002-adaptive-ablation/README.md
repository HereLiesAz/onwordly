# Experiment 002 — Adaptive mechanism ablation

**Status: prepared, not yet authorized to interpret or run as evidence.**

Experiment 002 exists so that a positive Experiment 001 result cannot immediately grow a mythology around itself.

## Question

If adaptive training beats the frozen baseline, which part actually bought the improvement?

## Regimes

The ablation keeps the same model, optimizer, train/eval partition, token budget, checkpoint schedule, and evaluation sets while separating mechanisms:

1. **static** — frozen supervised dataset;
2. **online-uniform** — generate new training tasks online, but sample operation/difficulty buckets uniformly and ignore competence;
3. **adaptive** — online generation plus competence-responsive bucket weighting;
4. **error-focused-uniform** — online uniform generation plus nearby variants after a failed base task;
5. **error-focused-adaptive** — competence-responsive sampling plus nearby failure variants.

This isolates three questions:

- Does merely generating fresh examples online matter?
- Does competence-responsive bucket selection add value beyond online generation?
- Do local failure variants add value independently of adaptive bucket selection?

## Full fine-tuning versus LoRA

`manifest-lora.json` is identical to `manifest.json` except for the update method:
a LoRA adapter (rank 16, alpha 32, no dropout, all attention and MLP projections)
on a frozen base, with learning rate 2e-4 instead of 2e-5. Running all five regimes
under both manifests asks whether any regime effect survives a change of update
method. The two manifests differ in two linked fields, so a difference between
them is attributed to "LoRA at its usual settings", not to rank or learning rate
alone.

- LoRA is prior art; it is a factor under test here, not a contribution.
- The rank, alpha and learning rate are common PEFT defaults, not tuned here.
  Record any pilot that changes them.
- Token budgets are identical; LoRA buys no extra tokens. It changes memory,
  speed and adapter size, which the summary records (`trainable_parameter_count`,
  peak memory, wall time).
- LoRA tends to disturb the base model less, which can raise out-of-range and
  transfer scores independently of curriculum. Compare regimes within a method
  first, then compare methods.

Run it by pointing `.kaggle-run` at `experiments/002-adaptive-ablation/manifest-lora.json`.

## Claim discipline

Nothing in these mechanisms is claimed as novel. The experiment measures contribution of established components inside the Onwordly training loop.

Do not run increasingly elaborate machinery merely because it exists. Experiment 002 should be promoted to a repeated real-model experiment only after Experiment 001 has a sane repeated-seed result worth explaining.

## Run locally

```bash
pip install -e '.[train,test]'
pytest
onwordly-arithmetic-ablation
```

Default output:

`results/002-adaptive-ablation/`
