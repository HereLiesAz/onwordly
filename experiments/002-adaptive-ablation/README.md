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
