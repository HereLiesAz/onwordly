# Experiment 001 results

**Status:** awaiting first real model run.

Do not manually invent values in this file. Generate the first report from the actual result JSON:

```bash
onwordly-arithmetic-report \
  results/001-arithmetic-curriculum/summary.json \
  --output experiments/001-arithmetic-curriculum/RESULTS.md
```

After the repeated-seed suite:

```bash
onwordly-arithmetic-report \
  results/001-arithmetic-curriculum-suite/aggregate.json \
  --output experiments/001-arithmetic-curriculum/RESULTS.md
```

The repeated-seed report should replace the single-run report once available.

Before drawing conclusions, inspect `checkpoints.csv` and the individual seed directories for inconsistent or seed-specific behavior.
