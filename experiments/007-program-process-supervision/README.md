# Experiment 007 — Exact process supervision for program execution

**Status: exact trace substrate and outcome-vs-trace training ablation implemented; real-model run not yet launched.**

This experiment uses the existing accumulator DSL because every intermediate
state is mechanically decidable. That makes it possible to test process
supervision without a learned process-reward model or subjective reasoning judge.

For a program such as:

```text
SET 3; ADD 4; MUL 2; NEG; SUB 1
```

the exact state trace is:

```text
3,7,14,-14,-15
```

## What this is testing

The eventual comparison should ask whether exact intermediate-state supervision
changes capability-per-token relative to outcome-only supervision on the same
program family.

Process supervision itself is established prior art. Onwordly is not claiming
the idea; this experiment is a controlled test of when exact process targets
are worth their extra target tokens and training cost.

## Implemented

- exact `execute_program_trace` interpreter;
- deterministic step-aligned trace target;
- strict comma-separated trace parser/verifier;
- trace task contract compatible with the generalized training harness;
- tests for alignment, missing steps, formatting, and final state;
- exact training-token utilization and examples-per-budget accounting;
- first-pass token-cost profiles for both supervision targets;
- explicit trace/outcome exposure ratios;
- Markdown result reporting;
- Kaggle single-run execution support, with repeated-seed mode deliberately gated.

## Prepared comparison

Both regimes see the **same prompt and same frozen program sequence**. The only
training-target difference is supervision density:

- outcome-only: `FINAL=-15`;
- trace-supervised: `TRACE=3,7,14,-14,-15;FINAL=-15`.

The existing equal-token harness counts the longer trace target against the same
model-token ceiling, so trace supervision necessarily purchases fewer training
examples when it costs more tokens. Both regimes are evaluated on final-answer
accuracy and exact-trace accuracy.

This confounds supervision density with example count. Exposure ratios measure
the confound; they do not control it. Before attributing any effect to process
supervision, add an example-matched outcome-only arm (same programs and count as
the trace arm, fewer tokens) alongside the token-matched comparison.

A full and smoke manifest are present. Before any full run:

1. let CI validate the scaffold;
2. run the smoke manifest;
3. inspect actual target-token/example counts and the trace/outcome exposure ratio;
4. render the result report;
5. add a repeated-seed suite only if the comparison behaves sanely.

No chain-of-thought data is required; the trace is the executable machine state,
not an unconstrained natural-language rationale.
