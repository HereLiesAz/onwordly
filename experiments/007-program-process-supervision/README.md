# Experiment 007 — Exact process supervision for program execution

**Status: exact trace substrate implemented; comparative training experiment not yet launched.**

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
- tests for alignment, missing steps, formatting, and final state.

## Before launching

The training ablation still needs to hold the comparison clean:

1. define outcome-only vs trace-supervised regimes with explicit token accounting;
2. ensure both evaluate on the same final-answer and trace sets;
3. report the extra supervised target tokens and generation cost separately;
4. add an ablation that distinguishes benefit from merely exposing more target tokens;
5. smoke-test before any full run.

No chain-of-thought data is required; the trace is the executable machine state,
not an unconstrained natural-language rationale.
