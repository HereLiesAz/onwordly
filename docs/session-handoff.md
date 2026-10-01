# Session handoff

This is the continuity document for a new session or contributor.

## Objective

Onwordly studies whether small language models can acquire useful techniques with substantially less static training data by replacing fixed corpora with executable curricula: generators that create tasks, observe behavior, verify outcomes, and adapt future training.

The project does **not** claim that active learning, hard-example mining, curriculum learning, counterexample refinement, process supervision, self-play, search, GRPO, LoRA, or distillation are new.

## Experiment 001

Experiment 001 compares:

1. **static** — frozen supervised examples;
2. **adaptive** — online generation weighted toward weaker operation/difficulty buckets;
3. **error-focused** — adaptive training plus nearby variants after failures.

All regimes receive the same training-token budget and the same pre-update generation/check step.

Leakage control, periodic held-out checkpoints, withheld-prompt evaluation, out-of-range evaluation, token-threshold measurements, and resource accounting are implemented.

## Kaggle execution

Experiment execution is centralized through `.github/workflows/kaggle-experiment.yml`.

The central workflow:

- uses the repository-level `KAGGLE_TOKEN`;
- submits a private Kaggle script;
- enables Internet access;
- requests `NvidiaTeslaT4`;
- polls the kernel to completion;
- downloads Kaggle outputs;
- preserves them as a GitHub Actions artifact.

The repository now owns the actual run intent in `.kaggle-run`. The central purpose profile calls `onwordly-kaggle`, so changing from a single run to the repeated-seed suite no longer requires editing the central controller.

Prepared run-plan combinations:

- Experiment 001 / single;
- Experiment 001 / suite;
- Experiment 002 / single;
- Experiment 003 / single;
- Experiment 003 / suite;
- Experiment 004 / single;
- Experiment 004 / suite;
- Experiment 005 / single;
- Experiment 005 / suite;
- Experiment 006 / single;
- Experiment 006 / suite.

Experiment 002 / suite is intentionally gated.

## Current Kaggle execution state

The runs previously described as the canonical live Kaggle jobs were not actually running new kernels. Their submit steps returned `Kernel push error: Maximum batch GPU session count of 2 reached.` The old workflow failed to treat that textual Kaggle CLI error as a submission failure and then polled inaccessible kernel refs. Those GitHub-side Kaggle watcher jobs have now been canceled.

Two earlier remote submissions to `azwashere/onwordly-experiment-001` did succeed: version 1 reached RUNNING after its 10:58 UTC submission, and version 2 reached RUNNING after its 11:08 UTC submission. The public Kaggle status interface does not expose the session IDs needed for supported API cancellation, so do not claim those remote Kaggle sessions themselves were canceled without separate confirmation.

Fresh transparent restarts are queued:

- Experiment 001 repeated-seed suite — target commit `d0401506fd75437851012c6b576fad447bc88b24`, tracker `36873051697`, run-plan revision 9.
- Experiment 002 five-regime ablation — target commit `ae7f715605aa9fca05f60624ef1fcf636d44abd7`, tracker `36873056937`, run-plan revision 10.

The central runner now requires a positive Kaggle push confirmation, treats textual push errors as failures, fails on inaccessible/not-found status responses, limits repeated unknown states, records kernel URL/ref/target SHA/accelerator/submission time, uploads control metadata before polling, and emits timestamped state transitions.

A queued GitHub tracker is not evidence of a live GPU experiment. Only a confirmed Kaggle push followed by QUEUED/RUNNING status counts.

## Work continuing while compute runs

Development is no longer waiting on the live Kaggle jobs.

Main now includes:

- arithmetic prompt-transfer isolation, duplicate diagnostics, synchronized timing, peak-memory capture, explicit cleanup, capability-gain-per-million-token reporting, and a smoke manifest;
- a generic trainable-task contract shared by exact domains;
- Experiment 003 symbolic transformations with stable partitions, adaptive/error-focused curricula, held-out, longer-sequence, and unseen two-step composition evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- Experiment 004 constrained string manipulation with exact verification, adaptive/error-focused curricula, held-out/longer-string evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- Experiment 005 simple program execution with an exact accumulator DSL interpreter, adaptive/error-focused curricula, held-out/longer-program evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- Experiment 006 formal logic with an exact propositional AST evaluator, adaptive/error-focused curricula, held-out/deeper-formula evaluation, full/smoke manifests, repeated-seed reporting, and Kaggle execution;
- experiment-specific Kaggle manifest defaults and report CLIs for all prepared domains;
- Experiment 007 exact program-state process supervision with a common prompt, outcome-only versus trace targets, exact final/trace verifiers, and full/smoke manifests.

The newer development commits are still queued for centralized CI behind the long-running experiment workload. Do not describe them as CI-validated until the latest queued validation completes.

## Exact next actions

1. Let transparent Experiment 001 run 9 and Experiment 002 run 10 dispatch through the fixed Kaggle runner; verify explicit successful-push output before calling either live.
2. If Kaggle still reports the two-GPU-session ceiling, preserve the fail-fast evidence and wait for the older Kaggle sessions to finish or stop them through Kaggle once their session IDs are available.
3. Inspect and fix the newest centralized CI result as soon as it executes.
4. Smoke-test Experiments 003–006 before any full new-domain spend.
5. Add composition-specific evaluation to Experiment 006; Experiments 003 and 004 already have composition splits, and Experiment 005 has a withheld-transition split.
6. CI-validate and smoke-test Experiment 007's common-prompt outcome-vs-trace ablation; do not give it repeated-seed compute until target-token/example accounting is inspected.
7. Only after cross-domain results exist decide whether search, distillation, or multi-agent language games deserve the next compute budget.

## Experiment map

See `docs/experiment-matrix.md` for the cross-domain comparison table, evidence ladder, and compute gates.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
