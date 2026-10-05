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
- packages and submits a private Kaggle script;
- enables Internet access;
- requests `NvidiaTeslaT4`;
- registers a signed, run-scoped callback with the central Cloudflare Worker;
- exits after Kaggle confirms the submission;
- leaves the target commit status pending while compute runs;
- has the Kaggle process report success/failure directly to the Worker;
- lets the Worker finalize the target commit status and dispatch a short result-collection action.

No GitHub Actions runner polls Kaggle or remains alive for the duration of model training.

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

Run-plan history after revisions 9 (001 suite) and 10 (002 ablation):

- revision 14 — Experiment 001 repeated-seed suite (`061e26e`, 2026-10-01);
- revision 15 — Experiment 002 single ablation (`8df8b6d`, 2026-10-01), the current `.kaggle-run`;
- Onwordly Actions were then disabled (`4d75572`, `bb3bbc3`) and workflow bindings re-centralized (`5791e7f`, `0414be8`, `db7c0d6`).

No outcome for revisions 9, 10, 14 or 15 is recorded in this repository. None counts as a live or completed run until a confirmed Kaggle push and status are recorded here.

Experiment 002 was launched before Experiment 001 produced the repeated-seed result its own gate requires. Treat any 002 output as a mechanics check, not evidence, until 001 is interpreted.

In `single` mode the `seeds:` line is ignored; the manifest `training_seed` is used.

The central runner requires a positive Kaggle push confirmation, treats textual push errors as submission failures, records kernel URL/ref/target SHA/accelerator/submission time, uploads control metadata, and then exits. The submitted Kaggle process owns terminal reporting through the signed Worker callback; GitHub no longer polls kernel state.

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

None of this is CI-validated. The 2026-10-03 audit found the local suite red (7 failures: Experiment 006 dataset builders raised on every call; two stale tests). Those are fixed; the suite now passes locally (95 tests). No smoke or real-model run of Experiments 003–007 is recorded.

## 2026-10-03 audit changes

- Experiment 006: static builder used an undefined name; composition builder requested depth-1 compositions. Both fixed. Logic partition now hashes the formula alone, so held-out means an unseen formula (previously only an unseen assignment).
- Integer and trace verifiers accept canonical ASCII decimals only (no `+`, leading zeros, `-0`, underscores, non-ASCII digits).
- `parse_supervised_program_final` rejects arbitrary text before `FINAL=`; a `TRACE=` prefix must be well formed.
- Symbolic and logic verifiers remain case-insensitive by design (documented in code).
- Harness records `generation_seconds`, `verifier_seconds`, `generated_characters` separately; `training_core_seconds` is now the update step alone. It also records `unique_examples`, `repeated_examples` and `unused_token_budget` so static pool cycling and budget shortfall are visible. Arithmetic and symbolic suite aggregates report them.
- Hugging Face adapter generates from the same `prompt + "\n"` format used in training (previously mismatched) and requests deterministic kernels when seeded.

Open, not fixed:

- `.github/workflows/ci.yml` and `.github/workflows/experiment.yml` are local implementations, contrary to the centralization rule; `.experiment-run` only feeds `experiment.yml`. Remove them through the central controller.
- Training dependencies (`torch`, `transformers`) are unpinned; record exact versions with every real-model result.
- Manifest hyperparameters (token budget, learning rate, holdout modulus, evaluation sizes) have no recorded provenance. A shared 100k-token budget yields different example counts per domain; compare domains on examples and tokens, not budget alone.
- Experiment 007 confounds supervision density with example count (trace targets cost more tokens). A token-matched outcome-only arm cannot separate them; an example-matched outcome-only arm is needed before attributing an effect to process supervision.
- Checkpoint evaluation reuses the first `checkpoint_evaluation_size` items of the final held-out set; tokens-to-threshold is not independent of final accuracy.
- Experiments 001 and 002 share an identical manifest, so their static baselines are the same condition, not independent replications.

## 2026-10-03 pre-compute checks

CPU-only, no GPU spend. Scripts live in `scripts/`.

- **End-to-end smoke** (`scripts/smoke_all.py`): every experiment's smoke manifest through the real Hugging Face adapter with `sshleifer/tiny-gpt2` — 001–007 single and 001/003/004/005/006 suite — all 12 pass (torch 2.14.1+cpu, transformers 5.18.0). This validates adapter, token accounting, serialization, reports and run-plan dispatch; it says nothing about learning.
- **Dataset and budget audit** (`scripts/audit_datasets.py`, report in `docs/dataset-audit.md`): full manifests, real Qwen tokenizer, no model.
  - Zero item overlap between static training data and every held-out, longer, composition, withheld-prompt and withheld-transition split. The only overlap is 001's prompt-transfer-only split, by design.
  - The 100k budget buys about half the 10k static pool in 001/002 (~5.3k examples), so static data is not cycled. The "repeated examples" seen there are duplicates already inside the pool: the 1–2 digit item space is small (10k rows hold 6.3k distinct items; held-out holds 659 distinct in 1000 rows). Online regimes repeat at the same rate.
  - Examples bought per 100k tokens vary about 5× across domains (001 ≈ 5.3k, 003 ≈ 3.0k, 006 ≈ 2.0k, 004 ≈ 1.8k, 005 ≈ 1.5k, 007 ≈ 1.0–1.2k). Compare domains on examples and tokens, not budget alone.
  - 007 confound quantified: trace supervision buys 1041 examples versus 1225 for outcome-only (−15%).
  - 006 answers are ~52–56% `false`; a constant-`false` model scores about 0.55, so judge logic accuracy against that floor.
- **Untrained baseline** (`scripts/baseline_eval.py`, report in `docs/baseline.md`): Qwen2.5-0.5B, greedy, 50 sampled rows per split.
  - Accuracy is 0–2% on every split in every domain. The base model does not follow the "return only the answer" format: it explains, restates the program, or emits multiple-choice text, and the exact parsers reject it (e.g. `The answer is 2.` for answer `2`).
  - Consequence: early in training, pre-update correctness is ~0 for every regime, so adaptive and error-focused sources start with no competence signal; error-focused regimes will spawn variants after nearly every task. Early gains will mostly be format acquisition, shared by all regimes, not skill.
  - Before interpreting curriculum effects, consider either a short shared format warm-up (identical for all regimes, counted in the budget) or a diagnostic lenient score (answer present anywhere in the response; reported only, never used for training) to separate format from capability.
  - Lenient scores on the same samples (answer anywhere in the response): arithmetic held-out 0.66, withheld prompts 0.32, prompt-transfer 0.28, out-of-range 0.00; logic 0.44–0.66; program 0.08–0.16; symbolic 0.00–0.14; string 0.02–0.10.
  - Reading: for 1–2 digit arithmetic, format is most of the gap — the base model often computes the answer and buries it in prose. Logic lenient sits at the constant-answer floor (~0.55), so it shows no capability; responses that mention both `true` and `false` can also count. Symbolic, string and program tasks show a genuine skill gap, not just format. Out-of-range arithmetic at 0.00 is likely truncation at 32 new tokens while the model explains.
  - Implication: in 001/002 early accuracy gains will be dominated by format acquisition; a format warm-up (or few-shot evaluation) matters most there. 003–005 are cleaner tests of curriculum effects.
- **LoRA on the real model**: the static regime of a 002 smoke ran on CPU with Qwen2.5-0.5B + the `manifest-lora.json` adapter config: 8.8M trainable of 503M parameters (1.7%), 17 examples, mean loss 1.39. The remaining regimes were not run (CPU time); they share the same adapter path.

## LoRA arm (prepared, not run)

`HuggingFaceCausalLMAdapter` accepts an optional `lora` config (via `peft`) and trains only adapter weights; arithmetic manifests accept a `lora` field. `experiments/002-adaptive-ablation/manifest-lora.json` differs from the full fine-tuning manifest only in `lora` and `learning_rate` (2e-4, an unpiloted PEFT default). Same five regimes, same budgets. See the 002 README. Gated with the rest of 002.

## Format controls (prepared, not run)

- **Lenient diagnostic score.** Every evaluation now also reports `lenient_correct` / `lenient_accuracy`: the expected answer appears as a standalone token anywhere in the response. Reported only; training and adaptive sources still see exact verification alone. It is an upper bound on capability that ignores format, and restated prompts can inflate it. `scripts/baseline_eval.py` reports both.
- **Shared format warm-up.** Manifests for 001–006 accept `format_warmup_tokens` (default 0 = off). Before each regime the harness trains, in order, on the tail of the static training pool until that many tokens are used, with no generation, verification or source observation. Warm-up tokens count toward `token_budget`, so totals stay matched, and every regime gets the identical warm-up. Results record `warmup_tokens` and `warmup_examples`. Experiment 007 rejects it, since each arm's output format is the variable under test.
- No manifest enables warm-up yet. Choose a size (for example 5–10% of the budget) from a pilot that shows exact accuracy rising on a held-out sample, and record that pilot as its provenance.

## GPU baselines

Untrained baselines now run as a Kaggle plan (`experiment: baseline`, keys `models`, `chat_template` = no/yes/both, `per_split`) or via `notebooks/baseline_colab.ipynb`. Audit and baseline code moved into `onwordly.diagnostics` (`onwordly-audit`, `onwordly-baseline`); `scripts/` keeps thin wrappers. Next baseline to run: base vs `Qwen/Qwen2.5-0.5B-Instruct`, raw prompts vs chat template, 200 rows per split. Qwen's default chat template turns a 13-token arithmetic prompt into 42 tokens (it adds a system message), so chat-format training would buy far fewer examples per budget.

## GPU baseline result (2026-10-04)

See `docs/baseline-gpu.md`. Summary: exact accuracy is ~0 in every domain for both Qwen2.5-0.5B and -Instruct, raw or chat-templated, except instruct + chat on non-canonical arithmetic prompts (0.21–0.23). Base + chat template degenerates; instruct without its template is worse than base. Symbolic, string and program tasks are a genuine skill gap under every variant.

Decision input: switching to instruct + chat buys a small exact-score head start on arithmetic only, at ~3× prompt tokens per example (13 → 42 for a short arithmetic prompt), so roughly a third as many examples per budget. The format warm-up on the base model with raw prompts remains the cheaper way to get a competence signal. If instruct is tried, it needs its own arm and its own baseline, not a silent swap.

## Using the baseline (2026-10-04)

- Decision: run Experiment 001 as designed, with no format warm-up. The warm-up would inflate absolute scores by teaching the test format; format acquisition is instead left to each regime's own budget and measured, not removed. `format_warmup_tokens` stays available (default 0) but no manifest enables it.
- Every arithmetic `RESULTS.md` now has an "Against the untrained baseline" table: step-0 (untrained) exact and lenient versus final held-out exact and lenient. A gain in lenient without exact is format-neutral capability; a gain in exact alone is mostly format.
- Arithmetic runs resume at regime granularity: a regime whose `<regime>.json` exists for the same manifest and code (SHA-256 over the manifest plus the installed `onwordly` sources) is loaded instead of re-run; any mismatch raises before datasets are rewritten. Results are written atomically (`.partial` then rename). Mid-regime resume is not supported.
- `notebooks/experiment_colab.ipynb` runs the 001 smoke manifest, then the full manifest, on a Colab T4, writing to Google Drive so a dropped session resumes. Kaggle: `.kaggle-run` with `experiment: 001`, `mode: single`.

## Kaggle run 16 (Experiment 001, prepared)

- `.kaggle-run` revision 16: Experiment 001, single, full manifest, no warm-up. Merging this to main dispatches it through the central Kaggle workflow.
- Both GPUs: Kaggle's `NvidiaTeslaT4` machine is two T4s. `onwordly-kaggle` now runs 001/002 regimes in parallel, one process per GPU (`onwordly.experiments.multi_gpu`), then assembles the summary from the regime files. Three regimes on two GPUs ≈ two regime-lengths (~2–2.5 h instead of ~3–3.5 h). Falls back to one process when fewer than two GPUs are visible. Suite mode uses the batch queue below; 003–007 still use one GPU.
- Results: everything under `/kaggle/working/results` is the kernel's saved output on Kaggle, plus a `<experiment>.tar.gz` archive of it. The central finalizer downloads that output into a GitHub Actions artifact (30-day retention). Durable GitHub storage (a release asset) needs a `release_tag` and `release_assets_glob: kaggle-output/results/*.tar.gz` on the Onwordly Kaggle profile in `HereLiesAz/workflows` (`scripts/semantic_catalog.py`); that profile currently has none, and a fixed tag would collide on the next run because release assets are immutable.

## Experiment 001 first real result (2026-10-04)

Kaggle run 16, one seed: held-out exact static 80.1%, adaptive 80.4%, error-focused 75.1% (from 0%). Static ≈ adaptive; error-focused is worse on every split and slow to 70%. Single seed: no training-advantage claim. Details in `experiments/001-arithmetic-curriculum/RESULTS.md`.

Next compute, in order: (1) Experiment 001 suite over seeds 3303, 4404, 5505; (2) Experiment 002 single run to locate the error-focused penalty (uniform vs adaptive sampling with failure variants).

## Batch runs across both GPUs (2026-10-04)

`.kaggle-run` (or the Kaggle notebook) accepts `experiment: batch` with `jobs:` drawn from `001`, `001-suite`, `002` and `seeds:`. Every regime of every job becomes one unit; each free GPU takes the next unit (`onwordly.experiments.multi_gpu.run_units`), then each job's summary/aggregate and `RESULTS.md` are assembled from the regime files. `experiment: 001` with `mode: suite` now uses the same queue. `notebooks/experiment_kaggle.ipynb` defaults to `jobs: 001-suite,002` with seeds 3303, 4404, 5505: 14 regime runs, roughly 6 hours on T4 x2 (one regime ≈ 48 min). The central workflow's `kaggle_timeout_seconds` is 21600 (6 h), so run this batch from the notebook, or split it, rather than through `.kaggle-run`.

## Experiment 008 — corrective language game (prepared 2026-10-04)

Motivated by 001 run 16: error-focused training adds more moves of the arithmetic game but never the game of correcting an answer. 008 interleaves corrective moves ("…A previous answer was 272. If it is wrong, return the correct integer; if it is right, return it unchanged.") into the static stream, showing either the model's own wrong answer from the pre-update attempt (`corrective-own`) or a synthetic plausible error (`corrective-synthetic`), plus confirm moves (p = 0.25) after right answers. Static and error-focused are references. New held-out measures: correction, confirmation, two-pass self-correction. Corrective moves only start once the model emits parseable integers (in 001 that took < 10k tokens). Kaggle: `experiment: 008` (single) or batch `jobs: 008`. See `experiments/008-corrective-language-game/README.md`.

## Publishing results from Kaggle/Colab

`onwordly.publish.publish_results` (CLI `onwordly-publish`) pushes a finished results tree, minus `.jsonl` datasets and archives, to `results/kaggle/<stamp>/` on a new branch `kaggle-results/<stamp>`; it never writes to main. The Kaggle notebook calls it in a `finally`, so results upload the moment the run stops, even after a failure. It needs a Kaggle secret `GITHUB_TOKEN` (contents: write). Kaggle keeps `/kaggle/working/results` as the notebook's output regardless.

## Experiment 008 first result (2026-10-05, one seed)

Neither corrective arm learned to correct (two-pass self-correction never improves). `corrective-own` learned to ignore the shown answer (correction = confirmation = arithmetic ≈ 74.7%); `corrective-synthetic` learned to copy it (confirmation 88.6%, correction 4.6%). Untrained regimes are suggestible: shown their own correct answer, static breaks 114/500. Next design candidate: an explicit, exactly verifiable verdict move (right/wrong) balanced so copying and ignoring both score at chance. Details: `experiments/008-corrective-language-game/RESULTS.md`.

## Experiment 009 — verdict game (prepared 2026-10-05)

Answer to 008's result: the move is a judgement. Shown a proposal (right with p = 0.5, independent of the model's attempt), the model replies `right` or `wrong: <correct integer>`; copying scores 50%, ignoring cannot answer. Arms: verdict-synthetic, verdict-mixed (own wrong answers when available), plus static and 008's corrective-synthetic as controls. Evaluation adds balanced verdict accuracy and a self-check (answer → judge own answer → keep or repair). Harness verification now dispatches per task (`verify_task`), identical for existing task types. Kaggle: `experiment: 009`. See `experiments/009-verdict-language-game/README.md`.

## Exact next actions

1. Establish what happened to run-plan revisions 14 and 15 (Kaggle push confirmation and final status) and record it here; Actions are currently disabled.
2. Remove the local `ci.yml` / `experiment.yml` implementations through the central controller and get a central `ci-validation` run green.
3. Run the batch (`jobs: 001-suite,002`, seeds 3303, 4404, 5505) on Kaggle T4 x2 and record the aggregate; Experiment 001 run 16 (post-fix code, one seed) is the first valid result.
4. Smoke-test Experiments 003–006 before any full new-domain spend; verify Experiment 006's withheld-composition dataset and aggregate/report path in that smoke.
5. CI-validate and smoke-test Experiment 007's common-prompt outcome-vs-trace ablation; inspect its exact token-utilization, examples-per-budget, first-pass cost, and exposure ratios before enabling repeated seeds.
6. Only after cross-domain results exist decide whether search, distillation, or multi-agent language games deserve the next compute budget.

## Experiment map

See `docs/experiment-matrix.md` for the cross-domain comparison table, evidence ladder, and compute gates.

## Interpretation rule

A higher final score is not enough. The question is whether an adaptive executable curriculum purchases more capability per constrained resource. A gain that disappears after accounting for training tokens, extra generation, verification, or repeated runs is not a shortcut; it is a bill wearing novelty glasses.
