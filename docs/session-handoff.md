# Session handoff

## Current state (2026-10-05)

- Clean slate. Onwordly is now the earned-trust / conscious-judgment / add-only-memory learner (`docs/model-design.md`). Work starts at Experiment 000.
- The LLM phase (Experiments 001–011, reports, notes, scripts) is archived on branch `archive/llm-phase` (commit `c35b185`). Findings: `docs/findings-llm-phase.md`.
- **Pending run:** Kaggle run 17 (verdict prior diagnostic, `experiment: prior`, Qwen2.5-0.5B and -Instruct, n = 200) was dispatched from main at `c35b185`. Results pending. That code exists only on the archive branch; record its outcome in `docs/findings-llm-phase.md` when it arrives.
- **Pending run:** Kaggle run 18 (Experiment 000 smoke manifest) dispatched from main at `b2c163d`. Results pending; inspect before the full run.
- Memory design in `docs/model-design.md` now covers consolidation-as-rewrite, deliberation-resolved contrasts, the S-curve size budget, time ranges, the top-down summary tree and pair summaries. The aive implementation of the lineage-bank memory merged as HereLiesAz/aive#449 and #450; the aive summary tree is not built yet.

## Experiment 000

Prepared and CPU-tested (tiny end-to-end run in `tests/learner/`); no full run. Question, arms, matched budget, evaluation, pre-registered reading, risks and runtime estimate: `experiments/000-onwordly-learner/README.md`. Code: `src/onwordly/learner/`, `src/onwordly/memory/`.

## Execution

- Central dispatch: pushing `.kaggle-run` to main triggers `.github/workflows/kaggle-experiment.yml` (managed by `HereLiesAz/workflows`; do not edit its body). The central profile runs the `onwordly-kaggle` console script, which now points to `onwordly.learner.kaggle:main` (Experiment 000 only; single mode only).
- Notebook: `notebooks/experiment_kaggle.ipynb` runs 000 and publishes results via `onwordly.publish` in a `finally` (upload failure is non-fatal).
- CI: `.github/workflows/ci.yml` (local; per the centralization rule it should move to the central controller). Torch-dependent tests skip when torch is not installed, and CI installs only `.[dev]`.

## Experiment 000 first full run (one seed)

Results on branch `claude/elegant-ritchie-d75ukh` (`experiments/000-onwordly-learner/RESULTS.md`): every register-reading arm collapsed (≤ 0.7% held-out solve); `learner-no-memory` 75.3% with earned-trust behaviour. Hypothesis: register is a train-time answer channel. Diagnostics built on `claude/000-memory-diagnostics` (training-frame probe, held-out second visit, `learner-self-memory`, `learner-memory-dropout`; see the 000 README "Memory diagnostics").

## Next actions

0. Run 000 with the diagnostic arms (`arms: learner,learner-no-memory,learner-self-memory,learner-memory-dropout`, ~2–2.5 h on T4) and read it against the pre-registered memory-diagnostics criteria before changing the memory design.

1. Collect run 17 results; record them in `docs/findings-llm-phase.md`.
2. Inspect run 18 (000 smoke).
3. Run 000 single with the full manifest; record the provisional reading against the pre-registered criteria.
4. Seeds (≥ 3) before attributing any effect to a component.
5. Move CI to the central controller; decide whether CI should install CPU torch so learner tests run there.
