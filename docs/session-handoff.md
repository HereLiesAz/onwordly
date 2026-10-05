# Session handoff

## Current state (2026-10-05)

- Clean slate. Onwordly is now the earned-trust / conscious-judgment / add-only-memory learner (`docs/model-design.md`). Work starts at Experiment 000.
- The LLM phase (Experiments 001–011, reports, notes, scripts) is archived on branch `archive/llm-phase` (commit `c35b185`). Findings: `docs/findings-llm-phase.md`.
- **Run 17 (verdict prior) never ran.** The `.kaggle-run` change merged at `c35b185`, but the Kaggle hand-off workflow (and CI) has been `disabled_manually` since 2026-10-01, so nothing was dispatched. Its code exists only on `archive/llm-phase`.
- **Run 18 (000 smoke) never ran** for the same reason (merged at `b2c163d`, workflow disabled). `.kaggle-run` on main still requests it; re-enabling the workflow and pushing a `.kaggle-run` change, or running the notebook by hand, starts it.
- Memory design in `docs/model-design.md` now covers consolidation-as-rewrite, deliberation-resolved contrasts, the S-curve size budget, time ranges, the top-down summary tree and pair summaries. The aive implementation of the lineage-bank memory merged as HereLiesAz/aive#449 and #450; the aive summary tree is not built yet.

## Experiment 000

Prepared and CPU-tested (tiny end-to-end run in `tests/learner/`); no full run. Question, arms, matched budget, evaluation, pre-registered reading, risks and runtime estimate: `experiments/000-onwordly-learner/README.md`. Code: `src/onwordly/learner/`, `src/onwordly/memory/`.

## Execution

- Central dispatch: pushing `.kaggle-run` to main triggers `.github/workflows/kaggle-experiment.yml` (managed by `HereLiesAz/workflows`; do not edit its body). The central profile runs the `onwordly-kaggle` console script, which now points to `onwordly.learner.kaggle:main` (Experiment 000 only; single mode only).
- Notebook: `notebooks/experiment_kaggle.ipynb` runs 000 and publishes results via `onwordly.publish` in a `finally` (upload failure is non-fatal).
- CI: `.github/workflows/ci.yml` (local; per the centralization rule it should move to the central controller). Torch-dependent tests skip when torch is not installed, and CI installs only `.[dev]`.

## Next actions

1. Re-enable `.github/workflows/kaggle-experiment.yml` (and CI) or run the notebook by hand; verify a run actually starts before recording it as dispatched.
2. Run and inspect the 000 smoke manifest.
3. Run 000 single with the full manifest; record the provisional reading against the pre-registered criteria.
4. Seeds (≥ 3) before attributing any effect to a component.
5. Move CI to the central controller; decide whether CI should install CPU torch so learner tests run there.
