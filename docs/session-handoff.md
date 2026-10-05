# Session handoff

## Current state (2026-10-05)

- Clean slate. Onwordly is now the earned-trust / conscious-judgment / add-only-memory learner (`docs/model-design.md`). Work starts at Experiment 000.
- The LLM phase (Experiments 001–011, reports, notes, scripts) is archived on branch `archive/llm-phase` (commit `c35b185`). Findings: `docs/findings-llm-phase.md`.
- **Pending run:** Kaggle run 17 (verdict prior diagnostic, `experiment: prior`, Qwen2.5-0.5B and -Instruct, n = 200) was dispatched from main at `c35b185`. Results pending. That code exists only on the archive branch; record its outcome in `docs/findings-llm-phase.md` when it arrives.
- `.kaggle-run` still holds the run-17 plan. The Experiment 000 runner rejects it (`experiment: prior`); set it to `experiment: 000` before the next dispatch.

## Experiment 000

Prepared and CPU-tested (tiny end-to-end run in `tests/learner/`); no full run. Question, arms, matched budget, evaluation, pre-registered reading, risks and runtime estimate: `experiments/000-onwordly-learner/README.md`. Code: `src/onwordly/learner/`, `src/onwordly/memory/`.

## Execution

- Central dispatch: pushing `.kaggle-run` to main triggers `.github/workflows/kaggle-experiment.yml` (managed by `HereLiesAz/workflows`; do not edit its body). The central profile runs the `onwordly-kaggle` console script, which now points to `onwordly.learner.kaggle:main` (Experiment 000 only; single mode only).
- Notebook: `notebooks/experiment_kaggle.ipynb` runs 000 and publishes results via `onwordly.publish` in a `finally` (upload failure is non-fatal).
- CI: `.github/workflows/ci.yml` (local; per the centralization rule it should move to the central controller). Torch-dependent tests skip when torch is not installed, and CI installs only `.[dev]`.

## Next actions

1. Collect run 17 results; record them in `docs/findings-llm-phase.md`.
2. Set `.kaggle-run` to the 000 smoke manifest (`experiment: 000`, `manifest: experiments/000-onwordly-learner/smoke-manifest.json`), dispatch, inspect.
3. Run 000 single with the full manifest; record the provisional reading against the pre-registered criteria.
4. Seeds (≥ 3) before attributing any effect to a component.
5. Move CI to the central controller; decide whether CI should install CPU torch so learner tests run there.
