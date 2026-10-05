# Session handoff

## Current state (2026-10-05)

- Clean slate. Onwordly is now the earned-trust / conscious-judgment / add-only-memory learner (`docs/model-design.md`). Work starts at Experiment 000.
- The LLM phase (Experiments 001–011, reports, notes, scripts) is archived on branch `archive/llm-phase` (commit `c35b185`). Findings: `docs/findings-llm-phase.md`.
- **Run 17 (verdict prior) never ran.** The `.kaggle-run` change merged at `c35b185`, but the Kaggle hand-off workflow (and CI) has been `disabled_manually` since 2026-10-01, so nothing was dispatched. Its code exists only on `archive/llm-phase`.
- **Experiment 000 first full run (by hand on Kaggle, one seed):** memory-reading arms collapsed (≤ 0.7% solve); learner-no-memory reached 75.3% solve and showed earned-trust behaviour (hold tracks corrector reliability by 0.488; post-challenge accuracy 86.9–97.8% vs plain 80.2%). Run 2 confirmed the register is an exact-frame answer channel (training probe 95.5% → 0.6% when emptied); dropout recovers 73.4% with earned trust intact. See `experiments/000-onwordly-learner/RESULTS.md`.
- Memory design in `docs/model-design.md` now covers consolidation-as-rewrite, deliberation-resolved contrasts, the S-curve size budget, time ranges, the top-down summary tree and pair summaries. The aive implementation of the lineage-bank memory merged as HereLiesAz/aive#449 and #450; the aive summary tree is not built yet.

## Experiment 000

One full run recorded (`RESULTS.md`); diagnostics merged. Question, arms, matched budget, evaluation, pre-registered reading, risks and runtime estimate: `experiments/000-onwordly-learner/README.md`. Code: `src/onwordly/learner/`, `src/onwordly/memory/`.

## Execution

- Central dispatch: pushing `.kaggle-run` to main triggers `.github/workflows/kaggle-experiment.yml` (managed by `HereLiesAz/workflows`; do not edit its body). The central profile runs the `onwordly-kaggle` console script, which now points to `onwordly.learner.kaggle:main` (Experiment 000 only; single mode only).
- Notebook: `notebooks/experiment_kaggle.ipynb` runs 000 and publishes results via `onwordly.publish` in a `finally` (upload failure is non-fatal).
- CI: `.github/workflows/ci.yml` (local; per the centralization rule it should move to the central controller). Torch-dependent tests skip when torch is not installed, and CI installs only `.[dev]`.

## Experiment 000 first full run (one seed)

Results in `experiments/000-onwordly-learner/RESULTS.md`: every register-reading arm collapsed (≤ 0.7% held-out solve); `learner-no-memory` 75.3% with earned-trust behaviour. Hypothesis: register is a train-time answer channel. Diagnostics merged (training-frame probe, held-out second visit, `learner-self-memory`, `learner-memory-dropout`; see the 000 README "Memory diagnostics").

## Recurring frames (prepared, not run)

Branch `claude/000-recurring-frames`: held-out frames revisited in a shuffled stream with gaps (`learner/recurring.py`), writes into a forked eval register (self + corrector fillers readable, verifier stored but not read, ledger frozen); per-visit solve/final/challenge metrics, learning curve, first-wrong-fixed. New arm `learner-first-visit` (revisits train the memory-reading passes only through the hold/change decision; solve loss from an extra empty-register pass). Manifests `recurring-manifest.json` (no-memory, memory-dropout, first-visit, plain, handcoded) and `recurring-smoke-manifest.json`. Pre-registered reading in the 000 README, "Recurring frames". Estimate ~40–50 min on one T4.

## Next actions

1. Run the recurring-frame evaluation by hand on Kaggle: `.kaggle-run` = `experiment: 000`, `mode: single`, `manifest: experiments/000-onwordly-learner/recurring-manifest.json`; record against the README's pre-registered reading. If no memory arm beats no-memory's curve, go to similarity-based recall.
2. Run 17 (verdict prior), if still wanted: by hand from `archive/llm-phase`. Run 18 (000 smoke) is superseded by the full run.
3. Run 000 single with the full manifest; record the provisional reading against the pre-registered criteria.
4. Seeds (≥ 3) before attributing any effect to a component.
5. Move CI to the central controller; decide whether CI should install CPU torch so learner tests run there.
