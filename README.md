# Onwordly

Onwordly is a small learner built from scratch around three ideas:

- **Earned trust.** Self-trust (per domain) and trust in each corrector are earned from a track record against an exact checker, never assumed.
- **Conscious judgment.** When a corrector contradicts the model, a reasoner sees both sides plus the evidence and decides to hold or change. The decision is recorded.
- **Add-only memory.** Attempts, challenges, corrections and decisions are appended, never overwritten. Contradictions are kept, linked and recalled together.

None of the components is new (see `docs/novelty-ledger.md`). The combination is the hypothesis, tested under exact verification and matched budgets.

## Layout

- `src/onwordly/memory/`: episode store, variant register, trust ledger, update-weight rule.
- `src/onwordly/learner/`: the Experiment 000 model, training, evaluation and Kaggle entry point.
- `src/onwordly/publish.py`: pushes a results tree to a `kaggle-results/<stamp>` branch.
- `experiments/000-onwordly-learner/`: README (question, arms, pre-registered reading) and manifests.
- `docs/model-design.md`: the design. `docs/findings-llm-phase.md`: what the earlier LLM phase found and why it led here.
- `notebooks/experiment_kaggle.ipynb`: runs Experiment 000 on Kaggle and publishes results.

## Running

```
pip install -e ".[train,dev]"
python -m pytest -q
python -m onwordly.learner.experiment --manifest experiments/000-onwordly-learner/smoke-manifest.json --output results/000-smoke
```

Remote runs go through the central workflow (`HereLiesAz/workflows`): `.kaggle-run` on main is read by the `onwordly-kaggle` console script, which runs Experiment 000.

## Archive

The previous phase (fine-tuning Qwen2.5-0.5B on executable curricula, Experiments 001–011, reports, notes and scripts) is on branch `archive/llm-phase` (commit `c35b185`). Its findings are summarised in `docs/findings-llm-phase.md`.
