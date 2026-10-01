# Onwordly session guide

Read these before changing the research design:

1. `README.md`
2. `docs/research-program.md`
3. `docs/novelty-ledger.md`
4. `docs/session-handoff.md`
5. the README and manifest for the experiment being changed

## Non-negotiable rules

- Do not rename established techniques and present them as Onwordly inventions.
- Keep training/evaluation partitions separate.
- Adaptive training must never receive held-out evaluation outcomes.
- Compare regimes under matched training-token budgets; record extra inference and verifier work separately.
- Prefer exact/programmatic verification over neural judges whenever possible.
- Add an ablation before attributing an effect to a new component.
- Keep repository Actions centralized through `HereLiesAz/workflows`; do not add one-off local workflow implementations.
- Update `docs/session-handoff.md` whenever the current state or next actions change.

## Current experiment

Experiment 001 tests whether adaptive generated curricula improve arithmetic capability per training token over a frozen static baseline.

The code is intended to make the boring comparison clean before adding PRMs, MCTS, multi-agent language games, or any other expensive furniture.
