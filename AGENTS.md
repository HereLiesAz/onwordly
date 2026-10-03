# Onwordly session guide

Read these before changing the research design:

1. `README.md`
2. `docs/research-program.md`
3. `docs/novelty-ledger.md`
4. `docs/experiment-matrix.md`
5. `docs/session-handoff.md`
6. the README and manifest for the experiment being changed

## Non-negotiable rules

- Do not rename established techniques and present them as Onwordly inventions.
- Keep training/evaluation partitions separate.
- Adaptive training must never receive held-out evaluation outcomes.
- Compare regimes under matched training-token budgets; record extra inference and verifier work separately.
- Prefer exact/programmatic verification over neural judges whenever possible.
- Add an ablation before attributing an effect to a new component.
- Keep repository Actions centralized through `HereLiesAz/workflows`; do not add one-off local workflow implementations.
- Remote model training/builds must be fire-and-forget from GitHub Actions: dispatch through the central Cloudflare Worker, never poll the external platform from a runner, and let the remote run report its own terminal state back through the Worker callback.
- Update `docs/session-handoff.md` whenever the current state or next actions change.

## Current program

Experiment 001 and the Experiment 002 arithmetic ablation own the active real-model compute. Experiments 003–006 extend the same exact-verification discipline across symbolic transformations, constrained strings, program execution, and formal logic. Experiment 007 prepares an exact process-supervision ablation using program-state traces.

Use `docs/experiment-matrix.md` to keep the evidence ladder and compute gates straight before launching anything expensive.
