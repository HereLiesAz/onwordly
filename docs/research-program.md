# Research program

## Thesis

A conventional language-model dataset is a frozen record of examples.

Onwordly investigates a different object: an **executable curriculum**. Instead of storing every lesson, it stores the machinery needed to create lessons, observe performance, verify outcomes, and choose what should happen next.

The wager is that useful supervision depends less on raw example count than on how much information each training event contributes.

## Operational principle

```text
generate task
    ↓
model attempts task
    ↓
verify observable result
    ↓
identify unresolved capability
    ↓
generate the next informative task
    ↓
train
    ↺
```

The system should spend progressively less training budget reproducing behavior the model already demonstrates reliably.

## Measurement discipline

Every experiment should report at least:

- held-out accuracy;
- compositional/generalization accuracy when applicable;
- training tokens consumed;
- generated examples consumed;
- wall-clock training time when real models are used;
- peak memory when available;
- parameter count;
- verifier calls;
- capability gained per million training tokens.

Equal-token comparisons are necessary but not sufficient: adaptive methods may spend extra inference or verification compute. That overhead must be recorded rather than smuggled into the wallpaper.

## Research phases

### Phase 0 — deterministic substrate

Build task generators and exact verifiers for domains where correctness is mechanically decidable.

Initial domains:

- integer arithmetic — training harness and live experiments implemented;
- symbolic transformations — deterministic generator/verifier substrate implemented;
- constrained string manipulation;
- simple program execution;
- formal logic.

### Phase 1 — curriculum efficiency

Compare static sampling with adaptive curricula while holding model, optimizer, training-token budget, evaluation set, and random-seed sets constant.

### Phase 2 — counterexamples and repair

Generate nearby failures, minimal perturbations, adversarial variants, and repair tasks. Treat these as established families of techniques unless an experiment introduces a genuinely new mechanism.

### Phase 3 — process supervision

Add structured intermediate states only where outcome supervision cannot localize failure cheaply.

### Phase 4 — search and distillation

Use expensive search during training to discover verified trajectories, then distill those trajectories into the small model so deployment remains cheap.

### Phase 5 — language games

Move from closed-form tasks into interactive environments where success depends on communication, tool use, coordination, and adaptation to another agent.

## Design constraints

1. **No novelty laundering.** Existing techniques keep their existing names.
2. **Ablate everything.** A new combination is not evidence until its parts are separately measured.
3. **Equal budgets.** Comparisons use matched training-token and, where practical, compute budgets.
4. **Prefer exact verification.** Neural judges are introduced only when programmatic verification is inadequate.
5. **Small first.** Experiments should fail cheaply.
6. **Keep inference cheap.** Expensive search is acceptable during training if it can later be distilled away.
7. **Freeze evaluation.** Adaptive generators never receive held-out evaluation outcomes.
