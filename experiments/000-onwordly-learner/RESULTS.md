# Experiment 000 results

Four runs so far, all one seed, run by hand on Kaggle (2026-10-05). Run 2 adds the memory diagnostics and two arms; the six original arms reproduced run 1 exactly. Run 3 adds recurring frames; run 4 adds `learner-child` and the eval-time memory-source ablation.


Matched budget: 4000 optimizer steps, 256000 training problems (same order every arm). Learner parameters 1844365; plain hidden width 917.

## Solve (held-out rule sets, exact)

| Arm | Params | Accuracy | Per-step accuracy | Fixes / breaks (steps) | Mean s | Conf ECE | Forward steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| learner | 1844365 | 0.7% | 2.0% → 1.9% → 1.2% → 0.7% | 6 / 19 | 0.551 | 0.619 | 2048000 |
| learner-flat | 1844365 | 0.0% | 0.0% → 0.0% → 0.0% → 0.0% | 0 / 0 | 0.535 | 0.720 | 2048000 |
| learner-no-memory | 1844365 | 75.3% | 71.3% → 72.4% → 74.7% → 75.3% | 62 / 22 | 0.951 | 0.020 | 2048000 |
| learner-no-trust | 1844365 | 0.1% | 0.3% → 0.2% → 0.3% → 0.1% | 3 / 5 | 0.574 | 0.581 | 2048000 |
| learner-self-memory | 1844365 | 3.6% | 5.2% → 5.0% → 3.4% → 3.6% | 63 / 79 | 0.651 | 0.253 | 2048000 |
| learner-memory-dropout | 1844365 | 73.4% | 70.8% → 72.0% → 72.8% → 73.4% | 40 / 14 | 0.947 | 0.029 | 2048000 |
| plain | 1846031 | 80.2% | 80.2% | 0 / 0 | 0.960 | 0.220 | 256000 |
| handcoded | 1846031 | 80.2% | 80.2% | 0 / 0 | 0.960 | 0.220 | 256000 |

## Fallible challenge (held out, exact)

| Arm | Corrector (error rate, seen) | Ledger reliability | Hold right vs wrong challenge | Change wrong under correct (to correct) | Hold when told wrong | Discrimination | Final | Decision ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner | A (0.1, seen) | 0.904 | 100.0% | 1.5% (1.5%) | 98.5% | 0.013 | 2.0% | 0.963 |
| learner | B (0.5, seen) | 0.471 | 100.0% | 1.8% (1.8%) | 98.2% | 0.009 | 1.6% | 0.976 |
| learner | C (0.3, unseen) | 0.500 | 100.0% | 1.5% (1.5%) | 98.5% | 0.010 | 1.7% | 0.973 |
| learner-flat | A (0.1, seen) | 0.897 | — | 0.0% (0.0%) | 100.0% | — | 0.0% | 1.000 |
| learner-flat | B (0.5, seen) | 0.485 | — | 0.0% (0.0%) | 100.0% | — | 0.0% | 1.000 |
| learner-flat | C (0.3, unseen) | 0.500 | — | 0.0% (0.0%) | 100.0% | — | 0.0% | 1.000 |
| learner-no-memory | A (0.1, seen) | 0.904 | 100.0% | 100.0% (100.0%) | 26.2% | 0.911 | 97.8% | 0.014 |
| learner-no-memory | B (0.5, seen) | 0.472 | 99.2% | 100.0% (100.0%) | 75.1% | 0.474 | 86.9% | 0.101 |
| learner-no-memory | C (0.3, unseen) | 0.500 | 100.0% | 100.0% (100.0%) | 56.5% | 0.717 | 93.0% | 0.053 |
| learner-no-trust | A (0.1, seen) | 0.896 | — | 3.5% (2.9%) | 96.5% | 0.031 | 2.7% | 0.955 |
| learner-no-trust | B (0.5, seen) | 0.493 | 100.0% | 3.3% (2.7%) | 96.7% | 0.017 | 1.5% | 0.972 |
| learner-no-trust | C (0.3, unseen) | 0.500 | — | 3.9% (3.5%) | 96.1% | 0.027 | 2.5% | 0.957 |
| learner-self-memory | A (0.1, seen) | 0.911 | 100.0% | 0.6% (0.4%) | 99.4% | 0.005 | 3.9% | 0.953 |
| learner-self-memory | B (0.5, seen) | 0.455 | 100.0% | 0.2% (0.0%) | 99.8% | 0.003 | 3.6% | 0.959 |
| learner-self-memory | C (0.3, unseen) | 0.500 | 100.0% | 0.2% (0.2%) | 99.8% | 0.002 | 3.7% | 0.958 |
| learner-memory-dropout | A (0.1, seen) | 0.914 | 98.8% | 94.3% (93.9%) | 28.9% | 0.863 | 96.2% | 0.023 |
| learner-memory-dropout | B (0.5, seen) | 0.525 | 100.0% | 89.0% (89.0%) | 75.4% | 0.455 | 85.5% | 0.120 |
| learner-memory-dropout | C (0.3, unseen) | 0.500 | 100.0% | 92.1% (92.1%) | 58.5% | 0.662 | 91.0% | 0.065 |
| handcoded | A (0.1, seen) | 0.904 | 0.0% | 100.0% (100.0%) | 0.0% | 0.783 | 89.9% | 0.101 |
| handcoded | B (0.5, seen) | 0.519 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |
| handcoded | C (0.3, unseen) | 0.500 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |

## Memory diagnostics (register as answer channel?)

Training reads: solve-pass register reads during training (before that visit's writes). Probe: fixed sample of training problems after training, eval mode, read-only. Second visit: held-out frames, attempt 2 reads only the model's own attempt-1 answer.

| Arm | Train reads non-empty | Train top other = target | Dropped | Probe as-is | Probe emptied | Drop | Probe non-empty | Held-out visit 1 | Visit 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner | 96.9% | 22.6% | 0.0% | 95.5% | 0.6% | 94.9% | 100.0% | 0.7% | 0.7% |
| learner-flat | 96.9% | 22.3% | 0.0% | 95.5% | 0.0% | 95.5% | 100.0% | 0.0% | 0.0% |
| learner-no-trust | 96.9% | 22.8% | 0.0% | 96.5% | 0.2% | 96.3% | 100.0% | 0.1% | 0.1% |
| learner-self-memory | 96.9% | 0.0% | 0.0% | 96.3% | 3.1% | 93.2% | 100.0% | 3.6% | 3.8% |
| learner-memory-dropout | 96.9% | 36.2% | 49.9% | 94.3% | 71.9% | 22.5% | 100.0% | 73.4% | 73.6% |

Hold tracks reliability (hold-when-told-wrong, unreliable minus reliable corrector):

- **learner**: -0.003
- **learner-flat**: 0.000
- **learner-no-memory**: 0.488
- **learner-no-trust**: 0.002
- **learner-self-memory**: 0.004
- **learner-memory-dropout**: 0.464
- **handcoded**: 1.000

## Provisional reading (run 2, one seed)

**The answer-channel hypothesis holds.** Pre-registered criteria met:
- **Probe (1):** on training frames, register-reading arms score 95.5–96.5% with the register as-is and 0.0–3.1% with it emptied. The network does not solve the problem; it reads the answer from memory of that exact frame.
- **Self-memory does not escape it (3.6%).** Its register never contains a corrector proposal (top-other = target 0.0%), yet it collapses as hard. Its own earlier answers on a revisited frame are enough of a channel, so the leak is revisitation itself, not corrector proposals.
- **Dropout recovers (73.4%)**, close to no-memory (75.3%), and keeps the earned-trust behaviour (hold tracks reliability 0.464 vs 0.488; post-challenge 85.5–96.2%).
- **Probe (2):** a second held-out visit with the model's own first answer in memory adds nothing (73.4% → 73.6% for dropout).

**What this means.** As built, memory is keyed to the exact frame (the rule set). Held-out evaluation uses frames never seen before, so memory can only ever be empty there; it cannot help by design. It can only shortcut training. The current evaluation never gives memory a chance to be useful.

**Two directions, not yet chosen:**
1. **Evaluate where memory can matter:** a stream in which held-out frames recur over time (first visit cold, later visits with history, challenges in between), so remembering one's own past attempts, corrections and contrasts on that frame can pay off. Train with dropout or with first-visit-only reads so the shortcut is not learnable.
2. **Make memory generalise:** recall by similarity (frames that share rules) rather than exact identity, so a new frame retrieves related experience.

Not a finding until repeated across seeds.


---

# Run 3 — recurring frames (one seed, by hand on Kaggle, 2026-10-05)

Matched budget: 4000 optimizer steps, 256000 training problems (same order every arm). Learner parameters 1844365; plain hidden width 917.

## Solve (held-out rule sets, exact)

| Arm | Params | Accuracy | Per-step accuracy | Fixes / breaks (steps) | Mean s | Conf ECE | Forward steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| learner-no-memory | 1844365 | 75.3% | 71.3% → 72.4% → 74.7% → 75.3% | 62 / 22 | 0.951 | 0.020 | 2048000 |
| learner-memory-dropout | 1844365 | 73.4% | 70.8% → 72.0% → 72.8% → 73.4% | 40 / 14 | 0.947 | 0.029 | 2048000 |
| learner-first-visit | 1844365 | 73.1% | 71.2% → 71.1% → 72.0% → 73.1% | 36 / 17 | 0.946 | 0.017 | 3040384 |
| plain | 1846031 | 80.2% | 80.2% | 0 / 0 | 0.960 | 0.220 | 256000 |
| handcoded | 1846031 | 80.2% | 80.2% | 0 / 0 | 0.960 | 0.220 | 256000 |

## Fallible challenge (held out, exact)

| Arm | Corrector (error rate, seen) | Ledger reliability | Hold right vs wrong challenge | Change wrong under correct (to correct) | Hold when told wrong | Discrimination | Final | Decision ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner-no-memory | A (0.1, seen) | 0.904 | 100.0% | 100.0% (100.0%) | 26.2% | 0.911 | 97.8% | 0.014 |
| learner-no-memory | B (0.5, seen) | 0.472 | 99.2% | 100.0% (100.0%) | 75.1% | 0.474 | 86.9% | 0.101 |
| learner-no-memory | C (0.3, unseen) | 0.500 | 100.0% | 100.0% (100.0%) | 56.5% | 0.717 | 93.0% | 0.053 |
| learner-memory-dropout | A (0.1, seen) | 0.914 | 98.8% | 94.3% (93.9%) | 28.9% | 0.863 | 96.2% | 0.023 |
| learner-memory-dropout | B (0.5, seen) | 0.525 | 100.0% | 89.0% (89.0%) | 75.4% | 0.455 | 85.5% | 0.120 |
| learner-memory-dropout | C (0.3, unseen) | 0.500 | 100.0% | 92.1% (92.1%) | 58.5% | 0.662 | 91.0% | 0.065 |
| learner-first-visit | A (0.1, seen) | 0.912 | 10.1% | 100.0% (16.0%) | 2.5% | 0.613 | 53.0% | 0.444 |
| learner-first-visit | B (0.5, seen) | 0.513 | 7.4% | 100.0% (15.6%) | 5.4% | 0.306 | 33.7% | 0.632 |
| learner-first-visit | C (0.3, unseen) | 0.500 | 9.0% | 100.0% (16.4%) | 4.8% | 0.462 | 42.8% | 0.552 |
| handcoded | A (0.1, seen) | 0.904 | 0.0% | 100.0% (100.0%) | 0.0% | 0.783 | 89.9% | 0.101 |
| handcoded | B (0.5, seen) | 0.519 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |
| handcoded | C (0.3, unseen) | 0.500 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |

## Memory diagnostics (register as answer channel?)

Training reads: solve-pass register reads during training (before that visit's writes). Probe: fixed sample of training problems after training, eval mode, read-only. Second visit: held-out frames, attempt 2 reads only the model's own attempt-1 answer.

| Arm | Train reads non-empty | Train top other = target | Dropped | Probe as-is | Probe emptied | Drop | Probe non-empty | Held-out visit 1 | Visit 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner-memory-dropout | 96.9% | 36.2% | 49.9% | 94.3% | 71.9% | 22.5% | 100.0% | 73.4% | 73.6% |
| learner-first-visit | 96.9% | 83.3% | 0.0% | 0.2% | 71.5% | -71.3% | 100.0% | 73.1% | 53.9% |

## Recurring frames (held out, exact)

500 held-out frames × 4 visits, shuffled stream, ≥ 25 other visits between two visits of a frame. Each visit: attempt (reads the frame's eval register), challenge (A/B/C per visit), hold/change, exact verifier; all written to a forked eval register (self + corrector fillers readable; verifier stored, not read; ledger frozen). Solve = before challenge, final = after. Δ = visit k − visit 1; vs no-mem = final minus learner-no-memory's final at the same visit (points).

| Arm | Visit | Register non-empty | Solve | Final | Hold right vs wrong | Change wrong under correct | Hold when told wrong | Final vs no-mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner-no-memory | 1 | not read | 77.6% | 92.6% | 99.2% | 100.0% | 60.5% | +0.0 |
| learner-no-memory | 2 | not read | 77.6% | 93.6% | 100.0% | 100.0% | 61.0% | +0.0 |
| learner-no-memory | 3 | not read | 77.6% | 91.8% | 99.1% | 100.0% | 60.9% | +0.0 |
| learner-no-memory | 4 | not read | 77.6% | 93.6% | 99.1% | 100.0% | 56.9% | +0.0 |
| learner-memory-dropout | 1 | 0.0% | 76.4% | 92.0% | 99.2% | 94.0% | 62.2% | -0.6 |
| learner-memory-dropout | 2 | 100.0% | 89.4% | 94.0% | 99.2% | 75.8% | 83.4% | +0.4 |
| learner-memory-dropout | 3 | 100.0% | 93.2% | 94.4% | 98.4% | 38.1% | 93.2% | +2.6 |
| learner-memory-dropout | 4 | 100.0% | 94.0% | 95.6% | 100.0% | 44.4% | 94.4% | +2.0 |
| learner-first-visit | 1 | 0.0% | 76.6% | 47.8% | 5.2% | 100.0% | 3.1% | -44.8 |
| learner-first-visit | 2 | 100.0% | 35.4% | 24.8% | 16.7% | 100.0% | 2.9% | -68.8 |
| learner-first-visit | 3 | 100.0% | 22.2% | 15.0% | 11.4% | 100.0% | 1.3% | -76.8 |
| learner-first-visit | 4 | 100.0% | 13.4% | 11.4% | 9.5% | 100.0% | 0.7% | -82.2 |
| plain | 1 | not read | 82.8% | — | — | — | — | — |
| plain | 2 | not read | 82.8% | — | — | — | — | — |
| plain | 3 | not read | 82.8% | — | — | — | — | — |
| plain | 4 | not read | 82.8% | — | — | — | — | — |
| handcoded | 1 | not read | 82.8% | 84.4% | 88.9% | 41.1% | 80.1% | -8.2 |
| handcoded | 2 | not read | 82.8% | 84.8% | 86.6% | 43.3% | 77.0% | -8.8 |
| handcoded | 3 | not read | 82.8% | 85.4% | 91.5% | 36.5% | 81.8% | -6.4 |
| handcoded | 4 | not read | 82.8% | 86.0% | 92.2% | 36.8% | 81.4% | -7.6 |

| Arm | Δ solve | Δ final | Δ final minus no-memory's Δ | First-visit-wrong frames | Right at visit k (final, by visit) |
| --- | ---: | ---: | ---: | ---: | --- |
| learner-no-memory | +0.0 | +1.0 | +0.0 | 112 | 67.9% → 71.4% → 63.4% → 71.4% |
| learner-memory-dropout | +17.6 | +3.6 | +2.6 | 118 | 66.1% → 85.6% → 88.1% → 94.1% |
| learner-first-visit | -63.2 | -36.4 | -37.4 | 117 | 12.8% → 6.8% → 5.1% → 5.1% |
| plain | +0.0 | — | — | 86 | 0.0% → 0.0% → 0.0% → 0.0% |
| handcoded | +0.0 | +1.6 | +0.6 | 86 | 26.7% → 33.7% → 26.7% → 29.1% |

Per corrector (final accuracy by visit; hold right vs wrong / change wrong under correct at the last visit):

| Arm | Corrector | Final by visit | Hold right (last) | Change wrong (last) |
| --- | --- | --- | ---: | ---: |
| learner-no-memory | A | 95.6% → 99.4% → 98.7% → 100.0% | 100.0% | 100.0% |
| learner-no-memory | B | 89.0% → 89.5% → 87.5% → 85.2% | 98.4% | 100.0% |
| learner-no-memory | C | 93.7% → 91.2% → 89.7% → 94.7% | 100.0% | 100.0% |
| learner-memory-dropout | A | 97.5% → 94.3% → 97.4% → 97.7% | 100.0% | 42.9% |
| learner-memory-dropout | B | 89.0% → 94.1% → 93.1% → 92.9% | 100.0% | 33.3% |
| learner-memory-dropout | C | 89.9% → 93.6% → 92.9% → 95.9% | 100.0% | 60.0% |
| learner-first-visit | A | 60.6% → 34.1% → 17.9% → 13.7% | 16.7% | 100.0% |
| learner-first-visit | B | 35.7% → 17.6% → 11.9% → 9.7% | 11.1% | 100.0% |
| learner-first-visit | C | 48.7% → 21.6% → 15.2% → 10.6% | 0.0% | 100.0% |
| handcoded | A | 88.1% → 87.5% → 91.7% → 94.9% | 0.0% | 100.0% |
| handcoded | B | 86.3% → 83.7% → 82.5% → 83.2% | 100.0% | 0.0% |
| handcoded | C | 78.5% → 83.0% → 82.6% → 79.4% | 100.0% | 0.0% |

Hold tracks reliability (hold-when-told-wrong, unreliable minus reliable corrector):

- **learner-no-memory**: 0.488
- **learner-memory-dropout**: 0.464
- **learner-first-visit**: 0.029
- **handcoded**: 1.000

## Provisional reading (run 3, one seed)

**Memory helps on revisits — but fails the pre-registered criterion.**
- `learner-memory-dropout` solve accuracy rose 76.4% → 94.0% across four visits (+17.6); final +3.6, beating no-memory by +2.6 at the last visit. Of frames it got wrong on visit 1, 94.1% were right by visit 4 (no-memory: 71.4%).
- Challenge behaviour did not stay intact: changing a wrong answer under a correct challenge fell 94.0% → 44.4%; holding when told wrong rose 62% → 94%. The pre-registered limit was a 5-point drop. **Criterion not met.**
- Reading: with memory the model becomes confident on revisited frames and stops yielding — right most of the time, stubborn on the rest. Trust in its own memory outruns its earned trust in correctors.
- **Unresolved:** how much of the gain is remembering a corrector's proposal (corrector fillers are readable on later visits) versus its own past attempts. The pre-registration excludes gains from simply adopting proposals. Needs an eval-register ablation: self fillers only vs self + corrector.

**`learner-first-visit` is broken by design.** Training the hold/change decision only with memory present left it untrained without memory: on visit 1 it holds right answers 5% of the time, and accuracy falls with every visit (47.8% → 11.4%). Drop this arm.

**Baselines.** No-memory flat across visits, as it must be. The hand-coded rule improves slightly (+1.6) and trails no-memory by 6–9 points.

# Run 4 — childhood arm + memory-source ablation (one seed, by hand on Kaggle, 2026-10-05)

Matched budget: 4000 optimizer steps, 256000 training problems (same order every arm). Learner parameters 1844365; plain hidden width 917.

## Solve (held-out rule sets, exact)

| Arm | Params | Accuracy | Per-step accuracy | Fixes / breaks (steps) | Mean s | Conf ECE | Forward steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| learner-no-memory | 1844365 | 75.3% | 71.3% → 72.4% → 74.7% → 75.3% | 62 / 22 | 0.951 | 0.020 | 2048000 |
| learner-memory-dropout | 1844365 | 73.4% | 70.8% → 72.0% → 72.8% → 73.4% | 40 / 14 | 0.947 | 0.029 | 2048000 |
| learner-child | 1844365 | 73.1% | 72.5% → 71.5% → 73.1% → 73.1% | 25 / 19 | 0.946 | 0.036 | 2048000 |
| plain | 1846031 | 80.2% | 80.2% | 0 / 0 | 0.960 | 0.220 | 256000 |
| handcoded | 1846031 | 80.2% | 80.2% | 0 / 0 | 0.960 | 0.220 | 256000 |

## Fallible challenge (held out, exact)

| Arm | Corrector (error rate, seen) | Ledger reliability | Hold right vs wrong challenge | Change wrong under correct (to correct) | Hold when told wrong | Discrimination | Final | Decision ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner-no-memory | A (0.1, seen) | 0.904 | 100.0% | 100.0% (100.0%) | 26.2% | 0.911 | 97.8% | 0.014 |
| learner-no-memory | B (0.5, seen) | 0.472 | 99.2% | 100.0% (100.0%) | 75.1% | 0.474 | 86.9% | 0.101 |
| learner-no-memory | C (0.3, unseen) | 0.500 | 100.0% | 100.0% (100.0%) | 56.5% | 0.717 | 93.0% | 0.053 |
| learner-memory-dropout | A (0.1, seen) | 0.914 | 98.8% | 94.3% (93.9%) | 28.9% | 0.863 | 96.2% | 0.023 |
| learner-memory-dropout | B (0.5, seen) | 0.525 | 100.0% | 89.0% (89.0%) | 75.4% | 0.455 | 85.5% | 0.120 |
| learner-memory-dropout | C (0.3, unseen) | 0.500 | 100.0% | 92.1% (92.1%) | 58.5% | 0.662 | 91.0% | 0.065 |
| learner-child | A (0.1, seen) | 0.901 | 98.6% | 64.1% (64.1%) | 50.7% | 0.556 | 88.0% | 0.101 |
| learner-child | B (0.5, seen) | 0.507 | 99.4% | 53.1% (53.1%) | 84.0% | 0.284 | 80.6% | 0.184 |
| learner-child | C (0.3, unseen) | 0.500 | 99.5% | 54.2% (54.2%) | 74.1% | 0.385 | 83.4% | 0.156 |
| handcoded | A (0.1, seen) | 0.904 | 0.0% | 100.0% (100.0%) | 0.0% | 0.783 | 89.9% | 0.101 |
| handcoded | B (0.5, seen) | 0.519 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |
| handcoded | C (0.3, unseen) | 0.500 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |

## Memory diagnostics (register as answer channel?)

Training reads: solve-pass register reads during training (before that visit's writes). Probe: fixed sample of training problems after training, eval mode, read-only. Second visit: held-out frames, attempt 2 reads only the model's own attempt-1 answer.

| Arm | Train reads non-empty | Train top other = target | Dropped | Probe as-is | Probe emptied | Drop | Probe non-empty | Held-out visit 1 | Visit 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner-memory-dropout | 96.9% | 36.2% | 49.9% | 94.3% | 71.9% | 22.5% | 100.0% | 73.4% | 73.6% |
| learner-child | 94.1% | 41.0% | 49.9% | 93.4% | 70.5% | 22.9% | 100.0% | 73.1% | 73.1% |

## Recurring frames (held out, exact)

500 held-out frames × 4 visits, shuffled stream, ≥ 25 other visits between two visits of a frame. Each visit: attempt (reads the frame's eval register), challenge (A/B/C per visit), hold/change, exact verifier; all written to a forked eval register (self + corrector fillers readable; verifier stored, not read; ledger frozen). Solve = before challenge, final = after. Δ = visit k − visit 1; vs no-mem = final minus learner-no-memory's final at the same visit (points). Rows `arm[sources]`: the same trained model, eval-register reads limited at evaluation to `self` (its own answers), `correctors` (what it was told) or `all` (both).

| Arm | Visit | Register non-empty | Solve | Final | Hold right vs wrong | Change wrong under correct | Hold when told wrong | Final vs no-mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learner-no-memory | 1 | not read | 77.6% | 92.6% | 99.2% | 100.0% | 60.5% | +0.0 |
| learner-no-memory | 2 | not read | 77.6% | 93.6% | 100.0% | 100.0% | 61.0% | +0.0 |
| learner-no-memory | 3 | not read | 77.6% | 91.8% | 99.1% | 100.0% | 60.9% | +0.0 |
| learner-no-memory | 4 | not read | 77.6% | 93.6% | 99.1% | 100.0% | 56.9% | +0.0 |
| learner-memory-dropout | 1 | 0.0% | 76.4% | 92.0% | 99.2% | 94.0% | 62.2% | -0.6 |
| learner-memory-dropout | 2 | 100.0% | 89.4% | 94.0% | 99.2% | 75.8% | 83.4% | +0.4 |
| learner-memory-dropout | 3 | 100.0% | 93.2% | 94.4% | 98.4% | 38.1% | 93.2% | +2.6 |
| learner-memory-dropout | 4 | 100.0% | 94.0% | 95.6% | 100.0% | 44.4% | 94.4% | +2.0 |
| learner-memory-dropout[self] | 1 | 0.0% | 76.4% | 92.8% | 98.4% | 98.8% | 59.8% | +0.2 |
| learner-memory-dropout[self] | 2 | 100.0% | 91.2% | 96.2% | 99.3% | 86.7% | 84.7% | +2.6 |
| learner-memory-dropout[self] | 3 | 100.0% | 94.2% | 96.0% | 100.0% | 64.3% | 94.6% | +4.2 |
| learner-memory-dropout[self] | 4 | 100.0% | 95.0% | 96.4% | 98.6% | 50.0% | 93.5% | +2.8 |
| learner-memory-dropout[correctors] | 1 | 0.0% | 76.4% | 92.6% | 98.4% | 98.8% | 59.8% | +0.0 |
| learner-memory-dropout[correctors] | 2 | 41.8% | 85.2% | 94.0% | 99.2% | 83.3% | 74.7% | +0.4 |
| learner-memory-dropout[correctors] | 3 | 61.2% | 83.0% | 90.0% | 97.7% | 67.2% | 77.4% | -1.8 |
| learner-memory-dropout[correctors] | 4 | 72.8% | 84.2% | 92.6% | 98.4% | 77.6% | 73.9% | -1.0 |
| learner-child | 1 | 0.0% | 75.8% | 87.0% | 100.0% | 68.3% | 71.4% | -5.6 |
| learner-child | 2 | 39.2% | 89.0% | 93.0% | 100.0% | 55.6% | 88.8% | -0.6 |
| learner-child | 3 | 58.2% | 92.8% | 94.6% | 98.6% | 47.8% | 92.0% | +2.8 |
| learner-child | 4 | 73.2% | 95.6% | 96.6% | 100.0% | 33.3% | 96.8% | +3.0 |
| plain | 1 | not read | 82.8% | — | — | — | — | — |
| plain | 2 | not read | 82.8% | — | — | — | — | — |
| plain | 3 | not read | 82.8% | — | — | — | — | — |
| plain | 4 | not read | 82.8% | — | — | — | — | — |
| handcoded | 1 | not read | 82.8% | 84.4% | 88.9% | 41.1% | 80.1% | -8.2 |
| handcoded | 2 | not read | 82.8% | 84.8% | 86.6% | 43.3% | 77.0% | -8.8 |
| handcoded | 3 | not read | 82.8% | 85.4% | 91.5% | 36.5% | 81.8% | -6.4 |
| handcoded | 4 | not read | 82.8% | 86.0% | 92.2% | 36.8% | 81.4% | -7.6 |

| Arm | Δ solve | Δ final | Δ final minus no-memory's Δ | First-visit-wrong frames | Right at visit k (final, by visit) |
| --- | ---: | ---: | ---: | ---: | --- |
| learner-no-memory | +0.0 | +1.0 | +0.0 | 112 | 67.9% → 71.4% → 63.4% → 71.4% |
| learner-memory-dropout | +17.6 | +3.6 | +2.6 | 118 | 66.1% → 85.6% → 88.1% → 94.1% |
| learner-memory-dropout[self] | +18.6 | +3.6 | +2.6 | 118 | 69.5% → 83.9% → 83.1% → 84.7% |
| learner-memory-dropout[correctors] | +7.8 | +0.0 | -1.0 | 118 | 69.5% → 85.6% → 81.4% → 88.1% |
| learner-child | +19.8 | +9.6 | +8.6 | 121 | 46.3% → 84.3% → 90.1% → 95.0% |
| plain | +0.0 | — | — | 86 | 0.0% → 0.0% → 0.0% → 0.0% |
| handcoded | +0.0 | +1.6 | +0.6 | 86 | 26.7% → 33.7% → 26.7% → 29.1% |

Per corrector (final accuracy by visit; hold right vs wrong / change wrong under correct at the last visit):

| Arm | Corrector | Final by visit | Hold when told wrong by visit | Hold right (last) | Change wrong (last) |
| --- | --- | --- | --- | ---: | ---: |
| learner-no-memory | A | 95.6% → 99.4% → 98.7% → 100.0% | 23.1% → 29.6% → 28.6% → 27.9% | 100.0% | 100.0% |
| learner-no-memory | B | 89.0% → 89.5% → 87.5% → 85.2% | 80.9% → 75.6% → 77.9% → 81.8% | 98.4% | 100.0% |
| learner-no-memory | C | 93.7% → 91.2% → 89.7% → 94.7% | 55.2% → 68.5% → 61.5% → 47.1% | 100.0% | 100.0% |
| learner-memory-dropout | A | 97.5% → 94.3% → 97.4% → 97.7% | 30.2% → 50.0% → 88.9% → 88.5% | 100.0% | 42.9% |
| learner-memory-dropout | B | 89.0% → 94.1% → 93.1% → 92.9% | 77.9% → 96.1% → 92.9% → 97.0% | 100.0% | 33.3% |
| learner-memory-dropout | C | 89.9% → 93.6% → 92.9% → 95.9% | 60.6% → 80.4% → 95.0% → 94.1% | 100.0% | 60.0% |
| learner-memory-dropout[self] | A | 98.1% → 95.5% → 98.7% → 97.7% | 27.9% → 61.5% → 75.0% → 87.5% | 100.0% | 40.0% |
| learner-memory-dropout[self] | B | 89.0% → 94.8% → 95.0% → 96.1% | 77.9% → 95.2% → 98.8% → 97.4% | 98.6% | 50.0% |
| learner-memory-dropout[self] | C | 91.8% → 98.2% → 94.6% → 95.3% | 54.9% → 80.6% → 94.0% → 90.5% | 98.1% | 55.6% |
| learner-memory-dropout[correctors] | A | 98.1% → 96.0% → 91.0% → 94.9% | 27.9% → 37.8% → 52.6% → 41.4% | 100.0% | 73.9% |
| learner-memory-dropout[correctors] | B | 89.0% → 90.2% → 87.5% → 87.7% | 77.9% → 93.3% → 85.2% → 85.0% | 97.1% | 90.9% |
| learner-memory-dropout[correctors] | C | 91.1% → 95.3% → 91.3% → 94.7% | 54.9% → 74.3% → 82.1% → 74.6% | 100.0% | 75.0% |
| learner-child | A | 92.5% → 94.3% → 96.2% → 95.4% | 37.5% → 69.2% → 76.0% → 94.4% | 100.0% | 16.7% |
| learner-child | B | 82.4% → 93.5% → 91.9% → 97.4% | 89.7% → 96.1% → 96.0% → 97.5% | 100.0% | 40.0% |
| learner-child | C | 86.7% → 91.2% → 95.7% → 97.1% | 68.1% → 92.1% → 93.7% → 96.5% | 100.0% | 50.0% |
| handcoded | A | 88.1% → 87.5% → 91.7% → 94.9% | 0.0% → 0.0% → 0.0% → 0.0% | 0.0% | 100.0% |
| handcoded | B | 86.3% → 83.7% → 82.5% → 83.2% | 100.0% → 100.0% → 100.0% → 100.0% | 100.0% | 0.0% |
| handcoded | C | 78.5% → 83.0% → 82.6% → 79.4% | 100.0% → 100.0% → 100.0% → 100.0% | 100.0% | 0.0% |

Hold tracks reliability (hold-when-told-wrong, unreliable minus reliable corrector):

- **learner-no-memory**: 0.488
- **learner-memory-dropout**: 0.464
- **learner-child**: 0.334
- **handcoded**: 1.000

## Provisional reading (run 4, one seed)

Shared arms reproduced run 3 exactly (`learner-no-memory`, `learner-memory-dropout`, `plain`, `handcoded`).

**Source of the run-3 gain: its own past answers.** `[self]` matches `[all]` (Δ solve +18.6 vs +17.6, Δ final +3.6 vs +3.6); `[correctors]` is well below (+7.8, +0.0). By the pre-registered reading, the revisit gain is remembering its own answers — the premature self-trust the mindset excludes. Stubbornness shows in both, stronger in `[self]`: change-wrong-under-correct 98.8 → 50.0% (`[self]`) vs 98.8 → 77.6% (`[correctors]`); hold-when-told-wrong 59.8 → 93.5% vs 59.8 → 73.9%.

**`learner-child`: revisit gains, but not child behaviour. Criterion not met.**
- Gain: Δ final +9.6 vs no-memory +1.0 (+8.6); solve 75.8 → 95.6%; final 87.0 → 96.6%, ahead of no-memory by +3.0 at visit 4. Of frames wrong on visit 1, 95.0% right by visit 4.
- Stubbornness anyway: change-wrong-under-correct 68.3 → 33.3% (limit: within 5 points). Hold-when-told-wrong rises with visits for every corrector — A 37.5 → 94.4%, B 89.7 → 97.5%, C 68.1 → 96.5% — so it tracks visit count, not reliability. B > C > A holds at visits 1–3; at visit 4 all three are ≥ 94%.
- Weaker yielding from the start: on the standard challenge, change-wrong-under-correct against A is 64.1% (dropout 94.3%, no-memory 100%), discrimination 0.556, hold-tracks-reliability 0.334 (dropout 0.464). Removing self-trust input and confidence × surprise weighting made it more stubborn, not less.
- Reading: this arm reads only corrector proposals, so stubbornness without self-trust means remembered proposals harden into held answers. A corrector's past word becomes the child's own conviction — "defending what it was taught" — and then it will not yield even to A, the reliable teacher. The mindset allows defending taught answers against others, not against the teacher. Nothing here distinguishes a teacher from any other corrector: A, B and C are all just correctors.
- Caveat: late-visit change-wrong rates rest on few frames (≈ 4% of 500 are still wrong at visit 4, about half of those get a correct challenge), so those percentages are coarse. The hold-when-told-wrong trend is on larger counts.

**Baselines** unchanged: no-memory flat; hand-coded trails no-memory by 6–9 points.

**Next.** The child arm has no notion of a teacher. Candidate (needs approval before building): mark one corrector as the teacher in training, so its fillers and challenges carry authority, and pre-register that the teacher can still move the model on every visit while others cannot. ≥ 3 seeds before attributing anything.
