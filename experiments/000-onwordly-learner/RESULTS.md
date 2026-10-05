# Experiment 000 — Onwordly learner (provisional, one seed)

Matched budget: 4000 optimizer steps, 256000 training problems (same order every arm). Learner parameters 1844365; plain hidden width 917.

## Solve (held-out rule sets, exact)

| Arm | Params | Accuracy | Per-step accuracy | Fixes / breaks (steps) | Mean s | Conf ECE | Forward steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| learner | 1844365 | 0.7% | 2.0% → 1.9% → 1.2% → 0.7% | 6 / 19 | 0.551 | 0.619 | 2048000 |
| learner-flat | 1844365 | 0.0% | 0.0% → 0.0% → 0.0% → 0.0% | 0 / 0 | 0.535 | 0.720 | 2048000 |
| learner-no-memory | 1844365 | 75.3% | 71.3% → 72.4% → 74.7% → 75.3% | 62 / 22 | 0.951 | 0.020 | 2048000 |
| learner-no-trust | 1844365 | 0.1% | 0.3% → 0.2% → 0.3% → 0.1% | 3 / 5 | 0.574 | 0.581 | 2048000 |
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
| handcoded | A (0.1, seen) | 0.904 | 0.0% | 100.0% (100.0%) | 0.0% | 0.783 | 89.9% | 0.101 |
| handcoded | B (0.5, seen) | 0.519 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |
| handcoded | C (0.3, unseen) | 0.500 | 100.0% | 0.0% (0.0%) | 100.0% | 0.000 | 80.2% | 0.198 |

Hold tracks reliability (hold-when-told-wrong, unreliable minus reliable corrector):

- **learner**: -0.003
- **learner-flat**: 0.000
- **learner-no-memory**: 0.488
- **learner-no-trust**: 0.002
- **handcoded**: 1.000

## Provisional reading (one seed, run by hand on Kaggle, 2026-10-05)

**Memory input broke learning.** Every arm that reads the register collapsed (learner 0.7%, flat 0.0%, no-trust 0.1% solve); the same network without memory reached 75.3% (plain one-pass: 80.2%). Not a property of trust or weighting — the three collapsed arms differ in those and share only the register input.

**Without memory, the earned-trust behaviour appeared.** learner-no-memory, reading only the trust ledger:
- held right answers against wrong challenges (99–100%) and changed every wrong answer under a correct one (100%, all to the correct answer);
- held when told wrong in proportion to corrector reliability: 26% for A (10% error), 57% for unseen C (30%), 75% for B (50%) — hold tracks reliability by 0.488;
- final accuracy after the challenge 97.8% / 93.0% / 86.9% (A / C / B), above plain's 80.2% and the hand-coded rule's 80.2–89.9%;
- calibrated decisions (decision ECE 0.014–0.101; solve confidence ECE 0.020 vs plain 0.220).
The hand-coded rule trusted A blindly and ignored B and C entirely (hold-tracks-reliability 1.000, but by thresholding, with no discrimination on B/C).

**Likely cause of the collapse (hypothesis, untested):** 20,000 training rule sets are each revisited ~13 times, so on most training visits the register already holds that frame's earlier fillers — including corrector proposals, which from A are right 90% of the time. The network learns to read the answer from memory. Held-out rule sets arrive with an empty register, a condition seen on only ~1 in 13 training visits. Memory as built is a train-time answer channel, not a test-time aid.

**Tests before any fix is credited:** (1) learner accuracy on training frames with populated vs emptied register; (2) held-out accuracy after a second visit (register populated by the model's own first attempt only); (3) register built from the model's own fillers only (no corrector proposals); (4) register dropout during training.

Not a finding until repeated across seeds.
