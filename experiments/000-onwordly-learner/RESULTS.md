# Experiment 000 results

Two runs so far, both one seed, run by hand on Kaggle (2026-10-05). Run 2 adds the memory diagnostics and two arms; the six original arms reproduced run 1 exactly.


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
