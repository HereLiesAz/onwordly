# Experiment 008 results

**Status:** one real-model seed (training seed 3303, Qwen2.5-0.5B, Kaggle T4 x2, 2026-10-05). Generated report below, unedited, followed by a provisional reading.



## Final evaluation

| Regime | Held-out | Prompt transfer only | Held-out + prompt transfer | Out-of-range | Train tokens | Examples | Total generation calls | Wall seconds | Peak GiB | Δ accuracy / 1M train tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 80.10% | 56.50% | 57.50% | 56.83% | 99,991 | 5,337 | 12,117 | 3,385.48 | 4.66 | 8.06 |
| corrective-own | 73.50% | 53.17% | 51.33% | 40.67% | 99,984 | 3,506 | 10,286 | 2,621.11 | 4.67 | 7.50 |
| corrective-synthetic | 78.70% | 55.83% | 57.00% | 50.50% | 99,990 | 3,580 | 10,360 | 2,677.66 | 4.67 | 8.00 |
| error-focused | 75.10% | 48.83% | 48.00% | 41.00% | 99,985 | 4,686 | 11,466 | 3,261.97 | 4.66 | 7.56 |

## Against the untrained baseline

Step 0 is the checkpoint evaluation before any training (the untrained
baseline on the checkpoint subset). Lenient = answer anywhere in the
response; diagnostic only, never used for training.

| Regime | Warm-up tokens | Step-0 exact | Step-0 lenient | Final held-out exact | Final held-out lenient |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 0 | 0.00% | 5.56% | 80.10% | 80.10% |
| corrective-own | 0 | 0.00% | 5.56% | 73.50% | 73.50% |
| corrective-synthetic | 0 | 0.00% | 5.56% | 78.70% | 78.70% |
| error-focused | 0 | 0.00% | 5.56% | 75.10% | 75.10% |

## Corrective language game (Experiment 008)

Correction: shown a synthetic wrong answer. Confirmation: shown the right answer.
Self-correction: answer, then see your own answer in a corrective prompt.

| Regime | Correction | Confirmation | Self-correction pass 1 | Pass 2 | Fixed | Broken | Corrective / confirm tasks trained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 60.80% | 59.40% | 81.40% | 58.80% | 1 | 114 | — |
| corrective-own | 74.80% | 74.60% | 74.60% | 74.40% | 1 | 2 | 736 / 421 |
| corrective-synthetic | 4.60% | 88.60% | 79.20% | 78.60% | 0 | 3 | 660 / 445 |
| error-focused | 49.20% | 41.00% | 75.60% | 40.80% | 2 | 176 | — |

## Tokens to checkpoint threshold

| Regime | 70% | 80% | 90% | 95% |
| --- | ---: | ---: | ---: | ---: |
| static | 10,006 | 60,012 | — | — |
| corrective-own | 60,036 | — | — | — |
| corrective-synthetic | 20,019 | 50,005 | — | — |
| error-focused | 70,012 | — | — | — |

## Pre-update accuracy on training moves

From the regime JSON (`bucket_stats`): how often the model already got each kind of move right before training on it.

| Regime | Arithmetic | Confirm moves | Correct moves |
| --- | ---: | ---: | ---: |
| corrective-own | 68.5% (2,350) | 99.8% (420) | 16.8% (736) |
| corrective-synthetic | 73.2% (2,475) | 96.4% (445) | 13.9% (660) |

## Reading (single seed; provisional)

- **Neither corrective arm learned to correct.** In two-pass self-correction, pass 2 never beats pass 1 (fixed 0–1, broken 2–3). Correction accuracy during training stayed at 14–17%.
- **corrective-own learned to ignore the shown answer.** Correction (74.8%), confirmation (74.6%) and plain arithmetic (74.6%) are the same number: it recomputes from scratch whatever it is shown. Its own early errors are rarely close to the truth, so the shown answer carries no usable signal and the cheapest rule is to discard it.
- **corrective-synthetic learned to copy the shown answer.** Confirmation 88.6% (above its own 79.2% arithmetic) but correction 4.6%: shown a near miss, it repeats it. Synthetic errors differ from the truth by one or two digits, so copying is almost right token by token and the training loss rewards it; exact verification does not.
- **Untrained regimes are suggestible.** Static and error-focused were never trained on the corrective format; shown their own *correct* answer and told to fix it if wrong, they change it and break 114 and 176 of 500 answers (pass 1 81.4% → pass 2 58.8% for static).
- **Arithmetic cost.** Corrective moves took ~1,100 of each arm's ~3,500 examples; held-out arithmetic fell 1.4 points (synthetic) and 6.6 points (own) against static.

**What this says about the game.** Both arms learned a rule that wins the surface game — "ignore it" or "copy it" — rather than the practice of checking an answer. Nothing in a corrective move forces a judgement: the target never depends on noticing whether the shown answer is right. A corrective game that teaches correction probably needs an explicit verdict move (`right` / `wrong`, exactly verifiable) before or instead of the repair, balanced so that neither copying nor ignoring scores above chance.

Single seed. Repeat before treating any of this as stable.
