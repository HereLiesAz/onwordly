# Experiment 009 results

**Status:** one real-model seed (Qwen2.5-0.5B, Kaggle T4 x2, 2026-10-05). Generated report below, unedited, followed by a provisional reading.


## Final evaluation

| Regime | Held-out | Prompt transfer only | Held-out + prompt transfer | Out-of-range | Train tokens | Examples | Total generation calls | Wall seconds | Peak GiB | Δ accuracy / 1M train tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 80.10% | 56.50% | 57.50% | 56.83% | 99,991 | 5,337 | 12,117 | 3,296.62 | 4.66 | 8.06 |
| verdict-synthetic | 75.80% | 56.33% | 56.00% | 38.17% | 99,963 | 3,702 | 10,482 | 2,541.98 | 4.68 | 7.50 |
| verdict-mixed | 74.60% | 54.83% | 53.67% | 46.00% | 99,997 | 3,716 | 10,496 | 2,498.95 | 4.68 | 7.72 |
| corrective-synthetic | 78.70% | 55.83% | 57.00% | 50.50% | 99,990 | 3,580 | 10,360 | 2,563.05 | 4.67 | 8.00 |

## Against the untrained baseline

Step 0 is the checkpoint evaluation before any training (the untrained
baseline on the checkpoint subset). Lenient = answer anywhere in the
response; diagnostic only, never used for training.

| Regime | Warm-up tokens | Step-0 exact | Step-0 lenient | Final held-out exact | Final held-out lenient |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 0 | 0.00% | 5.56% | 80.10% | 80.10% |
| verdict-synthetic | 0 | 0.00% | 5.56% | 75.80% | 75.80% |
| verdict-mixed | 0 | 0.00% | 5.56% | 74.60% | 74.60% |
| corrective-synthetic | 0 | 0.00% | 5.56% | 78.70% | 78.70% |

## Verdict game (Experiment 009)

Right shown: reply `right`. Wrong shown (synthetic near miss): reply `wrong: <answer>`;
judgement counts the verdict alone, repair also the number. Always answering `right`
scores 50% balanced. Self-check: answer, judge your own answer, keep or repair it.

| Regime | Right shown | Wrong: judgement | Wrong: repair | Balanced verdict | Self-check pass 1 | Final | Fixed | Broken | Verdict moves trained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 0.00% | 0.00% | 0.00% | 0.00% | 81.40% | 81.40% | 0 | 0 | — |
| verdict-synthetic | 90.40% | 13.40% | 13.40% | 51.90% | 75.80% | 76.00% | 1 | 0 | 414 right / 0 own / 475 synthetic |
| verdict-mixed | 100.00% | 0.00% | 0.00% | 50.00% | 75.80% | 75.80% | 0 | 0 | 423 right / 129 own / 337 synthetic |
| corrective-synthetic | 0.00% | 0.00% | 0.00% | 0.00% | 79.20% | 79.20% | 0 | 0 | — |

## Tokens to checkpoint threshold

| Regime | 70% | 80% | 90% | 95% |
| --- | ---: | ---: | ---: | ---: |
| static | 10,006 | 60,012 | — | — |
| verdict-synthetic | 10,003 | — | — | — |
| verdict-mixed | 20,010 | — | — | — |
| corrective-synthetic | 20,019 | 50,005 | — | — |

## Interpretation

Do not infer a training advantage from this run alone. Repeat across seeds and compare both accuracy and total resource cost.


## Provisional reading (one seed)

The verdict move collapsed to acceptance.

- **verdict-mixed:** `right` on every proposal — 100% on right, 0% on wrong. Pure always-accept; balanced accuracy 50.0%, exactly chance.
- **verdict-synthetic:** 90.4% right-shown, 13.4% wrong caught, every catch repaired correctly. Balanced 51.9%, barely above chance.
- **Self-check:** 1 fix, 0 breaks in 1,000 second passes. The model does not doubt its own answers.
- **Cost:** both verdict arms lose held-out accuracy (75.8 / 74.6 vs static 80.1) and out-of-range generalization (38.2 / 46.0 vs 56.8) at the same token budget.

Training labels were near-balanced (414 right vs 475 synthetic; 423 right vs 466 wrong), so the collapse is not a label-ratio artefact. When the model does reject, its repair is always correct: solving is not the bottleneck, rejecting is.

Third shortcut. 008 found copy (synthetic) and ignore (own); 009 finds accept. Under SFT at 0.5B and 100k tokens, the model takes whichever move is cheapest to imitate. Consistent with SCoRe's "no edits" collapse (arXiv:2409.12917) and with Zhang et al. (arXiv:2404.17140): small models do not learn to verify themselves from SFT.

Untested explanations, in order of cheapness:
1. Too few verdict examples (~890 per arm) — check the training curve and a larger verdict budget.
2. Near-miss wrong answers are too close to discriminate without recomputing — the model must solve before judging; the format lets it judge first.
3. SFT cannot install conditional rejection here at all — on-policy RL with the exact checker as reward is the next lever.

Not a finding until repeated across seeds.
