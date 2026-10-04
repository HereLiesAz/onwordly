# Experiment 001 results

**Status:** one real-model seed (Kaggle run 16, training seed 3303, Qwen2.5-0.5B,
two T4s, full manifest, no warm-up). Not yet evidence of a training advantage:
repeated seeds are required. Tables below are the generated `RESULTS.md` from
that run, unedited.

## Final evaluation

| Regime | Held-out | Prompt transfer only | Held-out + prompt transfer | Out-of-range | Train tokens | Examples | Total generation calls | Wall seconds | Peak GiB | Δ accuracy / 1M train tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 80.10% | 56.50% | 57.50% | 56.83% | 99,991 | 5,337 | 10,117 | 2,907.80 | 4.66 | 8.06 |
| adaptive | 80.40% | 60.33% | 60.50% | 55.33% | 99,985 | 4,750 | 9,530 | 2,894.78 | 4.66 | 8.06 |
| error-focused | 75.10% | 48.83% | 48.00% | 41.00% | 99,985 | 4,686 | 9,466 | 2,859.21 | 4.66 | 7.56 |

## Against the untrained baseline

| Regime | Warm-up tokens | Step-0 exact | Step-0 lenient | Final held-out exact | Final held-out lenient |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 0 | 0.00% | 5.56% | 80.10% | 80.10% |
| adaptive | 0 | 0.00% | 5.56% | 80.40% | 80.40% |
| error-focused | 0 | 0.00% | 5.56% | 75.10% | 75.10% |

## Tokens to checkpoint threshold

| Regime | 70% | 80% | 90% | 95% |
| --- | ---: | ---: | ---: | ---: |
| static | 10,006 | 60,012 | — | — |
| adaptive | 10,013 | 80,019 | — | — |
| error-focused | 70,012 | — | — | — |

## Reading (single seed; provisional)

- All regimes go from 0% to 75–80% held-out exact. Final lenient equals final
  exact, so the answer format is fully learned and the remaining errors are wrong
  answers, not buried right ones.
- Static and adaptive pass 70% by ~10k tokens: format is acquired almost
  immediately, so the regime comparison is decided over the remaining ~90k tokens.
- Static ≈ adaptive on held-out (80.1% vs 80.4%; ±~1.3 points standard error at
  n = 1000). Adaptive is ~3–4 points ahead on prompt transfer, within what one
  seed can produce by chance.
- Error-focused is behind everywhere (−5 held-out, −8 to −16 on transfer and
  out-of-range) and needed 70k tokens to reach 70%. That is the largest effect in
  the run and the first thing to test across seeds and in Experiment 002's
  ablation (error-focused-uniform vs error-focused-adaptive).
- Adaptive and error-focused buy fewer examples (4,750 and 4,686 vs 5,337) for the
  same tokens because they sample longer, harder tasks.
- Step-0 lenient is 5.56% here versus 65% in `docs/baseline.md` because this run
  generates at most 16 new tokens; the untrained model's explanations are cut off.
- Checkpoints evaluate the first 180 held-out items, so tokens-to-threshold is not
  independent of the final held-out score.
