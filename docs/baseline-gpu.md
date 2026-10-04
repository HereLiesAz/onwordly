# Untrained baseline on GPU: base vs instruct, raw vs chat template

Colab T4, `experiment: baseline` plan, 200 sampled rows per split (seed 0), greedy,
32 new tokens. Exact = experiment parser; lenient = answer appears anywhere
(diagnostic only). Full per-variant reports were produced by
`onwordly.diagnostics.baseline`; this file keeps the summary.

| Domain / split | Base raw (exact / lenient) | Base chat | Instruct raw | Instruct chat |
| --- | --- | --- | --- | --- |
| 001 heldout | 0.00 / 0.65 | 0.00 / 0.10 | 0.00 / 0.34 | 0.00 / 0.78 |
| 001 out-of-range | 0.00 / 0.00 | 0.00 / 0.07 | 0.00 / 0.00 | 0.00 / 0.34 |
| 001 prompt-transfer-only | 0.00 / 0.26 | 0.00 / 0.14 | 0.00 / 0.25 | **0.23** / 0.70 |
| 001 withheld-prompts | 0.00 / 0.34 | 0.00 / 0.15 | 0.00 / 0.17 | **0.21** / 0.69 |
| 003 (all splits) | 0.00 / ≤0.04 | 0.00 / ≤0.04 | 0.00 / ≤0.04 | 0.00 / ≤0.01 |
| 004 (all splits) | ≤0.01 / ≤0.06 | 0.00 / ≤0.06 | 0.00 / ≤0.05 | ≤0.01 / ≤0.07 |
| 005 (all splits) | 0.00 / 0.10–0.18 | 0.00 / 0.05–0.10 | 0.00 / 0.00–0.04 | 0.00 / 0.10 |
| 006 (all splits) | 0.00 / 0.39–0.64 | 0.00 / 0.24–0.77 | 0.00 / 0.24–0.34 | 0.00 / 0.04–0.22 |

## Reading

- **Instruct + chat template is the only variant with any exact score**, and only on
  arithmetic prompt styles other than the canonical one (0.21–0.23). On the
  canonical prompt it still answers in a sentence ("The result of 915 − 435 is …"),
  so exact stays 0.00 while lenient reaches 0.78.
- **Base model + chat template degenerates** (repeated junk tokens). Never use it.
- **Instruct without its template is worse than base raw.** If instruct is used, use
  the template.
- **Symbolic, string and program tasks are a real skill gap in every variant**
  (lenient ≤ 0.18). No model or prompt choice rescues them; training must.
  Instruct + chat collapses program answers to a single value (e.g. `-6`).
- **Logic lenient is not capability.** Responses that mention both `true` and
  `false` match either answer; treat 006 lenient as noise.
- **Out-of-range arithmetic is truncated.** Lenient 0.34 for instruct + chat versus
  0.00 elsewhere is consistent with long explanations hitting the 32-token limit.
