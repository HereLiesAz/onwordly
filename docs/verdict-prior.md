# Verdict prior diagnostic

**Status:** implemented and CPU-tested (fake adapter; tiny-gpt2 end to end); no real-model run.

## Question

Experiment 009 collapsed to accepting proposals (`right` for nearly everything).
Was that already the untrained model's preference? If the base model prefers
`right` about as much for a wrong proposal as for a correct one, 009's SFT
started from an acceptance prior with no discrimination to build on. No
training happens here.

## What it does

`src/onwordly/diagnostics/verdict_prior.py` loads an untrained model and takes
the first `n` held-out problems of 009 (the 009 manifest, eval partition,
`evaluation_seed`). The near misses are the ones 009's verdict evaluation shows,
drawn from the same seed in the same order. After the 009 verdict prompt, it
scores the summed log-probability of three complete replies (plus EOS, using the
same tokenisation as training):

- `right`;
- `wrong`, the verdict word alone. This is not a valid 009 reply; it isolates the word;
- `wrong: <correct answer>`, the full valid reply for a wrong proposal.

It does this under three conditions:

1. **correct_shown**: the true answer is proposed;
2. **near_miss_shown**: a synthetic near miss is proposed;
3. **own_wrong_shown**: the model's own greedy answer is proposed, only when that answer is parseable and wrong.

Margin = logp(`right`) − logp(wrong reply). Positive means the model prefers `right`.

## How to read it

- **Prefers right** is the fraction of problems with a positive margin. **Mean margin** is in nats.
- **Discrimination** asks whether the margin depends on whether the proposal is true:
  - **AUROC** is P(margin with the correct answer shown > margin with a wrong one shown), with ties counted half, over all pairs;
  - **paired higher** is the same comparison on the same problem;
  - **mean margin difference** is the average paired gap.
- The *full* margin has a length bias. `wrong: <n>` has more tokens than `right`, so its log-probability is lower, and "prefers right (full)" is close to 100% for almost any model. Read the *full* margin only through discrimination: on a given problem the continuation is identical across conditions, so the length bias cancels in the paired difference. The *bare* margin compares a single word with a single word and is the better read of the absolute lean.

| Pattern | Reading |
| --- | --- |
| High prefers-right in every condition, AUROC ≈ 0.5 | An acceptance prior with no discrimination. 009's collapse is the starting point, not something SFT created. Training must build the signal from nothing. |
| High prefers-right, AUROC clearly > 0.5 | The model already "notices" wrong proposals, but the notice is too weak to flip the verdict. A small push (budget, RL) may be enough. |
| AUROC near 0.5 for near misses but higher for own wrong answers (or the reverse) | Discrimination depends on where the error comes from. Read alongside 009's synthetic-vs-mixed arms. |
| Instruct model discriminates, base does not | The instruction-tuned prior helps. That is a candidate starting checkpoint, as a separate arm, never mixed into existing comparisons. |

`n` = 200 gives an AUROC standard error of roughly ±0.03. Treat differences below about 0.05 as noise.

## Running it

```
python -m onwordly.diagnostics.verdict_prior --n 200 --out results/verdict-prior \
    --model Qwen/Qwen2.5-0.5B --model Qwen/Qwen2.5-0.5B-Instruct
```

Kaggle notebook plan (one GPU, a few minutes per model):

```
experiment: prior
models: Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-0.5B-Instruct
n: 200
```

`models` and `n` are optional (defaults: `Qwen/Qwen2.5-0.5B`, 200). `manifest:` overrides the 009 manifest. Results go to `results/verdict-prior/` (`verdict-prior.json` with every per-problem score, and `RESULTS.md`) and are published like other runs. Both models use the raw training prompt format, with no chat template, so the instruct row measures the checkpoint, not the template.

Cost: three scored continuations per proposal (one forward pass each) and one greedy generation per problem. That is about 1,000 forward passes per model at n = 200.
