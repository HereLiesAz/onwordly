# Model design — earned trust, conscious judgment, add-only memory

Status: design. Nothing here is built or claimed. Components are established (see "Prior art"); the combination is the hypothesis.

## Why

Experiments 008–009 failed in one shape: copy, ignore, accept. A gradient update is memory that judges silently — a correction overwrites weights, the old belief is gone, no reasoning ever met the conflict. The shortcuts are what unconscious judgment looks like. The base model also arrives suggestible (broke 114/500 right answers when shown a wrong one).

## Principles

1. **No settled state.** The model acts with confidence and keeps going; nothing is final.
2. **Memory never judges.** Add-only. Contradictions are kept, both sides, linked. Nothing is overwritten or hidden.
3. **Judgment is conscious.** A reasoner meets the contradiction, weighs it, decides — and the decision is itself written to memory, open to reconsideration.
4. **Trust is earned, per domain.** From the track record: when I held, was I right; when I yielded, should I have. Confidence saturates toward a ceiling it never reaches and decays unless re-earned.
5. **Correctors are fallible.** Corrections are weighed against self-trust and the corrector's own track record, discounted for shared ancestry (agreement is not proof).
6. **Trust shapes learning.** Confident-and-wrong → largest update to answer and to trust. Held correctly under challenge → trust grows. Unsure-and-wrong → small update.
7. **Truth comes from outside.** The model never certifies itself; the exact checker is the ground truth trust is earned against.

## Components

| Component | Role | Rule |
| --- | --- | --- |
| Episode store | every attempt, challenge, correction, verifier result | add-only; correction = new record linked to the attempt, never a replacement |
| Divergence marker | deterministic: same task signature, different output | flags, never rules; marked pairs recalled as one unit ranking can't split |
| Reasoner | adjudicates divergences | must see both sides plus verifier evidence |
| Deliberation record | the reasoner's decision | typed; cites what it considered; never condensed, never supersedes |
| Trust ledger | per-domain self-trust; per-corrector reliability | computed from deliberation outcomes vs verifier; weights what the reasoner reads, never filters what memory returns; updates are append-only events |
| Learner | updates the model | step size scaled by confidence × surprise |

## Lessons taken from aive's memory workflow

Assessed against aive (`HereLiesAz/aive`, docs/Memory-layer.md and docs/architecture/). Its separation — clerks file, agents judge — is the right model. Its workflow leaks judgment in places this design must avoid:

- **Supersession hides.** Condensation keeps the medoid verbatim and supersedes the rest; superseded records leave recall. A merge of wording-level disagreements ("chose Postgres" / "chose MySQL" share a claim signature) is a clerk ruling on the claim. Here: supersede only on full coverage; otherwise the summary is an extra index entry and sources stay active.
- **Noticing was forbidden along with judging.** No non-judging component may say "these differ", so contradictions reach the reasoner only by luck of ranking and attention budget. Here: the divergence marker.
- **No handoff contract.** No trigger, timing, or evidence set; conclusions bank as untyped context and can be condensed into one side. Here: the deliberation record.
- **Popularity wins.** Condensation strengthens shared associations, so the majority wording outranks dissent. In a learner this makes the most common mistake the remembered default. Here: condense only verified-identical items.
- **Condensation drifted from its purpose.** It was designed for genuinely repeated events — the same event occurring again — and widened in code to "similar enough". Here: condensation folds true repeats only (identical after normalization), keeping a count, every occurrence's time and provenance. Near-repeats are not merged; they get a divergence marker. Repetition count is itself evidence: a mistake made ten times is a pattern for the reasoner, not a strengthened default.
- **Scale by hiding.** Growth pressure pushes toward more condensation. Here: tier access, keep everything recallable, cap the injected payload instead.
- **Erasure outside the model.** Here: the single audited exception — purge, tombstone, re-derive.

## Path

1. **Fallible-challenge arms in 010/011** (current architecture): turn 1 answer; turn 2 a corrector that is wrong 30% of the time; reply `hold` or `change`; graded reward (right held 1.0, wrong changed correctly 0.6, close 0.4, far 0.3, close held 0.2, far held 0.0, right caved −0.5). Tests whether held-vs-changed tracks being right. Cheapest test of earned trust.
2. **Explicit trust state** on the existing model: per-domain trust estimate from the episode store, fed to the reasoner, scaling learning rate. Ablate against aive-style hand-coded track records (success rate, three-strike circuit breaker).
3. **From-scratch construction**: episode store + divergence marker + reasoner + trust ledger as model components; local, confidence-weighted learning. Baselines: same-size standard network; same construction trained by backprop.

## Prior art (not ours)

Predictive coding and precision-weighted prediction error; equilibrium propagation (Scellier & Bengio 2017); calibration and metacognitive confidence; annotator-reliability models (Dawid–Skene); confidence-weighted and focal losses; sycophancy and challenge robustness (Sharma et al. arXiv:2310.13548; FlipFlop arXiv:2311.08596); reward shaping and improvement bonuses (SCoRe arXiv:2409.12917); noisy-OR evidence accumulation; Hebbian association. A related-work search is required before any component or the combination is called novel.
