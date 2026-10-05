# Model design — earned trust, conscious judgment, add-only memory

Status: design, built as Experiment 000 (`experiments/000-onwordly-learner/`); nothing claimed. Components are established (see "Prior art"); the combination is the hypothesis.

## Why

The archived Experiments 008–009 (`docs/findings-llm-phase.md`) failed in one shape: copy, ignore, accept. A gradient update is memory that judges silently — a correction overwrites weights, the old belief is gone, no reasoning ever met the conflict. The shortcuts are what unconscious judgment looks like. The base model also arrives suggestible (broke 114/500 right answers when shown a wrong one).

## Principles

1. **No settled state.** The model acts with confidence and keeps going; nothing is final.
2. **Memory never judges.** Add-only. Contradictions are kept, both sides, linked. Nothing is overwritten or hidden.
3. **Judgment is conscious.** A reasoner meets the contradiction, weighs it, decides — and the decision is itself written to memory, open to reconsideration.
4. **Trust is earned, per domain.** From the track record: when I held, was I right; when I yielded, should I have. Confidence saturates toward a ceiling it never reaches and decays unless re-earned.
5. **Correctors are fallible.** Corrections are weighed against self-trust and the corrector's own track record, discounted for shared ancestry (agreement is not proof).
6. **Trust shapes learning.** Confident-and-wrong → largest update to answer and to trust. Held correctly under challenge → trust grows. Unsure-and-wrong → small update.
7. **Truth comes from outside.** The model never certifies itself; the exact checker is the ground truth trust is earned against.

## Consolidation rewrites; storage does not forget

Consolidation is rewriting: the current memory is replaced by a consolidated version, and its sources become history — never deleted, reachable by lineage, faded from default recall (git-like: new commit, immutable past). Similar memories consolidate automatically. Contrasting memories consolidate only after conscious judgment: the model investigates (memories plus other evidence) and writes a deliberation naming the memory it judged correct; later consolidation absorbs that deliberation into the chosen memory, and the other side becomes history, linked to the reason. No deliberation, no consolidation of a contrast. Once resolved, the chosen memory is recalled as a single memory; its contradiction history fades with use — each access strengthens the memory and surfaces less of the old dispute, until only a link remains. Nothing is re-adjudicated on every recall; only new contrasting evidence reopens it.

Each current memory has a size budget (not the store as a whole): every rewrite must be smaller than the version before it, following an S-curve from the original size down to a floor that is a fixed fraction of the original — very little loss early in the memory's life and near the floor, the steepest loss just past the middle. When memories combine (consolidation, a resolved contradiction, new information folded in), the curve is re-based from every contributor, the newest included: the ceiling is the average of the new memory's size and the summary-so-far's size, weighted by each one's weight (use, recency, salience, deliberation citations); the floor is the same fraction of that ceiling; the life position is the same weighted average — so both sides of a contradiction shape the result. The rewrite still never reaches the size of the version before it. A memory never grows; new information that does not fit becomes a separate, linked memory. Each rewrite is degradation by summary, weighted so that detail that is used often, recently, salient, or cited by deliberations survives and rarely used detail compresses first. Higher levels of the summary tree get tighter budgets — gist toward the root. The full pre-summary version becomes history.

Time follows the same path: a raw memory keeps its exact timestamp; once consolidated, its current version carries a time range spanning its sources, widening with each consolidation. Exact events remain available in history — a flaw of human memory deliberately not copied — while the compression that makes routine forgettable and the unusual memorable is kept. Borrowed from systems consolidation, trace transformation and reconsolidation (Nadel & Moscovitch†; Nader et al. 2000†).

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
- **Similarity conflated frame with filler.** "Similar enough" is right — close to how people remember — but surface similarity treats "I chose Postgres" and "I chose MySQL" as near-identical (same frame; embeddings place co-hyponyms and antonyms close). To a person they are opposites. Here, similarity is two-part:

  | Frame (what it's about) | Filler (what it says) | Relation | Action |
  | --- | --- | --- | --- |
  | same | same or synonymous | similar | condense; keep count, times, provenance |
  | same | different | **contrast** | never merge; divergence marker; recall both sides together |
  | different | any | unrelated / associated | link at most |

  **Variant register.** The varying filler — "Postgres", "MySQL", and every filler like it the model encounters — is kept in a list per frame. Each entry: the filler; **age** (when it was encountered, and when it was said to hold — record time and event time are different clocks); **context** (situation, project, task); **subject** (who or what the frame is about); plus source and occurrence count. Nothing in the list is ranked as correct. The list *is* the contrast: one frame, many answers, each with its circumstances, for the reasoner to read. For the learner, a task frame's register holds its own answers, challenges, corrections and verifier outcomes — per-frame trust is read off it.

  Detection is structural, not a truth judgment: align, diff; shared structure + differing content words (entities, values, choices) = contrast; an alias table keeps synonyms ("Postgres"/"PostgreSQL") from counting. The clerk says "same question, different answer"; the reasoner decides. Repetition count from condensation is evidence: a mistake made ten times is a pattern for the reasoner, not a strengthened default.
- **Scale by hiding.** Growth pressure pushes toward more condensation. Here: tier access, keep everything recallable, cap the injected payload instead.
- **Erasure outside the model.** Here: the single audited exception — purge, tombstone, re-derive.

## Path

1. **Fallible-challenge arms in 010/011** (archived, built but never run; branch `archive/llm-phase`): an LLM-scale test of earned trust with a fallible corrector and graded reward. Their game and reward carry over to step 3.
2. **Explicit trust state on the existing model** (not pursued): superseded by step 3; the aive-style hand-coded rule survives as the `handcoded` baseline.
3. **From-scratch construction = Experiment 000** (current): episode store + variant register + reasoner + trust ledger as model components; confidence × surprise update weighting. Baselines: parameter-matched plain network; hand-coded trust rule; ablations of memory, trust and update rule.

## Borrow map

Borrow wherever possible. † = from memory, verify before citing.

| Component | Borrow from |
| --- | --- |
| Add-only episode store | Event sourcing / append-only logs; immutable databases (Datomic†, XTDB†) |
| Two clocks (record time, event time) | Bitemporal data modelling (SQL:2011 temporal tables); Zep/Graphiti bi-temporal knowledge graph† |
| Frame / filler | Frame semantics (Fillmore; FrameNet); semantic role labelling; Open IE subject–relation–object triples |
| Synonyms in fillers | Entity linking and canonicalisation; alias tables (e.g. Wikidata aliases) |
| Contrast detection | NLI contradiction classifiers, used only to *flag*; minimal-pair alignment |
| Keeping contradictions alive | Assumption-based truth maintenance (de Kleer's ATMS†): holds every consistent set of assumptions at once, nothing discarded |
| Conscious adjudication record | Argumentation frameworks (Dung†); design rationale / decision logs |
| Corrector reliability | Dawid–Skene; truth discovery with source dependence (Dong et al. 2009†) |
| Per-domain self-trust | Beta-Bernoulli reputation (Jøsang†); calibration |
| Update size from confidence × surprise | Precision-weighted prediction error; focal loss. Note: Kalman-style updates shrink when confident — the opposite of "confident-and-wrong learns most"; surprise must dominate |
| Fallible challenge | AI safety via debate (Irving et al. 2018†); sycophancy / challenge robustness (arXiv:2310.13548, 2311.08596) |
| Graded reward | Reward shaping; SCoRe improvement bonus (arXiv:2409.12917) |

What is left to build is the wiring between these, under exact verification and matched budgets.

## Prior art (not ours)

Predictive coding and precision-weighted prediction error; equilibrium propagation (Scellier & Bengio 2017); calibration and metacognitive confidence; annotator-reliability models (Dawid–Skene); confidence-weighted and focal losses; sycophancy and challenge robustness (Sharma et al. arXiv:2310.13548; FlipFlop arXiv:2311.08596); reward shaping and improvement bonuses (SCoRe arXiv:2409.12917); noisy-OR evidence accumulation; Hebbian association. A related-work search is required before any component or the combination is called novel.
