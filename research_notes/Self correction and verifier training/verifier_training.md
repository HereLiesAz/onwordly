# Training language models to verify/judge answers (and then repair them)

Verification status: items marked [V] were checked this session (search/fetch of arXiv or proceedings page). Items marked [M] are cited from model knowledge with the arXiv id I believe is correct but were NOT re-fetched this session. The report writer should treat [M] ids as needing a quick check before publication.

## Q1. Core verifier / reward-model / critique-then-revise literature (one-line finding + id)

### Takeaway
Verifier training has moved from discriminative scalar classifiers (Cobbe 2021) to process-level labels (Lightman 2023, Math-Shepherd) and then to generative verifiers that write a verdict as text (GenRM) and critics trained to produce critiques (CriticGPT, CFT). Revision without a reliable verifier is weak, a finding that recurs across papers.

### Cited Findings
- [V] GenRM: Zhang, Hosseini, Bansal, Kazemi, Kumar, Agarwal, "Generative Verifiers: Reward Modeling as Next-Token Prediction", ICLR 2025, arXiv:2408.15240. The verifier is trained with next-token prediction jointly on verification and solution generation. It beats discriminative, DPO and LLM-as-judge verifiers in Best-of-N: 5%→45.3% on algorithmic tasks, 73%→93.4% on GSM8K, and 28%→44.6% on easy-to-hard MATH. — [arXiv](https://arxiv.org/abs/2408.15240v2); [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/214308a2d5e3f83ef9ad2739e1cbc46d-Abstract-Conference.html)
- [V] Critique Fine-Tuning (CFT): Wang, Yue, Chen, arXiv:2501.17703. The model is trained to critique noisy responses, using 50K GPT-4o critiques of WebInstruct, instead of imitating correct answers. It gains 4–10% over SFT on six math benchmarks. — [arXiv](https://arxiv.org/abs/2501.17703v3)
- [V] Zhang et al. (Yunxiang Zhang et al., per ACL listing), "Small Language Models Need Strong Verifiers to Self-Correct Reasoning", ACL Findings 2024, arXiv:2404.17140. Correct solutions are used to guide critiques of a small model's own incorrect responses, then an SFT refiner is trained on them. Gains are large when a GPT-4 verifier decides when to correct, and limited with a weak self-verifier. — [arXiv](https://arxiv.org/abs/2404.17140)
- [V] Chen et al., "Learning to Self-Verify Makes Language Models Better Reasoners", 2026, arXiv:2602.07594 (ICML 2026 poster). Generation and self-verification are asymmetric: training on generation does not improve verification, but learning to self-verify improves generation. Both are trained as two objectives in multi-task RL. — [arXiv](https://arxiv.org/abs/2602.07594); [ICML](https://icml.cc/virtual/2026/poster/61313)
- [V] Ye, Xu, Li, Allen-Zhu, "Physics of Language Models Part 2.2: How to Learn From Mistakes on Grade-School Math Problems", ICLR 2025, arXiv:2408.16293. On synthetic math, pretraining on erroneous steps immediately followed by corrections beats the same amount of error-free data. The paper also studies masking error tokens, how much error to include, and deferring this data to fine-tuning. — [arXiv](https://arxiv.org/abs/2408.16293)
- [M] Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K), arXiv:2110.14168. An outcome verifier trained on correct/incorrect samples plus best-of-N beats fine-tuning alone, with a gain roughly equal to a 30x model-size increase.
- [M] Uesato et al., "Solving math word problems with process- and outcome-based feedback", arXiv:2211.14275. ORM and PRM give similar final-answer accuracy, but process supervision reduces reasoning errors among correct answers.
- [M] Lightman et al., "Let's Verify Step by Step", arXiv:2305.20050. Process supervision (PRM800K human step labels) beats outcome supervision on MATH best-of-N.
- [M] Wang et al., "Math-Shepherd", arXiv:2312.08935. Step labels come automatically from Monte Carlo rollouts (does a completion from this step reach the right answer?), with no human annotation.
- [M] McAleese et al., "LLM Critics Help Catch LLM Bugs" (CriticGPT), arXiv:2407.00215. An RLHF-trained critic for code is trained partly on bugs that humans deliberately inserted ("tampering") and catches more bugs than human contractors. It also produces nitpicks and hallucinated bugs, which is a precision/recall tradeoff.
- [M] Gou et al., "CRITIC", arXiv:2305.11738. Verify-then-correct using external tools (search, code interpreter). Gains depend on tool feedback; self-critique alone helps little.
- [M] Weng et al., "Large Language Models are Better Reasoners with Self-Verification", arXiv:2212.09561. Backward verification (re-deriving conditions from the answer) is used to rerank candidates.
- [M] Welleck et al., "Generating Sequences by Learning to Self-Correct", arXiv:2211.00053. A separate corrector model is trained on (worse→better) pairs.
- [M] Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet", arXiv:2310.01798. Without oracle feedback, intrinsic self-correction often hurts, because models flip correct answers to wrong ones.
- [M] Tyen et al., "LLMs cannot find reasoning errors, but can correct them given the error location", arXiv:2311.08516. Locating the error is the bottleneck; repair works once the location is given.
- [M] Kumar et al., "SCoRe: Training Language Models to Self-Correct via RL", arXiv:2409.12917. SFT on self-correction traces collapses (the model does not change its answer, or makes minimal edits) because of distribution mismatch. Multi-turn online RL with reward shaping fixes this.
- [M] Zheng et al., "ProcessBench", arXiv:2412.06559. A benchmark for locating the first erroneous step. Many PRMs generalize poorly to it.
- [V] Han et al., "Small Language Model Can Self-correct", arXiv:2401.07301 (Intrinsic Self-Correction via partial answer masking, PAM). Reports that an imbalance between bad and good cases did not hurt when PAM is used. — [arXiv](https://arxiv.org/pdf/2401.07301)

### Inferences
- The strongest pattern for a combined "judge then fix" model: the verifier is the bottleneck, so the judging head needs its own exact signal (programmatic checks) rather than self-assessment.

### Gaps
- One-line findings for the [M] items were not re-checked against their abstracts this session.

## Q2. Single model, combined "judge right/wrong → corrected answer", proposals balanced 50/50?

### Takeaway
I found no paper that explicitly trains one model on the exact format "verdict + corrected answer" with a stated 50/50 balance of correct and incorrect proposals. The closest work trains verification and generation jointly (GenRM, 2602.07594), or trains verify-then-revise (SCoRe, 2404.17140, ISC). Balance is usually mentioned only as preprocessing.

### Cited Findings
- GenRM trains verification and solution generation jointly in one model. — [arXiv](https://arxiv.org/abs/2408.15240v2)
- 2602.07594 builds its self-verification data with "data balancing, filtering, and diversity-aware sampling". The exact ratio was not confirmed. — [arXiv](https://arxiv.org/html/2602.07594v1)
- Wrong proposals come from different sources in different papers:
  - The model's own sampled errors: Cobbe [M], GenRM, and 2404.17140 ([arXiv](https://arxiv.org/abs/2404.17140)).
  - Deliberately inserted bugs: CriticGPT tampering [M].
  - Synthetic injected errors in synthetic math: Physics of LMs 2.2 ([arXiv](https://arxiv.org/abs/2408.16293)).
  - Third-party noisy responses: CFT on WebInstruct ([arXiv](https://arxiv.org/abs/2501.17703v3)).
- Physics of LMs 2.2 found that synthetic "retry" errors help, while the authors (per abstract) also examine how such data should be prepared. I recall it reporting that easy-to-generate fake mistakes are effective, but this is [M]-level detail.
- [M] SCoRe argues that off-policy correction data, meaning errors not produced by the current model, causes a train/test mismatch. That is evidence that the model's own errors matter for the repair step.

### Inferences
- A 50/50 balanced, exactly verified judge+repair format on arithmetic appears to be an open, testable design point. Per AGENTS.md, it should be framed as a combination of known techniques, not an invention.
- A recommended ablation: wrong proposals sampled from the model itself vs. synthetically perturbed, crossed with a balanced vs. natural ratio.

### Gaps
- No direct head-to-head comparison of on-policy vs. synthetic wrong proposals for verifier accuracy was verified this session.

## Q3. Shortcut and bias problems

### Takeaway
The documented failure modes are:
- Verdict bias toward the majority label.
- Self-correction policies that never change the answer, or that flip correct answers.
- Position bias in LLM judges.
- Over-optimization or hacking of learned reward models.
- Critics that hallucinate bugs.

### Cited Findings
- [V] 2404.17140: when a weak self-verifier decides whether to correct, it limits the gains. — [arXiv](https://arxiv.org/abs/2404.17140)
- [M] Huang et al. 2310.01798: self-correction without oracle feedback changes correct answers to wrong ones (an over-reject or over-revise bias).
- [M] SCoRe 2409.12917: SFT-trained correctors learn to copy the first attempt or make minimal edits (copy-the-proposal collapse).
- [M] Wang et al., "Large Language Models are not Fair Evaluators", arXiv:2305.17926: LLM judges show position bias in pairwise comparison.
- [M] Gao, Schulman, Hilton, "Scaling Laws for Reward Model Overoptimization", arXiv:2210.10760: policy optimization against a proxy RM raises the proxy score while the gold score falls.
- [M] CriticGPT 2407.00215: critics trade off hallucinated or nitpick bugs against comprehensiveness.

### Inferences
- With imbalanced data, a judge can reach high accuracy by always predicting the majority label. A 50/50 balance and per-class (balanced) accuracy prevent this. Exact checkers remove reward hacking of the verdict label itself.

### Gaps
- I did not verify a paper that quantifies always-accept vs. always-reject rates as a function of training label ratio.

## Q4. Small models (<1B) / toy arithmetic

### Takeaway
Physics of LMs 2.2 (small GPT-2-scale models on synthetic GSM-like data) is the closest controlled toy setting. The GenRM algorithmic tasks are another. Most other work uses models of 7B or more.

### Cited Findings
- Physics of LMs 2.2 uses a synthetic math dataset with error-correction pretraining. — [arXiv](https://arxiv.org/abs/2408.16293)
- GenRM reports results on algorithmic tasks (5%→45.3%), alongside GSM8K. — [arXiv](https://arxiv.org/abs/2408.15240v2)
- "Small" in 2404.17140 means ≤13B, not <1B. — [arXiv](https://arxiv.org/abs/2404.17140)

### Gaps
- Exact model sizes in Physics 2.2 were not confirmed this session (believed to be GPT-2-scale, ~100M parameters, [M]).
- No sub-1B judge-then-repair study on pure arithmetic was found.
