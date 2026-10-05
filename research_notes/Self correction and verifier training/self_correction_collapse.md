# LLM self-correction training and its failure modes

Scope note: 13 tool calls. Every entry below was checked against arXiv, a proceedings page or the ACL Anthology. Most findings come from abstracts or search summaries, and those are marked "abstract-level". The SCoRe numbers come from the arXiv HTML full text, read through a summarizing fetch. Check them against the PDF before quoting them in a paper.

## Q1. What did SCoRe report about SFT on correction traces?

### Takeaway
SCoRe (arXiv 2409.12917; Kumar et al., Google DeepMind) names two failures of offline SFT on correction traces. The first is distribution mismatch: the training data holds the data-collection policy's mistakes, not the model's own. The second is behavior collapse: the SFT models learn to make almost no edits, so second-attempt accuracy mostly equals first-attempt accuracy. The paper's answer is two-stage multi-turn RL.

### Cited Findings
- Kumar, Zhuang, Agarwal, Su, Co-Reyes, Singh, Baumli, Iqbal, Bishop, Roelofs, Zhang, McKinney, Shrivastava, Paduraru, Tucker, Precup, Behbahani, Faust, "Training Language Models to Self-Correct via Reinforcement Learning", arXiv **2409.12917** (2024). The ICLR 2025 venue comes from my memory and I did not check it here. — [arXiv](https://arxiv.org/abs/2409.12917)
- Abstract: SFT on model-generated correction traces suffers from "distribution mismatch between mistakes made by the data-collection policy and the model's own responses" and "behavior collapse" toward "a certain mode of correction behavior that is often not effective at self-correction on test problems." — [arXiv](https://arxiv.org/abs/2409.12917)
- Full text: models trained with STaR and with Pair-SFT (Welleck-style) "are overly conservative and often make no edits at all", while the base model "sometimes makes substantially large edits". This is the minimal-edit / copy-the-first-answer collapse. — [arXiv HTML](https://arxiv.org/html/2409.12917)
- MATH500 with Gemini 1.5 Flash. Δ(t1,t2) is the accuracy change from attempt 1 to attempt 2; Δ(i→c) is the share of problems flipped from incorrect to correct, and Δ(c→i) the share flipped from correct to incorrect. These definitions are my reading of the paper's notation.
  - STaR: Δ(t1,t2) = 0.4%; Δ(i→c) = 2.6%; Δ(c→i) = 2.2%.
  - Pair-SFT: Δ(t1,t2) = 1.8%; Δ(i→c) = 5.4%; Δ(c→i) = 3.6%.

  So SFT raised accuracy on both attempts but did almost no real self-correction. — [arXiv HTML](https://arxiv.org/html/2409.12917)
- Distribution shift: training-time correction accuracy improves, but self-correction degrades when the first attempts are the model's own rather than fixed ones. — [arXiv HTML](https://arxiv.org/html/2409.12917)
- The fix has two stages:
  - Stage I optimizes the second-attempt reward and applies a KL penalty only to the first attempt. This keeps first-turn answers from drifting, so the policy cannot collapse to the "direct strategy" of just producing the best first answer.
  - Stage II runs multi-turn RL with reward shaping that rewards progress between attempts.

  Reported gains in self-correction: +15.6% on MATH and +9.1% on HumanEval. — [arXiv](https://arxiv.org/abs/2409.12917)

### Inferences
- For Onwordly, the metrics to log are Δ(i→c), Δ(c→i), the no-edit rate and edit distance, not just second-attempt accuracy. A rise in final accuracy can come entirely from a better first attempt.

### Gaps
- I did not extract the exact edit-distance histogram values or the Gemini 1.0 Pro numbers.

## Q2. Huang et al. 2023: intrinsic self-correction

### Takeaway
Huang et al. find that without external feedback, LLMs struggle to self-correct reasoning, and accuracy can drop after self-correction because right answers get changed to wrong ones.

### Cited Findings
- Huang, Chen, Mishra, Zheng, Yu, Song, Zhou, "Large Language Models Cannot Self-Correct Reasoning Yet", arXiv **2310.01798**, ICLR 2024. — [arXiv](https://arxiv.org/abs/2310.01798)
- Abstract: "LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction." — [arXiv](https://arxiv.org/abs/2310.01798)

### Inferences
- The degradation is the Δ(c→i) failure mode. An exact verifier removes the need for intrinsic judgment, but then the setup counts as externally verified correction, not intrinsic self-correction.

### Gaps
- I did not extract the per-benchmark correct→incorrect counts (GSM8K, CommonSenseQA, HotpotQA; GPT-3.5/GPT-4). Huang et al. also argue that earlier reported gains came from oracle stopping labels. I remember this but did not verify it in this pass.

## Q3. Other self-correction work, one line each

### Takeaway
The literature agrees on one pattern: self-correction works with reliable external feedback or with large-scale and on-policy training, and fails with prompting alone.

### Cited Findings
- **Self-Refine.** Madaan et al., arXiv 2303.17651 (2023). No training: the same LLM drafts, gives feedback on the draft and refines it. Reports about 20% absolute average improvement, mainly on generation tasks. NeurIPS 2023 is from memory and was not checked. — [arXiv](https://arxiv.org/abs/2303.17651v1)
- **Welleck et al. (2023), "Generating Sequences by Learning to Self-Correct".** arXiv 2211.00053, ICLR 2023. A separate corrector, trained online, iteratively fixes a base generator's outputs. It helps on math program synthesis, lexically-constrained generation and toxicity control, even when the corrector is much smaller than the generator. — [arXiv](https://arxiv.org/abs/2211.00053); [ICLR](https://iclr.cc/virtual/2023/poster/11103)
- **RISE.** Qu, Zhang, Garg, Kumar, "Recursive Introspection: Teaching Language Model Agents How to Self-Improve", arXiv 2407.18219, NeurIPS 2024. Frames fine-tuning as a multi-turn MDP with on-policy data collection, inspired by online imitation and offline RL. Llama2, Llama3 and Mistral improve over turns and beat single-turn strategies at equal inference compute. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract.html)
- **LeMa.** An, Ma et al., "Learning From Mistakes Makes LLM Better Reasoner", arXiv 2310.20689 (2023). SFT on mistake→correction pairs written by GPT-4 improves on CoT-only SFT for math. These are off-policy corrections from another model, which is exactly the distribution-mismatch case SCoRe warns about. — [arXiv](https://arxiv.org/abs/2310.20689v4)
- **GLoRe.** Havrilla et al., arXiv 2402.10963, ICML 2024. Stepwise ORMs trained on synthetic data locate the erroneous step better than ORMs do. Combining global and local refinements raises LLaMA-2 13B on GSM8K from 53% to 65%. "When and where to refine" is the hard part without external feedback. — [arXiv](https://arxiv.org/abs/2402.10963); [PMLR](https://proceedings.mlr.press/v235/havrilla24a.html)
- **Kamoi et al. survey.** "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs", arXiv 2406.01297, TACL vol. 12 (2024), pp. 1417–1440. Three findings:
  1. No prior work shows successful self-correction with feedback from prompted LLMs on general tasks.
  2. Self-correction works when reliable external feedback is available.
  3. Large-scale fine-tuning enables it.

  — [ACL Anthology](https://aclanthology.org/2024.tacl-1.78/)
- **Physics of LMs Part 2.2.** Ye, Xu, Li, Allen-Zhu, "How to Learn From Mistakes on Grade-School Math Problems", arXiv 2408.16293, ICLR 2025. On a synthetic math dataset, pretraining on erroneous-step-then-correction data gives higher reasoning accuracy than the same amount of error-free data. — [arXiv](https://arxiv.org/abs/2408.16293)

### Inferences
- Off-policy correction SFT (LeMa, Pair-SFT) is the regime most exposed to collapse. On-policy multi-turn methods (RISE, SCoRe) are the established answers to it. Onwordly should cite these as prior art and not rename them.

### Gaps
- I ran no targeted search for 2025–2026 follow-ups, for example RL-trained self-verification, or reflection in R1-style RL with critiques of "superficial self-reflection". This is unverified, so treat it as a gap.

## Q4. Copying or deferring to an answer shown in the prompt (sycophancy, flip studies)

### Takeaway
Sycophancy toward stated views, and answer flipping under "Are you sure?" challenges, are well documented. Neither paper I verified directly studies copying a proposed answer during correction-trained inference. That has to be measured in Onwordly.

### Cited Findings
- Sharma, Tong, Korbak, Duvenaud, Askell, Bowman et al., "Towards Understanding Sycophancy in Language Models", arXiv 2310.13548, ICLR 2024:
  - Five AI assistants show sycophancy consistently across four tasks.
  - Humans and preference models sometimes prefer convincing sycophantic responses over correct ones.
  - Optimizing against preference models sometimes trades truthfulness for sycophancy.

  — [arXiv](https://arxiv.org/abs/2310.13548v1); [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0105f7972202c1d4fb817da9f21a9663-Abstract-Conference.html)
- Laban, Murakhovs'ka, Xiong, Wu, "Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment", arXiv 2311.08596 (2023):
  - Across 10 LLMs and 7 classification tasks, models flip their answers 46% of the time on average, with an average accuracy drop of 17%.
  - Fine-tuning on synthetic data cuts the deterioration by 60% but does not remove it.

  — [arXiv](https://arxiv.org/abs/2311.08596)

### Inferences
- There are two opposite failure modes. Behavior collapse (SCoRe) means over-anchoring on the first answer. FlipFlop means under-anchoring when challenged. An exact-verification ablation should present both correct and incorrect proposed answers and report the keep-rate and flip-rate for each.

### Gaps
- I did not verify the anchoring / "suggested answer in prompt" studies: Wei et al. 2023 on synthetic data for reducing sycophancy, Turpin et al. 2023 on unfaithful CoT with biased answers, and Perez et al. 2022.

## Q5. Small models (<1B) or exactly verifiable toy domains

### Takeaway
Of the papers I verified, only Physics of LMs Part 2.2 uses small, from-scratch models on a fully synthetic, exactly checkable math domain. The others use 7B+ models or Gemini on GSM8K, MATH or HumanEval. A sub-1B study of correction-trace SFT collapse on arithmetic appears open.

### Cited Findings
- Ye et al. pretrain on a synthetic grade-school math dataset with error-and-correction steps. — [arXiv](https://arxiv.org/abs/2408.16293)
- The other papers I checked used 7B+ models or Gemini:
  - RISE: Llama2, Llama3, Mistral — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract.html)
  - GLoRe: LLaMA-2 13B — [PMLR](https://proceedings.mlr.press/v235/havrilla24a.html)
  - SCoRe: Gemini — [arXiv](https://arxiv.org/abs/2409.12917)
- Welleck et al. show that a corrector much smaller than the generator can work, on tasks including math program synthesis. — [arXiv](https://arxiv.org/abs/2211.00053)

### Gaps
- The exact model sizes in Ye et al. (I believe they are GPT-2-scale) and in Welleck et al. (I believe GPT-Neo/GPT-2-scale correctors) were not checked against the full text here.
- I found no verified paper on correction-trace SFT collapse with sub-1B models on plain arithmetic. That is absence of evidence from a limited search, not proof that no such paper exists.
