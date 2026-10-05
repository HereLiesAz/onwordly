# Controlled small-model experiments on curricula, hard-example training, and correction/verification in exactly-verifiable domains

Scope note: Only papers whose title, authors, year and ID/venue were confirmed through search or fetch are cited. Details marked "(unverified detail)" come from memory and were not checked against the paper text in this session. Search date: 2026-10-05.

## Q1. Curriculum and competence-based curricula versus static or random data

### Takeaway
Competence-based curricula reported large savings in training time for NMT. Later controlled studies find the benefit depends on the setting: curricula help mainly when the training-time budget is limited or the labels are noisy, or at small scale (up to roughly 160M parameters). Under a standard full budget, random or i.i.d. ordering is usually competitive.

### Cited Findings
- Platanios, Stretcu, Neubig, Poczos, Mitchell (NAACL 2019), "Competence-based Curriculum Learning for Neural Machine Translation", ACL N19-1119. Samples are admitted according to an estimated difficulty and a competence function. The paper reports up to 70% less training time and up to +2.2 BLEU for both RNNs and Transformers. — [ACL Anthology](https://aclanthology.org/N19-1119/)
- Wu, Dyer, Neyshabur (ICLR 2021 oral), "When Do Curricula Work?", arXiv:2012.03107. The study runs thousands of orderings: curriculum, anti-curriculum, and a "random-curriculum" control where the dataset grows over training but the order within it is random. Curriculum (but not anti-curriculum) helps only under a **limited training-time budget** or with **noisy data**. In the standard setting the benefits are marginal, and the random-curriculum control captures much of the effect (the last point is from the paper as I remember it). This is the key compute-matched reference. — [arXiv](https://arxiv.org/abs/2012.03107v2); [Google Research](https://research.google/pubs/when-do-curricula-work/)
- Elgaar & Amiri (arXiv:2601.21698, Jan 2026, rev. May 2026), "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics". Pythia-style models from 14M to 1B parameters are trained on Age-of-Acquisition, word-frequency and Verb-Variation curricula and compared with random order **at matched compute**. Training passes through the same latent phases under every ordering; curricula only change exposure within a phase. At ≤160M parameters, random order shows more gradient noise, output-head saturation, and lower final accuracy. The gains shrink at larger scale. Reversing the VV curriculum removes most of its benefit. — [arXiv](https://arxiv.org/abs/2601.21698)
- In the BabyLM constrained-data/compute setting, Active Curriculum Language Modeling (ACLM) is reported to beat the official baselines. This is a workshop paper, and I did not check whether the baselines were budget-matched. — [ACL Anthology preview 2025.babylm-main.34](https://preview.aclanthology.org/setup/2025.babylm-main.34)
- Swayamdipta et al. (EMNLP 2020), "Dataset Cartography", arXiv:2009.10795. Training dynamics split examples into easy, ambiguous and hard-to-learn regions. Ambiguous examples contribute most to OOD generalization. Hard-to-learn examples often turn out to be **label errors**, so "train on the hardest" is risky when labels are noisy. — [ACL Anthology](https://aclanthology.org/2020.emnlp-main.746.pdf); [arXiv](https://arxiv.org/abs/2009.10795v1)

### Inferences
- Matched-budget comparisons should include a "random-curriculum" control (same dataset-size schedule, random order), following Wu et al. Otherwise a pacing effect can be mistaken for an ordering effect.
- Exact verification gives clean labels, so the noisy-label pathway does not apply to Onwordly domains. That leaves the limited-budget pathway as the main reason to expect curriculum gains.

### Gaps
- I did not verify a recent paper that directly tests competence-based or adaptive curricula for **SFT of sub-1B LMs on arithmetic** with matched tokens. I found none in this pass.
- I did not verify the specific hard-example-mining / active-learning papers for LM fine-tuning. Search did not surface a clean, controlled sub-1B study.

## Q2. Arithmetic learning in small transformers: format, sampling, generalization

### Takeaway
How the data is formatted is the dominant factor: reversed outputs, scratchpads, position tokens and positional embeddings. Sampling balance also matters. Length generalization needs either architectural help (relative positions, Abacus embeddings) or data tricks (priming, self-improvement). Naive static training does not extrapolate.

### Cited Findings
- Nogueira, Jiang, Lin (2021), arXiv:2102.13019. The surface form of numbers strongly affects accuracy. With subwords the model fails at 5-digit addition; with explicit position tokens it learns addition/subtraction up to 60 digits. No model learned length-independent rules, at any parameter count or data size. — [arXiv](https://arxiv.org/abs/2102.13019v1)
- Lee, Sreenivasan, J. Lee, K. Lee, Papailiopoulos (ICLR 2024), "Teaching Arithmetic to Small Transformers", arXiv:2307.03381. These are NanoGPT-scale models trained from scratch. Standard formatting is sub-optimal; **reversed output** or **detailed scratchpad** data improves accuracy and sample complexity. Accuracy jumps sharply with data scale, which the authors connect in some cases to low-rank matrix completion. Balanced sampling over digit lengths and carries also matters (unverified detail). — [arXiv](https://arxiv.org/pdf/2307.03381); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6bf82fdcbd92b6a7793b3894422d2437-Abstract-Conference.html)
- Jelassi et al. (2023), "Length Generalization in Arithmetic Transformers", arXiv:2306.15400. Relative position embeddings let a model trained on 5-digit addition do 15 digits. That fails for multiplication. **Train-set priming** (adding 10–50 long examples) lets 5×3 training generalize to 35×3. — [arXiv](https://www.arxiv.org/pdf/2306.15400)
- Zhou et al., "What Algorithms Can Transformers Learn? A Study in Length Generalization" (ICLR 2024). The RASP-L conjecture: transformers length-generalize on tasks that have simple RASP-L programs. — [MLAnthology](https://mlanthology.org/iclr/2024/zhou2024iclr-algorithms)
- McLeish et al. (NeurIPS 2024), "Transformers Can Do Arithmetic with the Right Embeddings", arXiv:2405.17399. Abacus embeddings encode digit position from the start of the number. Trained on ≤20-digit numbers (one GPU, one day), the model reaches up to 99% on 100-digit addition. Input injection and recurrence help further. — [arXiv](https://arxiv.org/pdf/2405.17399); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c35986bc1ee29b31c1011481b77fe540-Abstract.html)
- N. Lee, Cai, Schwarzschild, K. Lee, Papailiopoulos (ICML 2025), "Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges", arXiv:2502.01612. Models generate their own solutions to slightly harder problems and retrain on them. This works as an **adaptive, self-generated length curriculum**, taking a model from 10-digit to 100-digit addition, and it also covers string manipulation and mazes. Filtering the self-generated data (e.g., length filtering or majority voting) is important to stop errors from compounding (unverified detail). — [arXiv](https://www.arxiv.org/abs/2502.01612); [PMLR](https://proceedings.mlr.press/v267/lee25d.html)

### Inferences
- Any curriculum comparison in arithmetic must fix the data format (reversal, scratchpad, positional scheme) across arms, because format effects are larger than ordering effects.
- Self-improvement (Lee et al. 2025) is the closest existing "adaptive curriculum with programmatic filtering" result in a toy arithmetic domain. Onwordly should cite it as prior art rather than present such a loop as new.

### Gaps
- I could not confirm whether Lee et al. 2023/2025 match **training tokens** across format or curriculum arms. Their comparisons are mostly in number of samples or rounds.

## Q3. Compute/token-matched comparisons; training on own errors or hard neighbours

### Takeaway
Few papers match tokens explicitly. Wu et al. 2021 (time budget), Elgaar & Amiri 2026 (matched compute) and Ye et al. 2024 (same amount of data, error-free vs. with errors) are the closest. I found no verified, controlled sub-1B study showing that training on the model's own errors or hard neighbours **hurts** generalization. The adjacent negative evidence comes from label noise, not from error-focus itself.

### Cited Findings
- Ye, Xu, Li, Allen-Zhu (ICLR 2025), "Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems", arXiv:2408.16293. On a synthetic GSM-style dataset (iGSM), pretraining data with erroneous steps immediately followed by corrections ("retry" data) gives higher accuracy with plain autoregression **than the same amount of error-free data**. This is a data-matched comparison. The paper also studies whether to mask error tokens, how much error data to use, and fine-tuning vs. pretraining. As I remember the answers, masking is not needed, and adding retry data only at fine-tuning helps much less than in pretraining (unverified details). — [arXiv](https://arxiv.org/abs/2408.16293); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c239bac713017b0b2257b7622bf8aab3-Abstract-Conference.html)
- Label noise during fine-tuning causes the largest degradation compared with grammatical or typographical noise. This is a 2026 preprint, arXiv:2604.12469, "Analyzing the Effect of Noise in LLM Fine-tuning", and I saw only the search snippet. — [alphaXiv](https://www.alphaxiv.org/abs/2604.12469)
- Hard-to-learn examples often correspond to mislabeled data (Swayamdipta et al. 2020, see Q1). — [ACL Anthology](https://aclanthology.org/2020.emnlp-main.746.pdf)

### Inferences
- With exact verifiers, "hard" examples are genuinely hard rather than mislabeled. The usual argument that "hard-example mining amplifies noise" therefore probably does not carry over, and Onwordly would need its own ablation to show that error-focus hurts or helps.
- Following Ye et al., the natural control for corrective training is **equal-token error-free data**, plus a masked-error variant.

### Gaps
- No verified primary source was found for "training on the model's own errors / hard neighbours hurts generalization" in small models. Treat this as an open claim.

## Q4. Studies combining curricula/correction/verification in small models

### Takeaway
Several recent works train tiny or small models with explicit verify/correct moves on exact tasks (multiplication, Sudoku, iGSM). They find that verification helps when verifier error is bounded, and that RL tends to exploit surface patterns instead.

### Cited Findings
- Yu, Xia, Yan, Xu, Zhang, Du, Wang (NeurIPS 2025), "Self-Verifying Reflection Helps Transformers with CoT Reasoning", arXiv:2510.12157. Tiny transformers (a few million parameters, no natural language) get self-verifying reflection. They reach LLM-level performance on integer multiplication and Sudoku. The paper proves that reflection improves results if verification errors are bounded. RL improves in-distribution results and makes the model reflect more often, but mostly fits shallow patterns and does not reduce verification errors. — [arXiv](https://arxiv.org/abs/2510.12157); [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html)
- Ye et al. 2024 (Q3): error-plus-correction pretraining data on synthetic math. — [arXiv](https://arxiv.org/abs/2408.16293)
- Lee et al. 2025 (Q2): self-improvement with filtering, i.e. an adaptive curriculum plus programmatic selection. — [arXiv](https://www.arxiv.org/abs/2502.01612)
- "Small Language Models Need Strong Verifiers to Self-Correct Reasoning" (SCORE), arXiv:2404.17140 (2024). Small LMs fine-tuned on self-generated correction data improve at self-correction only when paired with **strong verifiers**. I did not verify the authors in this session. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2404.17140)
- Qu et al. (NeurIPS 2024), "Recursive Introspection (RISE)", arXiv:2407.18219. Multi-turn fine-tuning on revision data, with best-of-N under a success indicator, improves Llama2/3 and Mistral over turns. It beats single-turn strategies at **equal inference compute**. These are 7B-scale models, not small. — [arXiv](https://arxiv.org/pdf/2407.18219)
- "Project Aletheia: Verifier-Guided Distillation of Backtracking for Small Language Models", arXiv:2601.14290 (Jan 2026). It distills verified traces with [CONFLICT] tokens and backtracking into a 7B model. I did not check the authors. — [arXiv](https://arxiv.org/html/2601.14290v1)

### Inferences
- Ye et al. 2024, Lee et al. 2025 and Yu et al. 2025 together already cover "correction data", "adaptive self-generated curriculum" and "verification moves" in exact toy domains. What remains open for Onwordly is a **single training-token-matched factorial ablation** across static SFT, adaptive curriculum, error-focused sampling, and corrective/verification data on a held-out exact evaluation.

### Gaps
- No paper found that crosses all four regimes under matched training tokens in a sub-1B model.
- I did not confirm the authors of SCORE (2404.17140) or Aletheia (2601.14290).
