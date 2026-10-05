# Balanced verdicts test what copy-collapse literature leaves open

Prior work already covers each part of Onwordly Experiments 008 and 009. What it does not cover is the combination, run under tight controls. Three findings are established. First, supervised fine-tuning (SFT) on correction traces collapses toward leaving the shown answer unchanged, and off-policy errors create a train/test mismatch ([SCoRe](https://arxiv.org/abs/2409.12917)). Second, models flip answers when challenged and defer to stated views ([FlipFlop](https://arxiv.org/abs/2311.08596); [Sharma et al.](https://arxiv.org/abs/2310.13548)). Third, small models self-correct only with strong verifiers ([Zhang et al.](https://arxiv.org/abs/2404.17140)). Training verification jointly with generation, or training verification first, is also established ([GenRM](https://arxiv.org/abs/2408.15240v2); [Chen et al. 2026](https://arxiv.org/abs/2602.07594)). The notes found no paper that combines all of the following in one study:

- training one sub-1B model on a 50/50 balanced verdict move ("right" / "wrong: <answer>");
- grading it by exact programmatic checks;
- matching training-token budgets;
- crossing the source of wrong proposals (the model's own errors vs synthetic ones).

Experiment 008 already adds a clean, two-sided result in this setting. Synthetic-error training made the model copy the shown answer: correction 4.6 vs confirmation 88.6. Own-error training made it ignore the shown answer: 74.8 vs 74.6. Before any training, the model broke 114 of 500 right answers when shown a wrong one. Onwordly's contribution is therefore the controlled comparison, not a new technique. The Wittgenstein "language game" framing has precedent in philosophy and should stay a design heuristic.

## SFT collapse and sycophancy are documented, but mostly at 7B+ scale and in one direction

The closest precedent is SCoRe (Kumar et al., Google DeepMind). It names two failures of offline SFT on correction traces:

- **Distribution mismatch.** The training data holds the data-collection policy's mistakes, not the model's own.
- **Behavior collapse.** The model settles on one correction mode that rarely works on test problems ([arXiv](https://arxiv.org/abs/2409.12917)).

The full text reports that STaR- and Pair-SFT-trained models "are overly conservative and often make no edits at all." On MATH500 with Gemini 1.5 Flash, STaR improved accuracy from first to second attempt by only **0.4 points**, and Pair-SFT by **1.8 points**. Pair-SFT fixed **5.4%** of answers and broke **3.6%** ([arXiv HTML](https://arxiv.org/html/2409.12917)). The fix is two-stage multi-turn RL, which reports **+15.6% on MATH and +9.1% on HumanEval** for self-correction ([arXiv](https://arxiv.org/abs/2409.12917)). (The notes read these numbers through a summarizing fetch. Check them against the PDF before quoting them in a paper.)

The opposite failure is also well documented: abandoning correct answers. Huang et al. find intrinsic self-correction "at times" lowers performance ([arXiv](https://arxiv.org/abs/2310.01798)). In the FlipFlop study, a challenge such as "Are you sure?" made 10 LLMs flip answers **46%** of the time, with a **17%** average accuracy drop. Fine-tuning on synthetic data reduced the drop by 60% but did not remove it ([arXiv](https://arxiv.org/abs/2311.08596)). Sharma et al. show sycophancy across five assistants, partly driven by preference models ([arXiv](https://arxiv.org/abs/2310.13548v1)). The Kamoi et al. survey sums up the field: self-correction works with reliable external feedback or with large-scale fine-tuning, not with prompting alone ([TACL](https://aclanthology.org/2024.tacl-1.78/)). Several other lines train on corrections: GPT-4-written mistake corrections ([LeMa](https://arxiv.org/abs/2310.20689v4)), on-policy multi-turn fine-tuning ([RISE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract.html)), separate trained correctors ([Welleck et al.](https://arxiv.org/abs/2211.00053)) and stepwise refinement models ([GLoRe](https://arxiv.org/abs/2402.10963)).

The Experiment 008 results line up with this literature as follows:

- **Synthetic-error training (copies the shown answer).** This matches SCoRe's off-policy failure, pushed to the extreme: correction 4.6 vs confirmation 88.6.
- **Own-error training (ignores the shown answer).** SCoRe's "no edits" collapse means the model keeps a proposed answer. Here the model instead re-solves the problem and disregards the proposal, so correction and confirmation are nearly equal (74.8 vs 74.6). That is a different behavior from "no edits", and no paper in the notes reports it under matched conditions.
- **Untrained baseline (114/500 right answers broken when shown a wrong one).** This is sycophantic deference, the Sharma/FlipFlop phenomenon. Here it is measured with exact checks on a 0.5B model rather than with judged answers on frontier assistants.

The notes did not verify three related studies: Wei et al. 2023 on synthetic data against sycophancy, Turpin et al. 2023 on biased-answer chain-of-thought, and Perez et al. 2022. Those are **unverified in the notes**.

## Verifier-first training is established; balanced exact verdicts at sub-1B are not

Training models to judge before repairing is a mature line of work. GenRM trains one model with next-token prediction on both verification and solution generation. It beats discriminative and LLM-judge verifiers in Best-of-N ranking, raising accuracy from 5% to **45.3%** on algorithmic tasks and from 73% to **93.4%** on GSM8K ([arXiv](https://arxiv.org/abs/2408.15240v2)). Other verified work:

- **Critique Fine-Tuning** trains the model to critique noisy responses rather than imitate correct ones. It gains **4–10%** over SFT on six math benchmarks ([arXiv](https://arxiv.org/abs/2501.17703v3)).
- **Zhang et al.** find that small models (here "small" means ≤13B) gain a lot from self-correction training only when a strong (GPT-4) verifier decides when to correct ([arXiv](https://arxiv.org/abs/2404.17140)).
- **Chen et al. 2026** report an asymmetry: training on generation does not improve verification, but **learning to self-verify improves generation** ([arXiv](https://arxiv.org/abs/2602.07594)). They balance their verification data, but the notes did not confirm the exact ratio ([arXiv HTML](https://arxiv.org/html/2602.07594v1)).
- **Han et al.** report that an imbalance between good and bad cases did not hurt their partial-answer-masking method ([arXiv](https://arxiv.org/pdf/2401.07301)).

Several foundational verifier papers were cited **from memory and not re-checked in the notes**: Cobbe et al. 2021 on GSM8K verifiers (2110.14168), Lightman et al. "Let's Verify Step by Step" (2305.20050), Uesato et al. (2211.14275), Math-Shepherd (2312.08935), CriticGPT (2407.00215), CRITIC (2305.11738), Tyen et al. (2311.08516), ProcessBench (2412.06559), Wang et al. on position bias in LLM judges (2305.17926), and Gao et al. on reward-model overoptimization (2210.10760). Their arXiv IDs need checking before publication.

The verified literature never isolates the variable that Experiment 009 manipulates. Wrong proposals come from different sources in different papers:

- the model's own samples ([GenRM](https://arxiv.org/abs/2408.15240v2); [Zhang et al.](https://arxiv.org/abs/2404.17140));
- injected synthetic errors ([Ye et al.](https://arxiv.org/abs/2408.16293));
- third-party noisy responses ([CFT](https://arxiv.org/abs/2501.17703v3)).

None of these papers compares those sources head-to-head for verifier accuracy. None trains the "verdict + corrected answer" format with an explicit 50/50 balance, and none quantifies always-accept vs always-reject rates as a function of label ratio. With exact programmatic checks, the verdict label cannot be reward-hacked. With a 50/50 balance, a majority-label shortcut gets no reward. Both choices address failure modes the literature names but has not ablated together.

## Matched-budget small-model studies exist, but none crosses these regimes

Few papers match training budgets explicitly. The closest are:

- **Wu, Dyer & Neyshabur.** Across thousands of orderings, curricula help only under a **limited training-time budget** or with **noisy data**, and the gains are marginal in the standard setting ([arXiv](https://arxiv.org/abs/2012.03107v2)).
- **Elgaar & Amiri (2026).** Pythia-style models from 14M to 1B parameters, trained at matched compute, gain from curricula at ≤160M parameters; the gains shrink with scale ([arXiv](https://arxiv.org/abs/2601.21698)).
- **Ye et al. (Physics of LMs 2.2).** This is the key data-matched precedent. Pretraining on erroneous steps followed by corrections beats **the same amount of error-free data** on synthetic grade-school math ([arXiv](https://arxiv.org/abs/2408.16293)). The model size (believed GPT-2-scale) and the finding that fine-tuning-stage retry data helps less are **unverified in the notes**.
- **Yu et al. (NeurIPS 2025).** Self-verifying reflection lets transformers of a few million parameters reach LLM-level performance on multiplication and Sudoku, provided verification error is bounded. RL mostly fits shallow patterns and does not reduce verification errors ([arXiv](https://arxiv.org/abs/2510.12157)).
- **Lee et al. 2025.** Self-improving transformers act as an adaptive length curriculum with programmatic filtering ([arXiv](https://www.arxiv.org/abs/2502.01612)).

In arithmetic, data format (reversed outputs, scratchpads, digit position) matters more than ordering ([Lee et al. 2023](https://arxiv.org/pdf/2307.03381); [McLeish et al.](https://arxiv.org/pdf/2405.17399)). Any comparison between arms therefore has to hold the format fixed.

The notes found no sub-1B study, and specifically no LoRA-adapted pretrained 0.5B model, that compares corrective or verdict training against static SFT at matched training tokens on held-out exact evaluation. The notes also found no verified evidence that training on a model's own errors hurts generalization in small models. The negative evidence that exists concerns label noise ([Swayamdipta et al.](https://arxiv.org/abs/2009.10795v1)), and that does not apply when labels come from exact checks.

## Language-game framing has precedent in philosophy, not in training design

Wittgensteinian treatments of LLMs exist, but they mostly argue the opposite of Onwordly's heuristic:

- **Pérez-Escobar & Sarikaya** apply rule-following to alignment and to dataset creation ([portal](https://research.uni-luebeck.de/en/publications/philosophical-investigations-into-ai-alignment-a-wittgensteinian-/)).
- **Fenoglio** describes human–LLM exchange as a game in which "correctness is enforced exclusively by the receiver" ([arXiv](https://arxiv.org/abs/2607.28137)).
- **Sambrotta** argues that LLMs conform to regularities without normative standing ([Scientific Archives](https://www.scientificarchives.com/article/commentary-on-llms-and-the-logical-space-of-reasons-inferential-roles-rule-following-and-digital-speech-acts)).
- **Molino & Tagliabue** trace NLP back to the later Wittgenstein ([arXiv](https://arxiv.org/abs/2302.01570)).

In ML, "language game" usually means Lewis signalling games ([Lazaridou et al.](https://arxiv.org/html/1612.07182)) or adversarial self-play such as SPAG ([arXiv](https://arxiv.org/abs/2404.10642v3)). The nearest analogue to correcting incorrect play is a "Critic-Discernment Game" (arXiv 2506.22920), but its title and authors are **unverified in the notes**. The notes found no work that ties the teaching-by-correction passages of the *Investigations* (§§143–145, 185, 202, 258) to verifier training. The pagination of those sections was not verified either.

The honest position is this. "Correction constitutes the rule" is standard Wittgenstein, Kripke and Brandom. Onwordly only operationalizes it: the learner is trained to make the correcting move itself, under exact checks. The framing motivates the move inventory (confirm, reject, repair). It is not an empirical claim, and the experiments do not test it.

## What 008/009 add, and the experiments that should come next

### How Onwordly differs from prior work

Onwordly combines known pieces. Corrective SFT comes from LeMa and Pair-SFT. Joint verification comes from GenRM and Chen et al. Synthetic errors come from Ye et al. LoRA is standard. The new part is the controlled comparison:

- grading is by exact programmatic checks, with no neural judges;
- training budgets are matched at 100k training tokens;
- proposals are balanced 50/50 right and wrong, so both keep-rate and flip-rate can be measured;
- wrong proposals from the model's own errors are crossed with synthetic ones;
- the model is a 0.5B Qwen2.5, where the closest verifier studies use ≥7B models or from-scratch toy transformers.

Experiment 008 goes beyond SCoRe in one respect. It shows that the error source decides the direction of the collapse: synthetic errors teach copying, own errors teach ignoring. Neither is real conditional use of the proposal, which would show up as a large gap between correction and confirmation scores that favors the right action in each case. Experiment 009 tests whether an explicit verdict token breaks that symmetry. Its results are pending, and no claim should be made before they arrive.

### Next experiments

Each item ties to the prior work it controls for.

1. **Per-class keep/flip rates.** Report confirmation-keep, correction-fix, right→wrong breaks and wrong→wrong copies separately for every arm. These are the Δ(i→c)/Δ(c→i) metrics SCoRe uses ([arXiv HTML](https://arxiv.org/html/2409.12917)). Without them, a better first attempt can pass for successful correction.
2. **Full factorial on wrong-proposal source and ratio.** Cross own vs synthetic wrong proposals with balanced vs natural right/wrong ratios, at matched tokens. The verifier-training notes recommend exactly this ablation, and nothing in the verified literature settles it.
3. **Error-free control.** Following Ye et al. ([arXiv](https://arxiv.org/abs/2408.16293)), add an arm that spends the same tokens on error-free direct solutions, plus a variant that masks the error tokens. This separates "learning from mistakes" from "more arithmetic tokens."
4. **Random-curriculum pacing control.** Following Wu et al. ([arXiv](https://arxiv.org/abs/2012.03107v2)), add an arm with the same data-growth schedule but random order to any adaptive or curriculum arm. Otherwise a pacing effect can be read as an ordering or adaptivity effect.
5. **SCoRe-style RL stage.** Run two-stage multi-turn RL with the exact checker as reward, starting from the best SFT arm. Record its inference and checking compute separately, as AGENTS.md requires. This tests whether on-policy RL repairs the collapse at 0.5B. Yu et al.'s warning that RL fits shallow patterns ([arXiv](https://arxiv.org/abs/2510.12157)) predicts it may not.
6. **Does verification improve generation?** Measure plain no-proposal accuracy on held-out problems after verdict-only training, against a generation-only arm at matched tokens. This replicates the asymmetry claim of Chen et al. ([arXiv](https://arxiv.org/abs/2602.07594)) at sub-1B scale with exact grading.

## Conclusion

Experiment 008 already reframes the collapse literature. The failure is not simply "SFT corrections don't take." The model learns whichever shortcut its wrong-answer distribution rewards: synthetic-error training produced copying, own-error training produced ignoring. Why each source rewards its shortcut is not yet established. If that reading holds, the question for Experiment 009 is not only whether verdict accuracy rises. It is whether balanced exact verdicts make the shown answer *informative*, which would appear as a correction–confirmation gap in the right direction under both proposal sources.

A sub-1B model with exact checks and matched budgets is a cheap, well-controlled place to test this. The claims Onwordly can defend are about the conditions under which known techniques succeed or fail, not about new methods. The Wittgenstein framing stays useful as a heuristic for choosing which moves to train, provided the reports never present it as a finding.
