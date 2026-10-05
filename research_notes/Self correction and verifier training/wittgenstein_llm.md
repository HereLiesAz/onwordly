# Wittgensteinian framings of LLM training/evaluation (language games, rule-following, correction)

Scope note: ~15 tool calls; current as of 2026-10-05. Items marked [verified] had their abstract/landing page fetched; [snippet] were seen only in search results and need checking before citation.

## Which papers apply "language games" or Wittgensteinian rule-following to LLMs?

### Takeaway
A modest, mostly philosophical literature exists (Philosophy & Technology, AI & Society, an Anthem Press edited volume, arXiv position papers). Most of it uses rule-following/normativity to argue that LLMs only *conform to* regularities and do not *follow* rules, because correction happens outside the model (in the human or community).

### Cited Findings
- Pérez-Escobar & Sarikaya, "Philosophical Investigations into AI Alignment: A Wittgensteinian Framework", *Philosophy & Technology* 37(3), 2024. Argues that later-Wittgenstein rule-following matters for alignment, and that it can guide dataset creation and guardrails [verified] — [Lübeck research portal](https://research.uni-luebeck.de/en/publications/philosophical-investigations-into-ai-alignment-a-wittgensteinian-/)
- Fenoglio, "Asymmetric Communication: Large Language Models and Language Games", arXiv 2607.28137, July 2026. Human-LLM interaction is a language game in which "one side bears all normative activity"; "correctness is enforced exclusively by the receiver". Draws on Wittgenstein, Luhmann, Esposito and Brandom [verified] — [arXiv](https://arxiv.org/abs/2607.28137)
- "Language Models and the Private Language Argument: A Wittgensteinian Guide to Machine Learning", in *Wittgenstein and Artificial Intelligence* (Anthem Press, 2024; online 2025), pp. 145–164. Asks whether connectionist NLP answers worries raised by the private language argument, and covers context-dependence and language games. Authorship is uncertain: the page listed Giovanni Galli alongside Ball, Helliwell and Rossi, and these names may be the chapter authors rather than the volume editors. The fetched summary reports no discussion of correction [verified, authorship uncertain] — [Cambridge Core](https://www.cambridge.org/core/books/wittgenstein-and-artificial-intelligence/language-models-and-the-private-language-argument-a-wittgensteinian-guide-to-machine-learning/D8D89946EEF76DD542DB7ECFA8B2236F)
- Molino & Tagliabue, "Wittgenstein's influence on artificial intelligence", arXiv 2302.01570, 2023. Traces modern NLP progress back to the later Wittgenstein; originally a chapter in a Spanish Tractatus-centenary volume [verified] — [arXiv](https://arxiv.org/abs/2302.01570)
- Sambrotta, "Commentary on 'LLMs and the Logical Space of Reasons': Inferential Roles, Rule-Following, and Digital Speech Acts", *Archives of Information Technology* 1(1), 2026. Argues that LLMs show conformity to rule-like regularities without normative standing, and that their outputs are "degenerate speech acts" [verified] — [Scientific Archives](https://www.scientificarchives.com/article/commentary-on-llms-and-the-logical-space-of-reasons-inferential-roles-rule-following-and-digital-speech-acts)
- Ghojogh & Ghojogh, "Data Evolution by Wittgenstein's Rule Following", arXiv 2606.22674, June 2026. A machine-learning method (WRF) that extrapolates sequences of datasets, inspired by rule-following and family resemblance. It has no correction or verification component [verified] — [arXiv](https://arxiv.org/abs/2606.22674)
- An *AI & Society* 2025 article (DOI 10.1007/s00146-025-02663-6) appeared under a "meaning is use" query. The Springer page could not be fetched, so the title is unknown [unverified] — [Springer](https://link.springer.com/10.1007/s00146-025-02663-6)
- "Neither Ghost nor Parrot: Wittgenstein and the Philosophy of Artificial Intelligence" (PhilArchive TOTNGN). Covers Wittgensteinian influence on symbolic, connectionist and embodied AI and on the world-model debate. The page returned 403 [snippet] — [PhilArchive](https://philarchive.org/rec/TOTNGN)
- "Wittgenstein and Artificial Intelligence" conference, Skjolden, June 2022 (Pichler & Säätelä, Bergen). Topics included language games and AI rule-following [verified] — [UiB](https://www.uib.no/en/rg/philtext/154466/wittgenstein-and-artificial-intelligence-towards-update)
- A Philosophy & Technology 2024 article (DOI 10.1007/s13347-024-00761-9) appeared in a Kripkenstein/LLM search. It could not be fetched, and it may be the Pérez-Escobar & Sarikaya paper [unverified] — [Springer](https://link.springer.com/article/10.1007/s13347-024-00761-9)
- Non-academic: STRV blog "Language Games and LLMs: What Wittgenstein can teach AI engineers"; a LessWrong post "Meaning is use: a Wittgensteinian defense of LLM" [snippet; not scholarly] — [STRV](https://www.strv.com/blog/language-games-and-llms-what-wittgenstein-can-teach-ai-engineers), [LessWrong](https://www.lesswrong.com/posts/SGdBNPDzfPQL73W2i/meaning-is-use-a-wittgensteinian-defense-of-llm)
- WoLaLa 2025 paper 27: combines Wittgensteinian language games with game theory to test whether four LLMs follow game rules. It reports that dynamic games produce more rule violations than static ones [snippet] — [WoLaLa PDF](https://wolala.nytud.hu/wp-content/uploads/2025/11/WoLaLa-2025_paper_27-2.pdf)

### Inferences
- The dominant philosophical move is negative: correction and normative assessment are said to sit outside the model. This is the opposite of training the model to correct play.

### Gaps
- I did not search Synthese or Minds and Machines systematically. Titles for the two Springer DOIs are still unknown.

## Has anyone linked the teaching-by-correction passages (PI §§143–145, 185, 202, 258) to model training, verification, or self-correction?

### Takeaway
I found no work that ties these specific sections to LLM verifier or self-correction training. Some papers state the general Wittgensteinian point that a rule exists only where applications can be corrected, but they use it to deny that LLMs follow rules, not to design training.

### Cited Findings
- Search snippets from the rule-following/LLM literature say that following a rule "involves being embedded in a normative practice where one's performance can be evaluated, corrected, and justified by others". They also say, after Kripke, that no fact about an isolated system fixes the correct continuation [snippet, source page not pinned down; likely the Fenoglio or Sambrotta texts] — [Fenoglio arXiv](https://arxiv.org/pdf/2607.28137)
- Fenoglio places all correction on the human receiver's side [verified] — [arXiv](https://arxiv.org/abs/2607.28137)
- Searches combining Wittgenstein with LLM self-correction returned only technical self-correction papers that make no Wittgenstein connection, e.g. Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet" (ICLR 2024), and "Small LMs Need Strong Verifiers to Self-Correct Reasoning" (ACL Findings 2024) — [arXiv 2310.01798](https://arxiv.org/html/2310.01798v1), [ACL](https://preview.aclanthology.org/dashboard/2024.findings-acl.924)
- The Kripkenstein "quus" deviant-pupil scenario has been compared to DNNs that diverge from the intended function on new inputs [snippet] — [search result list](https://lists.philo.at/hyperkitty/list/news@lists.philo.at/message/FWKVPXN3L66T2AYT3TTWSUJQW5IEXXZK/)

### Inferences
- The specific claim that "learning a game includes learning to correct incorrect play", turned into a training signal through exact verifiers, looks unclaimed in what I found. Treat this as provisional, because the search was shallow (no PhilPapers query and no full-text checks).
- Onwordly should not claim novelty for the philosophical idea itself. The idea that correction constitutes rules is standard Wittgenstein/Kripke/Brandom. Any novelty would lie only in operationalising it with exact verification.

### Gaps
- I did not fetch the primary PI passages or the SEP rule-following entry. Pagination for §§143–145, 185, 202 and 258 is not verified here.
- I did not check Brandom-inspired ML work on "normative scorekeeping" or the "Abrichtung" (training/drill) literature, e.g. work by Meredith Williams. No hits connected Abrichtung to LLMs.

## ML work using "language games" as a training paradigm (vs Wittgensteinian framing)

### Takeaway
In ML, "language game" usually means a Lewis signalling or referential game (emergent communication), or an adversarial self-play game with a programmatic win condition. These use the term loosely. A win/loss check acts as an exact verifier, but it is not Wittgensteinian correction of play.

### Cited Findings
- Lazaridou, Peysakhovich & Baroni, "Multi-Agent Cooperation and the Emergence of (Natural) Language" (arXiv 1612.07182; ICLR 2017). Sender and receiver agents play referential games and develop a protocol from task success — [arXiv](https://arxiv.org/html/1612.07182)
- Lazaridou, Pham & Baroni, "Towards Multi-Agent Communication-Based Language Learning" (arXiv 1605.07133, 2016) — [arXiv](https://arxiv.org/pdf/1605.07133)
- "A Practical Guide to Studying Emergent Communication through Grounded Language Games" (arXiv 2004.09218, 2020; Steels-tradition naming game framework) — [arXiv](https://arxiv.org/pdf/2004.09218)
- "Emergent Linguistic Phenomena in Multi-Agent Communication Games" (arXiv 1901.08706, 2019). Frames these games as special or generalised cases of Lewis signalling games — [ar5iv](https://ar5iv.labs.arxiv.org/html/1901.08706)
- Cheng et al., "Self-playing Adversarial Language Game Enhances LLM Reasoning" (SPAG, arXiv 2404.10642, 2024; NeurIPS 2024). LLMs self-play Adversarial Taboo, then offline RL on winning episodes improves reasoning benchmarks — [arXiv](https://arxiv.org/abs/2404.10642v3)
- arXiv 2506.22920: self-play in a "Critic-Discernment Game", where a prover's solution is challenged by critiques. This is the closest in spirit to training correction of play, but it has no Wittgenstein framing [snippet] — [arXiv](https://arxiv.org/pdf/2506.22920.pdf)

### Inferences
- Emergent-communication papers sometimes mention "meaning as use" in passing (per snippets). The SPAG and critic-game papers are game-theoretic, not philosophical. The Critic-Discernment Game is the nearest ML analogue of learning to correct incorrect play.

### Gaps
- I did not verify the title or authors of 2506.22920. I also did not check whether SPAG or Lazaridou cite Wittgenstein explicitly.
