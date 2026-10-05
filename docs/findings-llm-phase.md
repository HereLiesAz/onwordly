# Findings from the LLM phase (archived)

Everything referenced here lives on branch `archive/llm-phase` (commit `c35b185`): experiment READMEs and `RESULTS.md` under `experiments/0xx-*/`, the literature review in `reports/Self correction and verifier training.md`, notes in `research_notes/`, and the former docs (`docs/positioning.md`, `docs/verdict-prior.md`, `docs/baseline*.md`, `docs/dataset-audit.md`, `docs/experiment-matrix.md`, `docs/research-program.md`). All results are **one seed** unless stated and are provisional.

## Setup

Qwen2.5-0.5B fine-tuned on executable curricula (generators + exact programmatic verifiers), regimes compared at matched training tokens (100k), inference and verifier work recorded separately. Kaggle T4 runs dispatched through the central workflow.

## What was found

- **Untrained baseline.** Exact accuracy ~0% in every domain for Qwen2.5-0.5B and -Instruct, raw or chat-templated. For 1–2 digit arithmetic most of the gap was format (lenient score 0.66); symbolic, string and program tasks were a genuine skill gap.
- **001 (arithmetic curriculum, run 16).** Held-out exact: static 80.1%, adaptive 80.4%, error-focused 75.1%. No curriculum advantage; error-focused worse on every split. A later rerun (same seed) reproduced every held-out and out-of-range figure exactly: the pipeline is deterministic.
- **008 (corrective language game).** Neither arm learned to correct. Own-error training learned to *ignore* the shown answer (correction 74.8 vs confirmation 74.6); synthetic-error training learned to *copy* it (correction 4.6 vs confirmation 88.6). The untrained model, shown a wrong answer, broke 114/500 answers it otherwise got right: it arrives suggestible.
- **009 (verdict game, balanced right/wrong proposals).** The verdict collapsed to acceptance. verdict-mixed said `right` to everything (100% right-shown accepted, 0% wrong caught; balanced 50.0%); verdict-synthetic caught 13.4% of wrong proposals (balanced 51.9%) and repaired every catch. Self-check: 1 fix, 0 breaks. Both verdict arms below static on held-out.
- **Three formats, three shortcuts:** copy, ignore, accept. None used the proposal conditionally.
- **Built, never run:** 010 (verdict repair: dense verdicts, answer-before-verdict, REINFORCE/GRPO-style on-policy verdicts, graded vs binary self-check reward), 011 (verdict on constrained strings, where checking is cheaper than producing), and the fallible-challenge arms in 010/011 (hold/change against a corrector wrong 30% of the time, graded reward). 002–007 (ablation, symbolic, strings, program execution, logic, process supervision) were prepared and smoke-tested only.
- **Verdict prior diagnostic** (does the untrained model already prefer `right` regardless of truth?) was dispatched as Kaggle run 17 from main at `c35b185`; results pending, see `docs/session-handoff.md`.

## Literature conclusions

- **SCoRe** (arXiv:2409.12917): SFT on correction traces collapses to "no edits"; off-policy errors cause train/test mismatch. 008 adds that the error source sets the collapse direction.
- **Sycophancy** (Sharma et al. arXiv:2310.13548; FlipFlop arXiv:2311.08596): models flip correct answers under pushback. Our 114/500 is an exact-checked instance at 0.5B.
- **GenRM** (arXiv:2408.15240): joint verify-and-solve training helps at larger scale; balanced verdict SFT did not reproduce it at 0.5B.
- **Verification asymmetry** (Song et al. arXiv:2412.02674; Stechly et al. arXiv:2310.12397): self-improvement by self-verification needs checking to be easier than producing. Arithmetic lacks that gap; constrained strings have it.
- Self-correction without external feedback can hurt (Huang et al. arXiv:2310.01798); small models self-correct only with a strong verifier (arXiv:2404.17140).

## Why this led to Experiment 000

A gradient update is memory that judges silently: a correction overwrites weights and no reasoning ever meets the conflict. The shortcuts above are what unconscious judgment looks like, and the base model's suggestibility is inherited, not learned. Experiment 000 therefore builds the judgment, memory and trust explicitly in a small model trained from scratch, on the constrained-string task (which has the verification asymmetry), with the fallible-challenge game and its graded reward carried over from 010/011. See `docs/model-design.md` and `experiments/000-onwordly-learner/README.md`.
