"""Memory diagnostics for Experiment 000 (CPU, tiny)."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from onwordly.learner.manifest import LearnerManifest
from onwordly.learner.model import Encoding
from onwordly.learner.task import Corrector, build_dataset, draw_challenge
from onwordly.learner.train import (
    VARIANTS,
    _inputs,
    make_memory,
    probe_sample,
    probe_second_visit,
    probe_training_frames,
    train_learner,
)
from onwordly.memory import frame_for

CPU = torch.device("cpu")


def _setup(**overrides):
    manifest = LearnerManifest.from_json("experiments/000-onwordly-learner/smoke-manifest.json")
    manifest = replace(manifest, **{"train_steps": 4, "train_size": 12, "eval_size": 10, "probe_size": 8, **overrides})
    kw = dict(lengths=manifest.lengths, alphabet=manifest.alphabet, optional_rules=manifest.optional_rules,
              modulus=manifest.holdout_modulus)
    train = build_dataset(seed=manifest.dataset_seed, size=manifest.train_size, partition="train", **kw)
    held = build_dataset(seed=manifest.evaluation_seed, size=manifest.eval_size, partition="eval", **kw)
    return manifest, train, held


def _populated_memory(manifest, tasks):
    encoding = Encoding(manifest.alphabet, manifest.slots)
    memory = make_memory(manifest, encoding, "test")
    rng = __import__("random").Random(0)
    for task in tasks:
        challenge = draw_challenge(task, task.witness[::-1], Corrector("A", 0.0), rng)
        link = memory.write_challenge(task, task.witness[::-1], challenge)
        memory.write_outcome(task, challenge, link, action="hold", value=challenge.first, outcome="held", reward=0.0,
                             final=challenge.first, decision_ok=False, confidence=0.5, inputs={})
    return encoding, memory


def test_empty_register_path_is_really_empty() -> None:
    manifest, train, _ = _setup()
    encoding, memory = _populated_memory(manifest, train[:4])
    g_full, x_full = _inputs(encoding, train[:4], memory, VARIANTS["learner"], None, CPU)
    g_empty, x_empty = _inputs(encoding, train[:4], memory, VARIANTS["learner"], None, CPU, empty=True)
    fresh = make_memory(manifest, encoding, "fresh")
    g_fresh, x_fresh = _inputs(encoding, train[:4], fresh, VARIANTS["learner"], None, CPU)
    assert not torch.equal(x_full, x_empty)
    # Emptied read == a genuinely empty register (trust features identical: both ledgers differ, so compare memory part).
    assert torch.equal(x_empty, x_fresh)
    assert torch.equal(x_empty, torch.zeros_like(x_empty))


def test_self_memory_excludes_corrector_and_verifier() -> None:
    manifest, train, _ = _setup()
    _, memory = _populated_memory(manifest, train[:1])
    task = train[0]
    sources = {e.source for e in memory.register.entries(frame_for(task))}
    assert {"self", "A", "verifier"} <= sources
    excluded = memory._excluded(task, "self")
    assert set(excluded) == sources - {"self"}
    summary = memory.register.summary(frame_for(task), exclude_sources=excluded)
    assert {s for _, s, _ in summary.entries} == {"self"}
    # Self read carries no "other" histogram; full read does (corrector A proposed the witness).
    _, per_slot = memory.read(task, "self")
    vocab = memory.encoding.vocab
    assert all(sum(row[vocab:]) == 0 for row in per_slot)
    _, per_slot_all = memory.read(task, "all")
    assert any(sum(row[vocab:]) > 0 for row in per_slot_all)
    nonempty, top_other = memory.register_stats(task, "self")
    assert nonempty and top_other is None
    assert memory.register_stats(task, "all")[1] == task.witness


def test_dropout_rate_respected_and_budget_matched() -> None:
    manifest, train, _ = _setup(memory_dropout=0.5, train_steps=40)
    _, _, log = train_learner(manifest, VARIANTS["learner-memory-dropout"], train, device=CPU, regime="d")
    reads = log.register_reads
    assert reads == 40 * manifest.batch_size == log.training_problems
    assert 0.35 < log.register_dropped / reads < 0.65
    _, _, base = train_learner(manifest, VARIANTS["learner"], train, device=CPU, regime="b")
    assert base.register_dropped == 0 and base.optimizer_steps == log.optimizer_steps
    for rate in (0.0, 1.0):
        m = replace(manifest, memory_dropout=rate, train_steps=3)
        _, _, l = train_learner(m, VARIANTS["learner-memory-dropout"], train, device=CPU, regime="d")
        assert l.register_dropped == rate * l.register_reads


def test_probes_do_not_leak_and_second_visit_sees_only_self() -> None:
    manifest, train, held = _setup()
    variant = VARIANTS["learner"]
    model, memory, _ = train_learner(manifest, variant, train, device=CPU, regime="p")
    before = (len(memory.store), {f: len(memory.register.entries(f)) for f in memory.register.frames()}, memory.step)
    tasks = probe_sample(manifest, train)
    probe = probe_training_frames(model, memory, variant, tasks, manifest.revision_steps, CPU)
    assert probe["view_records_written"] == 0
    assert probe["register_nonempty_fraction"] == 1.0  # every training frame was visited
    second = probe_second_visit(model, memory, variant, held, manifest.revision_steps, CPU)
    assert second["first_visit_empty_fraction"] == 1.0
    assert second["second_visit_register_sources"] == ["self"]
    after = (len(memory.store), {f: len(memory.register.entries(f)) for f in memory.register.frames()}, memory.step)
    assert before == after
    assert not any(frame_for(t) in memory.register.frames() for t in held)
    with pytest.raises(RuntimeError):
        memory.write_self_filler(held[0], "AAAAA")
