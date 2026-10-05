"""learner-child and the eval-time memory-source ablation (CPU, tiny)."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from onwordly.learner import train as train_mod
from onwordly.learner.experiment import run_learner_experiment
from onwordly.learner.manifest import LearnerManifest
from onwordly.learner.model import MEMORY_GLOBAL_DIM, Encoding
from onwordly.learner.recurring import recurring_frames, recurring_schedule, run_recurring
from onwordly.learner.task import Corrector, build_dataset
from onwordly.learner.train import VARIANTS, _inputs, learner_decider, make_memory, solve_learner, train_learner
from onwordly.memory import frame_for

CPU = torch.device("cpu")
CONDITIONS = [(Corrector("A", 0.1), True), (Corrector("B", 0.5), True), (Corrector("C", 0.3), False)]
SMOKE = "experiments/000-onwordly-learner/recurring-smoke-manifest.json"


def _setup(**overrides):
    manifest = replace(LearnerManifest.from_json(SMOKE), **{"train_steps": 4, "train_size": 12, "eval_size": 12, **overrides})
    kw = dict(lengths=manifest.lengths, alphabet=manifest.alphabet, optional_rules=manifest.optional_rules,
              modulus=manifest.holdout_modulus)
    train = build_dataset(seed=manifest.dataset_seed, size=manifest.train_size, partition="train", **kw)
    held = build_dataset(seed=manifest.evaluation_seed, size=manifest.eval_size, partition="eval", **kw)
    return manifest, train, held


def _seeded_view(manifest, task):
    """Eval view whose register for ``task`` holds one self and one corrector filler."""
    memory = make_memory(manifest, Encoding(manifest.alphabet, manifest.slots), "t")
    for _ in range(5):  # give the ledger a non-trivial self-trust
        memory.ledger.record("self", task.domain, True, step=0, partition="train")
    view = memory.eval_view("eval:test")
    view.write_self_filler(task, "AB")
    view._add(task, task.witness, "A")
    view._add(task, task.witness, "verifier")
    return memory, view


def test_child_variant_definition() -> None:
    child, base = VARIANTS["learner-child"], VARIANTS["learner-memory-dropout"]
    assert child.memory_dropout and base.memory_dropout
    assert not child.self_trust and child.weighting == "flat" and child.register_sources == "correctors"
    assert replace(child, self_trust=True, weighting=base.weighting, register_sources="all") == base


def test_child_has_no_self_trust_input() -> None:
    manifest, _, held = _setup()
    task = held[0]
    memory, view = _seeded_view(manifest, task)
    enc = Encoding(manifest.alphabet, manifest.slots)
    col = enc.problem_dim + MEMORY_GLOBAL_DIM  # trust[0] = self-trust
    g_child, _ = _inputs(enc, [task], view, VARIANTS["learner-child"], None, CPU)
    g_base, _ = _inputs(enc, [task], view, VARIANTS["learner-memory-dropout"], None, CPU)
    assert memory.ledger.self_trust(task.domain) > 0
    assert g_base[0, col].item() > 0 and g_child[0, col].item() == 0.0
    assert torch.equal(g_child[0, col + 1 :], g_base[0, col + 1 :])  # corrector reliability untouched


def test_child_update_weighting_ignores_own_confidence(monkeypatch) -> None:
    manifest, train, _ = _setup()
    calls = []
    real = train_mod.update_weight

    def spy(confidence, surprise, **kw):
        calls.append(kw["rule"])
        return real(confidence, surprise, **kw)

    monkeypatch.setattr(train_mod, "update_weight", spy)
    _, _, log = train_learner(manifest, VARIANTS["learner-child"], train, device=CPU, regime="c")
    assert calls and set(calls) == {"flat"}
    assert log.mean_weight_wrong in (None, 1.0) and log.mean_weight_right in (None, 1.0)
    # Same matched budget as learner-memory-dropout.
    _, _, base = train_learner(manifest, VARIANTS["learner-memory-dropout"], train, device=CPU, regime="d")
    assert (log.optimizer_steps, log.training_problems, log.forward_steps) == (
        base.optimizer_steps, base.training_problems, base.forward_steps)


def test_source_restricted_reads() -> None:
    manifest, _, held = _setup()
    task, task2 = recurring_frames(held, 2)
    _, view = _seeded_view(manifest, task)
    vocab = Encoding(manifest.alphabet, manifest.slots).vocab
    reads = {s: view.read(task, s)[1] for s in ("self", "correctors", "all")}
    mine = {s: any(v for row in r for v in row[:vocab]) for s, r in reads.items()}
    told = {s: any(v for row in r for v in row[vocab:]) for s, r in reads.items()}
    assert mine == {"self": True, "correctors": False, "all": True}
    assert told == {"self": False, "correctors": True, "all": True}
    for s in ("self", "correctors", "all"):
        assert "verifier" in view._excluded(task, s)
    assert "self" in view._excluded(task, "correctors")
    # Only self fillers present: the child (correctors) read is empty.
    view.write_self_filler(task2, "AB")
    assert view.register_stats(task2, "correctors") == (False, None)
    assert view.read(task2, "correctors") == view.empty_read()
    assert view.register_stats(task2, "self")[0]
    with pytest.raises(ValueError):
        view.read(task, "teacher")


def test_ablation_runs_read_only_their_sources_and_never_touch_training_memory() -> None:
    manifest, train, held = _setup()
    variant = VARIANTS["learner-memory-dropout"]
    model, memory, _ = train_learner(manifest, variant, train, device=CPU, regime="r")
    snap = lambda: (len(memory.store), memory.store.head, {f: len(memory.register.entries(f)) for f in memory.register.frames()},
                    memory.step, memory.ledger.snapshot())
    before = snap()
    tasks = recurring_frames(held, 6)
    stream = recurring_schedule(len(tasks), 3, seed="x", min_gap=2)
    for sources, allowed in (("self", {"self"}), ("correctors", {"A", "B", "C"}), ("all", {"self", "A", "B", "C"})):
        alt = replace(variant, register_sources=sources)
        seen = set()

        def solve(batch, view, alt=alt, sources=sources, seen=seen):
            for t in batch:
                s = view.register.summary(frame_for(t), exclude_sources=view._excluded(t, sources), cap=100)
                seen.update(src for _, src, _ in s.entries)
            return [row[-1] for row in solve_learner(model, view, alt, batch, manifest.revision_steps, CPU)[0]]

        _, view = run_recurring(tasks, stream, solve=solve, decide=learner_decider(model, alt, manifest.revision_steps, CPU),
                                memory=memory, correctors=CONDITIONS, seed=1, sources=sources)
        assert seen and seen <= allowed
        assert "verifier" not in seen
        assert snap() == before
        assert not any(frame_for(t) in memory.register.frames() for t in tasks)


def test_experiment_reports_ablation_rows(tmp_path) -> None:
    manifest = replace(
        LearnerManifest.from_json(SMOKE), train_steps=2, train_size=40, eval_size=24, recurring_frames=8,
        recurring_min_gap=2, probe_size=8, arms=("learner-memory-dropout", "learner-child"),
        recurring_source_ablation=("learner-memory-dropout", "learner-child"),
    )
    summary = run_learner_experiment(manifest, output_dir=tmp_path, device=CPU)
    dropout, child = summary["arms"]["learner-memory-dropout"], summary["arms"]["learner-child"]
    assert set(dropout["recurring_source_ablation"]) == {"self", "correctors"}
    assert set(child["recurring_source_ablation"]) == {"self", "all"}
    assert child["recurring_sources"] == "correctors"
    text = (tmp_path / "RESULTS.md").read_text()
    for row in ("| learner-memory-dropout[self] | 1 |", "| learner-memory-dropout[correctors] | 3 |", "| learner-child[all] |"):
        assert row in text
