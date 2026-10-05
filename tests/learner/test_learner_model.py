"""Workspace model mechanics, parameter matching, memory reads, leakage guards."""
import pytest

torch = pytest.importorskip("torch")

from onwordly.learner.manifest import LearnerManifest
from onwordly.learner.memory_io import LearnerMemory
from onwordly.learner.model import Encoding, OnwordlyLearner, PlainNet, matched_plain_hidden, parameter_count
from onwordly.learner.task import Challenge, build_dataset
from onwordly.learner.train import handcoded_decision, calibration

ENC = Encoding(tuple("ABCDE12345"), 10)
TASK = build_dataset(seed=5, size=1, lengths=[6], partition="train")[0]


def test_encoding_round_trip() -> None:
    assert ENC.decode(ENC.slots_of("AB12")) == "AB12" and len(ENC.slots_of("AB12")) == 10
    assert len(ENC.problem(TASK)) == ENC.problem_dim


def test_any_step_count_and_every_step_emits() -> None:
    torch.manual_seed(0)
    model = OnwordlyLearner(ENC, d_model=32, layers=1, heads=2)
    g, x = torch.randn(3, ENC.global_dim), torch.randn(3, ENC.slots, ENC.extra_dim)
    init = torch.zeros(3, ENC.slots, ENC.vocab)
    for steps in (1, 3, 7):  # no fixed endpoint
        outs = model(g, x, init, steps)
        assert len(outs) == steps
        assert outs[-1].logits.shape == (3, ENC.slots, ENC.vocab) and outs[-1].confidence_logit.shape == (3,)
    outs = model(g, x, init, 2)
    # Every slot is revised at every step.
    assert (outs[1].logits - outs[0].logits).abs().amax(-1).gt(0).all()
    with pytest.raises(ValueError):
        model(g, x, init, 0)


def test_parameter_matching() -> None:
    target = parameter_count(OnwordlyLearner(ENC, d_model=64, layers=2, heads=4))
    plain = PlainNet(ENC, hidden=matched_plain_hidden(ENC, target))
    assert abs(parameter_count(plain) - target) / target < 0.02


def test_memory_reads_and_eval_isolation() -> None:
    memory = LearnerMemory(ENC, context="train:t")
    g, s = memory.read(TASK)
    assert g == [0.0, 0.0, 0.0]
    bad = TASK.witness[::-1] + "A"
    challenge = Challenge("A", bad, False, True, TASK.witness)
    links = memory.write_challenge(TASK, bad, challenge)
    memory.write_outcome(TASK, challenge, links, action="change", value=TASK.witness, outcome="changed_correct",
                         reward=0.6, final=TASK.witness, decision_ok=True, confidence=0.8, inputs={})
    g, s = memory.read(TASK)
    assert g[2] == 1.0  # divergence flag: draft vs proposal
    # Verifier fillers are stored but never read back as input.
    assert any(e.source == "verifier" for e in memory.register.entries(next(iter(memory.register.frames()))))
    assert memory.trust(TASK, "A")[1] == 2 / 3 and memory.trust(TASK, None)[2] == 0.0
    before = (len(memory.store), memory.ledger.snapshot(), len(memory.register.entries(memory.register.frames()[0])))
    view = memory.eval_view("eval")
    links = view.write_challenge(TASK, bad, challenge)
    view.write_outcome(TASK, challenge, links, action="hold", value=bad, outcome="held", reward=0.0, final=bad,
                       decision_ok=False, confidence=0.5, inputs={})
    after = (len(memory.store), memory.ledger.snapshot(), len(memory.register.entries(memory.register.frames()[0])))
    assert before == after and memory.ledger.frozen
    with pytest.raises(RuntimeError):
        memory.ledger.record("self", TASK.domain, True, step=99, partition="train")
    assert memory.store.verify() and memory.report()["chain_verified"]


def test_handcoded_rule_and_breaker() -> None:
    memory = LearnerMemory(ENC, context="train:h")
    said_wrong = Challenge("A", TASK.witness, True, False, "X")
    assert handcoded_decision(memory, TASK, Challenge("A", "x", False, False, None), threshold=0.7, strikes=3) == "hold"
    for step in range(10):
        memory.ledger.record("corrector", "A", True, step=step, partition="train")
    assert handcoded_decision(memory, TASK, said_wrong, threshold=0.7, strikes=3) == "change"
    for step in range(10, 13):
        memory.ledger.record("corrector", "A", False, step=step, partition="train")
    assert handcoded_decision(memory, TASK, said_wrong, threshold=0.7, strikes=3) == "hold"  # breaker open


def test_calibration() -> None:
    assert calibration([(1.0, True), (0.0, False)])["ece"] == 0.0
    assert calibration([(1.0, False)])["brier"] == 1.0 and calibration([])["ece"] is None


def test_manifests_load() -> None:
    for name in ("manifest.json", "smoke-manifest.json"):
        manifest = LearnerManifest.from_json(f"experiments/000-onwordly-learner/{name}")
        assert len(manifest.correctors) >= 2 and manifest.unseen_corrector[0] not in dict(manifest.correctors)
    full = LearnerManifest.from_json("experiments/000-onwordly-learner/manifest.json")
    params = parameter_count(OnwordlyLearner(Encoding(full.alphabet, full.slots), d_model=full.d_model, layers=full.layers, heads=full.heads))
    assert 1_000_000 <= params <= 5_000_000
