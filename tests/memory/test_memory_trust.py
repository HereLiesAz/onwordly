"""Trust ledger math (Beta, decay, saturation), leakage guard, update weights."""
import math

import pytest

from onwordly.memory import BetaTrust, EpisodeStore, TrustLedger, normalize_weights, update_weight


def test_beta_updates_and_raw_counts() -> None:
    ledger = TrustLedger(EpisodeStore())
    assert ledger.self_trust("d") == 0.5
    for step, ok in enumerate([True, True, False, True]):
        ledger.record("self", "d", ok, step=step, partition="train")
    entry = ledger.entry("self", "d")
    assert (entry.alpha, entry.beta) == (4.0, 2.0) and math.isclose(entry.mean, 4 / 6)
    assert entry.success_rate == 0.75 and entry.trials == 4 and entry.failure_streak == 0
    for step in range(4, 7):
        ledger.record("corrector", "B", False, step=step, partition="train")
    assert ledger.entry("corrector", "B").failure_streak == 3
    assert ledger.store.counts()["trust_update"] == 7 and ledger.store.verify()


def test_decay_toward_prior_with_half_life() -> None:
    trust = BetaTrust(11.0, 1.0, 1.0, 1.0, step=0)
    half = trust.decayed(100, 100)
    assert math.isclose(half.alpha, 6.0) and math.isclose(half.beta, 1.0)
    assert math.isclose(trust.decayed(10_000, 100).mean, 0.5, abs_tol=1e-6)
    assert trust.decayed(100, None) is trust
    ledger = TrustLedger(EpisodeStore(), half_life=10)
    for step in range(20):
        ledger.record("self", "d", True, step=step, partition="train")
    high = ledger.self_trust("d")
    assert high > ledger.self_trust("d", step=1000) and math.isclose(ledger.self_trust("d", step=10**6), 0.5)


def test_saturates_below_one() -> None:
    ledger = TrustLedger(EpisodeStore(), max_evidence=50)
    for step in range(500):
        ledger.record("self", "d", True, step=step, partition="train")
    entry = ledger.entry("self", "d")
    assert entry.evidence <= 50 + 1e-9 and entry.mean < 1.0 and math.isclose(entry.mean, 51 / 52)


def test_training_only_and_freeze() -> None:
    ledger = TrustLedger(EpisodeStore())
    with pytest.raises(ValueError):
        ledger.record("self", "d", True, step=0, partition="eval")
    ledger.record("self", "d", True, step=0, partition="train")
    ledger.freeze()
    with pytest.raises(RuntimeError):
        ledger.record("self", "d", True, step=1, partition="train")
    assert ledger.self_trust("d") == 2 / 3
    with pytest.raises(ValueError):
        TrustLedger(EpisodeStore()).record("oracle", "d", True, step=0, partition="train")


def test_replay_from_store() -> None:
    store = EpisodeStore()
    ledger = TrustLedger(store, half_life=5, max_evidence=8)
    for step in range(30):
        ledger.record("self", "d", step % 3 != 0, step=step, partition="train")
        ledger.record("corrector", "A", step % 5 != 0, step=step, partition="train")
    replayed = TrustLedger.replay(store, half_life=5, max_evidence=8)
    assert replayed.snapshot() == ledger.snapshot()


def test_update_weight_rule() -> None:
    confident_wrong = update_weight(0.9, 1.0)
    unsure_wrong = update_weight(0.1, 1.0)
    confident_right = update_weight(0.9, 0.0)
    assert confident_wrong > unsure_wrong > confident_right == 1.0
    assert update_weight(1.0, 1.0, gain=10, max_weight=3) == 3.0
    assert update_weight(0.9, 1.0, rule="flat") == 1.0
    with pytest.raises(ValueError):
        update_weight(1.5, 0.0)
    with pytest.raises(ValueError):
        update_weight(0.5, 0.5, rule="kalman")
    weights = normalize_weights([1.0, 3.0])
    assert weights == [0.5, 1.5] and normalize_weights([]) == []
