"""Episode store: append-only, immutable records, hash chain."""
from dataclasses import FrozenInstanceError

import pytest

from onwordly.memory import GENESIS_HASH, EpisodeStore, Record


def filled() -> EpisodeStore:
    store = EpisodeStore()
    a = store.append("attempt", {"frame": "f", "answer": "AB"}, step=0)
    store.append("correction", {"proposed": "BA"}, step=1, links=(a.index,))
    store.append("verifier_result", {"right": False}, step=1, links=(0, 1))
    return store


def test_append_only_api() -> None:
    store = filled()
    for name in ("update", "delete", "remove", "pop", "clear", "insert", "__setitem__", "__delitem__"):
        assert not hasattr(store, name)
    with pytest.raises(TypeError):
        store[0] = store[1]  # type: ignore[index]
    with pytest.raises(TypeError):
        del store[0]  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        store[0].payload = "{}"  # type: ignore[misc]
    records = store.records
    assert isinstance(records, tuple) and len(store) == 3
    # Mutating what a read returns cannot reach the stored record.
    data = store[0].data
    data["answer"] = "ZZ"
    assert store[0].data["answer"] == "AB"


def test_hash_chain_links_and_tamper_detection(tmp_path) -> None:
    store = filled()
    assert store[0].prev_hash == GENESIS_HASH and store[1].prev_hash == store[0].hash
    assert store.verify() and store.head == store[2].hash
    assert store[1].links == (0,)
    path = tmp_path / "store.jsonl"
    store.dump_jsonl(path)
    assert EpisodeStore.load_jsonl(path).head == store.head
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0].replace("AB", "XY"), *lines[1:]]) + "\n")
    with pytest.raises(ValueError):
        EpisodeStore.load_jsonl(path)


def test_validation() -> None:
    store = EpisodeStore()
    with pytest.raises(ValueError):
        store.append("rewrite", {}, step=0)
    with pytest.raises(ValueError):
        store.append("attempt", {}, step=0, links=(0,))  # cannot link forward
    with pytest.raises(ValueError):
        store.append("attempt", {}, step=-1)
    store.append("attempt", {}, step=0)
    assert store.counts()["attempt"] == 1 and isinstance(store[0], Record)
