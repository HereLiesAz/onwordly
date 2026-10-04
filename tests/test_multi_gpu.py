from onwordly.experiments.multi_gpu import assign_regimes


def test_assign_regimes_round_robin() -> None:
    assert assign_regimes(["static", "adaptive", "error-focused"], 2) == [
        ["static", "error-focused"],
        ["adaptive"],
    ]
    assert assign_regimes(["a", "b"], 4) == [["a"], ["b"]]
    assert assign_regimes(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]
