"""Frames, variant register tagging, divergence markers."""
from onwordly.memory import VariantRegister, arithmetic_frame, frame_for, rules_frame
from onwordly.learner.task import build_dataset


def test_frames() -> None:
    assert arithmetic_frame("add", 3, 2, 1) == arithmetic_frame("add", 2, 3, 1)
    assert arithmetic_frame("subtract", 3, 2, 1) != arithmetic_frame("subtract", 2, 3, 1)
    task = build_dataset(seed=1, size=1, lengths=[5], partition="train")[0]
    assert frame_for(task) == rules_frame(task.rules) and frame_for(task).startswith("rules:")
    # Frame depends on the rules only, never on the filler (witness).
    assert rules_frame(tuple(reversed(task.rules))) == rules_frame(task.rules)


def test_register_tags_counts_and_divergence() -> None:
    reg = VariantRegister()
    entry, marks = reg.add("f", " 272 ", record_time=3, context="train:x", subject="add:2", source="self")
    assert (entry.filler, entry.record_time, entry.context, entry.subject, entry.source, entry.count) == (
        "272", 3, "train:x", "add:2", "self", 1
    ) and marks == ()
    _, marks = reg.add("f", 272, record_time=4, context="train:x", subject="add:2", source="self")
    assert marks == ()  # same filler after canonicalisation: no divergence
    entry, marks = reg.add("f", 270, record_time=5, context="train:x", subject="add:2", source="B")
    assert len(marks) == 1 and (marks[0].existing, marks[0].new, marks[0].source) == ("272", "270", "B")
    _, marks = reg.add("f", 270, record_time=6, context="train:x", subject="add:2", source="B")
    assert marks == () and reg.entries("f")[-1].count == 2
    reg.add("f", 272, record_time=7, context="train:x", subject="add:2", source="verifier")
    assert len(reg.entries("f")) == 5  # add-only: every observation kept
    summary = reg.summary("f", exclude_sources=("verifier",))
    assert summary.entries == (("272", "self", 2), ("270", "B", 2))  # first-seen order, not ranked
    assert summary.divergent and summary.distinct_fillers == 2 and summary.total == 4
    assert reg.summary("f", cap=1).entries == (("272", "verifier", 1),)
    assert reg.summary("other").total == 0 and not reg.summary("other").divergent


def test_fork_isolates_writes() -> None:
    reg = VariantRegister()
    reg.add("f", "A", record_time=0, context="c", subject="s", source="self")
    fork = reg.fork()
    fork.add("f", "B", record_time=1, context="eval", subject="s", source="C")
    assert len(reg.entries("f")) == 1 and len(fork.entries("f")) == 2
    assert reg.divergences() == () and len(fork.divergences("f")) == 1
