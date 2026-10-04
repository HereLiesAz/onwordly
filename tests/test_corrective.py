from random import Random

from onwordly.tasks.arithmetic import make_arithmetic_task
from onwordly.tasks.corrective import make_corrective_task, synthetic_wrong_answer
from onwordly.training.corrective_evaluation import evaluate_corrective
from onwordly.training.corrective_sources import CorrectiveArithmeticSource
from onwordly.verifiers.arithmetic import verify_arithmetic_answer


def test_corrective_task_targets_true_answer() -> None:
    base = make_arithmetic_task(47, 6, "multiply")
    task = make_corrective_task(base, 272)
    assert task.kind == "correct" and task.answer == 282 and "272" in task.prompt
    assert verify_arithmetic_answer(task, "282")
    confirm = make_corrective_task(base, 282)
    assert confirm.kind == "confirm" and confirm.target_text == "282"


def test_synthetic_wrong_answer_is_never_right() -> None:
    rng = Random(0)
    for left, right, op in [(0, 0, "add"), (9, 9, "multiply"), (5, 7, "subtract"), (47, 6, "multiply")]:
        base = make_arithmetic_task(left, right, op)
        for _ in range(50):
            assert synthetic_wrong_answer(base, rng) != base.answer


def _source(previous: str) -> CorrectiveArithmeticSource:
    tasks = [make_arithmetic_task(2, 3, "add"), make_arithmetic_task(4, 4, "multiply")]
    return CorrectiveArithmeticSource(tasks, previous=previous, confirm_probability=1.0, seed=1)


def test_own_arm_shows_model_answer_and_synthetic_arm_does_not() -> None:
    rng = Random(0)
    own, synthetic = _source("own"), _source("synthetic")
    base_own, base_syn = own.next_task(rng), synthetic.next_task(rng)
    own.observe_response(base_own, "7", False)
    synthetic.observe_response(base_syn, "7", False)
    shown_own = own.next_task(rng)
    shown_syn = synthetic.next_task(rng)
    assert shown_own.previous == 7 and shown_own.answer == 5
    assert shown_syn.kind == "correct" and shown_syn.answer == 5
    assert own.queued == synthetic.queued == {"correct": 1, "confirm": 0}


def test_unparseable_wrong_answers_and_corrective_tasks_queue_nothing() -> None:
    rng = Random(0)
    source = _source("own")
    base = source.next_task(rng)
    source.observe_response(base, "The answer is 5", False)
    assert not source.pending
    source.observe_response(base, "5", True)  # confirm_probability = 1
    confirm = source.next_task(rng)
    assert confirm.kind == "confirm"
    source.observe_response(confirm, "9", False)
    assert not source.pending


class EchoAdapter:
    """Answers arithmetic correctly but always repeats any shown previous answer."""

    def generate(self, prompt: str) -> str:
        if "previous answer was" in prompt:
            return prompt.split("previous answer was ")[1].split(".")[0]
        left, rest = prompt.split("Compute ")[1].split(" + ")
        return str(int(left) + int(rest.split(".")[0]))


def test_evaluation_separates_correction_from_confirmation() -> None:
    heldout = [make_arithmetic_task(a, b, "add") for a, b in [(1, 2), (3, 4), (5, 6)]]
    result = evaluate_corrective(EchoAdapter(), heldout, seed=0)
    assert result["correction_accuracy"] == 0.0
    assert result["confirmation_accuracy"] == 1.0
    self_correction = result["self_correction"]
    assert self_correction["first_pass_accuracy"] == self_correction["second_pass_accuracy"] == 1.0
    assert result["generation_calls"] == 3 * 3 + 3
