from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from random import Random
from typing import Literal, Sequence

StringOperation = Literal[
    "remove_vowels",
    "take_even_indices",
    "reverse_pairs",
    "duplicate_each",
]
StringPartition = Literal["train", "eval"]
STRING_OPERATIONS: tuple[StringOperation, ...] = (
    "remove_vowels",
    "take_even_indices",
    "reverse_pairs",
    "duplicate_each",
)

_RULES: dict[StringOperation, str] = {
    "remove_vowels": "remove every vowel (A, E, I, O, U)",
    "take_even_indices": "keep only characters at zero-based indices 0, 2, 4, ...",
    "reverse_pairs": "reverse each consecutive pair of characters independently; leave a final unpaired character unchanged",
    "duplicate_each": "replace every character with two copies of itself",
}


@dataclass(frozen=True, slots=True)
class StringTask:
    prompt: str
    answer: str
    operation: StringOperation
    length: int
    text: str

    @property
    def target_text(self) -> str:
        return self.answer

    @property
    def bucket_key(self) -> str:
        return f"{self.operation}:{self.length}"


def apply_string_operation(text: str, operation: StringOperation) -> str:
    if not text:
        raise ValueError("text cannot be empty")
    if operation == "remove_vowels":
        result = "".join(char for char in text if char not in "AEIOU")
        return result or "<EMPTY>"
    if operation == "take_even_indices":
        return text[::2]
    if operation == "reverse_pairs":
        chunks = [text[index:index + 2] for index in range(0, len(text), 2)]
        return "".join(chunk[::-1] for chunk in chunks)
    if operation == "duplicate_each":
        return "".join(char * 2 for char in text)
    raise ValueError(f"unsupported string operation: {operation}")


def make_string_task(text: str, operation: StringOperation) -> StringTask:
    normalized = text.upper()
    if not normalized or any(not char.isalnum() or not char.isascii() for char in normalized):
        raise ValueError("text must contain only ASCII letters and digits")
    answer = apply_string_operation(normalized, operation)
    prompt = (
        f"Transform {normalized} using this rule: {_RULES[operation]}. "
        "Return only the transformed string with no spaces or explanation. "
        "If the result is empty, return <EMPTY>."
    )
    return StringTask(
        prompt=prompt,
        answer=answer,
        operation=operation,
        length=len(normalized),
        text=normalized,
    )


def generate_string_task(
    rng: Random,
    operation: StringOperation,
    length: int,
    *,
    alphabet: Sequence[str] = tuple("ABCDE12345"),
) -> StringTask:
    if length < 1:
        raise ValueError("length must be positive")
    symbols = tuple(char.upper() for char in alphabet)
    if not symbols or any(len(char) != 1 or not char.isalnum() or not char.isascii() for char in symbols):
        raise ValueError("alphabet entries must be single ASCII alphanumeric characters")
    text = "".join(rng.choice(symbols) for _ in range(length))
    return make_string_task(text, operation)


def string_partition(
    task: StringTask,
    *,
    modulus: int = 5,
) -> StringPartition:
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    key = f"{task.operation}:{task.text}".encode("utf-8")
    residue = int.from_bytes(blake2b(key, digest_size=8).digest(), "big") % modulus
    return "eval" if residue == 0 else "train"
