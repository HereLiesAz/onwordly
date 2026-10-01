from __future__ import annotations

from typing import Protocol


class TrainableTask(Protocol):
    """Minimal task contract for equal-token supervised training."""

    prompt: str

    @property
    def target_text(self) -> str:
        ...

    @property
    def bucket_key(self) -> str:
        ...
