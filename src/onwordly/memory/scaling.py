"""Per-episode update weight from confidence x surprise (docs/model-design.md, principle 6).

``confidence`` in [0, 1] is how sure the learner was of the move it made;
``surprise`` in [0, 1] is how wrong that move turned out (1 = wrong against the
exact verifier). The ``confidence_surprise`` rule is

    w = min(max_weight, 1 + gain * confidence * surprise)

so confident-and-wrong gets the largest update, unsure-and-wrong and any
correct move get the base weight 1.

This deliberately INVERTS Kalman-gain behaviour. A Kalman filter's gain
K = P / (P + R) shrinks as the prior becomes precise: a confident estimator
moves less when surprised. Precision-weighted prediction error and focal loss
are the nearest established relatives (focal loss up-weights confidently
*wrong* examples via (1 - p_true)^gamma). That confident-and-wrong should learn
most is the hypothesis under test, not an established result. The ``flat``
rule (w = 1) is its ablation.
"""
from __future__ import annotations

from typing import Sequence

UPDATE_RULES: tuple[str, ...] = ("confidence_surprise", "flat")


def update_weight(
    confidence: float,
    surprise: float,
    *,
    rule: str = "confidence_surprise",
    gain: float = 2.0,
    max_weight: float = 3.0,
) -> float:
    if rule not in UPDATE_RULES:
        raise ValueError(f"unknown update rule: {rule}")
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= surprise <= 1.0:
        raise ValueError("confidence and surprise must be in [0, 1]")
    if gain < 0 or max_weight < 1:
        raise ValueError("gain must be >= 0 and max_weight >= 1")
    if rule == "flat":
        return 1.0
    return min(max_weight, 1.0 + gain * confidence * surprise)


def normalize_weights(weights: Sequence[float]) -> list[float]:
    """Rescale to mean 1, so the rule redistributes update mass across
    episodes without changing the total step size (keeps arms comparable)."""
    if not weights:
        return []
    mean = sum(weights) / len(weights)
    if mean <= 0:
        raise ValueError("weights must have a positive mean")
    return [weight / mean for weight in weights]
