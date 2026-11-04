"""Answer assembly: probabilities to typed answers, with TypeSafe's confidence formulas.

The two confidence functions are ported from typesafe-ai/system-one-adapter-python
(MIT License, Copyright (c) 2026 TypeSafe AI) and verified against Jev's published answers
(mean absolute error 0.013 for Choice, 0.005 for Score, within two-decimal rounding).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .schema import ChoiceAnswer, NoulAnswer, ScoreAnswer


def _normalize(probs: Sequence[float]) -> list[float]:
    if not len(probs):
        raise ValueError("at least one probability is required")
    values = [float(p) for p in probs]
    if any(not math.isfinite(p) or p < 0 for p in values):
        raise ValueError("probabilities must be finite and non-negative")
    scale = max(values)
    if scale == 0:
        return [1.0 / len(probs)] * len(probs)
    # Scaling first also handles finite weights whose sum would overflow.
    values = [p / scale for p in values]
    total = sum(values)
    return [p / total for p in values]


def choice_confidence(probs: Sequence[float]) -> float:
    """Peak probability rescaled from uniform (0) to certainty (1)."""
    p = _normalize(probs)
    if len(p) == 1:
        return 1.0
    uniform = 1.0 / len(p)
    return min(1.0, max(0.0, (max(p) - uniform) / (1.0 - uniform)))


def score_confidence(probs: Sequence[float]) -> float:
    """One minus the probability-weighted mean absolute deviation from the modal level,
    normalised by the mean absolute deviation of the uniform distribution over the levels."""
    p = _normalize(probs)
    if len(p) == 1:
        return 1.0
    k = len(p)
    mode = max(range(k), key=p.__getitem__)
    distance = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    center = (k - 1) / 2
    uniform_mad = sum(abs(i - center) for i in range(k)) / k
    return max(0.0, 1.0 - distance / uniform_mad)


def expected_score(probs: Sequence[float]) -> float:
    p = _normalize(probs)
    return float(sum(i * pi for i, pi in enumerate(p)))


def _round(x: float, nd: int = 6) -> float:
    return float(round(x, nd))


def make_noul(p_yes: float) -> NoulAnswer:
    if not math.isfinite(p_yes):
        raise ValueError("noul probability must be finite")
    return NoulAnswer(noul=_round(min(1.0, max(0.0, float(p_yes)))))


def make_choice(options: Sequence[str], probs: Sequence[float]) -> ChoiceAnswer:
    if len(options) != len(probs):
        raise ValueError("options and probabilities must have the same length")
    p = _normalize(probs)
    best = max(range(len(p)), key=p.__getitem__)
    return ChoiceAnswer(
        choice=options[best],
        probabilities={o: _round(pi) for o, pi in zip(options, p)},
        confidence=_round(choice_confidence(p)),
    )


def make_score(levels: Sequence[Any], probs: Sequence[float]) -> ScoreAnswer:
    if len(levels) != len(probs):
        raise ValueError("levels and probabilities must have the same length")
    p = _normalize(probs)
    return ScoreAnswer(
        score=_round(expected_score(p)),
        legend={str(i): lvl for i, lvl in enumerate(levels)},
        probabilities={str(i): _round(pi) for i, pi in enumerate(p)},
        confidence=_round(score_confidence(p)),
    )
