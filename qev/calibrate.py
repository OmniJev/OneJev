"""Calibration: temperatures per (primitive, option count), fitting, and probability metrics."""
from __future__ import annotations

import json
import logging
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class Calibration:
    """Temperature table. Keys are "kind" or "kind:K". Missing keys fall back to the primitive-level
    temperature, then to 1.0."""

    temperatures: dict[str, float] = field(default_factory=dict)
    source: str = "none"

    def __post_init__(self) -> None:
        self.temperatures = {k: _positive_temperature(v) for k, v in self.temperatures.items()}

    def temperature(self, kind: str, k: int) -> float:
        return self.temperatures.get(f"{kind}:{k}", self.temperatures.get(kind, 1.0))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(temperatures=data.get("temperatures", {}), source=str(path))

    def save(self, path: str | Path) -> None:
        from .prompt import PROMPT_VERSION

        Path(path).write_text(json.dumps({"temperatures": self.temperatures, "source": self.source,
                                          "prompt_version": PROMPT_VERSION}, indent=2))


def _positive_temperature(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("temperature must be finite and positive") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")
    return value


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    temperature = _positive_temperature(temperature)
    z = [float(x) for x in logits]
    if not z or any(math.isnan(x) or x == math.inf for x in z) or max(z) == -math.inf:
        raise ValueError("logits need a finite maximum and cannot contain NaN or positive infinity")
    m = max(z)
    w = [math.exp((x - m) / temperature) for x in z]
    s = sum(w)
    return [x / s for x in w]


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]],
                    lo: float = 0.05, hi: float = 50.0, iters: int = 60) -> float:
    """Fit one scalar temperature by minimising cross-entropy against (possibly soft) targets.
    Golden-section search on log T; no torch dependency so it runs anywhere."""
    _same_length(logits, targets)
    if not len(logits):
        raise ValueError("temperature fitting requires at least one example")
    lo, hi = _positive_temperature(lo), _positive_temperature(hi)
    if lo >= hi or iters <= 0:
        raise ValueError("temperature fitting needs lo < hi and positive iters")
    rows = []
    for z, y in zip(logits, targets):
        _same_length(z, y)
        if not len(z) or any(not math.isfinite(x) for x in z):
            raise ValueError("fitting logits must be non-empty and finite")
        if any(not math.isfinite(v) or v < 0 for v in y) or not math.isclose(sum(y), 1.0, abs_tol=1e-6):
            raise ValueError("fitting targets must be probability distributions")
        maximum = max(z)
        rows.append(([float(x) - maximum for x in z], list(y)))

    def nll(t: float) -> float:
        total = 0.0
        for z, y in rows:
            scaled = [x / t for x in z]
            log_partition = math.log(sum(math.exp(x) for x in scaled))
            total += sum(yi * (log_partition - zi) for yi, zi in zip(y, scaled) if yi)
        return total / len(rows)

    a, b = math.log(lo), math.log(hi)
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(math.exp(c)), nll(math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(math.exp(d))
    t = math.exp((a + b) / 2)
    if t >= hi * 0.99 or t <= lo * 1.01:
        logging.getLogger("qev.calibrate").warning(
            "temperature fit hit the %s bound (T=%.4f) on %d rows; the slot logits likely carry no signal for this group",
            "upper" if t >= hi * 0.99 else "lower", t, len(logits))
    return t


def _same_length(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError("paired values must have the same length")


def _check_ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int) -> None:
    _same_length(confidences, correct)
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if any(not 0 <= c <= 1 for c in confidences) or any(not 0 <= y <= 1 for y in correct):
        raise ValueError("confidences and correctness values must lie in [0, 1]")


def ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 15) -> float:
    _check_ece(confidences, correct, n_bins)
    n = len(confidences)
    if n == 0:
        return 0.0
    boundaries = [(b + 1) / n_bins for b in range(n_bins)]
    errors = [0.0] * n_bins
    for confidence, outcome in zip(confidences, correct):
        errors[bisect_left(boundaries, confidence)] += confidence - outcome
    return sum(abs(error) for error in errors) / n


def adaptive_ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 15) -> float:
    _check_ece(confidences, correct, n_bins)
    n = len(confidences)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: confidences[i])
    total = 0.0
    bins = min(n, n_bins)
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        avg_acc = sum(correct[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(avg_conf - avg_acc)
    return total


def brier(probs: Sequence[float], target: Sequence[float]) -> float:
    _same_length(probs, target)
    return sum((p - t) ** 2 for p, t in zip(probs, target))


def nll(probs: Sequence[float], target: Sequence[float]) -> float:
    _same_length(probs, target)
    return -sum(t * math.log(max(p, 1e-12)) for p, t in zip(probs, target))


def tvd(p: Sequence[float], q: Sequence[float]) -> float:
    _same_length(p, q)
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))
