"""Calibration: temperatures per (primitive, option count), fitting, and probability metrics."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class Calibration:
    """Temperature table. Keys are "kind" or "kind:K". Missing keys fall back to the primitive-level
    temperature, then to 1.0."""

    temperatures: dict[str, float] = field(default_factory=dict)
    source: str = "none"

    def temperature(self, kind: str, k: int) -> float:
        return self.temperatures.get(f"{kind}:{k}", self.temperatures.get(kind, 1.0))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(temperatures={k: float(v) for k, v in data.get("temperatures", {}).items()}, source=str(path))

    def save(self, path: str | Path) -> None:
        from .prompt import PROMPT_VERSION

        Path(path).write_text(json.dumps({"temperatures": self.temperatures, "source": self.source,
                                          "prompt_version": PROMPT_VERSION}, indent=2))


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    z = [float(x) / temperature for x in logits]
    m = max(z)
    w = [math.exp(x - m) for x in z]
    s = sum(w)
    return [x / s for x in w]


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]],
                    lo: float = 0.05, hi: float = 50.0, iters: int = 60) -> float:
    """Fit one scalar temperature by minimising cross-entropy against (possibly soft) targets.
    Golden-section search on log T; no torch dependency so it runs anywhere."""
    def nll(t: float) -> float:
        total = 0.0
        for z, y in zip(logits, targets):
            p = softmax(z, t)
            total -= sum(yi * math.log(max(pi, 1e-12)) for yi, pi in zip(y, p))
        return total / max(len(logits), 1)

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


def ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 15) -> float:
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c <= lo)]
        if idx:
            avg_conf = sum(confidences[i] for i in idx) / len(idx)
            avg_acc = sum(correct[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(avg_conf - avg_acc)
    return total


def adaptive_ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 15) -> float:
    n = len(confidences)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: confidences[i])
    total = 0.0
    size = max(1, n // n_bins)
    for start in range(0, n, size):
        idx = order[start:start + size]
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        avg_acc = sum(correct[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(avg_conf - avg_acc)
    return total


def brier(probs: Sequence[float], target: Sequence[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(probs, target))


def nll(probs: Sequence[float], target: Sequence[float]) -> float:
    return -sum(t * math.log(max(p, 1e-12)) for p, t in zip(probs, target))


def tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))
