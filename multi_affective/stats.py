from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quantiles:
    p50: float
    p80: float
    p90: float
    p95: float
    p99: float


def quantiles(values: list[int]) -> Quantiles:
    if not values:
        return Quantiles(0, 0, 0, 0, 0)
    xs = sorted(values)

    def _q(p: float) -> float:
        if len(xs) == 1:
            return float(xs[0])
        pos = p * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        if hi == lo:
            return float(xs[lo])
        w = pos - lo
        return xs[lo] * (1 - w) + xs[hi] * w

    return Quantiles(
        p50=_q(0.50),
        p80=_q(0.80),
        p90=_q(0.90),
        p95=_q(0.95),
        p99=_q(0.99),
    )

