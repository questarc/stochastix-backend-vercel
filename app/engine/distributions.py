"""Return distributions parameterised so mean and standard deviation match the CMA inputs."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

DISTRIBUTIONS: tuple[str, ...] = (
    "lognormal",
    "normal",
    "uniform",
    "triangular",
    "pert",
    "exponential",
    "binomial",
)

PERT_LAMBDA = 4.0


class Marginal:
    """A one-year return distribution exposing ``ppf``, ``mean`` and ``std``."""

    def __init__(self, name: str, mean: float, std: float):
        if name not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution '{name}'. Choose one of {list(DISTRIBUTIONS)}.")
        if std < 0:
            raise ValueError("Volatility must be non-negative.")
        self.name = name
        self.mean = float(mean)
        self.std = float(std)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        m, s = self.mean, self.std
        if s == 0:
            return np.full_like(u, m)
        if self.name == "lognormal":
            gross = 1.0 + m
            if gross <= 0:
                raise ValueError("Lognormal requires expected return greater than -100%.")
            sigma2 = math.log(1.0 + (s / gross) ** 2)
            mu = math.log(gross) - sigma2 / 2.0
            return np.exp(mu + math.sqrt(sigma2) * stats.norm.ppf(u)) - 1.0
        if self.name == "normal":
            return m + s * stats.norm.ppf(u)
        if self.name == "uniform":
            half = s * math.sqrt(3.0)
            return (m - half) + 2.0 * half * u
        if self.name == "triangular":
            half = s * math.sqrt(6.0)
            return stats.triang.ppf(u, c=0.5, loc=m - half, scale=2.0 * half)
        if self.name == "pert":
            a = 1.0 + PERT_LAMBDA / 2.0
            half = s * math.sqrt(2.0 * a + 1.0)
            return stats.beta.ppf(u, a, a, loc=m - half, scale=2.0 * half)
        if self.name == "exponential":
            return stats.expon.ppf(u, loc=m - s, scale=s)
        # binomial: two equally likely outcomes, m - s and m + s
        return np.where(u < 0.5, m - s, m + s)


def make_marginal(name: str, mean: float, std: float) -> Marginal:
    return Marginal(name, mean, std)
