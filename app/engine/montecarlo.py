"""Monte Carlo retirement projection engine."""
from __future__ import annotations

import json
import secrets
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .distributions import DISTRIBUTIONS, make_marginal
from .sampling import correlated_uniforms, uniform_samples, validate_sampling

PERCENTILES = (10, 25, 50, 75, 90)


@dataclass(frozen=True)
class CapitalMarketAssumptions:
    asset_classes: list[str]
    expected_return: dict[str, float]
    volatility: dict[str, float]
    correlation: np.ndarray
    distributions: dict[str, str]
    inflation: float = 0.025

    @classmethod
    def from_dict(cls, data: dict) -> "CapitalMarketAssumptions":
        classes = list(data["asset_classes"])
        dists = {ac: data.get("distributions", {}).get(ac, "lognormal") for ac in classes}
        return cls(
            asset_classes=classes,
            expected_return={ac: float(data["expected_return"][ac]) for ac in classes},
            volatility={ac: float(data["volatility"][ac]) for ac in classes},
            correlation=np.asarray(data["correlation"], dtype=float),
            distributions=dists,
            inflation=float(data.get("inflation", 0.025)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CapitalMarketAssumptions":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def with_distribution(self, name: str) -> "CapitalMarketAssumptions":
        if name not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution '{name}'. Choose one of {list(DISTRIBUTIONS)}.")
        return replace(self, distributions={ac: name for ac in self.asset_classes})

    def to_dict(self) -> dict:
        return {
            "inflation": self.inflation,
            "asset_classes": list(self.asset_classes),
            "expected_return": dict(self.expected_return),
            "volatility": dict(self.volatility),
            "correlation": self.correlation.tolist(),
            "distributions": dict(self.distributions),
        }


@dataclass
class SimulationInputs:
    start_value: float
    weights: dict[str, float]
    current_age: int
    retirement_age: int
    plan_end_age: int
    annual_contribution: float = 0.0
    annual_withdrawal: float = 0.0
    trials: int = 10_000
    seed: int | None = 42
    sampling: str = "prng"
    copula: str = "gaussian"
    copula_df: float = 5.0


@dataclass
class SimulationResult:
    ages: np.ndarray
    paths: np.ndarray
    success_probability: float
    percentiles: dict[int, np.ndarray]
    ending_values: np.ndarray
    ending_values_real: np.ndarray
    depletion_ages: np.ndarray
    settings: dict = field(default_factory=dict)


def _validate(inputs: SimulationInputs) -> None:
    validate_sampling(inputs.sampling, inputs.copula, inputs.copula_df)
    if inputs.start_value < 0 or inputs.annual_contribution < 0 or inputs.annual_withdrawal < 0:
        raise ValueError("Start value, contribution and withdrawal must not be negative.")
    if not (inputs.current_age <= inputs.retirement_age < inputs.plan_end_age):
        raise ValueError("Ages must satisfy current age <= retirement age < plan end age.")
    if inputs.trials <= 0:
        raise ValueError("Trials must be greater than zero.")


def _weight_vector(weights: dict[str, float], cma: CapitalMarketAssumptions) -> np.ndarray:
    unknown = [ac for ac in weights if ac not in cma.asset_classes]
    if unknown:
        raise ValueError(f"Unknown asset classes: {unknown}. Known: {cma.asset_classes}.")
    w = np.array([float(weights.get(ac, 0.0)) for ac in cma.asset_classes])
    total = w.sum()
    if total <= 0:
        raise ValueError("Portfolio weights must sum to more than zero.")
    return w / total


def simulate(inputs: SimulationInputs, cma: CapitalMarketAssumptions) -> SimulationResult:
    _validate(inputs)
    years = inputs.plan_end_age - inputs.current_age
    ages = np.arange(inputs.current_age, inputs.plan_end_age + 1)
    seed = inputs.seed if inputs.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)
    w = _weight_vector(inputs.weights, cma)
    n_assets = len(cma.asset_classes)

    dims = years * n_assets + (years if inputs.copula == "student_t" else 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u = uniform_samples(rng, inputs.sampling, inputs.trials, dims)
    cu = correlated_uniforms(u, cma.correlation, years, n_assets, inputs.copula, inputs.copula_df)

    asset_returns = np.empty_like(cu)
    for j, ac in enumerate(cma.asset_classes):
        marginal = make_marginal(cma.distributions[ac], cma.expected_return[ac], cma.volatility[ac])
        asset_returns[:, :, j] = marginal.ppf(cu[:, :, j])
    asset_returns = np.maximum(asset_returns, -1.0)
    port_returns = asset_returns @ w

    paths = np.empty((inputs.trials, years + 1))
    paths[:, 0] = inputs.start_value
    value = np.full(inputs.trials, float(inputs.start_value))
    for t in range(1, years + 1):
        age = inputs.current_age + t - 1
        r = port_returns[:, t - 1]
        inflation_factor = (1.0 + cma.inflation) ** t
        if age < inputs.retirement_age:
            value = value * (1.0 + r) + inputs.annual_contribution
        else:
            value = (value - inputs.annual_withdrawal * inflation_factor) * (1.0 + r)
        value = np.maximum(value, 0.0)
        paths[:, t] = value

    ending = paths[:, -1]
    success = float(np.mean(ending > 0))
    pct = {p: np.percentile(paths, p, axis=0) for p in PERCENTILES}
    ending_real = ending / (1.0 + cma.inflation) ** years
    failed = ending <= 0
    first_zero = np.argmax(paths[failed] <= 0, axis=1)
    depletion_ages = ages[first_zero] if failed.any() else np.array([], dtype=int)

    settings = {
        "seed": int(seed),
        "sampling": inputs.sampling,
        "copula": inputs.copula,
        "copula_df": float(inputs.copula_df),
        "trials": int(inputs.trials),
        "years": int(years),
        "distributions": dict(cma.distributions),
    }
    return SimulationResult(ages, paths, success, pct, ending, ending_real, depletion_ages, settings)
