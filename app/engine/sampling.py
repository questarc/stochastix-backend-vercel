"""Uniform sampling (PRNG / LHS / Sobol) and copulas (Gaussian / Student-t)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import qmc

SAMPLING_METHODS: tuple[str, ...] = ("prng", "lhs", "sobol")
COPULAS: tuple[str, ...] = ("gaussian", "student_t")

_EPS = 1e-12


def validate_sampling(sampling: str, copula: str, copula_df: float) -> None:
    if sampling not in SAMPLING_METHODS:
        raise ValueError(f"Unknown sampling method '{sampling}'. Choose one of {list(SAMPLING_METHODS)}.")
    if copula not in COPULAS:
        raise ValueError(f"Unknown copula '{copula}'. Choose one of {list(COPULAS)}.")
    if copula == "student_t" and copula_df <= 2:
        raise ValueError("Student-t copula needs degrees of freedom greater than 2.")


def uniform_samples(rng: np.random.Generator, sampling: str, n: int, d: int) -> np.ndarray:
    """Return an ``(n, d)`` array of uniforms in the open interval (0, 1)."""
    if sampling == "prng":
        u = rng.random((n, d))
    elif sampling == "lhs":
        u = qmc.LatinHypercube(d=d, seed=rng).random(n)
    elif sampling == "sobol":
        u = qmc.Sobol(d=d, scramble=True, seed=rng).random(n)
    else:
        raise ValueError(f"Unknown sampling method '{sampling}'.")
    return np.clip(u, _EPS, 1.0 - _EPS)


def correlated_uniforms(
    u: np.ndarray,
    correlation: np.ndarray,
    years: int,
    n_assets: int,
    copula: str,
    copula_df: float,
) -> np.ndarray:
    """Apply the copula. Returns uniforms of shape ``(trials, years, n_assets)``."""
    trials = u.shape[0]
    chol = np.linalg.cholesky(np.asarray(correlation, dtype=float))
    base = u[:, : years * n_assets].reshape(trials, years, n_assets)
    z = stats.norm.ppf(base) @ chol.T
    if copula == "gaussian":
        out = stats.norm.cdf(z)
    elif copula == "student_t":
        mix_u = u[:, years * n_assets : years * n_assets + years]
        chi2 = stats.chi2.ppf(mix_u, df=copula_df)
        t = z / np.sqrt(chi2 / copula_df)[:, :, None]
        out = stats.t.cdf(t, df=copula_df)
    else:
        raise ValueError(f"Unknown copula '{copula}'.")
    return np.clip(out, _EPS, 1.0 - _EPS)
