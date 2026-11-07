"""Application services shared by the API routers and the Advisor AI tools."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .domain import Account, ModelPortfolio, Position, allocation_from_positions, total_market_value
from .engine import CapitalMarketAssumptions, SimulationInputs, SimulationResult, simulate
from .sources.base import SourceAgent

logger = logging.getLogger(__name__)

HISTOGRAM_BINS = 60


@dataclass
class AccountPortfolio:
    source: str
    account: Account
    positions: list[Position]
    current_weights: dict[str, float]
    model: ModelPortfolio
    market_value: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "account": self.account.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "current_weights": dict(self.current_weights),
            "model": self.model.to_dict(),
            "market_value": self.market_value,
        }


def find_source_for_account(sources: dict[str, SourceAgent], account_id: str) -> SourceAgent:
    for source in sources.values():
        if source.has_account(account_id):
            return source
    raise KeyError(f"Unknown account '{account_id}'.")


def load_account_portfolio(
    sources: dict[str, SourceAgent], account_id: str, model_id: str | None = None
) -> AccountPortfolio:
    source = find_source_for_account(sources, account_id)
    account = source.get_account(account_id)
    positions = source.get_positions(account_id)
    resolved = resolve_model_id(source, model_id) if model_id else account.model_id
    model = source.get_model_portfolio(resolved)
    return AccountPortfolio(
        source=source.name,
        account=account,
        positions=positions,
        current_weights=allocation_from_positions(positions),
        model=model,
        market_value=total_market_value(positions),
    )


def resolve_model_id(source: SourceAgent, text: str) -> str:
    """Accept a model ID or a model name, case-insensitive."""
    needle = text.strip().lower()
    for model in source.list_model_portfolios():
        if model.model_id.lower() == needle or model.model_name.lower() == needle:
            return model.model_id
    raise KeyError(f"Unknown model portfolio '{text}'.")


def find_account_ids(sources: dict[str, SourceAgent], client_name: str) -> list[str]:
    """Case-insensitive 'contains' match on client name; 'all' returns every account."""
    needle = client_name.strip().lower()
    ids: list[str] = []
    for source in sources.values():
        for account in source.list_accounts():
            if needle == "all" or needle in account.client_name.lower():
                if account.account_id not in ids:
                    ids.append(account.account_id)
    return ids


def run_portfolio_simulation(
    portfolio: AccountPortfolio,
    cma: CapitalMarketAssumptions,
    weights: dict[str, float],
    retirement_age: int | None = None,
    plan_end_age: int | None = None,
    annual_contribution: float | None = None,
    annual_withdrawal: float | None = None,
    trials: int = 10_000,
    seed: int | None = 42,
    sampling: str = "prng",
    copula: str = "gaussian",
    copula_df: float = 5.0,
) -> SimulationResult:
    a = portfolio.account
    inputs = SimulationInputs(
        start_value=portfolio.market_value,
        weights=weights,
        current_age=a.age,
        retirement_age=a.retirement_age if retirement_age is None else int(retirement_age),
        plan_end_age=a.plan_end_age if plan_end_age is None else int(plan_end_age),
        annual_contribution=a.annual_contribution if annual_contribution is None else float(annual_contribution),
        annual_withdrawal=a.annual_withdrawal if annual_withdrawal is None else float(annual_withdrawal),
        trials=int(trials),
        seed=seed,
        sampling=sampling,
        copula=copula,
        copula_df=float(copula_df),
    )
    result = simulate(inputs, cma)
    logger.info("Simulation account=%s settings=%s success=%.4f", a.account_id, result.settings, result.success_probability)
    return result


def summarize_result(result: SimulationResult) -> dict:
    depl = result.depletion_ages
    return {
        "success_probability": float(result.success_probability),
        "median_ending_value_todays_dollars": float(np.median(result.ending_values_real)),
        "failed_trials": int(depl.size),
        "median_depletion_age": float(np.median(depl)) if depl.size else None,
        "earliest_depletion_age": int(depl.min()) if depl.size else None,
    }


def result_to_api(result: SimulationResult, sample_paths: int = 0) -> dict:
    """Compact API representation; never returns the full path matrix."""
    counts, edges = np.histogram(result.ending_values, bins=HISTOGRAM_BINS)
    if result.depletion_ages.size:
        d_ages, d_counts = np.unique(result.depletion_ages, return_counts=True)
    else:
        d_ages, d_counts = np.array([], dtype=int), np.array([], dtype=int)
    out = {
        "ages": [int(x) for x in result.ages],
        "bands": {f"p{p}": result.percentiles[p].tolist() for p in (10, 25, 50, 75, 90)},
        "ending_values_histogram": {"bin_edges": edges.tolist(), "counts": counts.tolist()},
        "depletion_ages_histogram": {"ages": d_ages.tolist(), "counts": d_counts.tolist()},
        "settings": result.settings,
        **summarize_result(result),
    }
    n = min(max(int(sample_paths or 0), 0), 100)
    if n:
        out["sample_paths"] = result.paths[:n].tolist()
    return out
