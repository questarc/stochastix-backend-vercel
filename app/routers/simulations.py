from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from ..deps import get_cma, get_sources
from ..engine import CapitalMarketAssumptions
from ..schemas import SimulationRequest, SimulationResponse
from ..services import AccountPortfolio, load_account_portfolio, result_to_api, run_portfolio_simulation

router = APIRouter(tags=["simulations"])


def effective_cma(cma: CapitalMarketAssumptions, req: SimulationRequest) -> CapitalMarketAssumptions:
    return cma.with_distribution(req.distribution) if req.distribution else cma


def simulate_weights(portfolio: AccountPortfolio, cma: CapitalMarketAssumptions,
                     req: SimulationRequest, weights: dict[str, float]):
    return run_portfolio_simulation(
        portfolio, effective_cma(cma, req), weights,
        retirement_age=req.retirement_age, plan_end_age=req.plan_end_age,
        annual_contribution=req.annual_contribution, annual_withdrawal=req.annual_withdrawal,
        trials=req.trials, seed=req.seed, sampling=req.sampling, copula=req.copula, copula_df=req.copula_df,
    )


def _run(req: SimulationRequest, sources: dict, cma: CapitalMarketAssumptions) -> dict:
    portfolio = load_account_portfolio(sources, req.account_id, req.model_id)
    current = simulate_weights(portfolio, cma, req, portfolio.current_weights)
    model = simulate_weights(portfolio, cma, req, portfolio.model.weights)
    return {
        "account_id": req.account_id,
        "model_id": portfolio.model.model_id,
        "current": result_to_api(current, req.sample_paths),
        "model": result_to_api(model, req.sample_paths),
    }


@router.post("/simulations", response_model=SimulationResponse, response_model_exclude_none=False)
async def run_simulations(req: SimulationRequest, sources: dict = Depends(get_sources),
                          cma: CapitalMarketAssumptions = Depends(get_cma)):
    return await run_in_threadpool(_run, req, sources, cma)
