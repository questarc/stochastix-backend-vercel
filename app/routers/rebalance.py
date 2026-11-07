from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from ..deps import get_cma, get_rebalance_settings, get_sources
from ..engine import CapitalMarketAssumptions
from ..rebalance import RebalanceSettings, propose_rebalance
from ..schemas import RebalanceRequest, RebalanceResponse, SimulationRequest
from ..services import load_account_portfolio
from .simulations import simulate_weights

router = APIRouter(tags=["rebalance"])


def build_proposal(req: RebalanceRequest, sources: dict, settings: RebalanceSettings):
    portfolio = load_account_portfolio(sources, req.account_id, req.model_id)
    effective = settings.override(req.drift_tolerance, req.min_trade_value, req.whole_shares)
    return portfolio, propose_rebalance(req.account_id, portfolio.positions, portfolio.model, effective)


def _run(req: RebalanceRequest, sources: dict, cma: CapitalMarketAssumptions, settings: RebalanceSettings) -> dict:
    portfolio, proposal = build_proposal(req, sources, settings)
    sim_req = req.simulation or SimulationRequest(account_id=req.account_id)
    sim_req = sim_req.model_copy(update={"sample_paths": 0})
    before = simulate_weights(portfolio, cma, sim_req, portfolio.current_weights).success_probability
    after = (simulate_weights(portfolio, cma, sim_req, proposal.post_trade_weights).success_probability
             if proposal.trades else before)
    return {**proposal.to_dict(), "success_current": before, "success_after": after}


@router.post("/rebalance/proposal", response_model=RebalanceResponse)
async def rebalance_proposal(
    req: RebalanceRequest,
    sources: dict = Depends(get_sources),
    cma: CapitalMarketAssumptions = Depends(get_cma),
    settings: RebalanceSettings = Depends(get_rebalance_settings),
):
    return await run_in_threadpool(_run, req, sources, cma, settings)


@router.get("/rebalance/settings")
def rebalance_settings(settings: RebalanceSettings = Depends(get_rebalance_settings)):
    return {
        "drift_tolerance": settings.drift_tolerance,
        "min_trade_value": settings.min_trade_value,
        "whole_shares": settings.whole_shares,
    }
