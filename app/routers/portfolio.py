from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_cma, get_sources
from ..engine import COPULAS, DISTRIBUTIONS, SAMPLING_METHODS, CapitalMarketAssumptions
from ..schemas import AccountPortfolio
from ..services import load_account_portfolio

router = APIRouter(tags=["portfolio"])

TRIAL_CHOICES = [1_000, 5_000, 10_000, 25_000, 50_000]


@router.get("/accounts/{account_id}/portfolio", response_model=AccountPortfolio)
def get_portfolio(account_id: str, model_id: str | None = None, sources: dict = Depends(get_sources)):
    return load_account_portfolio(sources, account_id, model_id).to_dict()


@router.get("/config/cma")
def get_cma_config(cma: CapitalMarketAssumptions = Depends(get_cma)):
    return cma.to_dict()


@router.get("/config/options")
def get_options():
    return {
        "sampling_methods": list(SAMPLING_METHODS),
        "copulas": list(COPULAS),
        "distributions": list(DISTRIBUTIONS),
        "trial_choices": TRIAL_CHOICES,
    }
