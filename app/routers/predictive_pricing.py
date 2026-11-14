from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..predictive_pricing import predict_stock_price

router = APIRouter(prefix="/predictive-pricing", tags=["Predictive Pricing Agent"])


class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=15)
    current_price: float = Field(gt=0, allow_inf_nan=False)


async def _predict_from_upstream(base_url: str, payload: dict) -> dict:
    endpoint = f"{base_url.rstrip('/')}/api/predictive-pricing/predict"
    try:
        async with httpx.AsyncClient(timeout=280.0) as client:
            response = await client.post(endpoint, json=payload)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Predictive Pricing service is unavailable.") from exc

    try:
        result = response.json()
    except ValueError:
        result = {}
    if response.is_error:
        detail = result.get("detail") if isinstance(result, dict) else None
        raise HTTPException(status_code=response.status_code, detail=detail or "Predictive Pricing request failed.")
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Predictive Pricing service returned an invalid response.")
    return result


@router.post("/predict")
async def predict(request: PredictionRequest):
    upstream_url = os.getenv("PREDICTIVE_PRICING_API_URL", "").strip()
    if upstream_url:
        return await _predict_from_upstream(upstream_url, request.model_dump())

    try:
        return await run_in_threadpool(predict_stock_price, request.symbol, request.current_price)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "yfinance"}:
            raise HTTPException(
                status_code=503,
                detail="Predictive Pricing is not installed here. Configure PREDICTIVE_PRICING_API_URL to a pricing backend.",
            ) from exc
        raise