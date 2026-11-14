from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..predictive_pricing import predict_stock_list, predict_stock_price

router = APIRouter(prefix="/predictive-pricing", tags=["Predictive Pricing Agent"])


class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=100)


class StockListRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=10)


async def _predict_from_upstream(base_url: str, payload: dict, endpoint: str = "/predict") -> dict:
    endpoint = f"{base_url.rstrip('/')}/api/predictive-pricing{endpoint}"
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


async def _run_local(predictor, *args):
    try:
        return await run_in_threadpool(predictor, *args)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "yfinance"}:
            raise HTTPException(
                status_code=503,
                detail="Predictive Pricing is not installed here. Configure PREDICTIVE_PRICING_API_URL to a pricing backend.",
            ) from exc
        raise


@router.post("/predict")
async def predict(request: PredictionRequest):
    upstream_url = os.getenv("PREDICTIVE_PRICING_API_URL", "").strip()
    if upstream_url:
        return await _predict_from_upstream(upstream_url, {"symbol": request.symbol})
    return await _run_local(predict_stock_price, request.symbol)


@router.post("/batch")
async def predict_batch(request: StockListRequest):
    upstream_url = os.getenv("PREDICTIVE_PRICING_API_URL", "").strip()
    if upstream_url:
        return await _predict_from_upstream(upstream_url, request.model_dump(), endpoint="/batch")
    return await _run_local(predict_stock_list, request.symbols)