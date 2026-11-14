from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..predictive_pricing import predict_stock_price

router = APIRouter(prefix="/predictive-pricing", tags=["Predictive Pricing Agent"])


class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=15)
    current_price: float = Field(gt=0, allow_inf_nan=False)


@router.post("/predict")
async def predict(request: PredictionRequest):
    return await run_in_threadpool(predict_stock_price, request.symbol, request.current_price)