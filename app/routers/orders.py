from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from ..deps import get_order_store, get_rebalance_settings, get_sources
from ..rebalance import RebalanceSettings, generate_orders, orders_to_csv, orders_to_json
from ..schemas import Order, RebalanceRequest
from ..stores import OrderStore
from .rebalance import build_proposal

router = APIRouter(tags=["orders"])


@router.post("/orders/generate", response_model=list[Order])
def generate(
    req: RebalanceRequest,
    sources: dict = Depends(get_sources),
    settings: RebalanceSettings = Depends(get_rebalance_settings),
    store: OrderStore = Depends(get_order_store),
):
    _, proposal = build_proposal(req, sources, settings)
    orders = generate_orders(proposal) if proposal.trades else []
    if orders:
        store.merge(orders)
    return orders


@router.get("/orders", response_model=list[Order])
def list_orders(store: OrderStore = Depends(get_order_store)):
    return store.list()


@router.delete("/orders")
def clear_orders(store: OrderStore = Depends(get_order_store)):
    store.clear()
    return {"cleared": True}


@router.get("/orders/export")
def export_orders(format: Literal["csv", "json"] = "csv", store: OrderStore = Depends(get_order_store)):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    orders = store.list()
    if format == "json":
        body, media = orders_to_json(orders), "application/json"
    else:
        body, media = orders_to_csv(orders), "text/csv"
    headers = {"Content-Disposition": f'attachment; filename="orders_{stamp}.{format}"'}
    return Response(content=body, media_type=media, headers=headers)
