"""Drift analysis, trade proposals and order generation."""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .domain import ModelPortfolio, Position, total_market_value

ORDER_FIELDS = (
    "order_id", "account_id", "model_id", "side", "ticker", "asset_class", "quantity",
    "order_type", "time_in_force", "est_price", "est_value", "status", "created_at",
)


@dataclass
class RebalanceSettings:
    drift_tolerance: float = 0.02
    min_trade_value: float = 500.0
    whole_shares: bool = True
    default_securities: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "RebalanceSettings":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            drift_tolerance=float(data.get("drift_tolerance", 0.02)),
            min_trade_value=float(data.get("min_trade_value", 500)),
            whole_shares=bool(data.get("whole_shares", True)),
            default_securities=dict(data.get("default_securities", {})),
        )

    def override(self, drift_tolerance=None, min_trade_value=None, whole_shares=None) -> "RebalanceSettings":
        return RebalanceSettings(
            drift_tolerance=self.drift_tolerance if drift_tolerance is None else float(drift_tolerance),
            min_trade_value=self.min_trade_value if min_trade_value is None else float(min_trade_value),
            whole_shares=self.whole_shares if whole_shares is None else bool(whole_shares),
            default_securities=self.default_securities,
        )


@dataclass
class Trade:
    ticker: str
    asset_class: str
    side: str
    quantity: float
    price: float
    value: float


@dataclass
class RebalanceProposal:
    account_id: str
    model_id: str
    model_name: str
    market_value: float
    max_drift: float
    rebalance_needed: bool
    trades: list[Trade]
    allocation: list[dict]
    total_sells: float
    total_buys: float
    uninvested_cash: float
    post_trade_weights: dict[str, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("post_trade_weights")
        return d


def _qty(raw: float, whole_shares: bool) -> float:
    return float(math.floor(raw + 1e-9)) if whole_shares else float(raw)


def propose_rebalance(
    account_id: str, positions: list[Position], model: ModelPortfolio, settings: RebalanceSettings
) -> RebalanceProposal:
    total = total_market_value(positions)
    if total <= 0:
        raise ValueError(f"Account '{account_id}' has no market value to rebalance.")
    classes = list(model.weights) + [p.asset_class for p in positions if p.asset_class not in model.weights]
    classes = list(dict.fromkeys(classes))
    current = {ac: sum(p.market_value for p in positions if p.asset_class == ac) for ac in classes}
    target = {ac: float(model.weights.get(ac, 0.0)) * total for ac in classes}
    drift = {ac: (current[ac] - target[ac]) / total for ac in classes}
    max_drift = max(abs(d) for d in drift.values())
    needed = max_drift > settings.drift_tolerance

    trades: list[Trade] = []
    post = dict(current)
    proceeds = 0.0
    if needed:
        for ac in classes:
            excess = current[ac] - target[ac]
            if excess <= 0 or current[ac] <= 0:
                continue
            for p in (p for p in positions if p.asset_class == ac):
                raw = p.quantity if target[ac] <= 0 else p.quantity * excess / current[ac]
                qty = min(_qty(raw, settings.whole_shares), p.quantity)
                value = qty * p.price
                if qty <= 0 or value < settings.min_trade_value:
                    continue
                trades.append(Trade(p.ticker, ac, "Sell", qty, p.price, value))
                proceeds += value
                post[ac] -= value

        shortfall = {ac: target[ac] - current[ac] for ac in classes if target[ac] - current[ac] > 0}
        total_short = sum(shortfall.values())
        fraction = min(1.0, proceeds / total_short) if total_short > 0 else 0.0
        for ac, short in shortfall.items():
            budget = short * fraction
            ticker, price = _buy_security(ac, positions, settings)
            if ticker is None or price <= 0:
                continue
            qty = _qty(budget / price, settings.whole_shares)
            value = qty * price
            if qty <= 0 or value < settings.min_trade_value:
                continue
            trades.append(Trade(ticker, ac, "Buy", qty, price, value))
            post[ac] += value

    total_sells = sum(t.value for t in trades if t.side == "Sell")
    total_buys = sum(t.value for t in trades if t.side == "Buy")
    uninvested = max(total_sells - total_buys, 0.0)
    if uninvested > 0:
        if "Cash" not in post:
            classes.append("Cash")
            post["Cash"] = 0.0
            current["Cash"] = 0.0
        post["Cash"] += uninvested
    post_weights = {ac: post[ac] / total for ac in classes}
    allocation = [
        {"asset_class": ac, "current": current[ac] / total,
         "model": float(model.weights.get(ac, 0.0)), "post_trade": post_weights[ac]}
        for ac in classes
    ]
    return RebalanceProposal(
        account_id, model.model_id, model.model_name, total, max_drift, needed, trades,
        allocation, total_sells, total_buys, uninvested, post_weights,
    )


def _buy_security(asset_class: str, positions: list[Position], settings: RebalanceSettings):
    default = settings.default_securities.get(asset_class)
    if default:
        ticker = default["ticker"]
        held = next((p for p in positions if p.ticker == ticker), None)
        return ticker, float(held.price if held else default["price"])
    in_class = [p for p in positions if p.asset_class == asset_class]
    if not in_class:
        return None, 0.0
    largest = max(in_class, key=lambda p: p.market_value)
    return largest.ticker, float(largest.price)


def generate_orders(proposal: RebalanceProposal, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d%H%M%S")
    ordered = [t for t in proposal.trades if t.side == "Sell"] + [t for t in proposal.trades if t.side == "Buy"]
    return [
        {
            "order_id": f"{proposal.account_id}-{stamp}-{i:03d}",
            "account_id": proposal.account_id,
            "model_id": proposal.model_id,
            "side": t.side,
            "ticker": t.ticker,
            "asset_class": t.asset_class,
            "quantity": t.quantity,
            "order_type": "Market",
            "time_in_force": "Day",
            "est_price": t.price,
            "est_value": t.value,
            "status": "Pending",
            "created_at": now.isoformat(),
        }
        for i, t in enumerate(ordered, start=1)
    ]


def merge_orders(existing: list[dict], new: list[dict]) -> list[dict]:
    """New orders replace Pending orders of the same accounts."""
    accounts = {o["account_id"] for o in new}
    kept = [o for o in existing if not (o["account_id"] in accounts and o.get("status") == "Pending")]
    return kept + list(new)


def orders_to_csv(orders: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(ORDER_FIELDS), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(orders)
    return buf.getvalue()


def orders_to_json(orders: list[dict]) -> str:
    return json.dumps([{k: o.get(k) for k in ORDER_FIELDS} for o in orders], indent=2)
