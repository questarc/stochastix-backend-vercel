"""Core domain objects: accounts, positions and model portfolios."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Account:
    account_id: str
    client_name: str
    advisor: str
    age: int
    retirement_age: int
    plan_end_age: int
    annual_contribution: float
    annual_withdrawal: float
    risk_profile: str
    model_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Position:
    account_id: str
    ticker: str
    asset_class: str
    quantity: float
    price: float

    @property
    def market_value(self) -> float:
        return float(self.quantity) * float(self.price)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["market_value"] = self.market_value
        return d


@dataclass(frozen=True)
class ModelPortfolio:
    model_id: str
    model_name: str
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "model_name": self.model_name, "weights": dict(self.weights)}


def total_market_value(positions: list[Position]) -> float:
    return float(sum(p.market_value for p in positions))


def allocation_from_positions(positions: list[Position]) -> dict[str, float]:
    """Current weights by asset class (fractions summing to 1)."""
    total = total_market_value(positions)
    if total <= 0:
        return {}
    weights: dict[str, float] = {}
    for p in positions:
        weights[p.asset_class] = weights.get(p.asset_class, 0.0) + p.market_value / total
    return weights
