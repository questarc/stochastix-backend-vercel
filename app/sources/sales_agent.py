"""Sales source agent: accounts, positions and models from CSV files."""
from __future__ import annotations

import pandas as pd

from ..domain import Account, ModelPortfolio, Position
from .base import SourceAgent, register


def read_accounts_csv(path) -> list[Account]:
    df = pd.read_csv(path, dtype={"account_id": str, "model_id": str})
    return [
        Account(
            account_id=str(r.account_id),
            client_name=str(r.client_name),
            advisor=str(r.advisor),
            age=int(r.age),
            retirement_age=int(r.retirement_age),
            plan_end_age=int(r.plan_end_age),
            annual_contribution=float(r.annual_contribution),
            annual_withdrawal=float(r.annual_withdrawal),
            risk_profile=str(r.risk_profile),
            model_id=str(r.model_id),
        )
        for r in df.itertuples(index=False)
    ]


def read_models_csv(path) -> list[ModelPortfolio]:
    df = pd.read_csv(path, dtype={"model_id": str})
    models: list[ModelPortfolio] = []
    for model_id, group in df.groupby("model_id", sort=False):
        weights = {str(r.asset_class): float(r.weight) for r in group.itertuples(index=False)}
        models.append(ModelPortfolio(str(model_id), str(group["model_name"].iloc[0]), weights))
    return models


@register("sales")
class SalesAgent(SourceAgent):
    def list_accounts(self) -> list[Account]:
        return read_accounts_csv(self.resolve_path(self.config["accounts_file"]))

    def get_positions(self, account_id: str) -> list[Position]:
        self.get_account(account_id)
        df = pd.read_csv(self.resolve_path(self.config["positions_file"]), dtype={"account_id": str})
        df = df[df["account_id"] == account_id]
        return [
            Position(str(r.account_id), str(r.ticker), str(r.asset_class), float(r.quantity), float(r.price))
            for r in df.itertuples(index=False)
        ]

    def list_model_portfolios(self) -> list[ModelPortfolio]:
        return read_models_csv(self.resolve_path(self.config["models_file"]))

    def get_model_portfolio(self, model_id: str) -> ModelPortfolio:
        for model in self.list_model_portfolios():
            if model.model_id == model_id:
                return model
        raise KeyError(f"Unknown model portfolio '{model_id}'.")
