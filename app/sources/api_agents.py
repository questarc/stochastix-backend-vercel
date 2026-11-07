"""A360 and Orion API source agents (placeholders)."""
from __future__ import annotations

from ..domain import Account, ModelPortfolio, Position
from .base import SourceAgent, register


class _PlaceholderApiAgent(SourceAgent):
    label = "API"

    def _not_implemented(self):
        raise NotImplementedError(f"The {self.label} source agent is not implemented yet.")

    def list_accounts(self) -> list[Account]:
        self._not_implemented()

    def get_positions(self, account_id: str) -> list[Position]:
        self._not_implemented()

    def get_model_portfolio(self, model_id: str) -> ModelPortfolio:
        self._not_implemented()

    def list_model_portfolios(self) -> list[ModelPortfolio]:
        self._not_implemented()


@register("a360_api")
class A360ApiAgent(_PlaceholderApiAgent):
    label = "A360 API"


@register("orion_api")
class OrionApiAgent(_PlaceholderApiAgent):
    label = "Orion API"
