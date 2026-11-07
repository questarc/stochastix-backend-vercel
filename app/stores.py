"""In-memory stores behind small interfaces so a database can replace them later."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from .domain import Position
from .rebalance import merge_orders


class OrderStore(ABC):
    @abstractmethod
    def list(self) -> list[dict]: ...

    @abstractmethod
    def merge(self, orders: list[dict]) -> list[dict]: ...

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryOrderStore(OrderStore):
    def __init__(self) -> None:
        self._orders: list[dict] = []
        self._lock = threading.Lock()

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(o) for o in self._orders]

    def merge(self, orders: list[dict]) -> list[dict]:
        with self._lock:
            self._orders = merge_orders(self._orders, orders)
            return list(orders)

    def clear(self) -> None:
        with self._lock:
            self._orders = []


class UploadStore(ABC):
    @abstractmethod
    def get(self, source: str, account_id: str) -> list[Position] | None: ...

    @abstractmethod
    def set(self, source: str, account_id: str, positions: list[Position]) -> None: ...

    @abstractmethod
    def reset(self, source: str, account_id: str) -> bool: ...


class InMemoryUploadStore(UploadStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[Position]] = {}
        self._lock = threading.Lock()

    def get(self, source: str, account_id: str) -> list[Position] | None:
        with self._lock:
            return self._data.get((source, account_id))

    def set(self, source: str, account_id: str, positions: list[Position]) -> None:
        with self._lock:
            self._data[(source, account_id)] = list(positions)

    def reset(self, source: str, account_id: str) -> bool:
        with self._lock:
            return self._data.pop((source, account_id), None) is not None


class InMemoryEtradeStatementStore:
    """Keep allowlisted statement analytics per browser session; never retain PDF or full extracted text."""

    def __init__(self) -> None:
        self._statements: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            statement = self._statements.get(session_id)
            return dict(statement) if statement else None

    def set(self, session_id: str, statement: dict) -> None:
        rows = [
            {
                "date": row["date"],
                "type": "Transfer",
                "amount": float(row["amount"]),
                "status": row["status"],
            }
            for row in statement["transactions"]
        ]
        summary = statement["summary"]
        structure_preview = statement["text"] if statement["document_type"] == "statement_structure" else ""
        portfolio_summary = statement.get("portfolio_summary")
        account_summaries = [
            {
                field: row.get(field)
                for field in (
                    "account_value",
                    "change_in_account_value",
                    "beginning_value",
                    "additions",
                    "subtractions",
                    "transaction_costs",
                    "investment_change",
                    "ending_value",
                    "accrued_interest",
                    "income_this_period",
                    "income_year_to_date",
                    "top_holdings_value",
                    "top_holdings_percent",
                )
            }
            for row in statement.get("account_summaries", [])
        ]
        holdings = [
            {
                field: row.get(field)
                for field in (
                    "beginning_market_value",
                    "quantity",
                    "price_per_unit",
                    "ending_market_value",
                    "cost_basis",
                    "unrealized_gain_loss",
                    "estimated_annual_income",
                    "estimated_yield_percent",
                )
            }
            for row in statement.get("holdings", [])
        ]
        activity_transactions = [
            {
                field: row.get(field)
                for field in (
                    "settlement_date",
                    "security_name",
                    "symbol_cusip",
                    "description",
                    "quantity",
                    "price",
                    "total_cost_basis",
                    "transaction_cost",
                    "amount",
                )
            }
            for row in statement.get("activity_transactions", [])
        ]
        safe_statement = {
            "filename": statement["filename"],
            "page_count": int(statement["page_count"]),
            "text": structure_preview,
            "character_count": len(structure_preview),
            "imported_at": statement["imported_at"],
            "document_type": statement["document_type"],
            "transactions": rows,
            "summary": dict(summary) if summary else None,
            "portfolio_summary": {
                "beginning_value": portfolio_summary.get("beginning_value"),
                "ending_value": portfolio_summary.get("ending_value"),
                "reported_change": portfolio_summary.get("reported_change"),
                "total_including_other_holdings": portfolio_summary.get("total_including_other_holdings"),
                "statement_period": portfolio_summary.get("statement_period"),
                "income_this_period": portfolio_summary.get("income_this_period"),
                "income_year_to_date": portfolio_summary.get("income_year_to_date"),
                "top_holdings_value": portfolio_summary.get("top_holdings_value"),
                "top_holdings_percent": portfolio_summary.get("top_holdings_percent"),
            } if portfolio_summary else None,
            "account_summaries": account_summaries,
            "holdings": holdings,
            "activity_transactions": activity_transactions,
        }
        with self._lock:
            self._statements[session_id] = safe_statement

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._statements.pop(session_id, None) is not None


class InMemoryFidelityPositionsStore:
    """Keep only displayed Fidelity position fields per browser session."""

    ALLOWED_FIELDS = (
        "symbol",
        "description",
        "quantity",
        "last_price",
        "last_price_change",
        "current_value",
        "todays_gain_loss_dollar",
        "todays_gain_loss_percent",
        "total_gain_loss_dollar",
        "total_gain_loss_percent",
        "percent_of_account",
        "cost_basis_total",
        "average_cost_basis",
        "type",
    )

    def __init__(self) -> None:
        self._positions: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict] | None:
        with self._lock:
            positions = self._positions.get(session_id)
            return [dict(row) for row in positions] if positions is not None else None

    def set(self, session_id: str, positions: list[dict]) -> None:
        safe_positions = [
            {field: row.get(field) for field in self.ALLOWED_FIELDS}
            for row in positions
        ]
        with self._lock:
            self._positions[session_id] = safe_positions

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._positions.pop(session_id, None) is not None
