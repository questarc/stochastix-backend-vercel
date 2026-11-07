"""Positions source agent: accounts and models from CSV, positions from A360 Positions exports."""
from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path

import pandas as pd

from ..domain import Position
from .base import register
from .sales_agent import SalesAgent

REQUIRED_COLUMNS = ("Qty", "Market Value", "Ticker", "Cusip")
FIDELITY_REQUIRED_COLUMNS = ("Symbol", "Description", "Quantity", "Last price", "Current value", "Type")
FIDELITY_CSV_DISPLAY_COLUMNS = (
    "Symbol",
    "Description",
    "Quantity",
    "Last price",
    "Last price change",
    "Current value",
    "Today's gain/loss dollar",
    "Today's gain/loss percent",
    "Total gain/loss dollar",
    "Total gain/loss percent",
    "Percent of account",
    "Cost basis total",
    "Average cost basis",
    "Type",
)
FIDELITY_CURRENCY_COMMA = re.compile(r"(?<=\$)\d{1,3}(?:,\d{3})+\.\d{2,4}")


def parse_number(value) -> float:
    """Parse '$1,234.50', '+12', '(123)' (= -123) and blanks (= 0)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return 0.0 if (isinstance(value, float) and math.isnan(value)) else float(value)
    text = str(value).strip().replace("$", "").replace(",", "").replace("+", "")
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    number = float(text) if text else 0.0
    return -number if negative else number


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _normalize_fidelity_csv_source(source):
    if isinstance(source, (bytes, bytearray)):
        text = source.decode("utf-8-sig")
    elif isinstance(source, str):
        try:
            is_path = Path(source).exists()
        except OSError:
            is_path = False
        if is_path:
            return source
        text = source
    elif isinstance(source, Path):
        return source
    else:
        return source

    header_line = next((line for line in text.splitlines() if line.strip()), "")
    headers = next(csv.reader([header_line]), [])
    normalized_headers = {header.strip().casefold() for header in headers}
    if {column.casefold() for column in FIDELITY_REQUIRED_COLUMNS}.issubset(normalized_headers):
        text = FIDELITY_CURRENCY_COMMA.sub(lambda match: match.group().replace(",", ""), text)
    return io.StringIO(text)


def classify_asset(ticker: str, sec_type: str, config: dict) -> str:
    ticker_map = {k.upper(): v for k, v in config.get("ticker_asset_class", {}).items()}
    if ticker.upper() in ticker_map:
        return ticker_map[ticker.upper()]
    sec_map = {k.lower(): v for k, v in config.get("sec_type_asset_class", {}).items()}
    if sec_type.lower() in sec_map:
        return sec_map[sec_type.lower()]
    return config.get("default_asset_class", "US Equity")


def parse_positions(source, account_id: str, config: dict) -> list[Position]:
    """Parse an A360 Positions export (path, bytes or text) into positions for ``account_id``."""
    source = _normalize_fidelity_csv_source(source)
    df = pd.read_csv(source, dtype=str, keep_default_na=False, skip_blank_lines=True, index_col=False)
    df.columns = [str(c).strip() for c in df.columns]
    normalized_columns = {str(column).strip().casefold(): column for column in df.columns}
    fidelity_columns = {column.casefold() for column in FIDELITY_REQUIRED_COLUMNS}
    if fidelity_columns.issubset(normalized_columns):
        return _parse_fidelity_positions(df, normalized_columns, account_id, config)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Not an A360 Positions file; missing columns: {missing}")

    positions: list[Position] = []
    for row in df.to_dict(orient="records"):
        qty = parse_number(row.get("Qty"))
        if qty == 0 and _text(row.get("Qty")) == "":
            qty = parse_number(row.get("Custodian Quantity"))
        ticker = _text(row.get("Ticker")) or _text(row.get("Cusip"))
        if qty <= 0 or not ticker:
            continue
        market_value = parse_number(row.get("Market Value"))
        price = parse_number(row.get("PM Price"))
        if price == 0:
            price = market_value / qty
        asset_class = classify_asset(ticker, _text(row.get("Sec Type")), config)
        positions.append(Position(account_id, ticker, asset_class, qty, price))
    return positions


def parse_fidelity_positions_csv(source) -> list[dict]:
    """Parse Fidelity's positions export, excluding account number/name columns."""
    normalized_source = _normalize_fidelity_csv_source(source)
    df = pd.read_csv(
        normalized_source,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
        index_col=False,
    )
    df.columns = [str(column).strip() for column in df.columns]
    columns = {str(column).strip().casefold(): column for column in df.columns}
    required = {column.casefold() for column in FIDELITY_CSV_DISPLAY_COLUMNS}
    missing = sorted(required - columns.keys())
    if missing:
        raise ValueError(f"Not a Fidelity positions CSV; missing columns: {missing}")

    positions = []
    numeric_fields = {
        "quantity": "quantity",
        "last_price": "last price",
        "last_price_change": "last price change",
        "current_value": "current value",
        "todays_gain_loss_dollar": "today's gain/loss dollar",
        "todays_gain_loss_percent": "today's gain/loss percent",
        "total_gain_loss_dollar": "total gain/loss dollar",
        "total_gain_loss_percent": "total gain/loss percent",
        "percent_of_account": "percent of account",
        "cost_basis_total": "cost basis total",
        "average_cost_basis": "average cost basis",
    }

    for row in df.to_dict(orient="records"):
        symbol = _text(row.get(columns["symbol"])).rstrip("*").strip()
        description = _text(row.get(columns["description"]))
        if not symbol or symbol.casefold() == "pending activity":
            continue
        parsed = {
            "symbol": symbol,
            "description": description,
            "type": _text(row.get(columns["type"])) or None,
        }
        for field, column in numeric_fields.items():
            value = _parse_nullable_fidelity_number(row.get(columns[column]))
            if field.endswith("_percent") or field == "percent_of_account":
                parsed[field] = value
            else:
                parsed[field] = value

        if "money market" in description.casefold():
            if parsed["quantity"] is None or parsed["quantity"] <= 0:
                parsed["quantity"] = parsed["current_value"]
            if parsed["last_price"] is None or parsed["last_price"] <= 0:
                parsed["last_price"] = 1.0
        positions.append(parsed)

    if not positions:
        raise ValueError("The Fidelity positions CSV contains no symbol rows.")
    return positions


def _parse_fidelity_positions(df, columns: dict[str, str], account_id: str, config: dict) -> list[Position]:
    positions: list[Position] = []
    for row in df.to_dict(orient="records"):
        ticker = _text(row.get(columns["symbol"])).rstrip("*").strip()
        if not ticker:
            continue
        description = _text(row.get(columns["description"]))
        quantity = _parse_fidelity_number(row.get(columns["quantity"]))
        last_price = _parse_fidelity_number(row.get(columns["last price"]))
        market_value = _parse_fidelity_number(row.get(columns["current value"]))

        is_money_market = "money market" in description.casefold()
        if is_money_market and quantity <= 0 and market_value > 0:
            quantity = market_value
            last_price = 1.0
        elif quantity <= 0:
            continue

        if last_price <= 0:
            if market_value <= 0:
                continue
            last_price = market_value / quantity

        asset_class = classify_asset(
            ticker,
            "Money Market" if is_money_market else "",
            config,
        )
        positions.append(Position(account_id, ticker, asset_class, quantity, last_price))
    return positions


def _parse_fidelity_number(value) -> float:
    if _text(value).casefold() in {"", "--", "—", "n/a", "n.a."}:
        return 0.0
    return parse_number(value)


def _parse_nullable_fidelity_number(value) -> float | None:
    text = _text(value).casefold().replace("%", "").strip()
    if text in {"", "--", "—", "n/a", "n.a."}:
        return None
    return parse_number(text)


@register("positions")
class A360PositionsAgent(SalesAgent):
    supports_upload = True

    def __init__(self, config: dict, base_dir):
        super().__init__(config, base_dir)
        self.upload_store = None

    def get_positions(self, account_id: str) -> list[Position]:
        self.get_account(account_id)
        if self.upload_store is not None:
            uploaded = self.upload_store.get(self.name, account_id)
            if uploaded is not None:
                return uploaded
        files = self.config.get("positions_files", {})
        if account_id not in files:
            return []
        return parse_positions(str(self.resolve_path(files[account_id])), account_id, self.config)

    def parse_upload(self, content: bytes, account_id: str) -> list[Position]:
        self.get_account(account_id)
        return parse_positions(content, account_id, self.config)
