import csv
import io
import json

import pytest

from app.domain import ModelPortfolio, Position
from app.rebalance import (
    ORDER_FIELDS, RebalanceSettings, generate_orders, merge_orders, orders_to_csv, orders_to_json, propose_rebalance,
)

MODEL = ModelPortfolio("M", "Model", {"US Equity": 0.6, "US Bonds": 0.4})
SETTINGS = RebalanceSettings(
    drift_tolerance=0.02, min_trade_value=500, whole_shares=True,
    default_securities={"US Equity": {"ticker": "VTI", "price": 300}, "US Bonds": {"ticker": "BND", "price": 73}},
)


def _positions(eq_qty, bond_qty):
    return [Position("A1", "VTI", "US Equity", eq_qty, 300.0), Position("A1", "BND", "US Bonds", bond_qty, 73.0)]


def test_no_trades_within_tolerance():
    p = propose_rebalance("A1", _positions(200, 548), MODEL, SETTINGS)  # ~60/40
    assert not p.rebalance_needed
    assert p.trades == []


def test_trades_are_funded_by_sells():
    p = propose_rebalance("A1", _positions(300, 300), MODEL, SETTINGS)
    assert p.rebalance_needed
    sells = [t for t in p.trades if t.side == "Sell"]
    buys = [t for t in p.trades if t.side == "Buy"]
    assert sells and buys
    assert p.total_buys <= p.total_sells
    assert p.uninvested_cash == pytest.approx(p.total_sells - p.total_buys)
    assert sum(r["post_trade"] for r in p.allocation) == pytest.approx(1.0)


def test_whole_shares_round_down():
    p = propose_rebalance("A1", _positions(300, 300), MODEL, SETTINGS)
    assert all(float(t.quantity).is_integer() for t in p.trades)


def test_fractional_shares_when_disabled():
    p = propose_rebalance("A1", _positions(300, 300), MODEL, SETTINGS.override(whole_shares=False))
    assert p.uninvested_cash == pytest.approx(0.0, abs=1e-6)


def test_min_trade_value_skips_small_trades():
    p = propose_rebalance("A1", _positions(205, 530), MODEL, SETTINGS.override(drift_tolerance=0.001,
                                                                               min_trade_value=10_000))
    assert p.rebalance_needed
    assert p.trades == []


def test_target_zero_sells_everything():
    positions = _positions(200, 548) + [Position("A1", "GLD", "Gold", 10, 200.0)]
    model = ModelPortfolio("M", "Model", {"US Equity": 0.6, "US Bonds": 0.4})
    p = propose_rebalance("A1", positions, model, SETTINGS.override(drift_tolerance=0.001, min_trade_value=0))
    gold = [t for t in p.trades if t.ticker == "GLD"]
    assert gold and gold[0].quantity == 10


def test_buy_uses_largest_holding_without_default():
    settings = SETTINGS.override()
    settings.default_securities = {}
    positions = [Position("A1", "VTI", "US Equity", 300, 300.0), Position("A1", "AGG", "US Bonds", 100, 100.0),
                 Position("A1", "BND", "US Bonds", 300, 73.0)]
    p = propose_rebalance("A1", positions, MODEL, settings)
    assert [t.ticker for t in p.trades if t.side == "Buy"] == ["BND"]


def test_generate_orders_sells_first_and_ids():
    p = propose_rebalance("A1", _positions(300, 300), MODEL, SETTINGS)
    orders = generate_orders(p)
    assert [o["side"] for o in orders] == sorted([o["side"] for o in orders], key=lambda s: s != "Sell")
    assert orders[0]["order_id"].startswith("A1-") and orders[0]["order_id"].endswith("-001")
    assert all(o["status"] == "Pending" and o["order_type"] == "Market" and o["time_in_force"] == "Day"
               for o in orders)


def test_merge_orders_replaces_pending_of_same_account():
    old = [{"order_id": "1", "account_id": "A1", "status": "Pending"},
           {"order_id": "2", "account_id": "A1", "status": "Filled"},
           {"order_id": "3", "account_id": "A2", "status": "Pending"}]
    new = [{"order_id": "4", "account_id": "A1", "status": "Pending"}]
    merged = merge_orders(old, new)
    assert [o["order_id"] for o in merged] == ["2", "3", "4"]


def test_export_csv_and_json():
    orders = generate_orders(propose_rebalance("A1", _positions(300, 300), MODEL, SETTINGS))
    rows = list(csv.reader(io.StringIO(orders_to_csv(orders))))
    assert tuple(rows[0]) == ORDER_FIELDS
    assert len(rows) == len(orders) + 1
    assert list(json.loads(orders_to_json(orders))[0]) == list(ORDER_FIELDS)
