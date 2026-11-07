import pytest

from app import deps
from app.market import MarketHeatmapAgent, MarketSettings
from app.market.sources import screener_rows

ROWS = [
    {"symbol": "NVDA", "name": "NVIDIA", "sector": "Technology", "marketCap": "5000000000000.00", "pctchange": "2.0%", "lastsale": "$200.00"},
    {"symbol": "AAPL", "name": "Apple", "sector": "Technology", "marketCap": "3000000000000.00", "pctchange": "-1.0%", "lastsale": "$250.00"},
    {"symbol": "GOOG", "name": "Alphabet C", "sector": "Technology", "marketCap": "2900000000000.00", "pctchange": "-0.5%"},
    {"symbol": "JPM", "name": "JPMorgan", "sector": "Finance", "marketCap": "700000000000.00", "pctchange": "0.5%"},
    {"symbol": "VTI", "name": "Held", "sector": "", "marketCap": "500000000000.00", "pctchange": "0.0%"},
    {"symbol": "TINY", "name": "Tiny", "sector": "Health Care", "marketCap": "1000.00", "pctchange": "9.0%"},
    {"symbol": "BAD", "name": "No cap", "sector": "Energy", "marketCap": "", "pctchange": "1%"},
]


class FakeHttp:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def get_json(self, url):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return {"data": {"rows": ROWS}}


def settings():
    return MarketSettings(news_feeds=[], earnings_url="", dividends_url="", splits_url="", heatmap_url="screener",
                          heatmap_exclude=["GOOG"])


HELD = lambda: {"VTI": ["A1001"]}  # noqa: E731


def test_screener_parsing_skips_rows_without_cap():
    rows = screener_rows({"data": {"rows": ROWS}})
    assert "BAD" not in [r["symbol"] for r in rows]
    vti = next(r for r in rows if r["symbol"] == "VTI")
    assert vti["sector"] == "Other" and vti["change_pct"] == 0.0


def test_heatmap_top_by_cap_sectors_breadth_and_cache():
    http = FakeHttp()
    agent = MarketHeatmapAgent(settings(), http, HELD)
    r = agent.heatmap(size=10)
    assert [i["symbol"] for i in r["items"]][:4] == ["NVDA", "AAPL", "JPM", "VTI"]  # GOOG excluded
    tech = next(s for s in r["sectors"] if s["sector"] == "Technology")
    assert tech["change_pct"] == pytest.approx((5 * 2.0 + 3 * -1.0) / 8, abs=1e-3) and tech["count"] == 2
    assert r["breadth"] == {"up": 3, "down": 1, "total": 5}
    assert next(i for i in r["items"] if i["symbol"] == "VTI")["held_by"] == ["A1001"]
    agent.heatmap(size=10)
    assert http.calls == 1


def test_heatmap_reports_source_failure():
    r = MarketHeatmapAgent(settings(), FakeHttp(fail=True), HELD).heatmap()
    assert r["items"] == [] and r["errors"][0]["source"] == "Nasdaq screener"


def test_heatmap_endpoint_and_chat_tool(client):
    from app.agent import build_tools

    agent = MarketHeatmapAgent(settings(), FakeHttp(), HELD)
    client.app.dependency_overrides[deps.get_heatmap_agent] = lambda: agent
    body = client.get("/api/market/heatmap", params={"size": 20}).json()
    assert body["agent"] == "MarketHeatmapAgent" and body["sectors"][0]["sector"] == "Technology"
    assert client.get("/api/market/heatmap", params={"size": 5}).status_code == 422
    tools = {t.name: t for t in build_tools(deps.get_sources(), deps.get_cma(), deps.get_rebalance_settings(),
                                            market_agents={"news": None, "earnings": None, "actions": None,
                                                           "heatmap": agent})}
    out = tools["get_market_heatmap"].invoke({})
    assert '"NVDA"' in out and "+2.00%" in out and "3 up, 1 down of 5" in out
