from datetime import date

import pytest

from app import deps
from app.market import CorporateActionsAgent, EarningsAgent, MarketAnalyzerAgent, MarketSettings, holdings_index
from app.market.sources import parse_money, parse_rss, tag_topics

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Fed holds interest rates steady</title><link>https://fed/1</link>
<description>&lt;p&gt;The FOMC kept rates&lt;/p&gt;</description><pubDate>Tue, 27 Oct 2026 18:00:00 GMT</pubDate></item>
<item><title>GDP grows 2.1%</title><link>https://bea/2</link><pubDate>Mon, 26 Oct 2026 12:30:00 GMT</pubDate></item>
</channel></rss>"""
DAY = date(2026, 10, 28)  # Wednesday


def earnings_payload(day):
    rows = [
        {"symbol": "MSFT", "name": "Microsoft Corporation", "time": "time-after-hours", "epsForecast": "$4.71",
         "lastYearEPS": "$4.13", "marketCap": "$3,955,314,920,000", "noOfEsts": "14", "fiscalQuarterEnding": "Sep/2026"},
        {"symbol": "TINY", "name": "Tiny Co", "time": "time-not-supplied", "epsForecast": "", "marketCap": "$1,000,000"},
        {"symbol": "VTI", "name": "Held Fund", "time": "time-pre-market", "epsForecast": "($0.10)", "marketCap": "N/A"},
    ]
    return {"data": {"rows": rows if day == "2026-10-28" else []}}


class FakeHttp:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def get_bytes(self, url):
        self.calls.append(url)
        if url in self.fail:
            raise ConnectionError("down")
        return RSS

    def get_json(self, url):
        self.calls.append(url)
        if any(f in url for f in self.fail):
            raise ConnectionError("down")
        if "earnings" in url:
            return earnings_payload(url.split("date=")[1])
        if "dividends" in url:
            return {"data": {"calendar": {"rows": [
                {"symbol": "BND", "companyName": "Vanguard Total Bond", "dividend_Ex_Date": "10/30/2026",
                 "payment_Date": "11/04/2026", "record_Date": "10/30/2026", "dividend_Rate": 0.2275},
                {"symbol": "OLD", "companyName": "Old", "dividend_Ex_Date": "01/01/2020", "dividend_Rate": 1}]}}}
        return {"data": {"rows": [{"symbol": "XYZ", "name": "XYZ Corp", "ratio": "2 : 1", "executionDate": "11/09/2026"}]}}


def settings(**kw):
    base = dict(news_feeds=[{"name": "Fed", "url": "f", "category": "Monetary policy"},
                            {"name": "BEA", "url": "b", "category": "Economic data"}],
                earnings_url="earnings?date={date}", dividends_url="dividends?date={date}", splits_url="splits",
                watchlist=["AAPL"])
    base.update(kw)
    return MarketSettings(**base)


HELD = lambda: {"VTI": ["A1001", "A1002"], "BND": ["A1001"]}  # noqa: E731


def test_rss_parsing_and_topics():
    items = parse_rss(RSS, "Fed", "Monetary policy")
    assert items[0]["title"] == "Fed holds interest rates steady" and items[0]["summary"] == "The FOMC kept rates"
    assert items[0]["published_at"].startswith("2026-10-27T18:00") and items[0]["url"] == "https://fed/1"
    assert "Growth" in tag_topics("GDP grows", "Economic data")
    assert parse_money("($0.10)") == -0.10 and parse_money("$3,955,314,920,000") == 3955314920000 and parse_money("N/A") is None


def test_news_dedupes_sorts_caches_and_reports_failed_feeds():
    http = FakeHttp(fail={"b"})
    agent = MarketAnalyzerAgent(settings(), http)
    r = agent.news()
    assert [i["title"] for i in r["items"]] == ["Fed holds interest rates steady", "GDP grows 2.1%"]
    assert r["errors"] == [{"source": "BEA", "error": "ConnectionError"}]
    calls = len(http.calls)
    agent.news()
    assert len(http.calls) == calls  # cached
    agent.news(refresh=True)
    assert len(http.calls) > calls
    assert [i["title"] for i in agent.news(topic="growth")["items"]] == ["GDP grows 2.1%"]


def test_earnings_filters_to_holdings_watchlist_and_large_caps():
    r = EarningsAgent(settings(), FakeHttp(), HELD).upcoming(days=2, today=DAY)
    syms = [i["symbol"] for i in r["items"]]
    assert syms == ["VTI", "MSFT"]  # held first; TINY dropped
    vti, msft = r["items"]
    assert vti["held_by"] == ["A1001", "A1002"] and vti["eps_forecast"] == -0.10 and vti["time"] == "Before market open"
    assert msft["eps_forecast"] == 4.71 and msft["time"] == "After market close" and msft["estimates"] == 14


def test_corporate_actions_window_holdings_first_and_errors():
    r = CorporateActionsAgent(settings(), FakeHttp(), HELD).upcoming(days=30, today=DAY)
    assert [(i["type"], i["symbol"]) for i in r["items"]] == [("Dividend", "BND"), ("Split", "XYZ")]
    assert r["items"][0]["held_by"] == ["A1001"] and r["items"][0]["amount"] == 0.2275
    assert r["items"][1]["ratio"] == "2 : 1" and r["items"][1]["ex_date"] == "2026-11-09"
    failed = CorporateActionsAgent(settings(), FakeHttp(fail={"splits"}), HELD).upcoming(days=5, today=DAY)
    assert {"source": "Nasdaq splits", "error": "ConnectionError"} in failed["errors"]


def test_holdings_index_uses_real_positions(client):
    held = holdings_index(deps.get_sources())
    assert "A1001" in held["VTI"] and "A2001" in held["VTI"]


@pytest.fixture
def market_client(client):
    http = FakeHttp()
    client.app.dependency_overrides[deps.get_market_agent] = lambda: MarketAnalyzerAgent(settings(), http)
    client.app.dependency_overrides[deps.get_earnings_agent] = lambda: EarningsAgent(settings(), http, HELD)
    client.app.dependency_overrides[deps.get_corporate_actions_agent] = lambda: CorporateActionsAgent(settings(), http, HELD)
    return client


def test_market_endpoints(market_client):
    news = market_client.get("/api/market/news").json()
    assert news["agent"] == "MarketAnalyzerAgent" and len(news["items"]) == 2 and "not investment advice" in news["notice"]
    earn = market_client.get("/api/market/earnings", params={"days": 3}).json()
    assert earn["agent"] == "EarningsAgent" and "updated_at" in earn
    acts = market_client.get("/api/market/corporate-actions").json()
    assert acts["agent"] == "CorporateActionsAgent"
    assert market_client.get("/api/market/earnings", params={"days": 99}).status_code == 422


def test_market_chat_tools_filter_by_account(client):
    from app.agent import build_tools

    http = FakeHttp()
    agents = {"news": MarketAnalyzerAgent(settings(), http),
              "earnings": EarningsAgent(settings(), http, HELD), "actions": CorporateActionsAgent(settings(), http, HELD)}
    tools = {t.name: t for t in build_tools(deps.get_sources(), deps.get_cma(), deps.get_rebalance_settings(),
                                            market_agents=agents)}
    acts = tools["get_corporate_actions"].invoke({"account_id": "A1001"})
    assert "BND" in acts and "XYZ" not in acts
    assert "Fed holds" in tools["get_market_news"].invoke({})
    assert '"error"' in tools["get_upcoming_earnings"].invoke({"account_id": "ZZZ"})
