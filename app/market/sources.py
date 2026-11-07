"""Free, no-key market data sources: public RSS feeds and Nasdaq's public calendars."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

TOPICS = [
    ("Monetary policy", r"\b(fed|fomc|federal reserve|interest rate|rates?|monetary|powell|central bank)\b"),
    ("Inflation", r"\b(inflation|cpi|pce|prices?|deflat)"),
    ("Jobs", r"\b(jobs?|employment|unemployment|payrolls?|labor|wages?)\b"),
    ("Growth", r"\b(gdp|growth|recession|output|economy|economic)\b"),
    ("Trade", r"\b(trade|tariffs?|exports?|imports?)\b"),
    ("Regulation", r"\b(sec|regulat|enforcement|rule|supervis)"),
    ("Markets", r"\b(stocks?|shares|s&p|dow|nasdaq|bonds?|treasur|yields?|earnings)\b"),
]


class MarketHttp:
    def __init__(self, user_agent: str, timeout: float = 10.0):
        self.headers = {"User-Agent": user_agent, "Accept": "application/json, application/rss+xml, text/xml, */*"}
        self.timeout = timeout

    def get_bytes(self, url: str) -> bytes:
        r = httpx.get(url, headers=self.headers, timeout=self.timeout, follow_redirects=True)
        r.raise_for_status()
        return r.content

    def get_json(self, url: str) -> dict:
        r = httpx.get(url, headers=self.headers, timeout=self.timeout, follow_redirects=True)
        r.raise_for_status()
        return r.json()


def _clean(text: str | None, limit: int = 280) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        d = parsedate_to_datetime(text.strip())
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def tag_topics(text: str, default: str) -> list[str]:
    found = [name for name, pattern in TOPICS if re.search(pattern, text, re.I)]
    return list(dict.fromkeys([default, *found]))[:3]


def parse_rss(content: bytes, source: str, category: str) -> list[dict]:
    root = ET.fromstring(content)
    items = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title"), 200)
        if not title:
            continue
        summary = _clean(item.findtext("description"))
        published = _parse_date(item.findtext("pubDate"))
        items.append({
            "title": title, "summary": summary if summary != title else "",
            "url": (item.findtext("link") or "").strip(), "source": source,
            "published_at": published.astimezone(timezone.utc).isoformat() if published else None,
            "topics": tag_topics(f"{title} {summary}", category),
        })
    return items


def parse_money(text) -> float | None:
    if text in (None, "", "N/A", "--"):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = re.sub(r"[^0-9.\-]", "", str(text).replace("(", "-"))
    try:
        return float(cleaned) if cleaned not in ("", "-", ".") else None
    except ValueError:
        return None


def parse_us_date(text: str | None) -> str | None:
    try:
        return datetime.strptime((text or "").strip(), "%m/%d/%Y").date().isoformat()
    except ValueError:
        return None


EARNINGS_TIME = {"time-pre-market": "Before market open", "time-after-hours": "After market close"}


def earnings_rows(payload: dict, day: date) -> list[dict]:
    rows = (payload.get("data") or {}).get("rows") or []
    return [{
        "date": day.isoformat(), "symbol": r.get("symbol", "").strip(), "name": _clean(r.get("name"), 80),
        "time": EARNINGS_TIME.get(r.get("time", ""), "Time not announced"),
        "eps_forecast": parse_money(r.get("epsForecast")), "last_year_eps": parse_money(r.get("lastYearEPS")),
        "market_cap": parse_money(r.get("marketCap")), "estimates": int(parse_money(r.get("noOfEsts")) or 0),
        "fiscal_quarter": r.get("fiscalQuarterEnding") or None,
    } for r in rows if r.get("symbol")]


def dividend_rows(payload: dict) -> list[dict]:
    data = payload.get("data") or {}
    rows = (data.get("calendar") or {}).get("rows") or data.get("rows") or []
    return [{
        "type": "Dividend", "symbol": r.get("symbol", "").strip(), "name": _clean(r.get("companyName"), 80),
        "ex_date": parse_us_date(r.get("dividend_Ex_Date")), "pay_date": parse_us_date(r.get("payment_Date")),
        "record_date": parse_us_date(r.get("record_Date")), "amount": parse_money(r.get("dividend_Rate")),
        "ratio": None,
    } for r in rows if r.get("symbol")]


def screener_rows(payload: dict) -> list[dict]:
    data = payload.get("data") or {}
    rows = data.get("rows") or (data.get("table") or {}).get("rows") or []
    out = []
    for r in rows:
        cap, change = parse_money(r.get("marketCap")), parse_money(r.get("pctchange"))
        if not r.get("symbol") or not cap or change is None:
            continue
        out.append({"symbol": r["symbol"].strip(), "name": _clean(r.get("name"), 60),
                    "sector": (r.get("sector") or "").strip() or "Other", "industry": _clean(r.get("industry"), 60),
                    "market_cap": cap, "change_pct": change, "price": parse_money(r.get("lastsale"))})
    return out


def split_rows(payload: dict) -> list[dict]:
    rows = (payload.get("data") or {}).get("rows") or []
    return [{
        "type": "Split", "symbol": r.get("symbol", "").strip(), "name": _clean(r.get("name"), 80),
        "ex_date": parse_us_date(r.get("executionDate")), "pay_date": None, "record_date": None,
        "amount": None, "ratio": (r.get("ratio") or "").strip() or None,
    } for r in rows if r.get("symbol")]
