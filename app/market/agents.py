"""MarketAnalyzerAgent, EarningsAgent and CorporateActionsAgent. Each caches its results and never invents items."""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .sources import MarketHttp, dividend_rows, earnings_rows, parse_rss, screener_rows, split_rows

logger = logging.getLogger(__name__)


@dataclass
class MarketSettings:
    news_feeds: list[dict]
    earnings_url: str
    dividends_url: str
    splits_url: str
    watchlist: list[str] = field(default_factory=list)
    earnings_min_market_cap: float = 50e9
    news_ttl_minutes: float = 15
    calendar_ttl_minutes: float = 360
    request_timeout_seconds: float = 10
    user_agent: str = "Mozilla/5.0"
    max_items: int = 30
    heatmap_url: str = ""
    heatmap_size: int = 100
    heatmap_ttl_minutes: float = 15
    heatmap_exclude: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "MarketSettings":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def holdings_index(sources: dict) -> dict[str, list[str]]:
    """{TICKER: [account_id, ...]} across every enabled source (current positions, including uploads)."""
    held: dict[str, list[str]] = {}
    for source in sources.values():
        for account in source.list_accounts():
            try:
                positions = source.get_positions(account.account_id)
            except Exception:  # one bad account must not hide the widgets
                continue
            for p in positions:
                held.setdefault(p.ticker.upper(), [])
                if account.account_id not in held[p.ticker.upper()]:
                    held[p.ticker.upper()].append(account.account_id)
    return held


class _Cached:
    """Result cache with a time-to-live and per-source error reporting."""

    def __init__(self, ttl_minutes: float, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl_minutes * 60
        self._clock = clock
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, build: Callable[[], dict], refresh: bool = False) -> dict:
        with self._lock:
            hit = self._cache.get(key)
            if hit and not refresh and self._clock() - hit[0] < self._ttl:
                return hit[1]
        result = build()
        result["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._cache[key] = (self._clock(), result)
        return result


class MarketAnalyzerAgent(_Cached):
    """Latest economic and market news from official and public RSS feeds."""

    def __init__(self, settings: MarketSettings, http: MarketHttp, clock=time.monotonic):
        super().__init__(settings.news_ttl_minutes, clock)
        self.settings, self.http = settings, http

    def _fetch(self, feed: dict) -> list[dict]:
        return parse_rss(self.http.get_bytes(feed["url"]), feed["name"], feed["category"])

    def news(self, refresh: bool = False, topic: str | None = None) -> dict:
        result = self.get("news", self._build, refresh)
        if topic:
            items = [i for i in result["items"] if any(topic.lower() in t.lower() for t in i["topics"])]
            return {**result, "items": items}
        return result

    def _build(self) -> dict:
        items, errors = [], []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {f["name"]: pool.submit(self._fetch, f) for f in self.settings.news_feeds}
        for name, fut in futures.items():
            try:
                items.extend(fut.result())
            except Exception as exc:
                errors.append({"source": name, "error": type(exc).__name__})
                logger.info("News feed %s failed: %s", name, type(exc).__name__)
        seen, unique = set(), []
        for item in sorted(items, key=lambda i: i["published_at"] or "", reverse=True):
            key = item["title"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return {"items": unique[: self.settings.max_items], "errors": errors}


def _weekdays(start: date, days: int) -> list[date]:
    return [start + timedelta(d) for d in range(days + 1) if (start + timedelta(d)).weekday() < 5]


class EarningsAgent(_Cached):
    """Upcoming earnings for client holdings, the watchlist and large companies (Nasdaq public calendar)."""

    def __init__(self, settings: MarketSettings, http: MarketHttp, holdings: Callable[[], dict], clock=time.monotonic):
        super().__init__(settings.calendar_ttl_minutes, clock)
        self.settings, self.http, self.holdings = settings, http, holdings

    def upcoming(self, days: int = 14, refresh: bool = False, today: date | None = None) -> dict:
        start = today or date.today()
        return self.get(f"earnings:{start}:{days}", lambda: self._build(start, days), refresh)

    def _build(self, start: date, days: int) -> dict:
        held, watch = self.holdings(), {t.upper() for t in self.settings.watchlist}
        items, errors = [], []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {d: pool.submit(self.http.get_json, self.settings.earnings_url.format(date=d.isoformat()))
                       for d in _weekdays(start, days)}
        for d, fut in futures.items():
            try:
                rows = earnings_rows(fut.result(), d)
            except Exception as exc:
                errors.append({"source": f"Nasdaq earnings {d}", "error": type(exc).__name__})
                continue
            for r in rows:
                sym = r["symbol"].upper()
                if sym in held or sym in watch or (r["market_cap"] or 0) >= self.settings.earnings_min_market_cap:
                    items.append({**r, "held_by": held.get(sym, []), "watchlist": sym in watch})
        items.sort(key=lambda r: (r["date"], -len(r["held_by"]), -(r["market_cap"] or 0)))
        return {"items": items[: self.settings.max_items * 2], "errors": errors, "source": "Nasdaq"}


class CorporateActionsAgent(_Cached):
    """Upcoming dividends and stock splits (Nasdaq public calendars); client holdings first."""

    def __init__(self, settings: MarketSettings, http: MarketHttp, holdings: Callable[[], dict], clock=time.monotonic):
        super().__init__(settings.calendar_ttl_minutes, clock)
        self.settings, self.http, self.holdings = settings, http, holdings

    def upcoming(self, days: int = 30, refresh: bool = False, today: date | None = None) -> dict:
        start = today or date.today()
        return self.get(f"actions:{start}:{days}", lambda: self._build(start, days), refresh)

    def _build(self, start: date, days: int) -> dict:
        held, errors, rows = self.holdings(), [], []
        with ThreadPoolExecutor(max_workers=6) as pool:
            divs = {d: pool.submit(self.http.get_json, self.settings.dividends_url.format(date=d.isoformat()))
                    for d in _weekdays(start, days)}
            splits = pool.submit(self.http.get_json, self.settings.splits_url)
        for d, fut in divs.items():
            try:
                rows.extend(dividend_rows(fut.result()))
            except Exception as exc:
                errors.append({"source": f"Nasdaq dividends {d}", "error": type(exc).__name__})
        try:
            rows.extend(split_rows(splits.result()))
        except Exception as exc:
            errors.append({"source": "Nasdaq splits", "error": type(exc).__name__})
        end = (start + timedelta(days)).isoformat()
        items, seen = [], set()
        for r in rows:
            key = (r["type"], r["symbol"], r["ex_date"])
            if key in seen or not r["ex_date"] or not (start.isoformat() <= r["ex_date"] <= end):
                continue
            seen.add(key)
            items.append({**r, "held_by": held.get(r["symbol"].upper(), [])})
        items.sort(key=lambda r: (not r["held_by"], r["ex_date"], r["symbol"]))
        return {"items": items[: self.settings.max_items * 2], "errors": errors, "source": "Nasdaq"}


class MarketHeatmapAgent(_Cached):
    """Daily % change of the largest US companies, grouped by sector (Nasdaq public screener)."""

    def __init__(self, settings: MarketSettings, http: MarketHttp, holdings: Callable[[], dict], clock=time.monotonic):
        super().__init__(settings.heatmap_ttl_minutes, clock)
        self.settings, self.http, self.holdings = settings, http, holdings

    def heatmap(self, size: int | None = None, refresh: bool = False) -> dict:
        n = min(max(size or self.settings.heatmap_size, 10), 300)
        return self.get(f"heatmap:{n}", lambda: self._build(n), refresh)

    def _build(self, size: int) -> dict:
        try:
            rows = screener_rows(self.http.get_json(self.settings.heatmap_url))
        except Exception as exc:
            logger.info("Heatmap source failed: %s", type(exc).__name__)
            return {"items": [], "sectors": [], "errors": [{"source": "Nasdaq screener", "error": type(exc).__name__}],
                    "source": "Nasdaq"}
        exclude = {s.upper() for s in self.settings.heatmap_exclude}
        held = self.holdings()
        top = sorted((r for r in rows if r["symbol"].upper() not in exclude), key=lambda r: -r["market_cap"])[:size]
        items = [{**r, "held_by": held.get(r["symbol"].upper(), [])} for r in top]
        sectors: dict[str, dict] = {}
        for i in items:
            s = sectors.setdefault(i["sector"], {"sector": i["sector"], "market_cap": 0.0, "weighted": 0.0, "count": 0})
            s["market_cap"] += i["market_cap"]
            s["weighted"] += i["market_cap"] * i["change_pct"]
            s["count"] += 1
        summary = [{"sector": s["sector"], "market_cap": s["market_cap"], "count": s["count"],
                    "change_pct": round(s["weighted"] / s["market_cap"], 3)} for s in sectors.values()]
        summary.sort(key=lambda s: -s["market_cap"])
        total = sum(i["market_cap"] for i in items)
        market = round(sum(i["market_cap"] * i["change_pct"] for i in items) / total, 3) if total else None
        up = sum(1 for i in items if i["change_pct"] > 0)
        return {"items": items, "sectors": summary, "errors": [], "source": "Nasdaq",
                "breadth": {"up": up, "down": sum(1 for i in items if i["change_pct"] < 0), "total": len(items)},
                "market_change_pct": market}

