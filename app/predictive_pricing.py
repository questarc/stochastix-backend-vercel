from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_HISTORY_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TICKER_PATTERN = re.compile(r"[A-Z0-9.^=_-]{1,15}")


def _history_for(ticker: str, yfinance: Any) -> pd.DataFrame:
    history = yfinance.Ticker(ticker).history(period="30d", interval="1d", auto_adjust=True)
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    if history.empty or any(column not in history.columns for column in REQUIRED_HISTORY_COLUMNS):
        return pd.DataFrame()
    return history[REQUIRED_HISTORY_COLUMNS].dropna().tail(30)


def _resolve_stock(query: str, yfinance: Any) -> tuple[str, str, pd.DataFrame]:
    query = query.strip()
    if not query or len(query) > 100:
        raise ValueError("Enter a stock ticker or company name (up to 100 characters).")

    if query.isupper() and TICKER_PATTERN.fullmatch(query):
        history = _history_for(query, yfinance)
        if len(history) >= 12:
            return query, query, history

    try:
        quotes = yfinance.Search(query, max_results=8).quotes
    except Exception as exc:
        raise ValueError(f"Could not find a stock for '{query}'.") from exc

    equities = [quote for quote in quotes if quote.get("quoteType") in {"EQUITY", "ETF"} and quote.get("symbol")]
    if not equities:
        raise ValueError(f"No matching stock was found for '{query}'.")
    match = next((quote for quote in equities if quote["symbol"].upper() == query.upper()), equities[0])
    ticker = match["symbol"].upper()
    history = _history_for(ticker, yfinance)
    company_name = match.get("shortname") or match.get("longname") or ticker
    return ticker, company_name, history


def predict_stock_price(query: str) -> dict:
    import torch
    from torch import nn
    import yfinance as yf

    ticker, company_name, history = _resolve_stock(query, yf)
    if len(history) < 12 or (history["Close"] <= 0).any():
        raise ValueError(f"At least 12 complete daily observations are required for {ticker}.")

    prices = history["Close"].to_numpy(dtype=np.float32)
    current_price = float(prices[-1])
    raw = history[REQUIRED_HISTORY_COLUMNS].to_numpy(dtype=np.float32)
    previous_close = np.roll(prices, 1)
    previous_close[0] = prices[0]
    features = np.column_stack((raw[:, :4] / previous_close[:, None] - 1, np.log1p(raw[:, 4])))
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    normalized = (features - feature_mean) / feature_scale

    sequence_length = 5
    sequences = np.stack([normalized[index - sequence_length:index] for index in range(sequence_length, len(history))])
    targets = features[sequence_length:, 3]
    torch.manual_seed(7)
    model = nn.LSTM(input_size=5, hidden_size=16, batch_first=True)
    output = nn.Linear(16, 1)
    optimizer = torch.optim.Adam([*model.parameters(), *output.parameters()], lr=0.01)
    x = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(targets[:, None], dtype=torch.float32)
    loss_function = nn.MSELoss()
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        hidden, _ = model(x)
        loss = loss_function(output(hidden[:, -1]), y)
        loss.backward()
        optimizer.step()

    model.eval()
    rolling = normalized[-sequence_length:].copy()
    forecast_returns: list[float] = []
    with torch.no_grad():
        for _ in range(5):
            hidden, _ = model(torch.tensor(rolling[None, :, :], dtype=torch.float32))
            predicted_return = float(output(hidden[:, -1]).item())
            predicted_return = float(np.clip(predicted_return, -0.2, 0.2))
            forecast_returns.append(predicted_return)
            next_features = np.array([predicted_return] * 4 + [float(features[:, 4].mean())])
            rolling = np.vstack((rolling[1:], (next_features - feature_mean) / feature_scale))

    forecast_prices: list[float] = []
    price = current_price
    for predicted_return in forecast_returns:
        price *= 1 + predicted_return
        forecast_prices.append(price)

    forecast_date = pd.Timestamp(history.index[-1]).date()
    forecasts = []
    for day, (predicted_return, forecast_price) in enumerate(zip(forecast_returns, forecast_prices), start=1):
        forecast_date += timedelta(days=1)
        while forecast_date.weekday() >= 5:
            forecast_date += timedelta(days=1)
        forecasts.append({
            "day": day,
            "date": forecast_date.isoformat(),
            "direction": "up" if predicted_return >= 0 else "down",
            "percent_change": predicted_return * 100,
            "expected_price": forecast_price,
        })

    return {
        "symbol": ticker,
        "company_name": company_name,
        "current_price": current_price,
        "history": [
            {"date": pd.Timestamp(date).date().isoformat(), "close": float(close)}
            for date, close in zip(history.index, prices)
        ],
        "forecasts": forecasts,
        "observations": len(history),
        "model": "PyTorch LSTM",
    }


def predict_stock_list(queries: list[str]) -> dict:
    results = []
    for query in queries:
        try:
            results.append({"query": query, "prediction": predict_stock_price(query), "error": None})
        except ModuleNotFoundError:
            raise
        except Exception as exc:
            results.append({"query": query, "prediction": None, "error": str(exc) or "Could not analyze this stock."})
    return {"results": results}