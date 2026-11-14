def test_predictive_pricing_endpoint_returns_five_forecasts(client, monkeypatch):
    from app.routers import predictive_pricing

    monkeypatch.delenv("PREDICTIVE_PRICING_API_URL", raising=False)
    monkeypatch.setattr(
        predictive_pricing,
        "predict_stock_price",
        lambda symbol: {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "current_price": 250,
            "history": [],
            "forecasts": [
                {"day": day, "date": f"2026-11-{14 + day:02d}", "direction": "up", "percent_change": 0.5,
                 "expected_price": 250 * (1 + 0.005 * day)}
                for day in range(1, 6)
            ],
            "observations": 20,
            "model": "PyTorch LSTM",
        },
    )

    response = client.post("/api/predictive-pricing/predict", json={"symbol": "Apple"})

    assert response.status_code == 200
    result = response.json()
    assert result["symbol"] == "AAPL"
    assert len(result["forecasts"]) == 5
    assert {forecast["direction"] for forecast in result["forecasts"]} <= {"up", "down"}
    assert all("percent_change" in forecast and "expected_price" in forecast for forecast in result["forecasts"])


def test_predictive_pricing_rejects_empty_stock_query(client):
    response = client.post("/api/predictive-pricing/predict", json={"symbol": ""})

    assert response.status_code == 422


def test_predictive_pricing_returns_503_when_optional_model_dependencies_are_missing(client, monkeypatch):
    from app.routers import predictive_pricing

    monkeypatch.delenv("PREDICTIVE_PRICING_API_URL", raising=False)

    def missing_torch(*_args):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(predictive_pricing, "predict_stock_price", missing_torch)
    response = client.post("/api/predictive-pricing/predict", json={"symbol": "AAPL"})

    assert response.status_code == 503
    assert "PREDICTIVE_PRICING_API_URL" in response.json()["detail"]


def test_predictive_pricing_uses_configured_upstream(client, monkeypatch):
    from app.routers import predictive_pricing

    monkeypatch.setenv("PREDICTIVE_PRICING_API_URL", "https://pricing.example.com/")
    expected = {"symbol": "AAPL", "current_price": 250, "history": [], "forecasts": []}

    async def fake_upstream(base_url, payload):
        assert base_url == "https://pricing.example.com/"
        assert payload == {"symbol": "AAPL"}
        return expected

    monkeypatch.setattr(predictive_pricing, "_predict_from_upstream", fake_upstream)
    response = client.post("/api/predictive-pricing/predict", json={"symbol": "AAPL"})

    assert response.status_code == 200
    assert response.json() == expected


def test_predictive_pricing_batch_returns_individual_analysis(client, monkeypatch):
    from app.routers import predictive_pricing

    monkeypatch.delenv("PREDICTIVE_PRICING_API_URL", raising=False)
    expected = {"results": [
        {"query": "Apple", "prediction": {"symbol": "AAPL", "current_price": 250, "forecasts": []}, "error": None},
        {"query": "Unknown", "prediction": None, "error": "No matching stock was found for 'Unknown'."},
    ]}
    monkeypatch.setattr(predictive_pricing, "predict_stock_list", lambda symbols: expected)

    response = client.post("/api/predictive-pricing/batch", json={"symbols": ["Apple", "Unknown"]})

    assert response.status_code == 200
    assert response.json() == expected


def test_predictive_pricing_batch_is_limited_to_ten_stocks(client):
    response = client.post("/api/predictive-pricing/batch", json={"symbols": ["AAPL"] * 11})

    assert response.status_code == 422


def test_resolve_stock_searches_company_name(monkeypatch):
    import pandas as pd
    import yfinance as yf
    from app.predictive_pricing import _resolve_stock

    dates = pd.date_range("2026-10-01", periods=15)
    history = pd.DataFrame(
        {column: [100.0] * 15 for column in ["Open", "High", "Low", "Close", "Volume"]},
        index=dates,
    )
    monkeypatch.setattr(yf, "Search", lambda query, max_results: type("SearchResult", (), {
        "quotes": [{"symbol": "AAPL", "shortname": "Apple Inc.", "quoteType": "EQUITY"}],
    })())
    monkeypatch.setattr(yf, "Ticker", lambda ticker: type("Stock", (), {
        "history": lambda self, **kwargs: history,
    })())

    symbol, company_name, prices = _resolve_stock("Apple", yf)

    assert symbol == "AAPL"
    assert company_name == "Apple Inc."
    assert len(prices) == 15