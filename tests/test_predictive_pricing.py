def test_predictive_pricing_endpoint_returns_five_forecasts(client, monkeypatch):
    from app.routers import predictive_pricing

    monkeypatch.delenv("PREDICTIVE_PRICING_API_URL", raising=False)
    monkeypatch.setattr(
        predictive_pricing,
        "predict_stock_price",
        lambda symbol, current_price: {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "history": [],
            "forecasts": [
                {"day": day, "date": f"2026-11-{14 + day:02d}", "direction": "up", "percent_change": 0.5,
                 "expected_price": current_price * (1 + 0.005 * day)}
                for day in range(1, 6)
            ],
            "observations": 20,
            "model": "PyTorch LSTM",
        },
    )

    response = client.post("/api/predictive-pricing/predict", json={"symbol": "AAPL", "current_price": 250})

    assert response.status_code == 200
    result = response.json()
    assert result["symbol"] == "AAPL"
    assert len(result["forecasts"]) == 5
    assert {forecast["direction"] for forecast in result["forecasts"]} <= {"up", "down"}
    assert all("percent_change" in forecast and "expected_price" in forecast for forecast in result["forecasts"])


def test_predictive_pricing_rejects_nonpositive_price(client):
    response = client.post("/api/predictive-pricing/predict", json={"symbol": "AAPL", "current_price": 0})

    assert response.status_code == 422


def test_predictive_pricing_returns_503_when_optional_model_dependencies_are_missing(client, monkeypatch):
    from app.routers import predictive_pricing

    monkeypatch.delenv("PREDICTIVE_PRICING_API_URL", raising=False)

    def missing_torch(*_args):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(predictive_pricing, "predict_stock_price", missing_torch)
    response = client.post("/api/predictive-pricing/predict", json={"symbol": "AAPL", "current_price": 250})

    assert response.status_code == 503
    assert "PREDICTIVE_PRICING_API_URL" in response.json()["detail"]


def test_predictive_pricing_uses_configured_upstream(client, monkeypatch):
    from app.routers import predictive_pricing

    monkeypatch.setenv("PREDICTIVE_PRICING_API_URL", "https://pricing.example.com/")
    expected = {"symbol": "AAPL", "current_price": 250, "history": [], "forecasts": []}

    async def fake_upstream(base_url, payload):
        assert base_url == "https://pricing.example.com/"
        assert payload == {"symbol": "AAPL", "current_price": 250}
        return expected

    monkeypatch.setattr(predictive_pricing, "_predict_from_upstream", fake_upstream)
    response = client.post("/api/predictive-pricing/predict", json={"symbol": "AAPL", "current_price": 250})

    assert response.status_code == 200
    assert response.json() == expected