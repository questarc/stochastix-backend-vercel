def test_predictive_pricing_endpoint_returns_five_forecasts(client, monkeypatch):
    from app.routers import predictive_pricing

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