from __future__ import annotations

from datetime import date, timedelta
import math

import pandas as pd
import pytest

from app.time_series import analyze_time_series_csv, preview_time_series_csv


def sample_csv(rows: int = 100) -> bytes:
    price = 100.0
    records = []
    start = date(2020, 1, 1)
    for index in range(rows):
        macro = math.sin(index * 0.23)
        momentum = math.cos(index * 0.11)
        records.append(
            {
                "Date": (start + timedelta(days=index)).isoformat(),
                "Asset": price,
                "Macro": macro,
                "Momentum": momentum,
            }
        )
        forward_return = 0.008 * macro + 0.006 * math.sin(index * 1.17)
        price *= 1 + forward_return
    return pd.DataFrame(records).to_csv(index=False).encode()


def test_preview_only_returns_column_metadata():
    preview = preview_time_series_csv(sample_csv())

    assert preview == {
        "columns": ["Date", "Asset", "Macro", "Momentum"],
        "date_columns": ["Date"],
        "numeric_columns": ["Asset", "Macro", "Momentum"],
        "row_count": 100,
    }


def test_linear_regression_uses_forward_returns_and_chronological_holdout():
    result = analyze_time_series_csv(sample_csv(), "Date", "Asset", ["Macro", "Momentum"], 1, "linear")

    assert result["model_type"] == "linear"
    assert result["observations"] == 99
    assert result["training_observations"] == 79
    assert result["test_observations"] == 20
    assert result["target_description"] == "1-observation forward simple return"
    assert result["test_series"][0]["date"] == "2020-03-20"
    assert math.isfinite(result["metrics"]["rmse"])
    assert {row["feature"] for row in result["coefficients"]} == {"(intercept)", "Macro", "Momentum"}


def test_logistic_regression_reports_probabilities_and_classification_metrics():
    result = analyze_time_series_csv(sample_csv(), "Date", "Asset", ["Macro", "Momentum"], 1, "logistic")

    assert result["model_type"] == "logistic"
    assert "accuracy" in result["metrics"]
    assert "roc_auc" in result["metrics"]
    positive_test_observations = sum(row["actual_up"] for row in result["test_series"])
    assert result["metrics"]["true_positive"] + result["metrics"]["false_negative"] == positive_test_observations
    assert len(result["test_series"]) == 20
    assert all(0 <= row["predicted_probability"] <= 1 for row in result["test_series"])
    assert all("odds_ratio" in coefficient for coefficient in result["coefficients"])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"Date,Value\n2024-01-01,a\n2024-01-02,b\n2024-01-03,c\n", "No numeric columns"),
        (b"Date,Asset,Asset\n2024-01-01,1,1\n", "unique"),
    ],
)
def test_preview_rejects_bad_csv(content, message):
    with pytest.raises(ValueError, match=message):
        preview_time_series_csv(content)


def test_model_rejects_date_leakage_and_constant_predictors():
    csv = sample_csv()
    with pytest.raises(ValueError, match="cannot be used as predictors"):
        analyze_time_series_csv(csv, "Date", "Asset", ["Date"], 1, "linear")
    with pytest.raises(ValueError, match="constant"):
        analyze_time_series_csv(
            pd.DataFrame(
                {
                    "Date": pd.date_range("2020-01-01", periods=40).astype(str),
                    "Asset": range(100, 140),
                    "Macro": [1] * 40,
                }
            ).to_csv(index=False).encode(),
            "Date",
            "Asset",
            ["Macro"],
            1,
            "linear",
        )


def test_time_series_endpoints_preview_analyze_and_validate(client):
    content = sample_csv()
    preview = client.post(
        "/api/time-series/preview",
        files={"file": ("history.csv", content, "text/csv")},
    )
    assert preview.status_code == 200
    assert preview.json()["row_count"] == 100

    result = client.post(
        "/api/time-series/analyze",
        files={"file": ("history.csv", content, "text/csv")},
        data={
            "date_column": "Date",
            "target_column": "Asset",
            "predictor_columns": "Macro",
            "horizon_periods": "1",
            "model_type": "linear",
        },
    )
    assert result.status_code == 200
    assert result.json()["model_type"] == "linear"

    invalid = client.post(
        "/api/time-series/analyze",
        files={"file": ("history.csv", content, "text/csv")},
        data={
            "date_column": "Date",
            "target_column": "Asset",
            "predictor_columns": "Date",
            "horizon_periods": "1",
            "model_type": "linear",
        },
    )
    assert invalid.status_code == 400
    assert "cannot be used as predictors" in invalid.json()["detail"]
