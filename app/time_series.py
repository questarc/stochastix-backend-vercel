"""Statistical regression for user-supplied, date-aligned time-series CSVs."""
from __future__ import annotations

import csv
from io import StringIO

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, t

MAX_ROWS = 100_000
MAX_PREDICTORS = 10
MIN_OBSERVATIONS = 30


def _read_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("The uploaded CSV is empty.")
    try:
        text = content.decode("utf-8-sig")
        header_rows = csv.reader(StringIO(text))
        raw_headers = next((row for row in header_rows if any(cell.strip() for cell in row)), [])
        frame = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
    except (csv.Error, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("The uploaded file must be a valid UTF-8 CSV.") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError("The CSV must contain at least one data row.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"The CSV exceeds the {MAX_ROWS:,}-row limit.")
    if not len(frame.columns) or any(not column for column in frame.columns):
        raise ValueError("Every CSV column must have a header.")
    normalized_headers = [header.strip() for header in raw_headers]
    if len(set(normalized_headers)) != len(normalized_headers):
        raise ValueError("CSV column names must be unique.")
    if normalized_headers != list(frame.columns):
        raise ValueError("The CSV header row is invalid.")
    return frame


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def preview_time_series_csv(content: bytes) -> dict:
    """Return column metadata without retaining or returning uploaded observations."""
    frame = _read_csv(content)
    date_columns = []
    numeric_columns = []
    for column in frame.columns:
        values = frame[column].astype(str).str.strip()
        parsed_dates = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
        if parsed_dates.notna().mean() >= 0.8:
            date_columns.append(column)
        numeric = _numeric_series(frame, column)
        if numeric.notna().sum() >= 3:
            numeric_columns.append(column)
    if not date_columns:
        raise ValueError("No date column was recognized. Include a date or timestamp column.")
    if not numeric_columns:
        raise ValueError("No numeric columns were recognized.")
    return {
        "columns": list(frame.columns),
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "row_count": len(frame),
    }


def analyze_time_series_csv(
    content: bytes,
    date_column: str,
    target_column: str,
    predictor_columns: list[str],
    horizon_periods: int,
    model_type: str,
) -> dict:
    """Fit a chronological holdout regression for forward asset returns."""
    frame = _read_csv(content)
    if date_column not in frame.columns or target_column not in frame.columns:
        raise ValueError("Choose a date column and target column from the uploaded CSV.")
    if date_column == target_column:
        raise ValueError("The date column cannot also be the asset price target.")
    if model_type not in {"linear", "logistic"}:
        raise ValueError("Model type must be linear or logistic.")
    if not 1 <= horizon_periods <= 30:
        raise ValueError("Prediction horizon must be between 1 and 30 observations.")
    if not predictor_columns or len(predictor_columns) > MAX_PREDICTORS:
        raise ValueError(f"Choose between 1 and {MAX_PREDICTORS} predictor columns.")
    if len(set(predictor_columns)) != len(predictor_columns):
        raise ValueError("Predictor columns must be unique.")
    if target_column in predictor_columns or date_column in predictor_columns:
        raise ValueError("The target and date columns cannot be used as predictors.")
    missing_columns = [column for column in [*predictor_columns, target_column] if column not in frame.columns]
    if missing_columns:
        raise ValueError("One or more selected columns are not in the uploaded CSV.")

    dates = pd.to_datetime(
        frame[date_column].astype(str).str.strip(),
        errors="coerce",
        format="mixed",
        utc=True,
    )
    target_prices = _numeric_series(frame, target_column)
    data = pd.DataFrame({"date": dates, "price": target_prices})
    for index, column in enumerate(predictor_columns):
        data[f"x{index}"] = _numeric_series(frame, column)
    data = data.sort_values("date", kind="stable").reset_index(drop=True)
    if data["date"].isna().any():
        raise ValueError("The selected date column contains invalid or blank dates.")
    if data["date"].duplicated().any():
        raise ValueError("Dates must be unique; combine duplicate timestamps before modeling.")
    if (data["price"].dropna() <= 0).any():
        raise ValueError("The target asset-price column must contain positive prices.")

    future_price = data["price"].shift(-horizon_periods)
    forward_return = future_price.div(data["price"]).sub(1)
    selected_features = [f"x{index}" for index in range(len(predictor_columns))]
    clean = data[["date", *selected_features]].copy()
    clean["target"] = forward_return
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(clean) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(clean)} complete forward-return observations remain; at least "
            f"{MIN_OBSERVATIONS} are required."
        )

    x = clean[selected_features].to_numpy(dtype=float)
    y = clean["target"].to_numpy(dtype=float)
    split = int(len(clean) * 0.8)
    if split < len(selected_features) + 3 or len(clean) - split < 5:
        raise ValueError("Not enough complete observations for the selected predictors and chronological test set.")
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    if (scales <= np.finfo(float).eps).any():
        constant_features = [
            predictor_columns[index]
            for index, scale in enumerate(scales)
            if scale <= np.finfo(float).eps
        ]
        raise ValueError(f"Predictors are constant in the training period: {', '.join(constant_features)}.")
    x_train = (x_train - means) / scales
    x_test = (x_test - means) / scales
    x_train_design = np.column_stack((np.ones(len(x_train)), x_train))
    x_test_design = np.column_stack((np.ones(len(x_test)), x_test))

    if model_type == "linear":
        return _linear_result(
            predictor_columns, clean, split, x_train_design, x_test_design, y_train, y_test, horizon_periods
        )
    return _logistic_result(
        predictor_columns, clean, split, x_train_design, x_test_design, y_train, y_test, horizon_periods
    )


def _linear_result(
    predictor_columns: list[str],
    clean: pd.DataFrame,
    split: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizon_periods: int,
) -> dict:
    coefficients, _, rank, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    if rank < x_train.shape[1]:
        raise ValueError("Predictors are linearly dependent in the training data; remove redundant columns.")
    predictions = x_test @ coefficients
    residuals = y_train - x_train @ coefficients
    degrees_freedom = len(y_train) - x_train.shape[1]
    if degrees_freedom <= 0:
        raise ValueError("Not enough training observations to estimate model uncertainty.")
    covariance = (residuals @ residuals / degrees_freedom) * np.linalg.pinv(x_train.T @ x_train)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.zeros_like(coefficients),
        where=standard_errors > 0,
    )
    p_values = 2 * t.sf(np.abs(statistics), degrees_freedom)
    test_residuals = y_test - predictions
    target_mean = float(np.mean(y_test))
    total_sum_squares = float(np.sum((y_test - target_mean) ** 2))
    r_squared = (
        1 - float(np.sum(test_residuals**2)) / total_sum_squares
        if total_sum_squares > 0
        else None
    )
    return {
        "model_type": "linear",
        "target_description": f"{horizon_periods}-observation forward simple return",
        "observations": len(clean),
        "training_observations": split,
        "test_observations": len(y_test),
        "metrics": {
            "r_squared": r_squared,
            "mae": float(np.mean(np.abs(test_residuals))),
            "rmse": float(np.sqrt(np.mean(test_residuals**2))),
        },
        "coefficients": [
            {
                "feature": "(intercept)" if index == 0 else predictor_columns[index - 1],
                "coefficient": float(coefficients[index]),
                "standard_error": float(standard_errors[index]),
                "p_value": float(p_values[index]),
            }
            for index in range(len(coefficients))
        ],
        "test_series": [
            {
                "date": clean["date"].iloc[split + index].date().isoformat(),
                "actual_return": float(y_test[index]),
                "predicted_return": float(predictions[index]),
            }
            for index in range(len(y_test))
        ],
    }


def _logistic_result(
    predictor_columns: list[str],
    clean: pd.DataFrame,
    split: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train_returns: np.ndarray,
    y_test_returns: np.ndarray,
    horizon_periods: int,
) -> dict:
    y_train = (y_train_returns > 0).astype(float)
    y_test = (y_test_returns > 0).astype(float)
    if np.unique(y_train).size != 2:
        raise ValueError("The training period must contain both positive and non-positive forward returns.")
    if np.unique(y_test).size != 2:
        raise ValueError("The chronological test period must contain both positive and non-positive forward returns.")

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x_train @ coefficients
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
        penalty = 0.5 * float(coefficients[1:] @ coefficients[1:])
        loss = float(np.sum(np.logaddexp(0, logits) - y_train * logits)) + penalty
        gradient = x_train.T @ (probabilities - y_train)
        gradient[1:] += coefficients[1:]
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(x_train.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2_000, "ftol": 1e-12},
    )
    if not fit.success:
        raise ValueError(f"Logistic regression did not converge: {fit.message}")

    logits = x_test @ fit.x
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
    predicted = probabilities >= 0.5
    true_positive = int(np.sum(predicted & (y_test == 1)))
    true_negative = int(np.sum(~predicted & (y_test == 0)))
    false_positive = int(np.sum(predicted & (y_test == 0)))
    false_negative = int(np.sum(~predicted & (y_test == 1)))
    auc = _roc_auc(y_test, probabilities)
    return {
        "model_type": "logistic",
        "target_description": f"probability of a positive {horizon_periods}-observation forward return",
        "observations": len(clean),
        "training_observations": split,
        "test_observations": len(y_test),
        "metrics": {
            "accuracy": float(np.mean(predicted == y_test)),
            "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
            "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
            "roc_auc": auc,
            "log_loss": float(-np.mean(y_test * np.log(np.clip(probabilities, 1e-12, 1)) + (1 - y_test) * np.log(np.clip(1 - probabilities, 1e-12, 1)))),
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "coefficients": [
            {
                "feature": "(intercept)" if index == 0 else predictor_columns[index - 1],
                "coefficient": float(fit.x[index]),
                "odds_ratio": float(np.exp(np.clip(fit.x[index], -35, 35))),
            }
            for index in range(len(fit.x))
        ],
        "test_series": [
            {
                "date": clean["date"].iloc[split + index].date().isoformat(),
                "actual_up": bool(y_test[index]),
                "predicted_probability": float(probabilities[index]),
            }
            for index in range(len(y_test))
        ],
    }


def _roc_auc(actual: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(np.sum(actual == 1))
    negatives = len(actual) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(probabilities)
    return float((np.sum(ranks[actual == 1]) - positives * (positives + 1) / 2) / (positives * negatives))
