from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..time_series import analyze_time_series_csv, preview_time_series_csv

router = APIRouter(prefix="/time-series", tags=["Time Series Statistical Models"])
MAX_CSV_BYTES = 20 * 1024 * 1024


async def _read_csv(file: UploadFile) -> bytes:
    content = await file.read(MAX_CSV_BYTES + 1)
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="The CSV exceeds the 20 MB upload limit.")
    return content


@router.post("/preview")
async def preview_csv(file: UploadFile = File(...)):
    content = await _read_csv(file)
    try:
        result = await run_in_threadpool(preview_time_series_csv, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    date_column: str = Form(...),
    target_column: str = Form(...),
    predictor_columns: list[str] = Form(...),
    horizon_periods: int = Form(1, ge=1, le=30),
    model_type: Literal["linear", "logistic"] = Form(...),
):
    content = await _read_csv(file)
    try:
        result = await run_in_threadpool(
            analyze_time_series_csv,
            content,
            date_column,
            target_column,
            predictor_columns,
            horizon_periods,
            model_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
