from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from ..deps import get_fidelity_positions_store, get_fidelity_statement_store
from ..etrade import MAX_PDF_BYTES, read_statement_pdf
from ..schemas import EtradeStatement, FidelityCsvPosition
from ..sources.positions_agent import parse_fidelity_positions_csv
from ..stores import InMemoryEtradeStatementStore, InMemoryFidelityPositionsStore

router = APIRouter(prefix="/fidelity", tags=["Fidelity"])
MAX_CSV_BYTES = 20 * 1024 * 1024


@router.get("/statement", response_model=EtradeStatement | None)
def get_statement(
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryEtradeStatementStore = Depends(get_fidelity_statement_store),
) -> EtradeStatement | None:
    return store.get(session_id)


@router.post("/statement/upload", response_model=EtradeStatement)
async def upload_statement(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryEtradeStatementStore = Depends(get_fidelity_statement_store),
):
    content = await file.read(MAX_PDF_BYTES + 1)
    statement = read_statement_pdf(file.filename or "statement.pdf", content, "Fidelity")
    del content
    store.set(session_id, statement)
    return statement


@router.delete("/statement")
def clear_statement(
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryEtradeStatementStore = Depends(get_fidelity_statement_store),
):
    return {"cleared": store.clear(session_id)}


@router.get("/positions", response_model=list[FidelityCsvPosition] | None)
def get_positions(
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryFidelityPositionsStore = Depends(get_fidelity_positions_store),
) -> list[FidelityCsvPosition] | None:
    return store.get(session_id)


@router.post("/positions/upload", response_model=list[FidelityCsvPosition])
async def upload_positions(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryFidelityPositionsStore = Depends(get_fidelity_positions_store),
) -> list[FidelityCsvPosition]:
    content = await file.read(MAX_CSV_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="The CSV exceeds the 20 MB upload limit.")
    try:
        positions = parse_fidelity_positions_csv(content)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a UTF-8 CSV file.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.set(session_id, positions)
    return positions


@router.delete("/positions")
def clear_positions(
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryFidelityPositionsStore = Depends(get_fidelity_positions_store),
):
    return {"cleared": store.clear(session_id)}
