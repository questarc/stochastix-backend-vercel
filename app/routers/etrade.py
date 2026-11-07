from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, UploadFile

from ..deps import get_etrade_statement_store
from ..etrade import MAX_PDF_BYTES, read_statement_pdf
from ..schemas import EtradeStatement
from ..stores import InMemoryEtradeStatementStore

router = APIRouter(prefix="/etrade", tags=["E*TRADE"])


@router.get("/statement", response_model=EtradeStatement | None)
def get_statement(
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryEtradeStatementStore = Depends(get_etrade_statement_store),
) -> EtradeStatement | None:
    return store.get(session_id)


@router.post("/statement/upload", response_model=EtradeStatement)
async def upload_statement(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryEtradeStatementStore = Depends(get_etrade_statement_store),
):
    content = await file.read(MAX_PDF_BYTES + 1)
    statement = read_statement_pdf(file.filename or "statement.pdf", content)
    del content
    store.set(session_id, statement)
    return statement


@router.delete("/statement")
def clear_statement(
    session_id: str = Header(..., alias="X-Statement-Session", min_length=16, max_length=128),
    store: InMemoryEtradeStatementStore = Depends(get_etrade_statement_store),
):
    return {"cleared": store.clear(session_id)}
