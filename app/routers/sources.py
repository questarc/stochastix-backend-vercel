from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..deps import get_sources, get_upload_store
from ..schemas import Account, ModelPortfolio, Position, SourceInfo
from ..sources import SourceAgent
from ..stores import UploadStore

router = APIRouter(tags=["sources"])


def lookup_source(sources: dict[str, SourceAgent], name: str) -> SourceAgent:
    if name not in sources:
        raise HTTPException(status_code=404, detail=f"Unknown source '{name}'.")
    return sources[name]


def _upload_source(sources: dict[str, SourceAgent], name: str) -> SourceAgent:
    source = lookup_source(sources, name)
    if not source.supports_upload:
        raise HTTPException(status_code=400, detail=f"Source '{name}' does not support uploads.")
    return source


@router.get("/sources", response_model=list[SourceInfo])
def list_sources(sources: dict = Depends(get_sources)):
    return [s.describe() for s in sources.values()]


@router.get("/sources/{source}/accounts", response_model=list[Account])
def list_accounts(source: str, sources: dict = Depends(get_sources)):
    return [a.to_dict() for a in lookup_source(sources, source).list_accounts()]


@router.get("/sources/{source}/models", response_model=list[ModelPortfolio])
def list_models(source: str, sources: dict = Depends(get_sources)):
    return [m.to_dict() for m in lookup_source(sources, source).list_model_portfolios()]


@router.post("/sources/{source}/accounts/{account_id}/positions/upload", response_model=list[Position])
async def upload_positions(
    source: str,
    account_id: str,
    file: UploadFile = File(...),
    sources: dict = Depends(get_sources),
    store: UploadStore = Depends(get_upload_store),
):
    agent = _upload_source(sources, source)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        positions = agent.parse_upload(content, account_id)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a UTF-8 CSV file.") from exc
    if not positions:
        raise HTTPException(status_code=400, detail="The file contains no positions with a quantity above zero.")
    store.set(agent.name, account_id, positions)
    return [p.to_dict() for p in positions]


@router.delete("/sources/{source}/accounts/{account_id}/positions/upload")
def reset_positions(
    source: str,
    account_id: str,
    sources: dict = Depends(get_sources),
    store: UploadStore = Depends(get_upload_store),
):
    agent = _upload_source(sources, source)
    agent.get_account(account_id)
    return {"reset": store.reset(agent.name, account_id)}
