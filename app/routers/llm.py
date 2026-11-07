from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from ..deps import get_llm_settings
from ..llm import LLMSettings, api_key_set, available_models
from ..schemas import LLMProvider

router = APIRouter(tags=["llm"])


def _providers(settings: LLMSettings) -> list[dict]:
    out = []
    for pid, cfg in settings.providers.items():
        models = available_models(settings, pid)
        default = settings.model_for(pid)
        out.append({
            "id": pid,
            "label": cfg.get("label", pid),
            "default_model": default,
            "models": models if default in models else [default, *models],
            "api_key_env": cfg.get("api_key_env"),
            "api_key_set": api_key_set(cfg),
        })
    default_id = settings.default_provider
    return sorted(out, key=lambda p: p["id"] != default_id)


@router.get("/llm/providers", response_model=list[LLMProvider])
async def list_providers(settings: LLMSettings = Depends(get_llm_settings)):
    return await run_in_threadpool(_providers, settings)
