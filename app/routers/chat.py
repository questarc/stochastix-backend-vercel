from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..deps import AgentCache, get_agent_cache, get_order_store
from ..llm import LLMConfigError
from ..schemas import ChatRequest, ChatResponse
from ..stores import OrderStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _ask(req: ChatRequest, cache: AgentCache, store: OrderStore) -> dict:
    agent = cache.get(req.provider, req.model)
    try:
        reply = agent.ask(req.message, req.thread_id, req.account_id, req.model_id)
    except LLMConfigError:
        raise
    except Exception as exc:
        logger.exception("Advisor AI request failed (provider=%s, model=%s)", req.provider, req.model)
        raise HTTPException(
            status_code=502,
            detail=f"The AI provider '{req.provider}' did not respond successfully. Please try again "
                   f"or choose another model. ({type(exc).__name__})",
        ) from exc
    if reply.orders:
        store.merge(reply.orders)
    return {
        "text": reply.text,
        "tool_calls": reply.tool_calls,
        "orders_added": len(reply.orders),
        "orders": reply.orders,
        "audio": reply.audio,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, cache: AgentCache = Depends(get_agent_cache),
               store: OrderStore = Depends(get_order_store)):
    return await run_in_threadpool(_ask, req, cache, store)


@router.delete("/chat/{thread_id}")
def reset_chat(thread_id: str, cache: AgentCache = Depends(get_agent_cache)):
    cache.reset_thread(thread_id)
    return {"reset": True}
