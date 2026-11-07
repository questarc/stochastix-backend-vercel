"""Advisor AI tools exposed over HTTP for agents that run in the browser (WebLLM)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.utils.function_calling import convert_to_openai_tool

from ..agent import SYSTEM_PROMPT
from ..deps import get_agent_tools, get_order_store
from ..schemas import AgentSpec, ToolRunRequest, ToolRunResponse
from ..stores import OrderStore

router = APIRouter(tags=["agent"])


@router.get("/agent/tools", response_model=AgentSpec)
def agent_spec(tools: dict = Depends(get_agent_tools)):
    return {"system_prompt": SYSTEM_PROMPT, "tools": [convert_to_openai_tool(t) for t in tools.values()]}


def _run(tool, name: str, args: dict, store: OrderStore) -> dict:
    orders, audio = [], []
    try:
        message = tool.invoke({"name": name, "args": args, "id": "browser", "type": "tool_call"})
        result = str(message.content)
        if name == "create_audio_summary":
            audio = [message.artifact] if message.artifact else []
        else:
            orders = list(message.artifact or [])
    except Exception as exc:  # invalid arguments are reported to the model, like tool errors
        result = json.dumps({"error": str(exc)})
    if orders:
        store.merge(orders)
    return {"tool": name, "args": args, "result": result, "orders_added": len(orders), "orders": orders,
            "audio": audio}


@router.post("/agent/tools/{name}", response_model=ToolRunResponse)
async def run_tool(name: str, req: ToolRunRequest, tools: dict = Depends(get_agent_tools),
                   store: OrderStore = Depends(get_order_store)):
    if name not in tools:
        raise HTTPException(status_code=404, detail=f"Unknown tool '{name}'.")
    return await run_in_threadpool(_run, tools[name], name, req.args, store)
