"""Advisor AI: a LangChain agent on a LangGraph runtime with per-thread memory."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ..engine import CapitalMarketAssumptions
from ..rebalance import RebalanceSettings
from .tools import build_tools

logger = logging.getLogger(__name__)

TOOL_RESULT_LIMIT = 2_000

SYSTEM_PROMPT = """You are Stochastix AI, an assistant for financial advisors inside the \
Stochastix AI app.

Rules:
1. Use the tools for every number. Never calculate, estimate or invent data yourself.
2. If the question names a client, account or model portfolio, use it. Otherwise use the account and \
model selected in the app, which are given in square brackets before the question.
3. For "what if" questions, call run_simulation with only the changed plan values and report the \
saved-plan baseline and the difference in percentage points.
4. Never say "no trades needed" unless the tool result shows "trades": "none".
5. When asked to "simulate and rebalance", call simulate_and_rebalance exactly once for all the \
requested accounts.
6. Orders you create are pending. They are not sent to the market; the advisor reviews them in the \
Orders tab.
7. Use the pre-formatted numbers from the tools exactly as given (for example "84.2%", "$1,234,567").
8. Keep answers short and clear, using bullet points or small tables.
9. Results are projections, not guarantees, and not investment advice. Say so when you report results.
10. When asked for an audio or voice summary for the client, call create_audio_summary with the summary \
you already gave, word for word, including its numbers. Do not add new figures.
11. For economic news, upcoming earnings or corporate actions (dividends, splits), use get_market_news, \
get_upcoming_earnings or get_corporate_actions. Report only items the tools return, with their source and date.
12. For how the market or sectors are doing today, use get_market_heatmap.
"""


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    audio: list[dict] = field(default_factory=list)


def selection_prefix(account_id: str | None, model_id: str | None) -> str:
    parts = []
    if account_id:
        parts.append(f"Account selected in the app: {account_id} (use only if the question names no client or account)")
    if model_id:
        parts.append(f"Model portfolio selected in the app: {model_id} (use only if the question names no model)")
    return f"[{'; '.join(parts)}]\n" if parts else ""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


class AdvisorAgent:
    def __init__(self, llm, sources: dict, cma: CapitalMarketAssumptions,
                 rebalance_settings: RebalanceSettings | None = None,
                 max_trials: int = 25_000, recursion_limit: int = 20, voice_agent=None, market_agents=None):
        self.recursion_limit = recursion_limit
        self.checkpointer = InMemorySaver()
        tools = build_tools(sources, cma, rebalance_settings or RebalanceSettings(), max_trials, voice_agent,
                            market_agents=market_agents)
        self.graph = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT,
                                  checkpointer=self.checkpointer, name="advisor_agent")

    def ask(self, question: str, thread_id: str, account_id: str | None = None,
            model_id: str | None = None) -> AgentReply:
        message = selection_prefix(account_id, model_id) + question
        state = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit},
        )
        messages = state["messages"]
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        turn = messages[start + 1:]

        results = {m.tool_call_id: m for m in turn if isinstance(m, ToolMessage)}
        tool_calls, orders, audio = [], [], []
        for m in turn:
            if isinstance(m, AIMessage):
                for call in m.tool_calls or []:
                    tm = results.get(call.get("id"))
                    result = _text(tm.content) if tm is not None else ""
                    tool_calls.append({"tool": call["name"], "args": call.get("args", {}),
                                       "result": result[:TOOL_RESULT_LIMIT]})
                    if tm is not None and call["name"] == "simulate_and_rebalance" and tm.artifact:
                        orders.extend(tm.artifact)
                    if tm is not None and call["name"] == "create_audio_summary" and tm.artifact:
                        audio.append(tm.artifact)
        final = next((m for m in reversed(turn) if isinstance(m, AIMessage) and _text(m.content).strip()), None)
        text = _text(final.content) if final is not None else ""
        logger.info("Advisor AI thread=%s tools=%s orders=%d", thread_id,
                    json.dumps([c["tool"] for c in tool_calls]), len(orders))
        return AgentReply(text=text, tool_calls=tool_calls, orders=orders, audio=audio)

    def reset(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)
