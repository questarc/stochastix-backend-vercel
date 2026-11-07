"""Cached application singletons, injected into routers with FastAPI ``Depends``."""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Callable

from .agent import AdvisorAgent, build_tools
from .engine import CapitalMarketAssumptions
from .llm import LLMSettings, build_chat_model
from .rebalance import RebalanceSettings
from .sources import SourceAgent, load_sources
from .sources.base import BASE_DIR
from .stores import (InMemoryEtradeStatementStore, InMemoryFidelityPositionsStore, InMemoryOrderStore,
                    InMemoryUploadStore, OrderStore, UploadStore)
from .market import (CorporateActionsAgent, EarningsAgent, MarketAnalyzerAgent, MarketHeatmapAgent, MarketHttp,
                     MarketSettings, holdings_index)
from .voice import AudioStore, PiperEngine, VoiceGenAgent

CONFIG_DIR = BASE_DIR / "config"


@lru_cache
def get_upload_store() -> UploadStore:
    return InMemoryUploadStore()


@lru_cache
def get_etrade_statement_store() -> InMemoryEtradeStatementStore:
    return InMemoryEtradeStatementStore()


@lru_cache
def get_fidelity_statement_store() -> InMemoryEtradeStatementStore:
    return InMemoryEtradeStatementStore()


@lru_cache
def get_fidelity_positions_store() -> InMemoryFidelityPositionsStore:
    return InMemoryFidelityPositionsStore()


@lru_cache
def get_order_store() -> OrderStore:
    return InMemoryOrderStore()


@lru_cache
def get_sources() -> dict[str, SourceAgent]:
    sources = load_sources(CONFIG_DIR / "sources.json")
    for source in sources.values():
        if hasattr(source, "upload_store"):
            source.upload_store = get_upload_store()
    return sources


@lru_cache
def get_cma() -> CapitalMarketAssumptions:
    return CapitalMarketAssumptions.load(CONFIG_DIR / "capital_market_assumptions.json")


@lru_cache
def get_rebalance_settings() -> RebalanceSettings:
    return RebalanceSettings.load(CONFIG_DIR / "rebalance.json")


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings.load(CONFIG_DIR / "llm.json")


@lru_cache
def get_speech_engine() -> PiperEngine:
    return PiperEngine()


@lru_cache
def get_audio_store() -> AudioStore:
    return AudioStore()


@lru_cache
def get_voice_agent() -> VoiceGenAgent:
    return VoiceGenAgent(get_speech_engine(), get_audio_store(), get_sources())


@lru_cache
def get_market_settings() -> MarketSettings:
    return MarketSettings.load(CONFIG_DIR / "market.json")


def _market_http() -> MarketHttp:
    s = get_market_settings()
    return MarketHttp(s.user_agent, s.request_timeout_seconds)


@lru_cache
def get_market_agent() -> MarketAnalyzerAgent:
    return MarketAnalyzerAgent(get_market_settings(), _market_http())


@lru_cache
def get_earnings_agent() -> EarningsAgent:
    return EarningsAgent(get_market_settings(), _market_http(), lambda: holdings_index(get_sources()))


@lru_cache
def get_corporate_actions_agent() -> CorporateActionsAgent:
    return CorporateActionsAgent(get_market_settings(), _market_http(), lambda: holdings_index(get_sources()))


@lru_cache
def get_heatmap_agent() -> MarketHeatmapAgent:
    return MarketHeatmapAgent(get_market_settings(), _market_http(), lambda: holdings_index(get_sources()))


def get_market_agents() -> dict:
    return {"news": get_market_agent(), "earnings": get_earnings_agent(), "actions": get_corporate_actions_agent(),
            "heatmap": get_heatmap_agent()}


class AgentCache:
    """One Advisor AI agent per (provider, model)."""

    def __init__(self, factory: Callable[[str, str | None], AdvisorAgent]):
        self._factory = factory
        self._agents: dict[tuple[str, str | None], AdvisorAgent] = {}
        self._lock = threading.Lock()

    def get(self, provider: str, model: str | None) -> AdvisorAgent:
        key = (provider, model)
        with self._lock:
            if key not in self._agents:
                self._agents[key] = self._factory(provider, model)
            return self._agents[key]

    def reset_thread(self, thread_id: str) -> None:
        with self._lock:
            agents = list(self._agents.values())
        for agent in agents:
            agent.reset(thread_id)


def _build_agent(provider: str, model: str | None) -> AdvisorAgent:
    settings = get_llm_settings()
    llm = build_chat_model(settings, provider, model)
    return AdvisorAgent(llm, get_sources(), get_cma(), get_rebalance_settings(),
                        max_trials=settings.max_trials_per_tool_call, voice_agent=get_voice_agent(),
                        market_agents=get_market_agents())


@lru_cache
def get_agent_cache() -> AgentCache:
    return AgentCache(_build_agent)


@lru_cache
def get_agent_tools() -> dict:
    """Advisor AI tools by name, for agents that run in the browser (WebLLM)."""
    tools = build_tools(get_sources(), get_cma(), get_rebalance_settings(),
                        get_llm_settings().max_trials_per_tool_call, voice_agent=get_voice_agent(),
                        market_agents=get_market_agents())
    return {t.name: t for t in tools}
