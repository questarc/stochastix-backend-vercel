from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import deps
from app.agent import AdvisorAgent
from app.deps import AgentCache


class FakeToolModel(BaseChatModel):
    """Replays scripted AI messages so tests do not need an LLM."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _cache(script, voice_agent=None):
    def factory(provider, model):
        return AdvisorAgent(FakeToolModel(script=script), deps.get_sources(), deps.get_cma(),
                            deps.get_rebalance_settings(), max_trials=1000, voice_agent=voice_agent)
    return AgentCache(factory)


def _call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_chat_runs_tools_and_adds_orders(client):
    script = [
        _call("simulate_and_rebalance", {"client_name": "Jane"}),
        AIMessage(content="Created pending orders for Jane Smith. Projections are not guarantees."),
    ]
    cache = _cache(script)
    client.app.dependency_overrides[deps.get_agent_cache] = lambda: cache
    r = client.post("/api/chat", json={"message": "Simulate and rebalance Jane", "thread_id": "t1",
                                       "provider": "ollama", "account_id": "A1001"})
    body = r.json()
    assert r.status_code == 200
    assert body["text"].startswith("Created pending orders")
    assert body["tool_calls"][0]["tool"] == "simulate_and_rebalance"
    assert body["orders_added"] > 0 and body["orders_added"] == len(body["orders"])
    assert len(client.get("/api/orders").json()) == body["orders_added"]


def test_chat_creates_audio_summary(client):
    from app.voice import AudioStore, VoiceGenAgent
    from tests.test_voice import FakeEngine

    voice = VoiceGenAgent(FakeEngine(), AudioStore(), deps.get_sources())
    script = [
        _call("create_audio_summary", {"summary_text": "Success is 76.8%.", "account_id": "A1001"}),
        AIMessage(content="Your audio summary is ready."),
    ]
    cache = _cache(script, voice)
    client.app.dependency_overrides[deps.get_agent_cache] = lambda: cache
    body = client.post("/api/chat", json={"message": "Make an audio summary", "thread_id": "t5",
                                          "provider": "ollama"}).json()
    assert len(body["audio"]) == 1 and "76.8 percent" in body["audio"][0]["script"]
    assert body["orders_added"] == 0


def test_chat_tool_error_is_returned_as_json(client):
    script = [_call("get_account_portfolio", {"account_id": "ZZZ"}), AIMessage(content="Unknown account.")]
    cache = _cache(script)
    client.app.dependency_overrides[deps.get_agent_cache] = lambda: cache
    body = client.post("/api/chat", json={"message": "x", "thread_id": "t2", "provider": "ollama"}).json()
    assert '"error"' in body["tool_calls"][0]["result"] and body["orders_added"] == 0


def test_chat_selection_prefix_and_reset(client):
    cache = _cache([AIMessage(content="Hello")])
    client.app.dependency_overrides[deps.get_agent_cache] = lambda: cache
    client.post("/api/chat", json={"message": "hi", "thread_id": "t3", "provider": "ollama",
                                   "account_id": "A1001", "model_id": "MOD-60-40"})
    agent = cache.get("ollama", None)
    state = agent.graph.get_state({"configurable": {"thread_id": "t3"}})
    first = state.values["messages"][0].content
    assert first.startswith("[Account selected in the app: A1001") and "MOD-60-40" in first
    assert client.delete("/api/chat/t3").json() == {"reset": True}
    assert not agent.graph.get_state({"configurable": {"thread_id": "t3"}}).values


def test_chat_provider_failure_returns_502(client):
    class Broken:
        def ask(self, *a, **k):
            raise RuntimeError("boom")

    class Cache:
        def get(self, provider, model):
            return Broken()

    client.app.dependency_overrides[deps.get_agent_cache] = lambda: Cache()
    r = client.post("/api/chat", json={"message": "x", "thread_id": "t4", "provider": "ollama"})
    assert r.status_code == 502


def test_agent_tools_spec_lists_all_tools(client):
    body = client.get("/api/agent/tools").json()
    names = [t["function"]["name"] for t in body["tools"]]
    assert len(names) == 14 and "get_upcoming_earnings" in names and "get_market_heatmap" in names
    run_sim = next(t for t in body["tools"] if t["function"]["name"] == "run_simulation")
    assert "account_id" in run_sim["function"]["parameters"]["properties"]
    assert "Use the tools for every number" in body["system_prompt"]


def test_agent_tool_run_formats_numbers_and_reports_errors(client):
    r = client.post("/api/agent/tools/compare_current_vs_model", json={"args": {"account_id": "A1001"}})
    assert r.status_code == 200 and "%" in r.json()["result"] and r.json()["orders_added"] == 0
    bad = client.post("/api/agent/tools/get_account_portfolio", json={"args": {"account_id": "ZZZ"}}).json()
    assert '"error"' in bad["result"]
    missing_arg = client.post("/api/agent/tools/run_simulation", json={"args": {}}).json()
    assert '"error"' in missing_arg["result"]
    assert client.post("/api/agent/tools/nope", json={"args": {}}).status_code == 404


def test_agent_tool_simulate_and_rebalance_adds_orders(client):
    body = client.post("/api/agent/tools/simulate_and_rebalance",
                       json={"args": {"client_name": "Jane"}}).json()
    assert body["orders_added"] > 0 and body["orders_added"] == len(body["orders"])
    assert len(client.get("/api/orders").json()) == body["orders_added"]
