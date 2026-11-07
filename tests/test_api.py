from app import deps
from app.llm import LLMConfigError

SIM = {"account_id": "A1001", "trials": 1000, "seed": 42}
A360_CSV = (
    "Ticker,Cusip,Sec Type,Qty,Custodian Quantity,PM Price,Market Value\n"
    'VTI,922908769,ETF,100,,$300.00,"$30,000.00"\n'
    "BND,921937835,ETF,200,,$73.00,\"$14,600.00\"\n"
)


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_sources_accounts_models(client):
    names = [s["name"] for s in client.get("/api/sources").json()]
    assert names == ["sales", "positions"]
    accounts = client.get("/api/sources/sales/accounts").json()
    assert accounts[0]["account_id"] == "A1001"
    models = client.get("/api/sources/sales/models").json()
    assert {m["model_id"] for m in models} == {"MOD-60-40", "CON-40-60", "AGG-80-20"}
    assert client.get("/api/sources/nope/accounts").status_code == 404


def test_portfolio(client):
    body = client.get("/api/accounts/A1001/portfolio", params={"model_id": "CON-40-60"}).json()
    assert body["model"]["model_id"] == "CON-40-60"
    assert body["market_value"] == sum(p["market_value"] for p in body["positions"])
    assert client.get("/api/accounts/ZZZ/portfolio").status_code == 404


def test_config_endpoints(client):
    assert client.get("/api/config/cma").json()["inflation"] == 0.025
    options = client.get("/api/config/options").json()
    assert options["sampling_methods"] == ["prng", "lhs", "sobol"]
    assert len(options["distributions"]) == 7


def test_simulations_reproducible_and_shaped(client):
    a = client.post("/api/simulations", json=SIM).json()
    b = client.post("/api/simulations", json=SIM).json()
    assert a["current"]["success_probability"] == b["current"]["success_probability"]
    cur = a["current"]
    assert len(cur["ages"]) == len(cur["bands"]["p50"])
    assert len(cur["ending_values_histogram"]["counts"]) == 60
    assert cur["settings"]["seed"] == 42 and "sample_paths" not in cur or cur["sample_paths"] is None


def test_simulations_sample_paths_and_distribution(client):
    body = client.post("/api/simulations", json={**SIM, "sample_paths": 5, "distribution": "normal"}).json()
    assert len(body["current"]["sample_paths"]) == 5
    assert set(body["current"]["settings"]["distributions"].values()) == {"normal"}


def test_simulations_validation(client):
    r = client.post("/api/simulations", json={**SIM, "retirement_age": 20})
    assert r.status_code == 400 and "Ages" in r.json()["detail"]
    assert client.post("/api/simulations", json={**SIM, "trials": 60_000}).status_code == 422
    assert client.post("/api/simulations", json={**SIM, "account_id": "ZZZ"}).status_code == 404


def test_rebalance_proposal(client):
    body = client.post("/api/rebalance/proposal", json={"account_id": "A1001", "simulation": SIM}).json()
    assert body["rebalance_needed"] is True and body["trades"]
    assert 0 <= body["success_current"] <= 1 and 0 <= body["success_after"] <= 1
    calm = client.post("/api/rebalance/proposal", json={"account_id": "A1001", "drift_tolerance": 0.5,
                                                         "simulation": SIM}).json()
    assert calm["rebalance_needed"] is False and calm["trades"] == []


def test_orders_lifecycle(client):
    orders = client.post("/api/orders/generate", json={"account_id": "A1001"}).json()
    assert orders and orders[0]["side"] == "Sell"
    client.post("/api/orders/generate", json={"account_id": "A1001"})
    assert len(client.get("/api/orders").json()) == len(orders)
    csv = client.get("/api/orders/export", params={"format": "csv"})
    assert csv.headers["content-type"].startswith("text/csv")
    assert csv.text.splitlines()[0].startswith("order_id,account_id")
    assert "orders_" in csv.headers["content-disposition"]
    assert len(client.get("/api/orders/export", params={"format": "json"}).json()) == len(orders)
    client.delete("/api/orders")
    assert client.get("/api/orders").json() == []


def test_upload_and_reset_positions(client):
    url = "/api/sources/positions/accounts/A2001/positions/upload"
    r = client.post(url, files={"file": ("p.csv", A360_CSV, "text/csv")})
    assert r.status_code == 200 and [p["ticker"] for p in r.json()] == ["VTI", "BND"]
    assert client.get("/api/accounts/A2001/portfolio").json()["market_value"] == 44_600
    bad = client.post(url, files={"file": ("p.csv", "a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 400 and "missing columns" in bad.json()["detail"]
    assert client.delete(url).json() == {"reset": True}
    assert client.get("/api/accounts/A2001/portfolio").json()["market_value"] > 44_600
    assert client.post("/api/sources/sales/accounts/A1001/positions/upload",
                       files={"file": ("p.csv", A360_CSV, "text/csv")}).status_code == 400


def test_methodology(client):
    titles = [s["title"] for s in client.get("/api/methodology").json()]
    assert titles == ["What it is", "How it works", "How it calculates", "How to measure the results", "Limitations"]


def test_llm_providers_never_return_keys(client, monkeypatch):
    monkeypatch.setattr("app.routers.llm.available_models", lambda s, p: s.providers[p].get("models", []))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-value")
    body = client.get("/api/llm/providers").json()
    assert body[0]["id"] == "nvidia"
    nvidia = next(p for p in body if p["id"] == "nvidia")
    assert nvidia["api_key_set"] is True and "nvidia-test-value" not in str(body)
    anthropic = next(p for p in body if p["id"] == "anthropic")
    assert anthropic["api_key_set"] is True and "test-value" not in str(body)


def test_nvidia_provider_uses_openai_compatible_endpoint(monkeypatch):
    from app.llm import LLMSettings, build_chat_model

    calls = {}

    def fake_init_chat_model(model, **kwargs):
        calls.update(model=model, **kwargs)
        return object()

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-value")
    settings = LLMSettings.load("config/llm.json")

    build_chat_model(settings, "nvidia")

    assert calls["model"] == "meta/llama-3.3-70b-instruct"
    assert calls["model_provider"] == "openai"
    assert calls["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert calls["api_key"] == "nvidia-test-value"


def test_chat_config_error_returns_400(client):
    class Cache:
        def get(self, provider, model):
            raise LLMConfigError("Set ANTHROPIC_API_KEY in your environment or .env file to use Anthropic.")

    client.app.dependency_overrides[deps.get_agent_cache] = lambda: Cache()
    r = client.post("/api/chat", json={"message": "hi", "thread_id": "t", "provider": "anthropic"})
    assert r.status_code == 400 and "ANTHROPIC_API_KEY" in r.json()["detail"]
