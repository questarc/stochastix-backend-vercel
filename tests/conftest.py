import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app import deps
    from app.main import app

    deps.get_order_store.cache_clear()
    deps.get_upload_store.cache_clear()
    deps.get_etrade_statement_store.cache_clear()
    deps.get_fidelity_statement_store.cache_clear()
    deps.get_fidelity_positions_store.cache_clear()
    deps.get_sources.cache_clear()
    deps.get_agent_tools.cache_clear()
    deps.get_voice_agent.cache_clear()
    deps.get_audio_store.cache_clear()
    deps.get_market_agent.cache_clear()
    deps.get_earnings_agent.cache_clear()
    deps.get_corporate_actions_agent.cache_clear()
    deps.get_heatmap_agent.cache_clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
