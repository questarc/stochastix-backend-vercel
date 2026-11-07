import io
import wave

import pytest

from app import deps
from app.voice import DISCLAIMER, AudioStore, SpeechEngine, VoiceGenAgent, build_script

SUMMARY = """### Jane Smith — current vs model
| Portfolio | Probability of success |
|---|---|
| Current | **76.8%** |
| Model | 77.8% (+1.0 pp) |

- Median ending value: $1,380,575
- Sells total $208,370; buys $1.2M
Results are projections, not guarantees, and not investment advice."""


class FakeEngine(SpeechEngine):
    name, voice = "fake", "test-voice"

    def __init__(self):
        self.texts = []

    def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * 22050 * 2)  # 2 seconds of silence
        return buf.getvalue()


def test_script_rewords_numbers_without_changing_them():
    script = build_script(SUMMARY, "Jane Smith", "Alex Morgan")
    assert script.startswith("Retirement plan update for Jane Smith.")
    for spoken in ("76.8 percent", "77.8 percent", "up 1.0 percentage points", "1,380,575 dollars",
                   "208,370 dollars", "1.2 million dollars"):
        assert spoken in script
    assert "Prepared by your advisor, Alex Morgan." in script


def test_script_strips_markdown_and_keeps_one_disclaimer():
    script = build_script(SUMMARY)
    for junk in ("|", "**", "###", "---", "$", "%"):
        assert junk not in script
    assert script.endswith(DISCLAIMER) and script.count(DISCLAIMER) == 1


def test_script_always_adds_disclaimer_and_rejects_empty():
    assert build_script("Success is 84.2%.").endswith(DISCLAIMER)
    assert "down 2.5 percent" in build_script("Drift −2.5% in bonds.")
    with pytest.raises(ValueError):
        build_script("  ** ")


def test_voice_agent_creates_named_clip(client):
    engine = FakeEngine()
    agent = VoiceGenAgent(engine, AudioStore(), deps.get_sources())
    clip = agent.create(SUMMARY, "A1001")
    assert clip.client_name == "Jane Smith" and clip.duration_seconds == pytest.approx(2.0)
    assert clip.file_name.startswith("A1001_summary_") and clip.file_name.endswith(".wav")
    assert engine.texts == [clip.script]
    with pytest.raises(KeyError):
        agent.create(SUMMARY, "ZZZ")


def test_audio_store_keeps_newest():
    agent = VoiceGenAgent(FakeEngine(), AudioStore(max_clips=2))
    ids = [agent.create(f"Note {i}.").audio_id for i in range(3)]
    assert agent.store.get(ids[0]) is None and agent.store.get(ids[2]) is not None


@pytest.fixture
def voice_client(client):
    engine = FakeEngine()
    store = AudioStore()
    client.app.dependency_overrides[deps.get_audio_store] = lambda: store
    client.app.dependency_overrides[deps.get_voice_agent] = lambda: VoiceGenAgent(engine, store, deps.get_sources())
    return client


def test_voice_endpoints_create_and_download(voice_client):
    r = voice_client.post("/api/voice", json={"text": SUMMARY, "account_id": "A1001"})
    assert r.status_code == 200
    info = r.json()
    assert info["url"] == f"/voice/{info['audio_id']}" and "76.8 percent" in info["script"]
    audio = voice_client.get(f"/api{info['url']}", params={"download": True})
    assert audio.headers["content-type"] == "audio/wav"
    assert f'attachment; filename="{info["file_name"]}"' == audio.headers["content-disposition"]
    assert audio.content[:4] == b"RIFF"
    assert voice_client.get("/api/voice/unknown").status_code == 404
    assert voice_client.post("/api/voice", json={"text": "**"}).status_code == 400
    assert voice_client.post("/api/voice", json={"text": "Hi.", "account_id": "ZZZ"}).status_code == 404


def test_missing_voice_model_returns_503(client, tmp_path):
    from app.voice import PiperEngine

    agent = VoiceGenAgent(PiperEngine(voices_dir=tmp_path), AudioStore())
    client.app.dependency_overrides[deps.get_voice_agent] = lambda: agent
    r = client.post("/api/voice", json={"text": "Hello."})
    assert r.status_code == 503 and "download_voices" in r.json()["detail"]


def test_create_audio_summary_tool_via_browser_endpoint(voice_client):
    deps.get_agent_tools.cache_clear()
    engine, store = FakeEngine(), AudioStore()
    original = deps.get_voice_agent
    deps.get_voice_agent = lambda: VoiceGenAgent(engine, store, deps.get_sources())
    try:
        body = voice_client.post("/api/agent/tools/create_audio_summary",
                                 json={"args": {"summary_text": SUMMARY, "account_id": "A1001"}}).json()
    finally:
        deps.get_voice_agent = original
        deps.get_agent_tools.cache_clear()
    assert body["orders_added"] == 0 and len(body["audio"]) == 1
    assert body["audio"][0]["file_name"].startswith("A1001_summary_")
