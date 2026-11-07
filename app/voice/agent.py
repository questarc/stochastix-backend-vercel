"""VoiceGenAgent: turns an Advisor AI summary into an audio file an advisor can share with the investor."""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .engine import SpeechEngine, wav_duration_seconds
from .script import build_script

logger = logging.getLogger(__name__)

MAX_STORED_CLIPS = 50


@dataclass
class AudioClip:
    audio_id: str
    account_id: str | None
    client_name: str | None
    script: str
    voice: str
    duration_seconds: float
    created_at: str
    file_name: str
    data: bytes = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "audio_id": self.audio_id, "account_id": self.account_id, "client_name": self.client_name,
            "script": self.script, "voice": self.voice, "duration_seconds": round(self.duration_seconds, 1),
            "created_at": self.created_at, "file_name": self.file_name, "url": f"/voice/{self.audio_id}",
        }


class AudioStore:
    """In-memory clip store (newest kept), behind a small interface like the order store."""

    def __init__(self, max_clips: int = MAX_STORED_CLIPS):
        self._clips: OrderedDict[str, AudioClip] = OrderedDict()
        self._max = max_clips
        self._lock = threading.Lock()

    def put(self, clip: AudioClip) -> None:
        with self._lock:
            self._clips[clip.audio_id] = clip
            while len(self._clips) > self._max:
                self._clips.popitem(last=False)

    def get(self, audio_id: str) -> AudioClip | None:
        with self._lock:
            return self._clips.get(audio_id)


class VoiceGenAgent:
    def __init__(self, engine: SpeechEngine, store: AudioStore, sources: dict | None = None):
        self.engine = engine
        self.store = store
        self.sources = sources or {}

    def _account(self, account_id: str | None):
        if not account_id:
            return None
        for source in self.sources.values():
            if source.has_account(account_id):
                return source.get_account(account_id)
        raise KeyError(f"Unknown account '{account_id}'.")

    def create(self, summary: str, account_id: str | None = None, now: datetime | None = None) -> AudioClip:
        account = self._account(account_id)
        script = build_script(summary, account.client_name if account else None, account.advisor if account else None)
        data = self.engine.synthesize(script)
        now = now or datetime.now(timezone.utc)
        prefix = account.account_id if account else "advisor"
        clip = AudioClip(
            audio_id=uuid.uuid4().hex, account_id=account.account_id if account else None,
            client_name=account.client_name if account else None, script=script, voice=self.engine.voice,
            duration_seconds=wav_duration_seconds(data), created_at=now.isoformat(),
            file_name=f"{prefix}_summary_{now.strftime('%Y%m%d_%H%M%S')}.wav", data=data,
        )
        self.store.put(clip)
        logger.info("Audio summary account=%s voice=%s seconds=%.1f", clip.account_id, clip.voice, clip.duration_seconds)
        return clip
