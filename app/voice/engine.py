"""Text-to-speech engines. Piper runs locally (free, offline); the text never leaves the server."""
from __future__ import annotations

import io
import os
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path

from ..sources.base import BASE_DIR

DEFAULT_VOICE = "en_US-lessac-medium"


class VoiceConfigError(Exception):
    """Raised when the speech engine is not installed or the voice model is missing."""


class SpeechEngine(ABC):
    name: str = ""
    voice: str = ""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return WAV bytes for ``text``."""


class PiperEngine(SpeechEngine):
    name = "piper"

    def __init__(self, voice: str | None = None, voices_dir: str | Path | None = None):
        self.voice = voice or os.getenv("MCAPP_VOICE", DEFAULT_VOICE)
        self.voices_dir = Path(voices_dir or os.getenv("MCAPP_VOICES_DIR", BASE_DIR / "voices"))
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise VoiceConfigError("Audio summaries require the 'piper-tts' package.") from exc
            path = self.voices_dir / f"{self.voice}.onnx"
            if not path.exists():
                raise VoiceConfigError(
                    f"Voice '{self.voice}' is not installed. Run "
                    f"`python -m piper.download_voices --download-dir voices {self.voice}` in the backend folder."
                )
            self._model = PiperVoice.load(str(path))
        return self._model

    def synthesize(self, text: str) -> bytes:
        with self._lock:  # one synthesis at a time per loaded model
            model = self._load()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                model.synthesize_wav(text, wav)
            return buf.getvalue()


def wav_duration_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate() or 1)
