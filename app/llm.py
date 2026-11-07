"""LLM provider settings and chat model construction."""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 60
_models_cache: dict[str, tuple[float, list[str]]] = {}


class LLMConfigError(Exception):
    """Raised when an LLM provider is misconfigured (missing key, package or model)."""


@dataclass
class LLMSettings:
    default_provider: str = "ollama"
    default_model: str | None = None
    temperature: float = 0.1
    timeout_seconds: float = 120
    max_trials_per_tool_call: int = 25_000
    providers: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "LLMSettings":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        providers = dict(data.get("providers", {}))
        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url and "ollama" in providers:
            providers["ollama"] = {**providers["ollama"], "base_url": base_url}
        return cls(
            default_provider=os.getenv("MCAPP_LLM_PROVIDER") or data.get("default_provider", "ollama"),
            default_model=os.getenv("MCAPP_LLM_MODEL") or None,
            temperature=float(data.get("temperature", 0.1)),
            timeout_seconds=float(data.get("timeout_seconds", 120)),
            max_trials_per_tool_call=int(data.get("max_trials_per_tool_call", 25_000)),
            providers=providers,
        )

    def provider(self, provider_id: str) -> dict:
        if provider_id not in self.providers:
            raise LLMConfigError(f"Unknown LLM provider '{provider_id}'. Choose one of {sorted(self.providers)}.")
        return self.providers[provider_id]

    def model_for(self, provider_id: str, model: str | None = None) -> str:
        if model:
            return model
        if self.default_model and provider_id == self.default_provider:
            return self.default_model
        return self.provider(provider_id)["default_model"]


def api_key_set(cfg: dict) -> bool:
    env = cfg.get("api_key_env")
    return bool(env and os.getenv(env))


def list_ollama_models(base_url: str, timeout: float = 3.0) -> list[str]:
    resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def list_anthropic_models(api_key: str, timeout: float = 5.0) -> list[str]:
    resp = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def available_models(settings: LLMSettings, provider_id: str) -> list[str]:
    cached = _models_cache.get(provider_id)
    if cached and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
        return list(cached[1])
    cfg = settings.provider(provider_id)
    models = list(cfg.get("models", [cfg["default_model"]]))
    try:
        if provider_id == "ollama":
            live = list_ollama_models(cfg.get("base_url", "http://localhost:11434"))
            models = live or models
        elif provider_id == "anthropic" and api_key_set(cfg):
            live = list_anthropic_models(os.environ[cfg["api_key_env"]])
            models = list(dict.fromkeys(models + live))
    except Exception as exc:  # network errors fall back to the configured list
        logger.info("Could not list models for %s: %s", provider_id, type(exc).__name__)
    _models_cache[provider_id] = (time.monotonic(), models)
    return list(models)


def build_chat_model(settings: LLMSettings, provider_id: str, model: str | None = None):
    cfg = settings.provider(provider_id)
    model_name = settings.model_for(provider_id, model)
    package = cfg.get("package")
    if package and importlib.util.find_spec(package) is None:
        raise LLMConfigError(f"The {cfg.get('label', provider_id)} provider requires the '{package.replace('_', '-')}' package.")
    env = cfg.get("api_key_env")
    if env and not os.getenv(env):
        raise LLMConfigError(f"Set {env} in your environment or .env file to use {cfg.get('label', provider_id)}.")

    from langchain.chat_models import init_chat_model

    kwargs: dict = {"temperature": settings.temperature}
    if provider_id == "ollama":
        base_url = cfg.get("base_url", "http://localhost:11434")
        try:
            pulled = list_ollama_models(base_url)
        except Exception as exc:
            raise LLMConfigError(f"Cannot reach Ollama at {base_url}. Is it running?") from exc
        if model_name not in pulled and f"{model_name}:latest" not in pulled:
            raise LLMConfigError(f"Model '{model_name}' is not available. Run `ollama pull {model_name}`.")
        kwargs.update(base_url=base_url, client_kwargs={"timeout": settings.timeout_seconds})
    else:
        kwargs["timeout"] = settings.timeout_seconds
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        if env:
            kwargs["api_key"] = os.environ[env]
    return init_chat_model(model_name, model_provider=cfg.get("model_provider", provider_id), **kwargs)
