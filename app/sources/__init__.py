"""Source agents loaded from ``config/sources.json``."""
from __future__ import annotations

import json
from pathlib import Path

from . import api_agents, positions_agent, sales_agent  # noqa: F401  (registers types)
from .base import BASE_DIR, SourceAgent, create, register, registered_types


def load_sources(config_path: str | Path | None = None, include_disabled: bool = False) -> dict[str, SourceAgent]:
    path = Path(config_path) if config_path else BASE_DIR / "config" / "sources.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sources: dict[str, SourceAgent] = {}
    for cfg in data.get("sources", []):
        if cfg.get("enabled", True) or include_disabled:
            sources[cfg["name"]] = create(cfg)
    return sources


__all__ = ["SourceAgent", "create", "load_sources", "register", "registered_types"]
