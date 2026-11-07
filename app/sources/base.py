"""Source agent interface and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ..domain import Account, ModelPortfolio, Position

BASE_DIR = Path(__file__).resolve().parents[2]

_REGISTRY: dict[str, type["SourceAgent"]] = {}


def register(source_type: str) -> Callable[[type["SourceAgent"]], type["SourceAgent"]]:
    def decorator(cls: type["SourceAgent"]) -> type["SourceAgent"]:
        _REGISTRY[source_type] = cls
        cls.source_type = source_type
        return cls

    return decorator


def create(config: dict, base_dir: Path | None = None) -> "SourceAgent":
    source_type = config.get("type")
    if source_type not in _REGISTRY:
        raise ValueError(f"Unknown source type '{source_type}'. Known: {sorted(_REGISTRY)}.")
    return _REGISTRY[source_type](config, base_dir or BASE_DIR)


def registered_types() -> list[str]:
    return sorted(_REGISTRY)


class SourceAgent(ABC):
    source_type: str = ""
    supports_upload: bool = False

    def __init__(self, config: dict, base_dir: Path):
        self.config = config
        self.name: str = config["name"]
        self.base_dir = Path(base_dir)

    def resolve_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p

    @abstractmethod
    def list_accounts(self) -> list[Account]: ...

    @abstractmethod
    def get_positions(self, account_id: str) -> list[Position]: ...

    @abstractmethod
    def get_model_portfolio(self, model_id: str) -> ModelPortfolio: ...

    @abstractmethod
    def list_model_portfolios(self) -> list[ModelPortfolio]: ...

    def get_account(self, account_id: str) -> Account:
        for account in self.list_accounts():
            if account.account_id == account_id:
                return account
        raise KeyError(f"Unknown account '{account_id}'.")

    def has_account(self, account_id: str) -> bool:
        return any(a.account_id == account_id for a in self.list_accounts())

    def describe(self) -> dict:
        return {"name": self.name, "type": self.source_type, "supports_upload": self.supports_upload}
