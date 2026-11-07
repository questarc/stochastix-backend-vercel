"""FastAPI application: CORS, routers and exception handlers."""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .llm import LLMConfigError
from .routers import ALL_ROUTERS
from .sources.base import BASE_DIR
from .voice import VoiceConfigError

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

logging.basicConfig(
    level=os.getenv("MCAPP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


def _message(exc: Exception) -> str:
    return str(exc.args[0]) if exc.args else str(exc)


def create_app() -> FastAPI:
    app = FastAPI(title="Stochastix AI", version="1.0.0")

    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": _message(exc)})

    @app.exception_handler(KeyError)
    async def key_error(_: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": _message(exc)})

    @app.exception_handler(LLMConfigError)
    async def llm_config_error(_: Request, exc: LLMConfigError):
        return JSONResponse(status_code=400, content={"detail": _message(exc)})

    @app.exception_handler(VoiceConfigError)
    async def voice_config_error(_: Request, exc: VoiceConfigError):
        return JSONResponse(status_code=503, content={"detail": _message(exc)})

    @app.exception_handler(NotImplementedError)
    async def not_implemented(_: Request, exc: NotImplementedError):
        return JSONResponse(status_code=501, content={"detail": _message(exc) or "Not implemented."})

    @app.get("/api/health", tags=["health"])
    def health():
        return {"status": "ok"}

    for router in ALL_ROUTERS:
        app.include_router(router, prefix="/api")

    return app


app = create_app()
