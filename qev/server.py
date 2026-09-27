"""FastAPI server exposing the TypeSafe-compatible System One API.

    POST /v1/systemone   evaluate a state against typed questions
    GET  /v1/models      list model names and aliases
    GET  /health
    GET  /               playground
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from . import __version__
from .engine import DecisionEngine
from .schema import ModelMetadata, ModelMetadataList, SystemOneRequest

log = logging.getLogger("qev.server")

ALIASES = {"jev-latest", "jev-preview", "qev-latest"}


def _validation_error(exc: ValidationError) -> JSONResponse:
    detail = [{"loc": ["body", *[str(x) for x in e["loc"]]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


def create_app(engine: DecisionEngine, model_name: str, description: str = "", release_date: str = "") -> FastAPI:
    app = FastAPI(title="Qev", version=__version__)
    app.state.engine = engine
    app.state.model_name = model_name
    app.state.debias = int(os.environ.get("QEV_DEBIAS", "1"))
    playground = Path(__file__).parent / "playground" / "index.html"

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-typesafe-request-id") or f"qev-{uuid.uuid4().hex[:16]}"
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-typesafe-request-id"] = rid
        response.headers["x-qev-server-time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    @app.get("/health")
    async def health() -> dict[str, Any]:
        info = engine.info
        return {"status": "ok", "model": model_name, "fork_mode": info.fork_mode, "prompt_version": info.prompt_version}

    @app.get("/v1/models")
    async def models() -> ModelMetadataList:
        items = [ModelMetadata(name=model_name, description=description or f"Qev decision model ({engine.model_type})",
                               release_date=release_date or "2026-09-18")]
        items += [ModelMetadata(name=a, description=f"alias for {model_name}", release_date=release_date or "2026-09-18")
                  for a in sorted(ALIASES)]
        return ModelMetadataList(models=items)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=422, content={"detail": [{"loc": ["body"], "msg": "invalid JSON", "type": "json_invalid"}]})
        try:
            req = SystemOneRequest.model_validate(body)
        except ValidationError as exc:
            return _validation_error(exc)
        requested = req.model or "jev-latest"
        if requested not in ALIASES and requested != model_name:
            return JSONResponse(status_code=404, content={"detail": f"model {requested!r} is not served; available: {model_name} plus aliases {sorted(ALIASES)}"})
        debug = bool(request.query_params.get("debug"))
        debias = int(request.query_params.get("debias", app.state.debias))
        try:
            response, meta = engine.decide(req, debias=debias, debug=debug)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": [{"loc": ["body"], "msg": str(exc), "type": "value_error"}]})
        response.model = model_name
        payload = response.model_dump()
        payload["qev"] = meta
        return JSONResponse(content=payload)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        if playground.exists():
            return playground.read_text()
        return "<h1>Qev</h1><p>POST /v1/systemone</p>"

    return app
