"""Tiny HTTP client for any System One endpoint (Qev or TypeSafe).

    from qev import Client, Choice, Score, Noul
    c = Client("http://localhost:8000")
    r = c.system_one(state, {"dept": Choice("Which team?", {"billing": "...", "tech": "..."})})
    r.answers["dept"].choice
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Noul:
    instructions: Any = None
    criteria: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "noul"}
        if self.instructions is not None:
            d["instructions"] = self.instructions
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d


@dataclass
class Choice:
    instructions: Any
    criteria: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


@dataclass
class Score:
    instructions: Any
    criteria: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": self.criteria}


class _Answer:
    def __init__(self, data: dict[str, Any]) -> None:
        self.raw = data
        for k, v in data.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"Answer({json.dumps(self.raw, ensure_ascii=False)})"


@dataclass
class Response:
    model: str
    answers: dict[str, _Answer]
    usage: dict[str, Any]
    request_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def nouls(self) -> dict[str, _Answer]:
        return {k: v for k, v in self.answers.items() if v.raw["type"] == "noul"}

    @property
    def choices(self) -> dict[str, _Answer]:
        return {k: v for k, v in self.answers.items() if v.raw["type"] == "choice"}

    @property
    def scores(self) -> dict[str, _Answer]:
        return {k: v for k, v in self.answers.items() if v.raw["type"] == "score"}


class Client:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str = "jev-latest", timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.environ.get("QEV_BASE_URL") or os.environ.get("TYPESAFE_BASE_URL") or "http://localhost:8000").rstrip("/")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or "local"
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
                                              "User-Agent": "qev-client"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read()), dict(resp.headers)

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "qev-client"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def system_one(self, state: Any, questions: dict[str, Any], model: str | None = None,
                   media: list[dict] | None = None, **params: Any) -> Response:
        """`media` (OneJev servers): images and videos the state refers to as <image:N> / <video:N>, see qev.media."""
        qs = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in questions.items()}
        path = "/v1/systemone"
        if params:
            path += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        body = {"state": state, "model": model or self.model, "questions": qs}
        if media is not None:
            body["media"] = media
        data, headers = self._post(path, body)
        return Response(model=data["model"], answers={k: _Answer(v) for k, v in data["answers"].items()},
                        usage=data.get("usage", {}), request_id=headers.get("x-typesafe-request-id"), meta=data.get("qev", {}))

    def models(self) -> list[dict[str, Any]]:
        return self._get("/v1/models")["models"]
