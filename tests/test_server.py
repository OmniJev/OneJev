import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from qev.schema import NoulAnswer, SystemOneResponse, Usage
from qev.server import create_app


BODY = {"state": "hello", "questions": {"ok": {"type": "noul"}}}


class StubEngine:
    model_type = "stub"
    info = SimpleNamespace(fork_mode="sequential", prompt_version="test")

    def __init__(self):
        self.calls = []
        self.error = None
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.thread_ids = []

    def decide(self, request, debias=1, debug=False):
        self.calls.append((request, debias, debug))
        self.thread_ids.append(threading.get_ident())
        self.started.set()
        if self.block:
            self.release.wait(timeout=1)
        if self.error:
            raise self.error
        return SystemOneResponse(
            model="stub", answers={"ok": NoulAnswer(noul=0.75)},
            usage=Usage(input_tokens=10, output_tokens=1),
        ), {"debug": debug}


@pytest.fixture
def engine():
    return StubEngine()


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine, "test-model")) as client:
        yield client


def test_decision_response_and_request_id(client, engine):
    response = client.post("/v1/systemone", json=BODY, headers={"x-typesafe-request-id": "test-123"})
    assert response.status_code == 200
    assert response.json()["model"] == "test-model"
    assert response.json()["answers"]["ok"]["noul"] == 0.75
    assert response.headers["x-typesafe-request-id"] == "test-123"
    assert float(response.headers["x-qev-server-time-ms"]) >= 0
    assert len(engine.calls) == 1


@pytest.mark.parametrize("query", ["debias=no", "debias=0", "debias=-1", "debug=maybe"])
def test_invalid_query_is_validation_error(client, engine, query):
    response = client.post(f"/v1/systemone?{query}", json=BODY)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][0] == "query"
    assert not engine.calls


@pytest.mark.parametrize("debug,expected", [("0", False), ("false", False), ("1", True), ("true", True)])
def test_debug_boolean(client, engine, debug, expected):
    assert client.post(f"/v1/systemone?debug={debug}", json=BODY).status_code == 200
    assert engine.calls[-1][2] is expected


@pytest.mark.parametrize("body", [None, [], {}, {"questions": {}}, {"questions": {"x": {"type": "unknown"}}}])
def test_invalid_body(client, engine, body):
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    assert not engine.calls


def test_invalid_json(client):
    assert client.post("/v1/systemone", content="{").status_code == 422


def test_unknown_model(client, engine):
    assert client.post("/v1/systemone", json={**BODY, "model": "other"}).status_code == 404
    assert not engine.calls


def test_engine_value_error(client, engine):
    engine.error = ValueError("too many tokens")
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 422
    assert "too many tokens" in response.text


def test_inference_runs_off_event_loop_and_leaves_health_available(engine):
    async def run():
        engine.block = True
        app = create_app(engine, "test-model")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            prediction = asyncio.create_task(client.post("/v1/systemone", json=BODY))
            try:
                assert await asyncio.to_thread(engine.started.wait, 2)
                assert not prediction.done()
                assert engine.thread_ids[0] != threading.get_ident()
                assert (await client.get("/health")).status_code == 200
                queued = asyncio.create_task(client.post("/v1/systemone", json=BODY))
                assert (await client.get("/v1/models")).status_code == 200
                assert len(engine.calls) == 1
            finally:
                engine.release.set()
                await prediction
            assert (await queued).status_code == 200
            assert len(engine.calls) == 2

    asyncio.run(run())
