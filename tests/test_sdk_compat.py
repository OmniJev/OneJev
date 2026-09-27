"""Drop-in compatibility: the official `typesafe-sdk` must work against a OneJev server unchanged.

Run with a server already up:  QEV_URL=http://localhost:8000 pytest tests/test_sdk_compat.py
"""
import os

import pytest

URL = os.environ.get("QEV_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set QEV_URL to a running OneJev server")


@pytest.fixture(scope="module")
def client():
    typesafe_sdk = pytest.importorskip("typesafe_sdk")
    os.environ.setdefault("TYPESAFE_API_KEY", "local")
    return typesafe_sdk.TypeSafeClient(base_url=URL, timeout=120.0)


def test_models_list(client):
    names = [m.name for m in client.models.list().models]
    assert "jev-latest" in names


def test_system_one_all_shapes(client):
    from typesafe_sdk import Choice, Noul, Score

    r = client.system_one(
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "bare": Noul(),
            "structured": Noul(instructions={"description": "urgent?", "examples": ["now"]}, criteria={"true": "yes desc", "false": None}),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": "mad"}),
            "sev": Score(instructions="How severe?", criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )
    assert r.model and r.request_id
    assert set(r.answers) == {"billing", "bare", "structured", "tone", "sev"}
    assert 0.0 <= r.answers["billing"].noul <= 1.0
    tone = r.answers["tone"]
    assert tone.choice in {"calm", "angry"} and abs(sum(tone.probabilities.values()) - 1) < 1e-3 and 0 <= tone.confidence <= 1
    sev = r.answers["sev"]
    assert set(sev.legend) == {0, 1, 2} and set(sev.probabilities) == {0, 1, 2}
    assert 0 <= sev.score <= 2 and 0 <= sev.confidence <= 1
    assert r.usage.input_tokens and r.usage.input_tokens > 0
    assert set(r.nouls) == {"billing", "bare", "structured"} and set(r.choices) == {"tone"} and set(r.scores) == {"sev"}


def test_validation_error_is_422(client):
    from typesafe_sdk import TypeSafeError

    with pytest.raises(TypeSafeError) as info:
        client.system_one(state="x", questions={"q": {"type": "score", "criteria": [str(i) for i in range(11)]}})
    assert "422" in str(info.value)
