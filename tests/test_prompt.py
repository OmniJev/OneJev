"""Rendering rules and schema tolerance."""
import pytest
from pydantic import ValidationError

from qev.prompt import LETTERS, MAX_SLOTS, SLOT_LABELS, build_messages, render_question, state_block
from qev.schema import SystemOneRequest

REQ = {
    "state": {"document": "I was charged twice. Please fix this ASAP."},
    "model": "jev-latest",
    "questions": {
        "billing": {"type": "noul", "instructions": "Is this about billing?"},
        "bare": {"type": "noul"},
        "structured": {"type": "noul", "instructions": {"description": "urgent?", "examples": ["now"]},
                       "criteria": {"true": "yes desc", "false": None}},
        "tone": {"type": "choice", "criteria": {"calm": None, "angry": "mad"}, "instructions": "What is the tone?"},
        "sev": {"type": "score", "criteria": ["Calm", "Frustrated", "Very angry"], "instructions": "How severe?"},
        "dictq": {"type": "noul", "instructions": "dict form", "extra_field": 123},
    },
}


def test_sdk_shaped_request_validates():
    req = SystemOneRequest.model_validate(REQ)
    assert set(req.questions) == set(REQ["questions"])


def test_question_ids_never_reach_the_model():
    body = dict(REQ, questions={f"qid_{i}_zx9v": q for i, q in enumerate(REQ["questions"].values())})
    req = SystemOneRequest.model_validate(body)
    for qid, q in req.questions.items():
        r = render_question(qid, q)
        text = "\n".join(m["content"] for m in build_messages(req.state, r.suffix))
        assert qid not in text


def test_letters_and_labels():
    req = SystemOneRequest.model_validate(REQ)
    r = render_question("sev", req.questions["sev"])
    assert r.kind == "score" and r.labels == ["0", "1", "2"]
    assert "A. level 0: Calm" in r.suffix and "C. level 2: Very angry" in r.suffix
    r = render_question("tone", req.questions["tone"])
    assert "A. calm\n" in r.suffix and "B. angry: mad" in r.suffix
    r = render_question("structured", req.questions["structured"])
    assert '"examples"' in r.suffix and "A. yes: yes desc" in r.suffix


def test_permutation_keeps_labels_aligned():
    req = SystemOneRequest.model_validate(REQ)
    r = render_question("sev", req.questions["sev"], order=[2, 0, 1])
    assert r.labels == ["2", "0", "1"] and r.suffix.index("A. level 2") < r.suffix.index("B. level 0")


def test_prefix_is_shared():
    req = SystemOneRequest.model_validate(REQ)
    blocks = {build_messages(req.state, render_question(k, q).suffix)[1]["content"].split("</state>")[0] for k, q in req.questions.items()}
    assert len(blocks) == 1 and state_block(req.state).startswith("<state>")


def test_limits():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": {}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "rank", "criteria": {}}}})
    too_many = {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(MAX_SLOTS)}}}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": too_many})
    from qev.schema import ChoiceQuestion
    with pytest.raises(ValueError):
        render_question("q", ChoiceQuestion.model_construct(type="choice", instructions=None,
                                                            criteria={f"o{i}": None for i in range(MAX_SLOTS + 1)}))


def test_labels_beyond_z():
    assert len(SLOT_LABELS) == MAX_SLOTS == 256 and len(set(SLOT_LABELS)) == 256
    assert SLOT_LABELS[:26] == tuple(LETTERS) and SLOT_LABELS[26] == "AA" and SLOT_LABELS[-1] == "JU"
    q26 = {"q": {"type": "choice", "criteria": {f"o{i}": f"option {i}" for i in range(26)}}}
    r26 = render_question("q", SystemOneRequest.model_validate({"state": "x", "questions": q26}).questions["q"])
    assert r26.suffix.endswith("Answer with one letter: " + ", ".join(LETTERS) + ".")
    q200 = {"q": {"type": "choice", "criteria": {f"o{i}": f"option {i}" for i in range(200)}}}
    r200 = render_question("q", SystemOneRequest.model_validate({"state": "x", "questions": q200}).questions["q"])
    lines = [l for l in r200.suffix.splitlines() if ". o" in l]
    assert len(lines) == 200 and lines[26].startswith("AA. o26:") and lines[-1].startswith(SLOT_LABELS[199] + ". o199:")
    assert r200.suffix.endswith("Answer with one label: " + ", ".join(SLOT_LABELS[:200]) + ".")
