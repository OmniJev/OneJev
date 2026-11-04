import gc
import threading
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from qev.calibrate import Calibration
from qev.engine import NONE_OPTION, BranchResult, DecisionEngine
from qev.prompt import DEFAULT_STYLE, build_messages, render_question
from qev.schema import SystemOneRequest
from train.shared import split_prefix


class FakeEngine(DecisionEngine):
    """Exercise request planning and answer assembly without loading model weights."""

    def __init__(self, max_slots=26):
        self._lock = threading.RLock()
        self.style = DEFAULT_STYLE
        self.max_slots = max_slots
        self.max_branch_tokens = 100_000
        self.max_request_tokens = 100_000
        self.info = SimpleNamespace(fork_mode="sequential")
        self.calibration = Calibration()
        self.encodes = 0
        self.fail = False

    def _encode(self, text):
        self.encodes += 1
        return list(text.encode())

    def _render_prompt(self, state, suffix):
        return "\n".join(m["content"] for m in build_messages(state, suffix)) + "\nassistant:"

    def _run_sequential(self, state, rendered):
        _, suffixes = self._prefix_and_suffix_ids(state, rendered)
        if self.fail:
            raise ValueError("forward failed")
        return [BranchResult(r.qid, r.kind, r.labels, r.descriptions, [0.0] * r.n_slots, 1.0, len(s))
                for r, s in zip(rendered, suffixes)]


def request(questions=None, state="hello"):
    return SystemOneRequest.model_validate({
        "state": state,
        "questions": questions or {"ok": {"type": "noul"}, "rating": {"type": "score", "criteria": ["bad", "good"]}},
    })


def test_tokenizes_each_branch_once_and_does_not_cache_between_requests():
    engine = FakeEngine()
    req = request()
    first, first_meta = engine.decide(req)
    assert engine.encodes == 3  # Two complete branches and one prefix boundary.
    second, second_meta = engine.decide(req)
    assert engine.encodes == 6
    assert first == second
    assert first_meta["prompt_sha256"] == second_meta["prompt_sha256"]


def test_debug_state_is_cleared_after_failure():
    engine = FakeEngine()
    engine.fail = True
    with pytest.raises(ValueError, match="forward failed"):
        engine.decide(request(), debug=True)
    assert not engine._want_mass
    engine.fail = False
    assert engine.decide(request())[0].answers["ok"].noul == 0.5


def test_prefix_marker_in_state_does_not_shorten_shared_prefix():
    engine = FakeEngine()
    req = request(state="user text QEV_PREFIX_MARK and more state text")
    rendered = [render_question(k, q) for k, q in req.questions.items()]
    prefix, suffixes = engine._prefix_and_suffix_ids(req.state, rendered)
    for r, suffix in zip(rendered, suffixes):
        assert bytes(suffix).decode().startswith("Question:" if r.kind == "noul" else "Rate the state:")
        assert prefix + suffix == engine._encode(engine._render_prompt(req.state, r.suffix))


def test_training_split_uses_the_same_boundary_with_literal_marker_in_state():
    engine = FakeEngine()
    bundle = SimpleNamespace(encode=engine._encode, render_prompt=engine._render_prompt)
    state = "user text QEV_PREFIX_MARK and more state text"
    prefix, suffixes = split_prefix(bundle, state, ["Question: first", "Question: second"])
    assert bytes(suffixes[0]).decode() == "Question: first\nassistant:"
    assert bytes(prefix).decode().endswith("</state>\n\n")


def test_chunked_choice_preserves_option_named_like_internal_none_slot():
    engine = FakeEngine(max_slots=4)
    options = {NONE_OPTION: "a valid user option", **{f"option-{i}": str(i) for i in range(6)}}
    req = request({"pick": {"type": "choice", "criteria": options}})
    response, _ = engine.decide(req)
    probs = response.answers["pick"].probabilities
    assert list(probs) == list(options)
    assert all(p > 0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1, abs=1e-5)


def test_debias_restores_original_choice_and_score_order():
    engine = FakeEngine()
    response, meta = engine.decide(request(), debias=10)
    assert response.answers["ok"].noul == 0.5
    assert response.answers["rating"].legend == {"0": "bad", "1": "good"}
    assert meta["branches"] == 4


@pytest.mark.parametrize("limit", ["max_branch_tokens", "max_request_tokens"])
def test_token_limits(limit):
    engine = FakeEngine()
    setattr(engine, limit, 1)
    with pytest.raises(ValueError, match="limit is"):
        engine.decide(request())


def test_out_of_memory_fallback_releases_failed_forward_temporaries():
    engine = FakeEngine()
    engine.info.fork_mode = "batched"
    references = []

    class Temporary:
        pass

    class OutOfMemoryError(RuntimeError):
        pass

    def batched(state, rendered):
        temporary = Temporary()
        references.append(weakref.ref(temporary))
        raise OutOfMemoryError("simulated allocation failure")

    sequential = engine._run_sequential

    def fallback(state, rendered):
        gc.collect()
        assert references[0]() is None, "failed-forward tensors must be freed before the fallback"
        return sequential(state, rendered)

    engine._run_batched = batched
    engine._run_sequential = fallback
    assert engine.decide(request())[0].answers["ok"].noul == 0.5


def test_incomplete_backend_result_is_not_a_successful_response():
    engine = FakeEngine()
    engine._run_sequential = lambda state, rendered: []
    with pytest.raises(RuntimeError, match="branches"):
        engine.decide(request())


def test_expanded_media_tokens_count_toward_limits_and_usage():
    engine = FakeEngine()
    req = request()
    before, before_meta = engine.decide(req)
    engine._prefix_token_count = lambda prefix: len(prefix) + 1000
    after, after_meta = engine.decide(req)
    assert after.usage.input_tokens == before.usage.input_tokens + 1000
    assert after_meta["prefix_tokens"] == before_meta["prefix_tokens"] + 1000
    engine.max_branch_tokens = 1000
    with pytest.raises(ValueError, match="limit is"):
        engine.decide(req)


def test_inference_selects_the_engine_cuda_device_and_restores_it(monkeypatch):
    import sys

    engine = FakeEngine()
    engine.device = SimpleNamespace(type="cuda", index=1)
    events = []

    @contextmanager
    def select(device):
        events.append(("enter", device.index))
        try:
            yield
        finally:
            events.append(("exit", device.index))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device=select)))
    engine.decide(request())
    engine.fail = True
    with pytest.raises(ValueError, match="forward failed"):
        engine.decide(request())
    assert events == [("enter", 1), ("exit", 1), ("enter", 1), ("exit", 1)]
