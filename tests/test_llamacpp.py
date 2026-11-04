import math
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from qev.llamacpp import LlamaCppEngine, LlamaServer, resolve_gguf, tokenizer_repo
from qev.prompt import render_question
from qev.schema import NoulQuestion


@pytest.mark.parametrize("name,expected", [
    ("repo/onejev1.6-0.8b-q8_0.gguf", "OmniJev/OneJev1.6-0.8B"),
    ("OneJev-27B-Q4_K_M.gguf", "OmniJev/OneJev-27B"),
    ("unknown.gguf", None),
])
def test_tokenizer_repository_detection(name, expected):
    assert tokenizer_repo(name) == expected


def test_local_gguf_cannot_be_a_directory(tmp_path):
    with pytest.raises(ValueError, match="file"):
        resolve_gguf(str(tmp_path))


def test_startup_failure_stops_the_child_process():
    process = MagicMock()
    process.poll.return_value = None
    with patch("qev.llamacpp.shutil.which", return_value="llama-server"), \
            patch("qev.llamacpp.resolve_gguf", return_value=("test.gguf", None)), \
            patch("qev.llamacpp.subprocess.Popen", return_value=process), \
            patch("qev.llamacpp.threading.Thread"), \
            patch.object(LlamaServer, "_wait", side_effect=TimeoutError("not ready")):
        with pytest.raises(TimeoutError, match="not ready"):
            LlamaServer("test.gguf")
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(10)


def test_stop_waits_after_killing_an_unresponsive_child():
    server = object.__new__(LlamaServer)
    server.proc = MagicMock()
    server.proc.poll.return_value = None
    server.proc.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 10), 0]
    server.stop()
    server.proc.kill.assert_called_once()
    assert server.proc.wait.call_count == 2


def test_media_prompt_keeps_literal_prefix_markers_in_state():
    engine = object.__new__(LlamaCppEngine)
    engine._media = [{"type": "image"}]
    engine._images = ["image-data"]
    engine.media_marker = "<__media__>"
    engine._render_prompt = lambda state, suffix: f"literal QEV_PREFIX_MARK {state} {suffix} assistant:"
    rendered = [render_question("q", NoulQuestion(type="noul"))]
    prefix, prompts = engine._prompts("screen", rendered)
    assert prefix["prompt_string"] == "literal QEV_PREFIX_MARK screen "
    assert prompts[0]["prompt_string"] == prefix["prompt_string"] + rendered[0].suffix + " assistant:"


@pytest.mark.parametrize("reply", [[], [{"completion_probabilities": []}], [{}, {}]])
def test_invalid_completion_responses_fail_clearly(reply):
    engine = object.__new__(LlamaCppEngine)
    engine._post = MagicMock(return_value=reply)
    with pytest.raises(RuntimeError, match="completion|probabilit"):
        engine._complete(["prompt"], {"n_probs": 2})


def test_completion_response_order_follows_prompt_indices():
    engine = object.__new__(LlamaCppEngine)
    engine._post = MagicMock(return_value=[
        {"index": 1, "completion_probabilities": [{"value": "second"}]},
        {"index": 0, "completion_probabilities": [{"value": "first"}]},
    ])
    assert engine._complete(["first", "second"], {}) == [{"value": "first"}, {"value": "second"}]


def test_single_question_needs_one_backend_completion():
    engine = object.__new__(LlamaCppEngine)
    engine.slot_ids = [5, 7]
    engine._prompts = lambda state, rendered: ([1], [[1, 2, 3]])
    calls = []

    def post(path, body):
        calls.append(body)
        if body["n_predict"] == 0:
            return {"id_slot": 0}
        return {"completion_probabilities": [{"top_logprobs": [
            {"id": 5, "logprob": math.log(0.7)}, {"id": 7, "logprob": math.log(0.3)},
        ]}]}

    engine._post = post
    rendered = [render_question("q", NoulQuestion(type="noul"))]
    result = engine._run_sequential(None, rendered)
    assert len(result) == 1
    assert result[0].logits == pytest.approx([math.log(0.7), math.log(0.3)])
    assert [body["n_predict"] for body in calls] == [1]


def test_prepared_image_token_count_matches_resized_patch_grid():
    from PIL import Image

    from qev.media import open_image

    engine = object.__new__(LlamaCppEngine)
    engine.factor, engine.min_pixels, engine.max_pixels = 32, 65536, 16777216
    encoded, tokens = engine._prepare_image(Image.new("RGB", (128, 96), "red"))
    image = open_image({"data": encoded})
    assert image.size == (320, 224)
    assert tokens == 70
    engine._vision_token_extra = tokens - 1
    assert engine._prefix_token_count(list(range(71))) == 140
