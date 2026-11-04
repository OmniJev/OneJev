"""Optional CUDA checks against locally cached weights.

Set QEV_TEST_MODEL to a local OneJev directory to run these tests. The ordinary
test suite does not download weights or require PyTorch or a GPU.
"""
import os
from pathlib import Path

import pytest

from qev.calibrate import softmax
from qev.prompt import render_question
from qev.schema import SystemOneRequest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def engine():
    model = os.environ.get("QEV_TEST_MODEL")
    if not model:
        pytest.skip("set QEV_TEST_MODEL to a local OneJev checkpoint")
    assert Path(model).is_dir(), "QEV_TEST_MODEL must be a local directory"
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the inference integration tests")
    from qev.mm_engine import MMDecisionEngine

    engine = MMDecisionEngine(
        model, device=os.environ.get("QEV_TEST_DEVICE", "cuda:0"),
        fork_mode="sequential", cuda_graphs=False,
    )
    yield engine
    engine._set_media([])


def make_request(state):
    return SystemOneRequest.model_validate({
        "state": state,
        "questions": {
            "red": {"type": "noul", "instructions": "There is a red object"},
            "color": {"type": "choice", "instructions": "What color is the object?",
                      "criteria": {"red": "red", "blue": "blue", "other": "another color"}},
            "amount": {"type": "score", "instructions": "How many objects are present?",
                       "criteria": ["none", "one", "several"]},
        },
    })


def assert_branch_parity(engine, request, media):
    rendered = [render_question(k, q) for k, q in request.questions.items()]
    with engine._lock:
        engine._set_media(media)
        try:
            reference = engine._full_logits(request.state, rendered)
            sequential = engine._run_sequential(request.state, rendered)
            batched = engine._run_batched(request.state, rendered)
        finally:
            engine._set_media([])
    for expected, seq, bat in zip(reference, sequential, batched):
        assert softmax(seq.logits) == pytest.approx(softmax(expected), abs=0.05)
        assert softmax(bat.logits) == pytest.approx(softmax(expected), abs=0.05)


def test_text_branch_parity(engine):
    assert_branch_parity(engine, make_request("There is one red ball on the table."), [])


def test_image_branch_parity_with_literal_prefix_marker(engine):
    from PIL import Image

    request = make_request({"note": "QEV_PREFIX_MARK is literal user text", "screen": "<image:1>"})
    media = [{"type": "image", "image": Image.new("RGB", (128, 96), "red")}]
    assert_branch_parity(engine, request, media)
    response, meta = engine.decide(request, media=media, debias=2, debug=True)
    assert response.answers["red"].noul > 0.8
    assert response.answers["color"].choice == "red"
    assert meta["media"]["images"] == 1
    assert meta["branches"] == 6
    assert engine._prep is None
    assert not engine._media
    assert not engine._want_mass


def test_video_branch_parity(engine):
    from PIL import Image

    media = [{"type": "video", "frames": [Image.new("RGB", (96, 96), color) for color in ("red", "blue")],
              "fps": 2.0}]
    assert_branch_parity(engine, make_request({"recording": "<video:1>"}), media)


def test_cuda_graphs_capture_and_replay(engine):
    import torch
    from PIL import Image

    from qev.mm_engine import DecoderGraphs, ForkGraphs

    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(engine.device)
    if free < 10 * 2**30:
        pytest.skip(f"graph captures need 10 GiB free alongside the model; {free / 2**30:.1f} GiB available")
    decoder = engine.model.model.language_model
    graphs = DecoderGraphs(decoder, sizes=(256, 512, 1024))
    forks = ForkGraphs(decoder, max_len=1024, fork=engine._fork, pad_id=engine.pad_id, max_graphs=2)
    engine._graphs, engine._fork_graphs = graphs, forks
    try:
        for color in ("red", "blue", "red"):
            request = make_request({"screen": "<image:1>"})
            media = [{"type": "image", "image": Image.new("RGB", (96, 96), color)}]
            assert_branch_parity(engine, request, media)
            single = SystemOneRequest(state=request.state, questions={"color": request.questions["color"]})
            response, _ = engine.decide(single, media=media)
            assert response.answers["color"].choice == color
        assert graphs.graphs, "the decoder graph path must actually capture a graph"
        assert forks.branches, "the shared-prefix graph path must actually capture a graph"
    finally:
        engine._graphs = engine._fork_graphs = None
        graphs.clear()
        forks.clear()
