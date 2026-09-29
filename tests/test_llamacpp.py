"""llama.cpp backend pieces that need no server."""
import random

import pytest

from qev.llamacpp import VISION_BLOCK, smart_resize, tokenizer_repo


def test_smart_resize_matches_the_processor():
    qwen = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    rng = random.Random(0)
    sizes = [(720, 1280), (32, 32), (10, 1900), (5000, 7000)] + [(rng.randint(20, 6000), rng.randint(20, 6000)) for _ in range(2000)]
    for h, w in sizes:
        if max(h, w) / min(h, w) > 200:
            continue
        assert tuple(smart_resize(h, w, 32, 65536, 16777216)) == tuple(
            qwen.smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216))


def test_tokenizer_repo_from_names():
    assert tokenizer_repo("mradermacher/OneJev-4B-GGUF:Q4_K_M") == "OmniJev/OneJev-4B"
    assert tokenizer_repo("/models/OneJev-0.8B.Q8_0.gguf") == "OmniJev/OneJev-0.8B"
    assert tokenizer_repo("someone/onejev-27b-gguf") == "OmniJev/OneJev-27B"
    assert tokenizer_repo("Qwen3.5-4B.gguf") is None


def test_vision_blocks_become_markers():
    text = 'a "<|vision_start|><|image_pad|><|vision_end|>" b <|vision_start|><|image_pad|><|vision_end|>'
    assert VISION_BLOCK.sub("<m>", text) == 'a "<m>" b <m>'
