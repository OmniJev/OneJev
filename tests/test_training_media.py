import gc
import weakref
from types import SimpleNamespace

import pytest

from train.mm import TokenCounter


class ShapeProcessor:
    image_processor = SimpleNamespace(merge_size=2, patch_size=16,
                                      size={"shortest_edge": 1024, "longest_edge": 4096})

    def __init__(self):
        self.calls = 0

    def tokenizer(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}

    def __call__(self, **kwargs):
        self.calls += 1
        return {"input_ids": [[1, 2, 3, 4, 5]]}


def test_video_shape_cache_does_not_retain_discarded_processor():
    pytest.importorskip("torch")
    processor = ShapeProcessor()
    counter = TokenCounter(processor)
    assert counter.video_extra(2, 32, 32, 1.0) == 2
    assert counter.video_extra(2, 32, 32, 1.0) == 2
    assert processor.calls == 1
    ref_counter, ref_processor = weakref.ref(counter), weakref.ref(processor)
    del counter, processor
    gc.collect()
    assert ref_counter() is None
    assert ref_processor() is None
