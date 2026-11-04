from unittest.mock import patch

import pytest

from qev import media
from qev.mm_engine import MMDecisionEngine


def test_failed_media_load_clears_request_data_and_cached_pixels():
    engine = object.__new__(MMDecisionEngine)
    engine.media_root = None
    engine.allow_local_paths = False
    engine.max_image_bytes = media.MAX_IMAGE_BYTES
    engine._media = [{"type": "image", "data": "previous"}]
    engine._loaded = (["previous pixels"], [], [])
    engine._prep = object()
    with patch("qev.mm_engine.mmedia.load", side_effect=media.MediaError("bad image")):
        with pytest.raises(media.MediaError, match="bad image"):
            engine._set_media([{"type": "image", "data": "invalid"}])
    assert engine._media == []
    assert engine._loaded == ([], [], [])
    assert engine._prep is None


def test_media_byte_limit_is_forwarded_by_the_engine():
    engine = object.__new__(MMDecisionEngine)
    engine.media_root = None
    engine.allow_local_paths = False
    engine.max_image_bytes = 1024
    items = [{"type": "image", "data": "test"}]
    with patch("qev.mm_engine.mmedia.load", return_value=(["pixels"], [], [])) as load:
        engine._set_media(items)
    assert load.call_args.kwargs["max_image_bytes"] == 1024
    assert engine._media == items
