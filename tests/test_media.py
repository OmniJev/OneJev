"""Media items and placeholders, without a model."""
import base64
import io

import pytest
from PIL import Image

from qev.media import MediaError, check_items, content_parts, data_uri, open_image


def _png() -> Image.Image:
    return Image.new("RGB", (8, 6), (200, 40, 40))


def test_data_uri_round_trip():
    uri = data_uri(_png(), fmt="PNG")
    assert uri.startswith("data:image/png;base64,")
    im = open_image(uri)
    assert im.size == (8, 6) and im.getpixel((0, 0)) == (200, 40, 40)


def test_open_image_from_item_and_raw_base64():
    raw = io.BytesIO()
    _png().save(raw, format="PNG")
    b64 = base64.b64encode(raw.getvalue()).decode()
    assert open_image({"data": b64}).size == (8, 6)


def test_check_items_accepts_images_and_videos():
    items = [{"type": "image", "url": "https://example.com/a.png"}, {"type": "video", "frames": [{"data": "x"}], "fps": 2}]
    assert check_items(items) == items
    assert check_items(None) == []


@pytest.mark.parametrize("bad", ["image", [{"type": "audio"}], [{"type": "image"}], [{"type": "video", "frames": []}]])
def test_check_items_rejects_bad_media(bad):
    with pytest.raises(MediaError):
        check_items(bad)


def test_local_paths_can_be_refused(tmp_path):
    path = tmp_path / "a.png"
    _png().save(path)
    assert open_image(str(path)).size == (8, 6)
    with pytest.raises(MediaError):
        open_image(str(path), allow_paths=False)


def test_paths_stay_inside_the_media_root(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    _png().save(root / "ok.png")
    _png().save(tmp_path / "outside.png")
    assert open_image("ok.png", root=root).size == (8, 6)
    with pytest.raises(MediaError):
        open_image("../outside.png", root=root)


def test_content_parts_splits_at_placeholders():
    media = [{"type": "image", "url": "u"}, {"type": "video", "frames": ["f"]}]
    parts = content_parts("look <image:1> then <video:2> end", media)
    kinds = [p["type"] for p in parts]
    assert kinds.count("image") == 1 and kinds.count("video") == 1
    assert kinds[0] == "text" and kinds[-1] == "text"
