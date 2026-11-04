import base64
import io
import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from qev.media import MediaError, check_items, content_parts, data_uri, load, open_image


@pytest.fixture
def png():
    buf = io.BytesIO()
    Image.new("RGBA", (3, 2), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("source", ["data", "uri", "pil", "path", "dict_path"])
def test_open_image_sources(png, tmp_path, source):
    path = tmp_path / "image.png"
    path.write_bytes(png)
    ref = {
        "data": {"data": base64.b64encode(png).decode()},
        "uri": "data:image/png;base64," + base64.b64encode(png).decode(),
        "pil": Image.open(io.BytesIO(png)),
        "path": path,
        "dict_path": {"path": "image.png"},
    }[source]
    image = open_image(ref, root=tmp_path)
    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("ref", [
    {"data": None}, {"data": "!!!!"}, {"data": "a"}, {"data": ""},
    "data:image/png;base64", "data:image/png,abc", {"data": "bm90IGFuIGltYWdl"},
    {}, 42,
])
def test_invalid_image_is_media_error(ref):
    with pytest.raises(MediaError):
        open_image(ref, allow_paths=False)


def test_base64_whitespace(png):
    encoded = base64.encodebytes(png).decode()
    assert open_image({"data": encoded}).size == (3, 2)


def test_data_uri_round_trip(tmp_path, png):
    path = tmp_path / "image.png"
    path.write_bytes(png)
    uri = data_uri(path, fmt="PNG")
    assert uri.startswith("data:image/png;base64,")
    assert open_image(uri).size == (3, 2)


def test_local_paths_are_opt_in(tmp_path, png):
    path = tmp_path / "image.png"
    path.write_bytes(png)
    with pytest.raises(MediaError, match="local paths"):
        open_image(str(path), allow_paths=False)


def test_media_root_rejects_parent_and_symlink_escape(tmp_path, png):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(png)
    (root / "link.png").symlink_to(outside)
    for ref in ("../outside.png", "link.png", str(outside)):
        with pytest.raises(MediaError, match="outside the media root"):
            open_image(ref, root=root)


@pytest.mark.parametrize("fps", [0, -1, "bad", float("inf"), float("nan"), True])
def test_invalid_video_fps(fps):
    with pytest.raises(MediaError, match="fps"):
        check_items([{"type": "video", "frames": ["frame.png"], "fps": fps}])


@pytest.mark.parametrize("media", ["wrong", [{}], [{"type": "image"}], [{"type": "video", "frames": []}]])
def test_invalid_media_structure(media):
    with pytest.raises(MediaError):
        check_items(media)


@pytest.mark.parametrize("text", ["<image:0>", "<video:1>", "<image:1><image:1>", "missing"])
def test_invalid_placeholders(text):
    with pytest.raises(MediaError):
        content_parts(text, [{"type": "image"}])


def test_mixed_placeholders_preserve_text():
    media = [{"type": "image"}, {"type": "video"}]
    assert content_parts("before <image:1> between <video:2> after", media) == [
        {"type": "text", "text": "before "}, {"type": "image"},
        {"type": "text", "text": " between "}, {"type": "video"},
        {"type": "text", "text": " after"},
    ]


def test_images_do_not_require_transformers(png):
    with patch.dict(sys.modules, {"transformers.video_utils": None}):
        images, videos, metas = load([{"type": "image", "data": base64.b64encode(png).decode()}])
    assert len(images) == 1
    assert not videos and not metas


def test_remote_image_size_is_bounded(png):
    response = MagicMock()
    response.headers = {"Content-Length": str(len(png))}
    response.read.return_value = png
    with patch("urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = response
        with pytest.raises(MediaError, match="byte limit"):
            open_image("https://example.test/image.png", max_bytes=10)


def test_base64_image_size_is_bounded(png):
    with pytest.raises(MediaError, match="byte limit"):
        open_image({"data": base64.b64encode(png).decode()}, max_bytes=10)
