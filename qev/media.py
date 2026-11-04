"""Images and videos in a System One request (OneJev, the multimodal Qev).

A request may carry `media`, a list of items the state refers to by placeholders `<image:N>` / `<video:N>` (1-based,
each item once, in order of appearance). This is the convention of the OneJev training rows, and
the rendering below matches the training prompts byte for byte, so a
request is seen exactly as training rows were:

    {"type": "image", "path" | "url" | "data": ...}
    {"type": "video", "frames": [ref, ...], "fps": 2.0}

A reference is a local path (relative to the engine's media root), an http(s) URL, base64 data (a `data:` URI or the
bare string under "data"), or, from Python, a PIL image. Video frames are pre-extracted stills; `fps` sets the
timestamps the model sees (default 1.0, as in training).
"""
from __future__ import annotations

import base64
import binascii
import io
import math
import re
import urllib.request
from pathlib import Path
from typing import Any, Sequence

PLACEHOLDER = re.compile(r"<(image|video):(\d+)>")
VIDEOS_KWARGS = {"do_sample_frames": False, "cap_pixels_per_frame": True}
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class MediaError(ValueError):
    pass


def content_parts(text: str, media: Sequence[dict]) -> list[dict]:
    """Split a user-turn text at the media placeholders into chat content parts."""
    parts, pos, seen = [], 0, []
    for m in PLACEHOLDER.finditer(text):
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        kind, n = m.group(1), int(m.group(2))
        if n < 1 or n > len(media) or media[n - 1]["type"] != kind:
            raise MediaError(f"placeholder {m.group(0)} does not match media {n} of {len(media)}")
        parts.append({"type": kind})
        seen.append(n)
        pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    if seen != list(range(1, len(media) + 1)):
        raise MediaError(f"media must be referenced once each, in order, by <image:N> / <video:N>; saw {seen} for "
                         f"{len(media)} items")
    return parts


def render_prompt(processor, template_kwargs: dict, messages: list[dict], media: Sequence[dict]) -> str:
    """Chat-rendered prompt with one vision block per media item (pads are expanded later by the processor). Without
    media the text is exactly the text-only prompt."""
    user = messages[-1]["content"]
    msgs = messages[:-1] + [{"role": "user", "content": content_parts(user, media) if media else user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)


def check_items(media: Any) -> list[dict]:
    if media is None:
        return []
    if not isinstance(media, list):
        raise MediaError("media must be a list of {type: image | video, ...} items")
    out = []
    for i, m in enumerate(media):
        if not isinstance(m, dict) or m.get("type") not in ("image", "video"):
            raise MediaError(f"media[{i}] must be an object with type image or video")
        if m["type"] == "image" and not any(k in m for k in ("path", "url", "data", "image")):
            raise MediaError(f"media[{i}] (image) needs one of path, url or data")
        if m["type"] == "video":
            if not (isinstance(m.get("frames"), list) and m["frames"]):
                raise MediaError(f"media[{i}] (video) needs a non-empty frames list")
            _video_fps(m.get("fps"), f"media[{i}]")
        out.append(m)
    return out


def _video_fps(value: Any, context: str = "video") -> float:
    try:
        fps = 1.0 if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        raise MediaError(f"{context} fps must be a finite positive number") from None
    if isinstance(value, bool) or not math.isfinite(fps) or fps <= 0:
        raise MediaError(f"{context} fps must be a finite positive number")
    return fps


def _decode_data(s: Any, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    if not isinstance(s, str) or not s:
        raise MediaError("image data must be a non-empty base64 string")
    if s.startswith("data:"):
        header, separator, s = s.partition(",")
        if not separator or not header.lower().endswith(";base64"):
            raise MediaError("image data URI must contain base64 data")
    s = "".join(s.split())
    if len(s) > 4 * ((max_bytes + 2) // 3):
        raise MediaError(f"image exceeds the {max_bytes} byte limit")
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise MediaError("invalid base64 image data") from None
    if len(raw) > max_bytes:
        raise MediaError(f"image exceeds the {max_bytes} byte limit")
    return raw


def open_image(ref: Any, root: str | Path | None = None, allow_paths: bool = True, timeout: float = 20.0,
               max_bytes: int = MAX_IMAGE_BYTES):
    """One RGB PIL image from a path, URL, base64 string or PIL image. Encoded files default to a 32 MiB limit;
    invalid references, failed reads and decoding errors raise MediaError."""
    from PIL import Image

    if max_bytes <= 0:
        raise MediaError("image byte limit must be positive")
    try:
        return _open_image(ref, root, allow_paths, timeout, max_bytes)
    except MediaError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise MediaError(f"could not load image: {type(exc).__name__}: {exc}") from exc


def _open_image(ref: Any, root, allow_paths: bool, timeout: float, max_bytes: int):
    from PIL import Image

    if isinstance(ref, dict):
        if "data" in ref:
            with Image.open(io.BytesIO(_decode_data(ref["data"], max_bytes))) as im:
                return im.convert("RGB")
        for key in ("image", "url", "path"):
            if key in ref:
                return open_image(ref[key], root, allow_paths, timeout, max_bytes)
        raise MediaError("image reference needs one of path, url or data")
    if isinstance(ref, Image.Image):
        return ref.convert("RGB")
    if isinstance(ref, Path):
        ref = str(ref)
    if not isinstance(ref, str):
        raise MediaError(f"unsupported image reference of type {type(ref).__name__}")
    if ref.startswith("data:"):
        raw = _decode_data(ref, max_bytes)
    elif ref.startswith(("http://", "https://")):
        req = urllib.request.Request(ref, headers={"User-Agent": "qev"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            try:
                length = int(r.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length > max_bytes:
                raise MediaError(f"image exceeds the {max_bytes} byte limit")
            raw = r.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise MediaError(f"image exceeds the {max_bytes} byte limit")
    else:
        if not allow_paths:
            raise MediaError("local paths are not accepted by this server; send a URL or base64 data")
        p = Path(ref)
        if root is not None:
            base = Path(root).resolve()
            p = (p if p.is_absolute() else base / p).resolve()
            if not p.is_relative_to(base):
                raise MediaError(f"path {ref!r} is outside the media root")
        if p.stat().st_size > max_bytes:
            raise MediaError(f"image exceeds the {max_bytes} byte limit")
        with Image.open(p) as im:
            return im.convert("RGB")
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB")


def data_uri(image: Any, fmt: str = "JPEG", quality: int = 95) -> str:
    """A local image file or PIL image as a base64 `data:` URI, for sending to a server."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            image = im.convert("RGB")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def load(media: Sequence[dict], root: str | Path | None = None, allow_paths: bool = True,
         max_image_bytes: int = MAX_IMAGE_BYTES):
    """(images, videos, video metadata) in media order, the way training built them."""
    images, videos, metas = [], [], []
    for m in media:
        if m["type"] == "image":
            images.append(open_image(m, root, allow_paths, max_bytes=max_image_bytes))
        else:
            from transformers.video_utils import VideoMetadata

            fps = _video_fps(m.get("fps"))
            frames = [open_image(f, root, allow_paths, max_bytes=max_image_bytes) for f in m["frames"]]
            n = len(frames)
            videos.append(frames)
            metas.append(VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)), duration=n / fps))
    return images, videos, metas


def processor_kwargs(images, videos, metas) -> dict:
    kw: dict[str, Any] = {}
    if images:
        kw["images"] = images
    if videos:
        kw["videos"] = videos
        kw["video_metadata"] = metas
        kw["videos_kwargs"] = dict(VIDEOS_KWARGS)
    return kw
