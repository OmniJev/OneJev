"""Multimodal rows for training and in-training evaluation.

A row may carry `media`: images and pre-extracted video frames, referenced in the state by placeholders
`<image:N>` / `<video:N>` (1-based, in order of appearance). The prompt is the frozen text prompt of qev.prompt with
each placeholder replaced by the model's vision block, so a row without media renders exactly like a text row.

Token lengths are exact without decoding pixels: text tokens of the rendered prompt (one pad token per media item)
plus, per image, the grid the processor's own smart_resize gives for its (h, w), and per video a probe of the
processor on blank frames of the same count, size and fps (cached). `check_lengths` compares this with the processor
on real rows.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("train.mm")

PLACEHOLDER = re.compile(r"<(image|video):(\d+)>")
MM_ROOT = Path(os.environ.get("QEV_MM_ROOT", "data/onejev"))


def set_root(root: str | Path) -> None:
    global MM_ROOT
    MM_ROOT = Path(root)


def content_parts(text: str, media: Sequence[dict]) -> list[dict]:
    """Split a user-turn text at the media placeholders into chat content parts."""
    parts, pos, seen = [], 0, []
    for m in PLACEHOLDER.finditer(text):
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        kind, n = m.group(1), int(m.group(2))
        if n < 1 or n > len(media) or media[n - 1]["type"] != kind:
            raise ValueError(f"placeholder {m.group(0)} does not match media {n} of {len(media)}")
        parts.append({"type": kind})
        seen.append(n)
        pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    if seen != list(range(1, len(media) + 1)):
        raise ValueError(f"media must be referenced once each, in order; saw {seen} for {len(media)} items")
    return parts


def render_prompt_mm(processor, template_kwargs, messages_fn, state: Any, suffix: str, media: Sequence[dict]) -> str:
    """Chat-rendered prompt with one vision block per media item (pads expanded later by the processor)."""
    msgs = messages_fn(state, suffix)
    user = msgs[-1]["content"]
    msgs = msgs[:-1] + [{"role": "user", "content": content_parts(user, media) if media else user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)


def load_media(media: Sequence[dict]):
    """PIL images, frame lists and video metadata, in media order."""
    from PIL import Image
    from transformers.video_utils import VideoMetadata

    images, videos, metas = [], [], []
    for m in media:
        if m["type"] == "image":
            with Image.open(MM_ROOT / m["path"]) as im:
                images.append(im.convert("RGB"))
        else:
            frames = []
            for p in m["frames"]:
                with Image.open(MM_ROOT / p) as im:
                    frames.append(im.convert("RGB"))
            n = len(frames)
            fps = float(m.get("fps") or 1.0)
            videos.append(frames)
            metas.append(VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)), duration=n / fps))
    return images, videos, metas


class TokenCounter:
    """Exact prompt length for a rendered prompt plus its media, without decoding any pixels."""

    def __init__(self, processor):
        self.processor = processor
        self.tok = processor.tokenizer
        ip = processor.image_processor
        self.merge = ip.merge_size
        self.factor = ip.patch_size * ip.merge_size
        self.min_pixels = ip.size["shortest_edge"]
        self.max_pixels = ip.size["longest_edge"]

    def image_tokens(self, h: int, w: int) -> int:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        hh, ww = smart_resize(h, w, factor=self.factor, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        return (hh // self.factor) * (ww // self.factor)

    @functools.lru_cache(maxsize=4096)
    def video_extra(self, n: int, h: int, w: int, fps: float) -> int:
        """Tokens a video block adds beyond its single pad token, probed on blank frames (cached per shape)."""
        from PIL import Image
        from transformers.video_utils import VideoMetadata

        frames = [Image.new("RGB", (w, h)) for _ in range(n)]
        base = "a<|vision_start|><|video_pad|><|vision_end|>b"
        md = VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)), duration=n / fps)
        out = self.processor(text=[base], videos=[frames], video_metadata=[md],
                             videos_kwargs={"do_sample_frames": False, "cap_pixels_per_frame": True})
        return len(out["input_ids"][0]) - len(self.tok(base, add_special_tokens=False)["input_ids"])

    def media_extra(self, media: Sequence[dict]) -> int:
        """Tokens the media add beyond the single pad token each one has in the rendered prompt."""
        n = 0
        for m in media:
            if m["type"] == "image":
                n += self.image_tokens(int(m["h"]), int(m["w"])) - 1
            else:
                n += self.video_extra(len(m["frames"]), int(m["h"]), int(m["w"]), float(m.get("fps") or 1.0))
        return n

    def length(self, prompt: str, media: Sequence[dict]) -> int:
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"]) + self.media_extra(media)


def mm_inputs(processor, prompts: Sequence[str], medias: Sequence[Sequence[dict]], device):
    """Processor tensors for a right-padded batch, and the readout index per row."""
    images, videos, metas = [], [], []
    for media in medias:
        im, vi, md = load_media(media)
        images += im
        videos += vi
        metas += md
    kw = {}
    if images:
        kw["images"] = images
    if videos:
        kw["videos"] = videos
        kw["video_metadata"] = metas
        kw["videos_kwargs"] = {"do_sample_frames": False, "cap_pixels_per_frame": True}
    processor.tokenizer.padding_side = "right"
    out = processor(text=list(prompts), padding=True, return_tensors="pt", **kw)
    out = {k: v.to(device) for k, v in out.items()}
    ends = (out["attention_mask"].sum(dim=1) - 1).tolist()
    return out, ends


def mm_readout_logits(model, inputs: dict, ends: Sequence[int]):
    """Full-vocabulary logits at each row's last real token (right padding), like common.readout_logits."""
    import torch

    keep = sorted(set(int(e) for e in ends))
    ltk = torch.tensor(keep, dtype=torch.long, device=inputs["input_ids"].device)
    out = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=ltk)
    rows = torch.arange(len(ends), device=ltk.device)
    pos = torch.tensor([keep.index(int(e)) for e in ends], dtype=torch.long, device=ltk.device)
    return out.logits[rows, pos]


def check_lengths(bundle, rows: Sequence[dict], render, limit: int = 50) -> list[tuple[str, int, int]]:
    """(id, counted, processor) for rows whose counted length differs from the processor's. Loads real media."""
    bad = []
    for i, row in enumerate(rows[:limit]):
        prompt = render(row, i)
        media = row.get("media") or []
        counted = bundle.counter.length(prompt, media)
        out, _ = mm_inputs(bundle.processor, [prompt], [media], "cpu")
        real = int(out["input_ids"].shape[1])
        if counted != real:
            bad.append((row.get("id"), counted, real))
    return bad
