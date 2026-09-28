"""Prefix-shared training step: encode a state once, train its questions as branches, with exact
gradients through the shared prefix. This is the engine's batched inference path made
differentiable, so a state with 14 questions costs one forward over the state plus 14 short
suffixes instead of 14 full forwards.

The branches are packed into ONE sequence: [prefix | suffix_1 | suffix_2 | ...]. Two things make
it equivalent to running each branch alone: position ids restart at len(prefix) for every suffix,
and a 4D attention mask lets prefix tokens attend causally among themselves, and suffix tokens
attend to the whole prefix plus (causally) their own suffix and nothing else. gemma-4 mixes sliding
window and full attention layers, so the sliding mask is built the same way with the window
applied to the branch-local positions, and both are handed to the model as the per-layer-type mask
mapping its forward accepts. No cache is involved, so gradient checkpointing works unchanged and
the memory is that of one sequence of length P + sum(suffix lengths).

`tests/test_shared_prefix.py` checks that logits and adapter gradients match the per-row path.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

log = logging.getLogger("train.shared")


def split_prefix(bundle, state: Any, suffix_texts: Sequence[str]) -> tuple[list[int], list[list[int]]]:
    """Tokenize every branch fully, then take the longest common token prefix that ends at or before
    the end of the state block (same rule as `DecisionEngine._prefix_and_suffix_ids`)."""
    full = [bundle.encode(bundle.render_prompt(state, s)) for s in suffix_texts]
    marked = bundle.render_prompt(state, "QEV_PREFIX_MARK")
    prefix_ids = bundle.encode(marked[: marked.index("QEV_PREFIX_MARK")])
    n = len(prefix_ids)
    while n > 0 and not all(ids[:n] == prefix_ids[:n] for ids in full):
        n -= 1
    if n == 0:
        raise ValueError("branches share no token prefix")
    suffixes = [ids[n:] for ids in full]
    if any(len(s) == 0 for s in suffixes):
        raise ValueError("a branch has an empty suffix; state block boundary mismatch")
    return prefix_ids[:n], suffixes


def _text_config(config):
    return config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config


def branch_masks(P: int, suffix_lengths: Sequence[int], layer_types: Sequence[str], sliding_window: int | None, device):
    """Position ids and the per-layer-type boolean masks [1, 1, T, T] for the packed sequence."""
    import torch

    T = P + sum(suffix_lengths)
    pos = torch.empty(T, dtype=torch.long, device=device)
    seg = torch.empty(T, dtype=torch.long, device=device)
    pos[:P] = torch.arange(P, device=device)
    seg[:P] = -1
    t = P
    for k, L in enumerate(suffix_lengths):
        pos[t : t + L] = torch.arange(P, P + L, device=device)
        seg[t : t + L] = k
        t += L
    q_pos, k_pos = pos[:, None], pos[None, :]
    q_seg, k_seg = seg[:, None], seg[None, :]
    causal = k_pos <= q_pos
    same_branch = (k_seg == -1) | (k_seg == q_seg)
    full = causal & same_branch
    masks = {}
    for lt in set(layer_types):
        if lt == "sliding_attention":
            if sliding_window is None:
                raise ValueError("model has sliding-window layers but no sliding_window in its config")
            masks[lt] = (full & (k_pos > q_pos - sliding_window))[None, None]
        else:
            masks[lt] = full[None, None]
    return pos[None], masks


def shared_readout_logits(model, config, prefix_ids: Sequence[int], suffixes: Sequence[Sequence[int]],
                          pad_id: int, device, checkpointing: bool = True):
    """Full-vocabulary logits at the last token of every branch, with gradients through the shared
    prefix. Returns a [n, V] tensor in branch order. `pad_id` and `checkpointing` are accepted for
    interface stability; the packed sequence needs neither."""
    import torch

    tc = _text_config(config)
    layer_types = list(getattr(tc, "layer_types", None) or ["full_attention"] * tc.num_hidden_layers)
    P = len(prefix_ids)
    lengths = [len(s) for s in suffixes]
    ids = list(prefix_ids)
    ends = []
    for s in suffixes:
        ids.extend(s)
        ends.append(len(ids) - 1)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    position_ids, masks = branch_masks(P, lengths, layer_types, getattr(tc, "sliding_window", None), device)
    keep = torch.tensor(ends, dtype=torch.long, device=device)
    out = model(input_ids=input_ids, attention_mask=masks, position_ids=position_ids, use_cache=False,
                return_dict=True, logits_to_keep=keep)
    return out.logits[0]


def state_key(row: dict) -> str:
    import json

    return json.dumps(row.get("state"), sort_keys=True, ensure_ascii=False)


def group_rows(rows: Sequence[dict], min_rows: int = 2) -> tuple[list[list[int]], list[int]]:
    """Indices of rows grouped by identical state (groups with at least `min_rows` rows), plus the
    indices left over for ordinary batching."""
    by_state: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_state.setdefault(state_key(row), []).append(i)
    groups, singles = [], []
    for idx in by_state.values():
        if len(idx) >= min_rows:
            groups.append(idx)
        else:
            singles.extend(idx)
    return groups, singles


def render_group(rows: Sequence[dict], idx: Sequence[int], seed: int, augment: bool):
    """Render the rows of one state group: the shared state serialization (one augmentation choice
    for the whole group) and one suffix per row with its own option permutation. Returns
    (state, suffix_texts, labels, targets, kinds), aligned with `idx`."""
    import random

    from qev.prompt import render_question
    from train.common import build_question, serialize_state, target_vector

    rng = random.Random(seed)
    compact = augment and rng.random() < 0.5
    state = serialize_state(rows[idx[0]].get("state"), compact)
    suffixes, labels, targets, kinds = [], [], [], []
    for j, i in enumerate(idx):
        question = build_question(rows[i]["question"])
        base = render_question("q", question)
        order = list(range(base.n_slots))
        random.Random(seed + 7919 * (j + 1)).shuffle(order)
        r = render_question("q", question, order=order)
        suffixes.append(r.suffix)
        labels.append(list(r.labels))
        targets.append(target_vector(rows[i]["target"], r.labels))
        kinds.append(r.kind)
    return state, suffixes, labels, targets, kinds
