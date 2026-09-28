"""Fine-tune a decoder LM at the answer-slot readout position only.

The only supervised position in a sequence is the last one, where the engine reads the answer
letter. The loss there is cross-entropy against the (possibly soft) target distribution over the
K answer slots plus a Brier term on the same restricted softmax. No generation target anywhere.

Full fine-tune, data parallel over the GPUs of one node:
    torchrun --nproc-per-node 4 -m train.sft --config train/configs/onejev_4b_full.yaml

OneJev-27B, sharded with FSDP over 8 GPUs:
    torchrun --nproc-per-node 8 -m train.sft --config train/configs/onejev_27b_full.yaml

Drop full_finetune from a config to train a LoRA adapter instead.

Every rank plans the same epoch (same seed), so the micro-batches are split without communication:
a step takes micro-batches until it holds --accum-rows rows over all ranks, and they are dealt to the
ranks by padded token count (plan_steps). Each micro-batch loss is weighted by its share of the
step's rows, so the summed gradient is the mean over every row in the step, whatever the world size.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import time
from pathlib import Path

from train import common

log = logging.getLogger("train.sft")
OPTION_POOL: dict = {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA SFT at the answer-slot readout position")
    p.add_argument("--config", default=None, help="YAML file with any of the flags below (CLI wins)")
    p.add_argument("--model", default=None, help="base model path or hub id")
    p.add_argument("--train", dest="train_file", default=None, help="training JSONL in the unified format")
    p.add_argument("--split", default=None, help="keep only rows with this split value")
    p.add_argument("--limit", type=int, default=None, help="use at most this many rows")
    p.add_argument("--name", default="run", help="run name; artifacts go to <runs-dir>/<name>")
    p.add_argument("--runs-dir", default="train/runs", help="parent directory for run artifacts")
    p.add_argument("--device", default="cuda:0", help="device for the single-GPU run")
    p.add_argument("--dtype", default="bfloat16", help="compute dtype")
    p.add_argument("--lr", type=float, default=1e-4, help="peak learning rate")
    p.add_argument("--epochs", type=int, default=1, help="passes over the training file")
    p.add_argument("--max-len", type=int, default=4096, help="drop examples longer than this many tokens")
    p.add_argument("--batch-tokens", type=int, default=16384, help="rows times padded width budget per step")
    p.add_argument("--max-rows", type=int, default=64, help="hard cap on rows per step")
    p.add_argument("--brier-weight", type=float, default=1.0, help="weight of the Brier term in the loss")
    p.add_argument("--warmup-frac", type=float, default=0.03, help="fraction of total steps spent warming up")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping norm")
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--target-modules", default=None, help="comma separated module names; default is every linear projection")
    p.add_argument("--augment", action="store_true", default=True, help="state serialization augmentation on (default)")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="turn the state serialization augmentation off")
    p.add_argument("--seed", type=int, default=0, help="base seed for permutations and batch order")
    p.add_argument("--log-every", type=int, default=20, help="log every this many optimizer steps")
    p.add_argument("--save-every", type=int, default=500, help="save the adapter every this many steps")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 means no limit)")
    p.add_argument("--merge", action="store_true", help="merge LoRA into the base and save a full model directory")
    p.add_argument("--merged-dir", default=None, help="where to write the merged model (default <run dir>/merged)")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
                   default=True, help="disable gradient checkpointing")
    p.add_argument("--share-prefix", action="store_true",
                   help="encode each state once and train its questions as branches off the cache (train/shared.py); "
                        "rows that share a state form one unit, other rows are batched as usual")
    p.add_argument("--step-rows", type=int, default=24,
                   help="with --share-prefix: accumulate units until a step holds at least this many rows")
    p.add_argument("--accum-rows", type=int, default=0,
                   help="accumulate micro-batches (over all ranks) until a step holds at least this many rows; "
                        "0 means one micro-batch per rank per step")
    p.add_argument("--dist-backend", default="nccl", help="torch.distributed backend under torchrun (gloo for tests)")
    p.add_argument("--eval", dest="eval_files", action="append", default=[],
                   help="NAME=PATH of an eval file (unified rows, or benchmark rows with options/label); repeatable")
    p.add_argument("--eval-every", type=int, default=0, help="evaluate every this many steps (0: only at the end)")
    p.add_argument("--eval-limit", type=int, default=0, help="use at most this many rows of each eval file")
    p.add_argument("--eval-at-start", action="store_true", help="also evaluate before the first step")
    p.add_argument("--eval-max-len", type=int, default=0, help="skip eval rows longer than this (0: --max-len)")
    p.add_argument("--pad-options", type=float, default=0.0,
                   help="probability that a choice row of an entity family (train.common.PAD_FAMILIES) gets extra "
                        "wrong options from other rows of its family, up to 27..255 options (trains every answer slot)")
    p.add_argument("--lora-autocast", action="store_true",
                   help="run forward passes under bf16 autocast and feed the LoRA layers bf16 inputs; the fp32 adapter "
                        "weights stay the master copy. Removes the per-module fp32 copies of every LoRA input")
    p.add_argument("--multimodal", action="store_true",
                   help="vision model + processor (train/mm.py); rows may carry media")
    p.add_argument("--media-root", default=None, help="root that media paths are relative to (default train.mm.MM_ROOT)")
    p.add_argument("--chat-template", default=None, help="chat template .jinja for base checkpoints that ship none")
    p.add_argument("--full-finetune", action="store_true",
                   help="train every language-model weight (fp32 master weights, bf16 autocast forward) instead of LoRA")
    p.add_argument("--train-vision", action="store_true", help="with --full-finetune: also train the vision tower")
    p.add_argument("--fsdp", action="store_true",
                   help="with --full-finetune under torchrun: shard weights, gradients and AdamW state over the ranks "
                        "(train/fsdp.py) instead of a full fp32 copy per GPU; needed for models the size of Qwen3.8-27B")
    p.add_argument("--no-autocast", action="store_true", help="tests only: run a full fine-tune without bf16 autocast")
    p.add_argument("--plan-only", action="store_true",
                   help="no model, no GPU: render and measure every row, plan the epoch, list the worst cases, exit")
    p.add_argument("--worst-case", default=None,
                   help="pre-launch check: JSON from train.preflight --worst-out, or 'auto' to pick them from the plan; "
                        "train one step on each listed micro-batch (one GPU), peak memory reset per step, no final eval")
    p.add_argument("--dump-first-grad", default=None,
                   help="test hook: after the first step's all-reduce, rank 0 saves the adapter gradients here")
    p.add_argument("--disable-fla-kernels", action="store_true",
                   help="force the reference PyTorch gated delta rule instead of the fla kernel "
                        "(needed on H100/H200 with Triton in [3.4.0, 3.7.1); correct but much slower)")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        common.apply_config(parser, pre.config)
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required (flag or config)")
    if not args.train_file:
        parser.error("--train is required (flag or config)")
    return args


def token_lengths(bundle, rows, args, epoch: int, rank: int = 0, world: int = 1) -> list[int]:
    """Token length of every row as rendered in this epoch. Under data parallel each rank measures
    rows rank, rank + world, ... and the lengths are all-gathered, so no rank tokenizes everything."""
    mine = list(range(rank, len(rows), world))
    lens: list[int] = []
    for chunk in common.chunked(mine, 256):
        texts = []
        for i in chunk:
            seed = args.seed * 1000003 + epoch * 1000033 + i
            texts.append(common.render_example(rows[i], bundle.template_for(rows[i]), i, seed=seed, augment=args.augment,
                                               option_pool=OPTION_POOL, pad_prob=args.pad_options).prompt)
        enc = bundle.tokenizer(texts, add_special_tokens=False)["input_ids"]
        if bundle.multimodal:
            lens.extend(len(x) + bundle.counter.media_extra(rows[i].get("media") or []) for x, i in zip(enc, chunk))
        else:
            lens.extend(len(x) for x in enc)
    if world == 1:
        return lens
    import torch.distributed as dist

    gathered = [None] * world
    dist.all_gather_object(gathered, lens)
    lengths = [0] * len(rows)
    for r, part in enumerate(gathered):
        for j, L in zip(range(r, len(rows), world), part):
            lengths[j] = L
    return lengths


def plan_epoch(bundle, rows, args, epoch: int, rank: int = 0, world: int = 1):
    """Render the epoch, measure token lengths, and cut length-sorted batches.

    Rendering is a deterministic function of (seed, epoch, row index), so the training loop can
    re-render a batch on the fly instead of holding every tokenized epoch in memory.
    """
    lengths = token_lengths(bundle, rows, args, epoch, rank, world)
    usable = [i for i in range(len(rows)) if lengths[i] <= args.max_len]
    dropped = len(rows) - len(usable)
    sub_lengths = [lengths[i] for i in usable]
    batches = common.length_bucket_batches(sub_lengths, args.batch_tokens, args.max_rows,
                                           shuffle_seed=args.seed + epoch)
    batches = [[usable[j] for j in b] for b in batches]
    return batches, dropped, lengths


def plan_steps(batches, lengths, world: int, accum_rows: int):
    """Group micro-batches into optimizer steps and split each step over the ranks.

    A step takes micro-batches in (shuffled) order until it holds at least `accum_rows` rows and at
    least `world` micro-batches. Its micro-batches are then dealt to the ranks longest first, each to
    the rank with the least padded tokens so far, so that no GPU waits long for another at the
    all-reduce. Which rank runs which micro-batch does not change the step's gradient (a sum).
    Returns a list of steps, each a list of `world` lists of micro-batches."""
    def cost(b):
        return len(b) * max(lengths[i] for i in b)

    def deal(mbs):
        per = [[] for _ in range(world)]
        load = [0] * world
        for b in sorted(mbs, key=cost, reverse=True):
            r = min(range(world), key=load.__getitem__)
            per[r].append(b)
            load[r] += cost(b)
        return per

    steps, cur, n = [], [], 0
    for b in batches:
        cur.append(b)
        n += len(b)
        if n >= max(accum_rows, 1) and len(cur) >= world:
            steps.append(deal(cur))
            cur, n = [], 0
    if cur:
        steps.append(deal(cur))
    return steps


def plan_epoch_shared(bundle, rows, args, epoch: int):
    """Like plan_epoch, but rows that share a state become prefix-shared units. Returns a list of
    steps; each step is a list of units, a unit being ("group", [row idx...]) or ("batch", [row idx...]).
    Groups whose prefix would exceed max_len, or whose rows do, are dropped like any other long row."""
    from train import shared

    groups, singles = shared.group_rows(rows)
    units = []
    dropped = 0
    for idx in groups:
        seed = args.seed * 1000003 + epoch * 1000033 + idx[0]
        state, sufs, _, _, _ = shared.render_group(rows, idx, seed=seed, augment=args.augment)
        try:
            prefix, suffix_ids = shared.split_prefix(bundle, state, sufs)
        except ValueError as exc:
            log.warning("group of %d rows falls back to per-row batching: %s", len(idx), exc)
            singles.extend(idx)
            continue
        keep = [i for i, sfx in zip(idx, suffix_ids) if len(prefix) + len(sfx) <= args.max_len]
        dropped += len(idx) - len(keep)
        if len(keep) >= 2:
            units.append(("group", keep))
        else:
            singles.extend(keep)
    if singles:
        sub = [rows[i] for i in singles]
        batches, d, _ = plan_epoch(bundle, sub, args, epoch)
        dropped += d
        units.extend(("batch", [singles[j] for j in b]) for b in batches)
    random.Random(args.seed + epoch).shuffle(units)
    steps, cur, n_rows = [], [], 0
    for unit in units:
        cur.append(unit)
        n_rows += len(unit[1])
        if n_rows >= args.step_rows:
            steps.append(cur)
            cur, n_rows = [], 0
    if cur:
        steps.append(cur)
    return steps, dropped, len(groups)


def auto_worst_case(rows, batches, lengths, bundle, args, per_cat: int = 5) -> list[dict]:
    """The most demanding micro-batches of the planned epoch, by category (what train.preflight --worst-out writes),
    computed in-process: most padded tokens, longest row, most rows, most padding, most options, most media tokens."""
    def n_opts(i):
        q = rows[i]["question"]
        if args.pad_options > 0 and OPTION_POOL:
            q = common.pad_options(rows[i], OPTION_POOL, args.pad_options, args.seed * 1000003 + i)
        return len(q["criteria"]) if q["type"] != "noul" else 2

    def media_tok(i):
        return bundle.counter.media_extra(rows[i].get("media") or []) if bundle.multimodal else 0

    def cost(b):
        return len(b) * max(lengths[i] for i in b)

    cats = {
        "padded_tokens": cost,
        "longest_row": lambda b: max(lengths[i] for i in b),
        "most_rows": len,
        "most_padding": lambda b: cost(b) - sum(lengths[i] for i in b),
        "most_options": lambda b: max(n_opts(i) for i in b),
        "most_media": lambda b: sum(media_tok(i) for i in b),
    }
    out, seen = [], set()
    for cat, key in cats.items():
        k = 0
        for b in sorted(batches, key=key, reverse=True):
            if tuple(b) in seen:
                continue
            seen.add(tuple(b))
            out.append({"category": cat, "rows": list(b), "padded_tokens": cost(b), "max_len": max(lengths[i] for i in b),
                        "n": len(b), "max_options": max(n_opts(i) for i in b), "media_tokens": sum(media_tok(i) for i in b)})
            k += 1
            if k >= per_cat:
                break
    return out


def plan_only(bundle, rows, args, world: int) -> int:
    """Every row rendered and measured exactly as the run will (padding included), the epoch planned for `world`
    ranks, the worst cases listed. Any row that cannot be rendered raises here instead of in the multi-GPU job."""
    t0 = time.perf_counter()
    batches, dropped, lengths = plan_epoch(bundle, rows, args, 0, 0, 1)
    steps = plan_steps(batches, lengths, int(os.environ.get("PLAN_WORLD", "4")), args.accum_rows)
    s = sorted(lengths)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    print(f"plan-only: {len(rows)} rows rendered in {time.perf_counter() - t0:.0f}s; tokens p50 {q(.5)} p90 {q(.9)} "
          f"p99 {q(.99)} max {s[-1]}; {dropped} rows over max-len {args.max_len}; {len(batches)} micro-batches, "
          f"{len(steps)} steps for {os.environ.get('PLAN_WORLD', '4')} ranks; total tokens {sum(lengths):,}", flush=True)
    for w in auto_worst_case(rows, batches, lengths, bundle, args):
        print(f"  worst {w['category']:<14} rows {w['n']:>3} max_len {w['max_len']:>6} padded {w['padded_tokens']:>6} "
              f"options {w['max_options']:>3} media tokens {w['media_tokens']:>6}", flush=True)
    print("PLAN_OK", flush=True)
    return 0


def use_autocast(args) -> bool:
    return bool(args.lora_autocast or (args.full_finetune and not args.no_autocast))


def collect_mm(bundle, rows, max_len: int, batch_tokens: int, max_rows: int, device, sync_batches=None):
    """common.collect_slot_logits for a multimodal bundle: frozen production rendering (no permutation, indent 2),
    exact lengths from the token counter, media loaded per batch. `sync_batches` as in collect_slot_logits (--fsdp):
    the rank pads to the busiest rank's batch count with one-token text forwards, under no_grad."""
    import torch

    from train import mm

    examples = [common.render_example(r, bundle.template_for(r), i, seed=None, augment=False) for i, r in enumerate(rows)]
    lengths = [bundle.prompt_length(ex.prompt, rows[ex.idx]) for ex in examples]
    usable = [i for i, L in enumerate(lengths) if L <= max_len]
    batches = common.length_bucket_batches([lengths[i] for i in usable], batch_tokens, max_rows, shuffle_seed=None)
    out = []
    n_total = sync_batches(len(batches)) if sync_batches is not None else len(batches)
    with (torch.no_grad() if sync_batches is not None else torch.inference_mode()):
        for _ in range(n_total - len(batches)):
            ids = torch.tensor([[bundle.tokenizer.pad_token_id]], dtype=torch.long, device=device)
            common.readout_logits(bundle.model, ids, torch.ones_like(ids), [0])
        for b in batches:
            exs = [examples[usable[j]] for j in b]
            inputs, ends = mm.mm_inputs(bundle.processor, [ex.prompt for ex in exs],
                                        [rows[ex.idx].get("media") or [] for ex in exs], device)
            vocab_logits = mm.mm_readout_logits(bundle.model, inputs, ends)
            sel, _ = common.slot_view(vocab_logits, bundle.slot_ids, [len(ex.labels) for ex in exs])
            sel = sel.float().cpu().tolist()
            for r, ex in enumerate(exs):
                out.append(common.Scored(kind=ex.kind, k=len(ex.labels), logits=sel[r][: len(ex.labels)],
                                         target=list(ex.target), source=ex.source, task=ex.task,
                                         labels=list(ex.labels), idx=ex.idx))
    return out


def save_weights(model, bundle, out_dir: Path, full: bool, sharded: bool = False, is_main: bool = True) -> None:
    """LoRA: the adapter. Full fine-tuning: the whole model in bf16 (from the fp32 master weights, copied to CPU),
    with tokenizer and processor, loadable like the base checkpoint. Sharded (--fsdp): every rank must call it,
    the weights are gathered to rank 0's CPU and written there as HF safetensors shards."""
    import torch

    if sharded:
        from train import fsdp

        sd = fsdp.full_state_dict(model, torch.bfloat16, is_main=is_main)
        if not is_main:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir), state_dict=sd, safe_serialization=True)
        del sd
        (bundle.processor or bundle.tokenizer).save_pretrained(str(out_dir))
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if full:
        sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items()}
        model.save_pretrained(str(out_dir), state_dict=sd, safe_serialization=True)
        del sd
    else:
        model.save_pretrained(str(out_dir))
    (bundle.processor or bundle.tokenizer).save_pretrained(str(out_dir))


def slot_loss(sel, mask, targets, brier_weight: float):
    """Cross-entropy against a soft target plus a Brier term, both over the K answer slots."""
    import torch

    logp = torch.log_softmax(sel, dim=-1)
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    ce = -(targets * logp).sum(dim=-1)
    probs = torch.softmax(sel, dim=-1)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    brier = ((probs - targets) ** 2).sum(dim=-1)
    loss = (ce + brier_weight * brier).mean()
    return loss, ce.mean(), brier.mean(), probs


def dist_setup(args) -> tuple[int, int]:
    """(rank, world). Under torchrun each rank drives its own GPU (LOCAL_RANK); a gloo test may put
    several ranks on one device, in which case --device is kept."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 and not (args.fsdp and "RANK" in os.environ):
        return 0, 1
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    if args.dist_backend == "nccl":
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}"
    torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group(args.dist_backend, timeout=datetime.timedelta(minutes=90))
    return rank, world


def allreduce_grads(params) -> None:
    """Sum the LoRA gradients over ranks, in flat buckets. Each rank's micro-batch losses are already
    weighted by their share of the global step's rows, so the sum is the global mean gradient."""
    import torch
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    bucket_bytes = 256 * 2**20
    bucket: list = []
    size = 0

    def flush():
        nonlocal bucket, size
        if bucket:
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat)
            for g, synced in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                g.copy_(synced)
        bucket, size = [], 0

    for prm in params:
        if prm.grad is None:
            prm.grad = torch.zeros_like(prm)
        g = prm.grad
        nbytes = g.numel() * g.element_size()
        if nbytes >= bucket_bytes:
            dist.all_reduce(g)
            continue
        if size + nbytes > bucket_bytes:
            flush()
        bucket.append(g)
        size += nbytes
    flush()


def load_eval_rows(path: str, limit: int = 0) -> list[dict]:
    """Unified rows pass through. Benchmark rows ({state, question, options, label}, e.g. the
    DecisionBench and TypeSafe fixtures) are mapped with train.common.row_to_question and
    a one-hot target on the labelled option; their state_id is kept for state-macro accuracy. Rows the
    frozen prompt cannot render (DecisionBench has 16 per subset with more than 26 options, which only the
    engine's chunked path handles) are skipped and counted; the official scores come from a full benchmark run."""
    from train.common import row_to_question
    from qev.prompt import render_question

    rows = []
    skipped: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "options" in r and "label" in r and "target" not in r:
                q = row_to_question(r)
                opt = str(r["options"][int(r["label"])]["id"])
                key = ("yes" if opt == "true" else "no") if q["type"] == "noul" else opt
                r = {"id": r["id"], "source": r.get("source", "bench"), "task": r.get("primitive") or q["type"],
                     "state": r["state"], "question": q, "target": {key: 1.0},
                     "meta": {"state_id": r.get("state_id")}}
            try:
                rendered = render_question("q", common.build_question(r["question"]))
                common.target_vector(r["target"], rendered.labels)
            except Exception as exc:
                skipped[str(exc).split(" of prompt")[0][-40:]] = skipped.get(str(exc).split(" of prompt")[0][-40:], 0) + 1
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    if skipped:
        log.warning("eval file %s: skipped rows that cannot be rendered for in-training eval: %s", path, skipped)
    return rows


def run_evals(model, bundle, eval_sets, args, rank: int, world: int, step: int, logf, is_main: bool) -> None:
    """Frozen production rendering (no permutation, indent 2) over each eval set, rows split across
    ranks. Reports accuracy, state-macro accuracy (rows grouped by state_id, else trajectory, else id),
    NLL, ECE (10 bins) and accuracy per question type."""
    import torch

    from qev.calibrate import nll as nll_fn
    from qev.calibrate import softmax as softmax_fn

    was_training = model.training
    model.eval()
    max_len = args.eval_max_len or args.max_len
    sync = None
    if args.fsdp:
        def sync(n: int) -> int:
            if world == 1:
                return n
            import torch.distributed as dist

            t = torch.tensor([n], dtype=torch.long, device=args.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return int(t.item())
    for name, erows in eval_sets:
        t0 = time.perf_counter()
        mine = erows[rank::world]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast(args)):
            if bundle.multimodal:
                scored = collect_mm(bundle, mine, max_len, args.batch_tokens, args.max_rows, args.device,
                                    sync_batches=sync)
            else:
                scored = common.collect_slot_logits(bundle, mine, max_len=max_len, batch_tokens=args.batch_tokens,
                                                    max_rows=args.max_rows, device=args.device, quiet=True,
                                                    sync_batches=sync)
        recs = []
        for sc in scored:
            prob = softmax_fn(sc.logits, 1.0)
            gold = max(range(len(sc.target)), key=sc.target.__getitem__)
            pred = max(range(len(prob)), key=prob.__getitem__)
            meta = mine[sc.idx].get("meta") or {}
            key = meta.get("state_id") or meta.get("traj_id") or mine[sc.idx].get("id")
            recs.append((key, sc.kind, int(pred == gold), max(prob), nll_fn(prob, sc.target)))
        if world > 1:
            import torch.distributed as dist

            parts = [None] * world
            dist.all_gather_object(parts, recs)
            recs = [x for part in parts for x in part]
        if not is_main:
            continue
        n = len(recs)
        if n == 0:
            log.warning("eval %s: no rows under max-len %d", name, max_len)
            continue
        per_state: dict = {}
        per_kind: dict = {}
        for key, kind, ok, _, _ in recs:
            per_state.setdefault(key, []).append(ok)
            per_kind.setdefault(kind, []).append(ok)
        bins = [[] for _ in range(10)]
        for _, _, ok, conf, _ in recs:
            bins[min(9, int(conf * 10))].append((conf, ok))
        ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b)
        rec = {"event": "eval", "step": step, "set": name, "n": n, "skipped": len(erows) - n,
               "acc": sum(r[2] for r in recs) / n,
               "state_macro": sum(sum(v) / len(v) for v in per_state.values()) / len(per_state),
               "nll": sum(r[4] for r in recs) / n, "ece": ece,
               "by_type": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_kind.items())},
               "seconds": time.perf_counter() - t0}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"eval step {step} {name}: n {n} acc {rec['acc']:.4f} state_macro {rec['state_macro']:.4f} "
              f"nll {rec['nll']:.4f} ece {rec['ece']:.4f} {rec['by_type']} ({rec['seconds']:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    if was_training:
        model.train()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    rank, world = dist_setup(args)
    is_main = rank == 0
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)
    if args.share_prefix and world > 1:
        raise SystemExit("--share-prefix is single-GPU only")
    if args.fsdp and not args.full_finetune:
        raise SystemExit("--fsdp shards a full fine-tune; add --full-finetune")
    if args.fsdp and args.share_prefix:
        raise SystemExit("--fsdp is implemented without --share-prefix")
    if args.disable_fla_kernels:
        common.block_fla_kernels()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    if world > 1:
        import torch.distributed as dist

    run_dir = Path(args.runs_dir) / args.name
    log_path = run_dir / "log.jsonl"
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.json").write_text(json.dumps({**vars(args), "world_size": world}, indent=2))

    rows = common.load_rows(args.train_file, limit=args.limit, split=args.split)
    if not rows:
        raise SystemExit(f"no rows in {args.train_file}")
    log.info("loaded %d rows from %s (world size %d)", len(rows), args.train_file, world)
    if args.pad_options > 0:
        OPTION_POOL.update(common.build_option_pool(rows))
        log.info("option padding p=%.2f, pool %s", args.pad_options, {k: len(v) for k, v in OPTION_POOL.items()})
    eval_sets = []
    for spec in args.eval_files:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        eval_sets.append((name, load_eval_rows(path, args.eval_limit)))
        log.info("eval set %s: %d rows from %s", name, len(eval_sets[-1][1]), path)

    if args.media_root:
        from train import mm as _mm

        _mm.set_root(args.media_root)
    load_dtype = "float32" if args.full_finetune and not args.fsdp else args.dtype
    bundle = common.load_bundle(args.model, device=args.device, dtype=load_dtype, train=True,
                                multimodal=args.multimodal, chat_template=args.chat_template,
                                load_model=not args.plan_only)
    if args.plan_only:
        return plan_only(bundle, rows, args, world)
    log.info("model_type=%s template_kwargs=%s", bundle.model_type, bundle.template_kwargs)

    if args.full_finetune:
        model = bundle.model
        for name, prm in model.named_parameters():
            prm.requires_grad = args.train_vision or not (".visual." in f".{name}." or name.startswith("visual"))
        log.info("full fine-tuning: fp32 master weights, bf16 autocast forward, vision tower %s",
                 "trained" if args.train_vision else "frozen")
    else:
        targets_arg = ([m.strip() for m in args.target_modules.split(",")] if args.target_modules
                       else common.linear_projection_modules(bundle.model))
        log.info("LoRA target modules: %d modules, kinds %s", len(targets_arg), common.short_names(targets_arg))
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          bias="none", task_type="CAUSAL_LM", target_modules=targets_arg)
        torch.manual_seed(args.seed)
        model = get_peft_model(bundle.model, lcfg)
        for _, param in model.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()
        if args.lora_autocast:
            from peft.tuners.tuners_utils import BaseTunerLayer

            n_layers = 0
            for module in model.modules():
                if isinstance(module, BaseTunerLayer):
                    module.cast_input_dtype_enabled = False
                    n_layers += 1
            log.info("LoRA autocast: %d adapter layers take bf16 inputs; adapter weights stay fp32", n_layers)
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1 and not args.full_finetune:
        for prm in params:
            dist.broadcast(prm.data, 0)

    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    log.info("trainable %.2fM of %.2fM parameters (%.3f%%)", trainable / 1e6, total / 1e6, 100 * trainable / total)

    if args.gradient_checkpointing:
        if not args.full_finetune:
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    if args.fsdp:
        from train import fsdp

        fsdp.shard_model(model, world, bf16_compute=use_autocast(args))
        params = [p for p in model.parameters() if p.requires_grad]

    pad_id = bundle.tokenizer.pad_token_id
    device = torch.device(args.device)

    def plan(epoch):
        """Steps of this epoch, the rows dropped over max-len, and the token length of every row
        (None on the shared path)."""
        if args.share_prefix:
            steps, dropped, n_groups = plan_epoch_shared(bundle, rows, args, epoch)
            log.info("prefix sharing: %d state groups, %d steps of at least %d rows", n_groups, len(steps), args.step_rows)
            return steps, dropped, None
        batches, dropped, lengths = plan_epoch(bundle, rows, args, epoch, rank, world)
        return plan_steps(batches, lengths, world, args.accum_rows), dropped, lengths

    def step_units(step_plan):
        """This rank's units of a step, and the step's row count over all ranks."""
        if args.share_prefix:
            return step_plan, sum(len(u[1]) for u in step_plan)
        return [("batch", b) for b in step_plan[rank]], sum(len(b) for per in step_plan for b in per)

    t_plan = time.perf_counter()
    steps0, dropped0, lengths0 = plan(0)
    if args.worst_case:
        if world > 1 and not args.fsdp:
            raise SystemExit("--worst-case runs on one GPU (or under --fsdp, where every rank runs the same batch)")
        if args.worst_case == "auto":
            wc = auto_worst_case(rows, [b for st in steps0 for per in st for b in per], lengths0, bundle, args)
        else:
            wc = json.loads(Path(args.worst_case).read_text())
        for w in wc:
            got = max(lengths0[i] for i in w["rows"])
            if got != w["max_len"]:
                raise SystemExit(f"worst-case batch {w['category']} max_len {w['max_len']} but trainer measures {got}")
            log.info("worst case %-14s rows %3d max_len %6d padded %6d options %3d media tokens %6d", w["category"],
                     len(w["rows"]), w["max_len"], w["padded_tokens"], w.get("max_options", 0), w.get("media_tokens", 0))
        steps0 = [[[w["rows"]]] * world for w in wc]
        log.info("worst-case mode: %d single micro-batch steps from %s (lengths match the preflight)", len(steps0),
                 args.worst_case)
    steps_per_epoch = len(steps0)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(round(args.warmup_frac * total_steps)))
    log.info("%d steps per epoch, %d total steps, %d warmup, %d rows dropped over max-len, planned in %.0fs",
             steps_per_epoch, total_steps, warmup, dropped0, time.perf_counter() - t_plan)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
                            fused=bool(args.full_finetune))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=max(total_steps, 1))

    logf = open(log_path, "a", encoding="utf-8") if is_main else open(os.devnull, "w")
    step = 0
    seen = 0
    t_start = time.perf_counter()

    def new_window():
        return {"loss": 0.0, "ce": 0.0, "brier": 0.0, "acc": 0.0, "gnorm": 0.0, "n": 0, "rows": 0, "tokens": 0,
                "t": time.perf_counter()}

    window = new_window()
    stop = False

    def unit_forward(unit, epoch):
        """Slot logits, target and option counts for one unit: a plain batch or a prefix-shared group."""
        kind, idx = unit
        if kind == "group":
            from train import shared

            seed = args.seed * 1000003 + epoch * 1000033 + idx[0]
            state, sufs, labels, targets, _ = shared.render_group(rows, idx, seed=seed, augment=args.augment)
            prefix, suffix_ids = shared.split_prefix(bundle, state, sufs)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast(args)):
                vocab_logits = shared.shared_readout_logits(model, bundle.config, prefix, suffix_ids, pad_id, device,
                                                            checkpointing=args.gradient_checkpointing)
            ks = [len(l) for l in labels]
        else:
            examples = []
            for i in idx:
                seed = args.seed * 1000003 + epoch * 1000033 + i
                examples.append(common.render_example(rows[i], bundle.template_for(rows[i]), i, seed=seed,
                                                      augment=args.augment, option_pool=OPTION_POOL,
                                                      pad_prob=args.pad_options))
            if bundle.multimodal:
                from train import mm

                inputs, ends = mm.mm_inputs(bundle.processor, [ex.prompt for ex in examples],
                                            [rows[i].get("media") or [] for i in idx], device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast(args)):
                    vocab_logits = mm.mm_readout_logits(model, inputs, ends)
            else:
                id_lists = [bundle.encode(ex.prompt) for ex in examples]
                input_ids, attention_mask, ends = common.pad_batch(id_lists, pad_id, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast(args)):
                    vocab_logits = common.readout_logits(model, input_ids, attention_mask, ends)
            ks = [len(ex.labels) for ex in examples]
            targets = [ex.target for ex in examples]
        kmax = max(ks)
        tgt = torch.zeros((len(ks), kmax), dtype=torch.float32, device=device)
        for r, t in enumerate(targets):
            tgt[r, : len(t)] = torch.tensor(t, dtype=torch.float32, device=device)
        sel, mask = common.slot_view(vocab_logits, bundle.slot_ids, ks)
        return sel, mask, tgt

    if eval_sets and args.eval_at_start:
        run_evals(model, bundle, eval_sets, args, rank, world, 0, logf, is_main)

    for epoch in range(args.epochs):
        if epoch == 0:
            steps_plan, lengths = steps0, lengths0
        else:
            steps_plan, _, lengths = plan(epoch)
        for step_plan in steps_plan:
            if args.worst_case:
                torch.cuda.reset_peak_memory_stats(device)
            units, n_step_rows = step_units(step_plan)
            n_dummy = max(len(per) for per in step_plan) - len(units) if args.fsdp else 0
            grad_scale = world if args.fsdp else 1
            agg = torch.zeros(4, dtype=torch.float64, device=device)
            try:
                for unit in units:
                    sel, mask, tgt = unit_forward(unit, epoch)
                    loss, ce, brier, probs = slot_loss(sel, mask, tgt, args.brier_weight)
                    w = tgt.shape[0] / n_step_rows
                    (loss * (w * grad_scale)).backward()
                    with torch.no_grad():
                        correct = (probs.argmax(dim=-1) == tgt.argmax(dim=-1)).float().sum()
                        agg += torch.stack([loss.detach() * w, ce.detach() * w, brier.detach() * w, correct]).double()
                    del sel, mask, tgt, loss, ce, brier, probs
                for _ in range(n_dummy):
                    fsdp.dummy_pass(model, pad_id, device, use_autocast(args))
            except torch.OutOfMemoryError:
                if not args.worst_case:
                    raise
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                log.warning("worst-case step %d OUT OF MEMORY (peak before failure %.1f GB, rows %d, tokens %d)",
                            step + 1, peak, n_step_rows, sum(lengths[i] for per in step_plan for b in per for i in b))
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                step += 1
                continue
            if world > 1:
                if not args.fsdp:
                    allreduce_grads(params)
                dist.all_reduce(agg)
            if args.dump_first_grad and step == 0 and args.fsdp:
                grads = fsdp.full_grads([(n, p) for n, p in model.named_parameters() if p.requires_grad])
                if is_main:
                    torch.save(grads, args.dump_first_grad)
                del grads
            elif args.dump_first_grad and step == 0 and is_main:
                names = [n for n, p in model.named_parameters() if p.requires_grad]
                torch.save({n: p.grad.detach().float().cpu() for n, p in zip(names, params)}, args.dump_first_grad)
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if hasattr(gnorm, "full_tensor"):
                gnorm = gnorm.full_tensor()
            try:
                opt.step()
            except torch.OutOfMemoryError:
                if not args.worst_case:
                    raise
                log.warning("worst-case step %d OUT OF MEMORY in the optimizer step (peak %.1f GB)", step + 1,
                            torch.cuda.max_memory_allocated(device) / 2**30)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                step += 1
                continue
            sched.step()
            opt.zero_grad(set_to_none=True)

            step += 1
            seen += n_step_rows
            a = agg.tolist()
            window["loss"] += a[0]
            window["ce"] += a[1]
            window["brier"] += a[2]
            window["acc"] += a[3] / n_step_rows
            window["gnorm"] += float(gnorm)
            window["n"] += 1
            window["rows"] += n_step_rows
            if lengths is not None:
                window["tokens"] += sum(lengths[i] for per in step_plan for b in per for i in b)

            if step % args.log_every == 0 or step == total_steps or step <= 5:
                n = max(window["n"], 1)
                dt = time.perf_counter() - window["t"]
                rec = {
                    "step": step, "epoch": epoch, "examples": seen,
                    "loss": window["loss"] / n, "ce": window["ce"] / n,
                    "brier": window["brier"] / n, "slot_acc": window["acc"] / n,
                    "grad_norm": window["gnorm"] / n,
                    "lr": sched.get_last_lr()[0], "rows_per_step": window["rows"] / n,
                    "examples_per_s": window["rows"] / max(dt, 1e-9),
                    "tokens_per_s": window["tokens"] / max(dt, 1e-9),
                    "sec_per_step": dt / n,
                    "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                    "elapsed_s": time.perf_counter() - t_start,
                }
                if is_main:
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    eta = (total_steps - step) * rec["sec_per_step"]
                    print(common.progress(step, total_steps, prefix="train ") +
                          f" loss {rec['loss']:.4f} ce {rec['ce']:.4f} brier {rec['brier']:.4f} "
                          f"slot_acc {rec['slot_acc']:.3f} gnorm {rec['grad_norm']:.3f} lr {rec['lr']:.2e} "
                          f"rows/s {rec['examples_per_s']:.2f} tok/s {rec['tokens_per_s']:.0f} "
                          f"s/step {rec['sec_per_step']:.1f} mem {rec['peak_mem_gb']:.1f}G eta {eta / 3600:.1f}h",
                          flush=True)
                window = new_window()

            if (is_main or args.fsdp) and args.save_every and step % args.save_every == 0:
                ckpt = run_dir / (f"ckpt-step{step}" if args.full_finetune else f"adapter-step{step}")
                save_weights(model, bundle, ckpt, args.full_finetune, sharded=args.fsdp, is_main=is_main)
                log.info("saved %s to %s", "model" if args.full_finetune else "adapter", ckpt)

            if eval_sets and args.eval_every and step % args.eval_every == 0 and step != total_steps:
                t_eval = time.perf_counter()
                run_evals(model, bundle, eval_sets, args, rank, world, step, logf, is_main)
                window["t"] += time.perf_counter() - t_eval

            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if eval_sets and not args.worst_case:
        run_evals(model, bundle, eval_sets, args, rank, world, step, logf, is_main)

    elapsed = time.perf_counter() - t_start
    final_dir = run_dir / ("final" if args.full_finetune else "adapter")
    if args.fsdp:
        save_weights(model, bundle, final_dir, True, sharded=True, is_main=is_main)
    if is_main:
        if not args.fsdp:
            save_weights(model, bundle, final_dir, args.full_finetune)
        log.info("saved final %s to %s", "model" if args.full_finetune else "adapter", final_dir)
        print(f"done: {step} steps, {seen} examples, {elapsed:.1f}s, {seen / max(elapsed, 1e-9):.2f} examples/s", flush=True)
        logf.write(json.dumps({"event": "done", "steps": step, "examples": seen, "elapsed_s": elapsed,
                               "examples_per_s": seen / max(elapsed, 1e-9)}) + "\n")
    logf.close()
    if world > 1 or args.fsdp:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    if not is_main:
        return 0

    if args.merge and not args.full_finetune:
        merged_dir = Path(args.merged_dir) if args.merged_dir else run_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_disable()
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        (bundle.processor or bundle.tokenizer).save_pretrained(str(merged_dir))
        log.info("merged model written to %s", merged_dir)
        print(f"merged model: {merged_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
