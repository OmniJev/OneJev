"""Sharded full fine-tuning (FSDP2) for models too large to replicate per GPU.

`train.sft --full-finetune` without this keeps an fp32 copy of every weight, its gradient and both
AdamW moments on every rank: 16 bytes per parameter, 434 GB for Qwen3.8-27B, which no GPU holds.
`--fsdp` shards all four over the ranks (16 bytes / world per parameter) and all-gathers one unit at a
time in bf16 for compute.

Units: every decoder layer, the input embedding and the LM head each become one FSDP unit, and the root
holds what is left (the final norm, and on a multimodal model the frozen vision tower). The 248k-token vocabulary makes the embedding and the head 1.27B
parameters each on Qwen3.8-27B; as their own units they are gathered only while they run instead of
staying unsharded on the root for the whole step.

The model is loaded in bf16 and each unit is cast to fp32 right before it is sharded, so no rank ever
holds a full fp32 copy: the peak during setup is the bf16 model plus one fp32 layer.

Mixed precision: fp32 sharded master weights, gradients and AdamW state; bf16 all-gathered weights for
the forward and backward; gradients reduce-scattered in fp32. FSDP averages gradients over ranks, while
train.sft weights every micro-batch loss by its share of the global step's rows and wants the sum, so under
--fsdp the backward loss is multiplied by the world size. A custom divide factor would do
the same through NCCL's PREMUL_SUM, which gloo (the test backend) does not have.

Multimodal (Qwen3_5ForConditionalGeneration, --multimodal): the vision tower stays frozen and inside the root
unit. The root is gathered once per forward whatever the batch holds, so a rank with images and a rank with
text only run the same collectives; a vision unit of its own would all-gather only on ranks whose batch has
media and deadlock the others. Frozen parameters get no gradient and are never reduce-scattered; they cost
4 bytes / world per parameter at rest and one bf16 copy (about 0.9 GB for the 27-block tower) while gathered.

Every rank must run the same number of forward and backward passes, since each one all-gathers and
reduce-scatters. train.sft pads a rank's micro-batches in a step with zero-weight dummy passes
(`dummy_pass`), and evaluation pads its batches the same way (common.collect_slot_logits `sync_batches`).
"""
from __future__ import annotations

import logging

log = logging.getLogger("train.fsdp")


def decoder_layers(model):
    """The decoder layer list of a causal LM (Qwen3.5 text: model.model.layers)."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(getattr(inner, "language_model", None), "layers", None)
    if layers is None:
        raise SystemExit(f"--fsdp: cannot find the decoder layers of {type(model).__name__}")
    return layers


def shard_model(model, world: int, bf16_compute: bool = True):
    """Shard `model` in place over the default process group. Call after requires_grad and gradient
    checkpointing are set, and before the optimizer is built (the parameters become DTensors).
    bf16_compute=False (tests, with --no-autocast) gathers and computes in fp32."""
    import torch
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if bf16_compute else None, reduce_dtype=torch.float32)
    kw = {"mp_policy": mp}
    import torch.distributed as dist

    if dist.get_backend() == "gloo":
        from torch.distributed.device_mesh import DeviceMesh

        kw["mesh"] = DeviceMesh.from_group(dist.group.WORLD, device_type="cuda")

    def to_fp32(module):
        for p in module.parameters(recurse=True):
            if p.is_floating_point() and p.dtype != torch.float32:
                p.data = p.data.float()

    layers = decoder_layers(model)
    for layer in layers:
        to_fp32(layer)
        fully_shard(layer, **kw)
    units = len(layers)
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head is not None and head.weight is embed.weight:
        raise SystemExit("--fsdp: tied input and output embeddings are not handled")
    for module in (embed, head):
        if module is not None:
            to_fp32(module)
            fully_shard(module, **kw)
            units += 1
    to_fp32(model)
    fully_shard(model, **kw)
    torch.cuda.empty_cache()
    n = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("FSDP: %d units + root over %d ranks, %.2fB parameters (%.2fB trained, %.2fB frozen), "
             "%.1f GB of fp32 weights+grads+AdamW per rank", units, world, n / 1e9, n_train / 1e9, (n - n_train) / 1e9,
             (16 * n_train + 4 * (n - n_train)) / world / 1e9)
    return model


def dummy_pass(model, pad_id: int, device, autocast: bool) -> None:
    """A zero-weight forward and backward over one token: every FSDP unit all-gathers and reduce-scatters
    exactly as in a real pass and adds nothing to the gradient. Pads ranks that hold fewer micro-batches."""
    import torch

    from train import common

    ids = torch.tensor([[pad_id]], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits = common.readout_logits(model, ids, torch.ones_like(ids), [0])
    (logits.float().sum() * 0.0).backward()


def gather_full(v):
    """The unsharded tensor of an FSDP DTensor (dim-0 shards in torch.chunk layout), on every rank; other
    tensors pass through. Uses c10d all_gather_into_tensor, the collective FSDP itself trains with:
    DTensor.full_tensor() goes through functional collectives, which crash with gloo on CUDA (the test backend)."""
    import torch
    import torch.distributed as dist

    if not hasattr(v, "to_local"):
        return v
    from torch.distributed.tensor import Shard

    if tuple(v.placements) != (Shard(0),):
        raise SystemExit(f"--fsdp: unexpected placement {v.placements}")
    local = v.to_local()
    n0, world = v.shape[0], dist.get_world_size()
    chunk = -(-n0 // world)
    tail = tuple(v.shape[1:])
    padded = torch.zeros((chunk,) + tail, dtype=local.dtype, device=local.device)
    padded[: local.shape[0]].copy_(local)
    out = torch.empty((chunk * world,) + tail, dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(out, padded)
    return out[:n0]


def full_state_dict(model, dtype=None, is_main: bool = True) -> dict:
    """The whole unsharded state dict on rank 0's CPU (empty on the other ranks). Collective: every rank
    gathers the tensors one at a time in the same order, so only one full tensor is on the GPU at once."""
    import torch

    out = {}
    for k, v in model.state_dict().items():
        v = gather_full(v)
        if is_main:
            v = v.detach()
            if dtype is not None and v.is_floating_point():
                v = v.to(dtype)
            out[k] = v.to("cpu").clone()
        del v
    torch.cuda.empty_cache()
    return out


def full_grads(named_params) -> dict:
    """Unsharded gradients by name, on CPU in fp32 (test hook). Collective."""
    out = {}
    for name, p in named_params:
        if p.grad is None:
            continue
        out[name] = gather_full(p.grad).detach().float().cpu()
    return out


def plain_float(x) -> float:
    """float() of a scalar that may be a DTensor (clip_grad_norm_ returns one under FSDP)."""
    if hasattr(x, "full_tensor"):
        x = x.full_tensor()
    return float(x)
