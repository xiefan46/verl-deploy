"""Repro 3: Magi cache corruption when (Q, K, V) come from an FSDP2-wrapped HF model.

Hypothesis: FSDP2's all-gather/reshard stream activity, plus the autocast
cast_forward_inputs path, interacts with Magi's cached position_ids
tensors in a way that produces drift only when Magi cache holds multiple
mgrs. This is what verl's actor engine does on every forward.

Compare:
  - repro 1 (no FSDP, random tensors):           cache_size=1000 PASSED
  - repro 2 (no FSDP, but vLLM ran first):       ?
  - repro 3 (THIS, FSDP2 wrap, no vLLM):         ?

We do NOT actually run prefix-tree attention through the wrapped model.
We use the FSDP2-wrapped model just to source (q, k, v) tensors — one full
forward through q_proj/k_proj/v_proj of layer_0 — and feed those into
calc_attn directly. That isolates "where the tensors come from" as the
only new variable vs repro 1.

Usage
-----
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1000 \
        torchrun --standalone --nproc_per_node=1 repro3_magi_with_fsdp.py

    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1 \
        torchrun --standalone --nproc_per_node=1 repro3_magi_with_fsdp.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


NUM_HEADS_Q = 14
NUM_HEADS_KV = 2
HEAD_DIM = 64


def _build_key(seq_len: int):
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges([(0, seq_len)]),
        k_ranges=AttnRanges.from_ranges([(0, seq_len)]),
        attn_mask_type=[AttnMaskType.FULL],
        total_seqlen_q=seq_len,
        total_seqlen_k=seq_len,
        num_heads_q=NUM_HEADS_Q,
        num_heads_kv=NUM_HEADS_KV,
        head_dim=HEAD_DIM,
        pad_size=0,
        cp_group_or_mesh=dist.group.WORLD,
        dist_attn_config=DistAttnConfig(
            dispatch_config=DispatchConfig(uneven_shard=True),
        ),
    )


def _build_fsdp_qwen(model_path: str):
    """Mirror FSDPEngine._build_module + _build_fsdp_module for actor."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from transformers import Qwen2ForCausalLM

    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict

    print(f"[REPRO3] Loading {model_path} in fp32")
    model = Qwen2ForCausalLM.from_pretrained(model_path, dtype=torch.float32).cuda()
    model.to(torch.float32)

    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    fsdp_kwargs = {
        "mesh": mesh,
        "mp_policy": mp_policy,
        "offload_policy": None,
        "reshard_after_forward": True,
    }
    full_state = model.state_dict()
    apply_fsdp2(model, fsdp_kwargs, config={})
    fsdp2_load_full_state_dict(model, full_state, mesh, None)
    return model


def _qkv_from_fsdp_model(model, seq_len: int, qkv_seed: int):
    """Run input → embed_tokens → layer_0.input_layernorm → q/k/v_proj.

    Returns (q, k, v) in shape (seq_len, n_heads, head_dim), bf16 — the
    same layout calc_attn expects from monkey_patch._magi_prefix_tree_attention_forward.
    """
    g = torch.Generator(device="cuda").manual_seed(qkv_seed)
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq_len), device="cuda", generator=g)

    root = model.model
    l0 = root.layers[0]

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), autocast_ctx:
        x = root.embed_tokens(input_ids)         # (1, S, H)
        x = l0.input_layernorm(x)                # (1, S, H)
        q = l0.self_attn.q_proj(x).view(1, seq_len, NUM_HEADS_Q, HEAD_DIM)
        k = l0.self_attn.k_proj(x).view(1, seq_len, NUM_HEADS_KV, HEAD_DIM)
        v = l0.self_attn.v_proj(x).view(1, seq_len, NUM_HEADS_KV, HEAD_DIM)

    # Reshape to (S, nh, hd) and cast to bf16 — same shape calc_attn ate
    # in repro 1 and 2.
    return (
        q.reshape(seq_len, NUM_HEADS_Q, HEAD_DIM).to(torch.bfloat16),
        k.reshape(seq_len, NUM_HEADS_KV, HEAD_DIM).to(torch.bfloat16),
        v.reshape(seq_len, NUM_HEADS_KV, HEAD_DIM).to(torch.bfloat16),
    )


def _forward_one(model, seq_len: int, qkv_seed: int) -> tuple[float, float]:
    from magi_attention.api import calc_attn

    key = _build_key(seq_len)
    q, k, v = _qkv_from_fsdp_model(model, seq_len, qkv_seed)
    o = calc_attn(q, k, v, key)[0]
    of = o.detach().float()
    return float(of.sum().item()), float(of.abs().max().item())


def main() -> int:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    cache_size_env = os.environ.get(
        "MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE", "default(1000)"
    )

    if dist.get_rank() == 0:
        print(f"[REPRO3] MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE={cache_size_env}")

    model_path = os.environ.get(
        "MODEL_PATH", os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    )
    if not os.path.exists(model_path):
        if dist.get_rank() == 0:
            print(f"[REPRO3] SKIP: model not found at {model_path}")
        dist.destroy_process_group()
        return 0

    model = _build_fsdp_qwen(model_path)

    REF_SEQ = 1024
    QKV_SEED = 42
    DISTRACTORS = [512, 768, 1280, 1536, 896, 1100, 800, 1408, 640, 1200, 960, 1320, 720, 1184, 1056]

    if dist.get_rank() == 0:
        print(f"[REPRO3] REF_SEQ={REF_SEQ}  QKV_SEED={QKV_SEED}  distractors={len(DISTRACTORS)}")
        print()
        print("[REPRO3] Pass A (cache empty, REF input)")
    sum_a, max_a = _forward_one(model, REF_SEQ, QKV_SEED)

    if dist.get_rank() == 0:
        print(f"[REPRO3] {len(DISTRACTORS)} distractor forwards")
    for s in DISTRACTORS:
        _forward_one(model, s, qkv_seed=s)

    if dist.get_rank() == 0:
        print("[REPRO3] Pass C (cache loaded, REF input)")
    sum_c, max_c = _forward_one(model, REF_SEQ, QKV_SEED)

    if dist.get_rank() == 0:
        print()
        print(f"[REPRO3] Pass A:  sum={sum_a:.6f}   abs_max={max_a:.6f}")
        print(f"[REPRO3] Pass C:  sum={sum_c:.6f}   abs_max={max_c:.6f}")
        diff = abs(sum_c - sum_a)
        rel = diff / (abs(sum_a) + 1e-9)
        print(f"[REPRO3] |A - C|     = {diff:.6f}")
        print(f"[REPRO3] |A - C|/|A| = {rel * 100:.4f}%")
        print()

        GREEN = "\033[0;32m"
        RED = "\033[0;31m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        if rel < 1e-3:
            print(f"  {GREEN}{BOLD}VERDICT: A ≈ C — FSDP2 alone is not enough to trigger{RESET}")
        else:
            print(f"  {RED}{BOLD}VERDICT: DRIFT — Magi + FSDP2 alone is enough{RESET}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
