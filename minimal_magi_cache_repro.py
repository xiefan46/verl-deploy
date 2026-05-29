"""Minimal repro for MagiAttention runtime-dict cache corruption.

Setup-free: no verl, no FSDP, no vLLM, no Ray. Just magi_attention + PyTorch.

What we're testing
------------------
We forward the SAME (Q, K, V, key-content) through ``calc_attn`` twice.
The ONLY variable between the two forwards is the state of Magi's
per-cp_group runtime-dict cache:

  - Pass A: cache empty when key_X is first registered → mgr_X is the sole
    occupant → forward gives output_A.

  - Pass B: between Pass A and Pass B we register 15 unrelated keys (each
    forces a calc_attn so the mgrs are fully materialized). Then we
    re-register key_X. Depending on ``MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE``:
      * size=1:    cache holds 1 mgr at all times. key_X gets a fresh
        rebuild. output_B should equal output_A.
      * size>=16:  cache holds 16 mgrs simultaneously. key_X may even
        cache-hit. output_B should still equal output_A under any sane
        kernel — multiple coexisting mgrs are not supposed to leak state.

If Pass A and Pass C diverge with size>=16 but agree with size=1, the
corruption is isolated to magi (no verl / FSDP / vLLM needed).

Usage
-----
    # Reproducing default-Magi bug:
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1000 \
        torchrun --standalone --nproc_per_node=1 minimal_magi_cache_repro.py

    # Verifying our fix:
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1 \
        torchrun --standalone --nproc_per_node=1 minimal_magi_cache_repro.py

Or use the bundled wrapper:
    bash verify_magi_cache_bug.sh
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


# Shape constants — Qwen2.5-0.5B-Instruct-like (GQA 14/2, head_dim=64).
NUM_HEADS_Q = 14
NUM_HEADS_KV = 2
HEAD_DIM = 64


def _build_key(seq_len: int):
    """Register one DistAttnRuntimeKey for the given seq_len.

    Same seq_len ⇒ same key hash ⇒ cache hit on second register.
    Different seq_len ⇒ different key hash ⇒ new mgr inserted.
    """
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


def _forward_one(seq_len: int, qkv_seed: int) -> tuple[float, float]:
    """Build (Q, K, V) deterministically from qkv_seed, run calc_attn, return (sum, abs_max)."""
    from magi_attention.api import calc_attn

    key = _build_key(seq_len)

    # Deterministic Q/K/V per (seq_len, qkv_seed). Using a Generator avoids
    # leaking global RNG state across forwards.
    g = torch.Generator(device="cuda").manual_seed(qkv_seed)
    q = torch.randn(seq_len, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(seq_len, NUM_HEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(seq_len, NUM_HEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)

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

    # The reference forward. Same seq_len + same QKV_SEED on both A and B,
    # so q/k/v/key-content are bit-exact identical.
    REF_SEQ = 1024
    QKV_SEED = 42

    # 15 distractor seq_lens — different total_seqlen_q ⇒ different key hash.
    # Each gets a real calc_attn so the mgr is fully populated in cache.
    DISTRACTORS = [512, 768, 1280, 1536, 896, 1100, 800, 1408, 640, 1200, 960, 1320, 720, 1184, 1056]

    if dist.get_rank() == 0:
        print(f"[REPRO] MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE={cache_size_env}")
        print(f"[REPRO] REF_SEQ={REF_SEQ}  QKV_SEED={QKV_SEED}  distractors={len(DISTRACTORS)}")
        print()

    # ── Pass A: cache empty, register REF, forward ──
    sum_a, max_a = _forward_one(REF_SEQ, QKV_SEED)

    # ── Pass B: register 15 distractor keys (real calc_attn each so they
    #    materialize fully — position_ids tensors etc) ──
    for s in DISTRACTORS:
        _forward_one(s, qkv_seed=s)

    # ── Pass C: forward REF again. SAME q/k/v/key-content as Pass A. ──
    sum_c, max_c = _forward_one(REF_SEQ, QKV_SEED)

    if dist.get_rank() == 0:
        print(f"[REPRO] Pass A (cache empty when REF runs):")
        print(f"          sum={sum_a:.6f}   abs_max={max_a:.6f}")
        print(f"[REPRO] Pass C (cache has {len(DISTRACTORS)} other mgrs when REF re-runs):")
        print(f"          sum={sum_c:.6f}   abs_max={max_c:.6f}")
        print()

        diff = abs(sum_c - sum_a)
        denom = abs(sum_a) + 1e-9
        rel = diff / denom

        BOLD = "\033[1m"
        GREEN = "\033[0;32m"
        RED = "\033[0;31m"
        RESET = "\033[0m"

        print(f"[REPRO] |A - C|     = {diff:.6f}")
        print(f"[REPRO] |A - C|/|A| = {rel * 100:.4f}%")
        print()

        # bf16 noise floor is ~1e-3 relative at this magnitude.
        if rel < 1e-3:
            print(f"  {GREEN}{BOLD}VERDICT: A ≈ C (within bf16 noise floor){RESET}")
            print(f"  {GREEN}No cache corruption at this cache size.{RESET}")
        else:
            print(f"  {RED}{BOLD}VERDICT: DRIFT (rel diff {rel * 100:.2f}% >> bf16 noise floor){RESET}")
            print(f"  {RED}Cache corruption REPRODUCED in isolation. No verl/FSDP/vLLM needed.{RESET}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
