"""Repro 2: Magi cache corruption after vLLM init+sleep.

Hypothesis: vLLM's CUDA allocator activity (capturing graphs, building KV
cache pools, then sleep()ing to free) leaves the PyTorch caching-allocator
pool in a state that subsequent Magi forwards reuse. With Magi cache size
>= 2, multiple coexisting mgrs hold position_ids tensors in this poisoned
pool. forward output drifts.

Compare repro 1 (no vLLM): cache_size=1000 PASSED → Magi alone is clean.
If THIS script DRIFTS at cache_size=1000 but PASSES at cache_size=1, the
bug requires vLLM as a precondition.

Setup
-----
Single process, single GPU. We init vLLM with a tiny model (Qwen2.5-0.5B,
which the rest of the verl pipeline already needs), then sleep it, then
run the same Pass-A / distractors / Pass-C sequence as repro 1.

If model isn't on disk we skip and report — repro 1 is enough to falsify
the simple "Magi alone" hypothesis.

Usage
-----
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1000 \
        torchrun --standalone --nproc_per_node=1 repro2_magi_with_vllm.py

    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=1 \
        torchrun --standalone --nproc_per_node=1 repro2_magi_with_vllm.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


# Same shape constants as repro 1.
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


def _forward_one(seq_len: int, qkv_seed: int) -> tuple[float, float]:
    from magi_attention.api import calc_attn

    key = _build_key(seq_len)
    g = torch.Generator(device="cuda").manual_seed(qkv_seed)
    q = torch.randn(seq_len, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(seq_len, NUM_HEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(seq_len, NUM_HEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    o = calc_attn(q, k, v, key)[0]
    of = o.detach().float()
    return float(of.sum().item()), float(of.abs().max().item())


def _try_vllm_cycle() -> str:
    """Init vLLM with a tiny model, run one prompt, then sleep.

    Returns a status string describing what happened.
    """
    model_path = os.environ.get(
        "MODEL_PATH", os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    )
    if not os.path.exists(model_path):
        return f"SKIPPED (model not found at {model_path}; set MODEL_PATH or download Qwen2.5-0.5B-Instruct)"

    try:
        # Use the same vllm import path verl uses. Keep params minimal so init
        # is cheap (~30s for graph capture).
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.4,
            enforce_eager=False,  # keep CUDA graph capture, like the e2e
            max_model_len=512,
            max_num_seqs=4,
            enable_sleep_mode=True,
        )
        sp = SamplingParams(max_tokens=8, temperature=0.0)
        _ = llm.generate(["Hello world"], sp)
        llm.sleep(level=1)
        # Force PyTorch to actually release whatever vLLM freed.
        torch.cuda.synchronize()
        return "OK (init + 1 prompt + sleep)"
    except Exception as e:
        return f"FAILED: {type(e).__name__}: {e}"


def main() -> int:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    cache_size_env = os.environ.get(
        "MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE", "default(1000)"
    )

    if dist.get_rank() == 0:
        print(f"[REPRO2] MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE={cache_size_env}")
        print()

    # ── Run vLLM lifecycle ONCE before any Magi work ──
    if dist.get_rank() == 0:
        print("[REPRO2] Stage 1: bring up vLLM, run a prompt, sleep it...")
    vllm_status = _try_vllm_cycle()
    if dist.get_rank() == 0:
        print(f"[REPRO2]   vLLM cycle: {vllm_status}")
        print()

    REF_SEQ = 1024
    QKV_SEED = 42
    DISTRACTORS = [512, 768, 1280, 1536, 896, 1100, 800, 1408, 640, 1200, 960, 1320, 720, 1184, 1056]

    # Pass A
    if dist.get_rank() == 0:
        print("[REPRO2] Stage 2: Pass A (REF, cache empty)")
    sum_a, max_a = _forward_one(REF_SEQ, QKV_SEED)

    # 15 distractors
    if dist.get_rank() == 0:
        print(f"[REPRO2] Stage 3: {len(DISTRACTORS)} distractor forwards")
    for s in DISTRACTORS:
        _forward_one(s, qkv_seed=s)

    # Pass C
    if dist.get_rank() == 0:
        print("[REPRO2] Stage 4: Pass C (REF, cache has distractors)")
    sum_c, max_c = _forward_one(REF_SEQ, QKV_SEED)

    if dist.get_rank() == 0:
        print()
        print(f"[REPRO2] Pass A:  sum={sum_a:.6f}   abs_max={max_a:.6f}")
        print(f"[REPRO2] Pass C:  sum={sum_c:.6f}   abs_max={max_c:.6f}")
        diff = abs(sum_c - sum_a)
        rel = diff / (abs(sum_a) + 1e-9)
        print(f"[REPRO2] |A - C|     = {diff:.6f}")
        print(f"[REPRO2] |A - C|/|A| = {rel * 100:.4f}%")
        print()

        GREEN = "\033[0;32m"
        RED = "\033[0;31m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        if rel < 1e-3:
            print(f"  {GREEN}{BOLD}VERDICT: A ≈ C — no drift even with vLLM in the loop{RESET}")
        else:
            print(f"  {RED}{BOLD}VERDICT: DRIFT — Magi + vLLM is enough to reproduce{RESET}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
