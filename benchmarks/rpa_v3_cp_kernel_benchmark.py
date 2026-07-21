#!/usr/bin/env python3
"""
Kernel-level benchmark for rpa_v3_cp DCP optimizations.

Decode DCP path follows attention_interface.forward_with_dcp:
  Phase 1 (context): skip_current_attn=True, update_kv_cache=False
  Phase 2 (current): skip_cache_attn=True,   update_kv_cache=True
  (dcp_alltoall / merge_attn_states are network ops, omitted on single chip)

Prefill DCP path mirrors decode: context phase attends to existing KV cache
(no causal mask), current phase attends to the new chunk (causal mask).

Per-chip head counts match TP sharding (DCP shards context dim, not heads):
  q_heads  = NUM_Q_HEADS  / TP_SIZE
  kv_heads = NUM_KV_HEADS / TP_SIZE

Calibration note: isolated kernel benchmarks run ~2x slower than xprof for
cache-read-heavy ops (HBM cold-start in isolation vs warm pipeline in production).
DCP current (no cache reads) and compute-bound prefill are closer to production.

Usage:
  NEW_MODEL_DESIGN=1 python3 benchmarks/rpa_v3_cp_kernel_benchmark.py

Env overrides:
  PAGE_SIZE=16   CP_GROUP_SIZE=2   WARMUP=5   BENCH=200   SKIP_PREFILL=0
"""

import os, sys, subprocess, time
import jax, jax.numpy as jnp
import numpy as np
import math

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import (
    get_kv_cache_shape, ragged_paged_attention)
from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import (
    ragged_paged_attention as ragged_paged_attention_baseline)

# ── Config ────────────────────────────────────────────────────────────────────
PAGE_SIZE     = int(os.environ.get("PAGE_SIZE",     16))
CP_GROUP_SIZE = int(os.environ.get("CP_GROUP_SIZE",  2))  # DCP group size
TP_SIZE = 8

NUM_Q_HEADS   = 64
NUM_KV_HEADS  = 4
HEAD_DIM      = 128
KV_DTYPE      = jnp.bfloat16

WARMUP        = int(os.environ.get("WARMUP",         5))
BENCH         = int(os.environ.get("BENCH",        200))
SKIP_PREFILL  = bool(int(os.environ.get("SKIP_PREFILL", 0)))
SUBPROCESS    = bool(int(os.environ.get("SUBPROCESS",   1)))

BATCH_SIZE    = 1

DECODE_KV_LENS = [128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]
PREFILL_CHUNK  = 8 * 1024
PREFILL_CACHES = [0, 32 * 1024, 128 * 1024, 256 * 1024]


# ── Helpers ───────────────────────────────────────────────────────────────────

def cdiv(a, b):
    return (a + b - 1) // b


def make_base_inputs(global_kv_len, q_len, *, q_heads, kv_heads, cp_group_size=CP_GROUP_SIZE):
    """Positional + scalar arrays shared by both DCP phases and baseline."""
    if cp_group_size is None:
        local_kv_len = global_kv_len
    else:
        local_kv_len = cdiv(global_kv_len, cp_group_size)

    pages_per_seq = cdiv(local_kv_len + q_len, PAGE_SIZE)
    total_pages   = BATCH_SIZE * pages_per_seq

    kv_cache = jnp.zeros(
        get_kv_cache_shape(total_pages, PAGE_SIZE, kv_heads, HEAD_DIM, KV_DTYPE),
        dtype=KV_DTYPE)
    q = jnp.zeros((q_len, q_heads, HEAD_DIM), dtype=KV_DTYPE)
    k = jnp.zeros((q_len, kv_heads, HEAD_DIM), dtype=KV_DTYPE)
    v = jnp.zeros((q_len, kv_heads, HEAD_DIM), dtype=KV_DTYPE)

    kv_lens      = jnp.array([global_kv_len + q_len], dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens    = jnp.array([0, q_len], dtype=jnp.int32)

    return dict(
        queries=q, keys=k, values=v, kv_cache=kv_cache,
        kv_lens=kv_lens, page_indices=page_indices,
        cu_q_lens=cu_q_lens,
    )



def _time_step(step_fn, init_kv):
    """Measure per-kernel time using fori_loop (all iterations in one XLA dispatch)."""
    # Warmup: JIT-compile step_fn first.
    kv = init_kv
    for _ in range(WARMUP):
        kv = step_fn(kv)
    jax.block_until_ready(kv)

    # Compile a fori_loop: BENCH iterations run entirely on-device without
    # host round-trips between kernels, matching production execution pattern.
    bench_fn = jax.jit(
        lambda kv: jax.lax.fori_loop(0, BENCH, lambda _, kv: step_fn(kv), kv))

    # Warmup the fori_loop compilation.
    kv = bench_fn(kv)
    jax.block_until_ready(kv)

    # Timed run.
    t0 = time.perf_counter()
    kv = bench_fn(kv)
    jax.block_until_ready(kv)
    mean_ms = (time.perf_counter() - t0) * 1e3 / BENCH
    return mean_ms, kv


def bench_label(label, mean_ms, width=62):
    print(f"  {label:<{width}s}  {mean_ms*1e3:7.1f} us  ({mean_ms:.4f} ms)", flush=True)


def bench_dcp_decode(global_kv_len):
    kv_k = global_kv_len // 1024
    dist_decode = jnp.array([BATCH_SIZE, BATCH_SIZE, BATCH_SIZE], dtype=jnp.int32)
    cp_rank = jnp.array([0], dtype=jnp.int32)

    # DCP shards the KV cache context dimension, not Q heads.
    # Per-chip Q/KV head count is determined by TP sharding only.
    q_heads_per_chip  = math.ceil(NUM_Q_HEADS  / TP_SIZE)  # 8
    kv_heads_per_chip = math.ceil(NUM_KV_HEADS / TP_SIZE)  # 1
    base = make_base_inputs(global_kv_len, 1, q_heads=q_heads_per_chip, kv_heads=kv_heads_per_chip, cp_group_size=None)
    base_dcp = make_base_inputs(global_kv_len, 1, q_heads=q_heads_per_chip, kv_heads=kv_heads_per_chip, cp_group_size=CP_GROUP_SIZE)

    # ── Baseline: no-DCP, single call ────────────────────────────────────────
    def step_nodcp(kv_cache):
        attn_out, new_kv = ragged_paged_attention_baseline(
            base["queries"], base["keys"], base["values"], kv_cache,
            base["kv_lens"], base["page_indices"], base["cu_q_lens"],
            dist_decode,
            update_kv_cache=False,
        )
        return new_kv

    mean, _ = _time_step(step_nodcp, base["kv_cache"])
    bench_label(f"no-DCP  decode cache={kv_k:>4}k  q=1  [baseline]", mean)
    r_nodcp = mean

    # ── DCP: two-phase (follows forward_with_dcp) ─────────────────────────────
    # Phase 1 — context attention: attend to cache, do NOT update cache
    # Phase 2 — query attention  : attend to new tokens, DO update cache
    def step_context(kv_cache):
        _, phase1_kv, _ = ragged_paged_attention(
            base_dcp["queries"], base_dcp["keys"], base_dcp["values"], kv_cache,
            base_dcp["kv_lens"], base_dcp["page_indices"], base_dcp["cu_q_lens"],
            dist_decode,
            cp_rank=cp_rank, cp_group_size=CP_GROUP_SIZE,
            skip_current_attn=True, update_kv_cache=False, return_lse=True,
        )
        return phase1_kv

    def step_current(kv_cache):
        _, new_kv, _ = ragged_paged_attention(
            base_dcp["queries"], base_dcp["keys"], base_dcp["values"], kv_cache,
            base_dcp["kv_lens"], base_dcp["page_indices"], base_dcp["cu_q_lens"],
            dist_decode,
            cp_rank=cp_rank, cp_group_size=CP_GROUP_SIZE,
            skip_cache_attn=True, update_kv_cache=False, return_lse=True,
            d_block_sizes=(1, PAGE_SIZE, 1, PAGE_SIZE)
        )
        return new_kv

    mean_ctx, phase1_kv = _time_step(step_context, base_dcp["kv_cache"])
    bench_label(f"DCP     decode cache={kv_k:>4}k  q=1  [context]", mean_ctx)

    mean_cur, _ = _time_step(step_current, phase1_kv)
    bench_label(f"DCP     decode cache={kv_k:>4}k  q=1  [current]", mean_cur)

    mean_combine = mean_ctx + mean_cur
    bench_label(f"DCP     decode cache={kv_k:>4}k  q=1  [combine=ctx+cur]", mean_combine)

    print(flush=True)


def bench_dcp_prefill(existing_cache):
    cache_k = existing_cache // 1024
    q_k     = PREFILL_CHUNK // 1024
    cp_rank = jnp.array([0], dtype=jnp.int32)
    dist_mixed = jnp.array([0, 0, BATCH_SIZE], dtype=jnp.int32)

    q_heads_per_chip  = math.ceil(NUM_Q_HEADS  / TP_SIZE)
    kv_heads_per_chip = math.ceil(NUM_KV_HEADS / TP_SIZE)
    base     = make_base_inputs(existing_cache, PREFILL_CHUNK,
                               q_heads=q_heads_per_chip, kv_heads=kv_heads_per_chip,
                               cp_group_size=None)
    base_dcp = make_base_inputs(existing_cache, PREFILL_CHUNK,
                               q_heads=q_heads_per_chip, kv_heads=kv_heads_per_chip)

    def step_nodcp(kv_cache):
        _, new_kv = ragged_paged_attention_baseline(
            base["queries"], base["keys"], base["values"], kv_cache,
            base["kv_lens"], base["page_indices"], base["cu_q_lens"],
            dist_mixed,
            update_kv_cache=False,
        )
        return new_kv

    # Phase 1: attend to KV cache only (no causal mask needed — cache is all before chunk)
    def step_context(kv_cache):
        _, phase1_kv, _ = ragged_paged_attention(
            base_dcp["queries"], base_dcp["keys"], base_dcp["values"], kv_cache,
            base_dcp["kv_lens"], base_dcp["page_indices"], base_dcp["cu_q_lens"],
            dist_mixed,
            cp_rank=cp_rank, cp_group_size=CP_GROUP_SIZE,
            skip_current_attn=True, update_kv_cache=False, return_lse=True,
            use_causal_mask=False,
        )
        return phase1_kv

    # Phase 2: attend to current chunk tokens only (causal mask within chunk)
    def step_current(kv_cache):
        _, new_kv, _ = ragged_paged_attention(
            base_dcp["queries"], base_dcp["keys"], base_dcp["values"], kv_cache,
            base_dcp["kv_lens"], base_dcp["page_indices"], base_dcp["cu_q_lens"],
            dist_mixed,
            cp_rank=cp_rank, cp_group_size=CP_GROUP_SIZE,
            skip_cache_attn=True, update_kv_cache=True, return_lse=True,
            
        )
        return new_kv

    mean_nd, _ = _time_step(step_nodcp, base["kv_cache"])
    bench_label(f"no-DCP  prefill  cache={cache_k:>4}k  q={q_k}k  [baseline]", mean_nd)

    mean_ctx, phase1_kv = _time_step(step_context, base_dcp["kv_cache"])
    bench_label(f"DCP     prefill  cache={cache_k:>4}k  q={q_k}k  [context]", mean_ctx)

    mean_cur, _ = _time_step(step_current, phase1_kv)
    bench_label(f"DCP     prefill  cache={cache_k:>4}k  q={q_k}k  [current]", mean_cur)

    mean_combine = mean_ctx + mean_cur
    bench_label(f"DCP     prefill  cache={cache_k:>4}k  q={q_k}k  [combine=ctx+cur]", mean_combine)

    print(flush=True)



def main():

    print("=" * 72, flush=True)
    print("  rpa_v3_cp kernel benchmark  — async LSE DMA, two-phase DCP decode")
    print("=" * 72, flush=True)
    print(f"  Q={NUM_Q_HEADS} KV={NUM_KV_HEADS} head_dim={HEAD_DIM} "
          f"page_size={PAGE_SIZE} DCP_GROUP_SIZE={CP_GROUP_SIZE} "
          f"warmup={WARMUP} bench={BENCH}", flush=True)
    print(flush=True)

    print("━" * 72, flush=True)
    print("  DECODE  (two-phase DCP: context + query)")
    print("━" * 72, flush=True)
    for cache_len in DECODE_KV_LENS:
        bench_dcp_decode(cache_len)

    # print("━" * 72, flush=True)
    # print(f"  PREFILL  (MIXED, DCP, chunk q={PREFILL_CHUNK//1024}k)")
    # print("━" * 72, flush=True)
    # for cache_len in PREFILL_CACHES:
    #     bench_dcp_prefill(cache_len)


if __name__ == "__main__":
    main()
