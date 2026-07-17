# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pallas kernel for writing decode tokens into a paged KV cache.

Phase A of the DCP decode forward pass: each DCP device writes the new decode
token for every sequence it owns (write_pos % cp_group_size == cp_rank) into
the correct page slot of the local KV cache shard.  All I/O is via DMA so no
HBM-wide data formatting is triggered.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _write_decode_kv_kernel(
    # ── Scalar prefetch (loaded into SMEM before kernel starts) ─────────────
    kv_lens_ref,          # i32[max_num_seqs]
    page_indices_ref,     # i32[max_num_seqs * pages_per_seq]
    cu_q_lens_ref,        # i32[max_num_seqs + 1]
    distribution_ref,     # i32[3]
    cp_rank_ref,          # i32[1]
    # ── HBM arrays (full-tensor refs, accessed via async DMA) ───────────────
    merged_kv_hbm_ref,    # bf16[max_num_tokens, nh_x2//kp, kp, head_dim]
    kv_cache_hbm_ref,     # bf16[local_pages, local_page_size, nh_x2//kp, kp, hd]
    updated_kv_cache_hbm_ref,  # same shape, aliased with kv_cache (in-place)
    # ── VMEM scratch ────────────────────────────────────────────────────────
    kv_slot_ref,          # VMEM bf16[1, nh_x2//kp, kp, head_dim]  staging buf
    sem,                  # DMA semaphore [1]
    *,
    # ── Static params ───────────────────────────────────────────────────────
    cp_group_size: int,
    pages_per_seq: int,
):
    # grid dim 0 == seq_idx; one invocation per (padded) sequence.
    # @pl.when is at the TOP LEVEL of the kernel body — not inside a fori_loop.
    # Nesting @pl.when + DMA inside lax.fori_loop causes Pallas to track DMA
    # semaphore effects as extra while_loop carry, producing a pytree mismatch
    # when the false branch has no matching DMA effects.
    seq_idx = pl.program_id(0)

    local_page_size = kv_cache_hbm_ref.shape[1]
    cp_rank = cp_rank_ref[0]

    # Flatten [local_pages, local_page_size, ...] → [local_pages*local_page_size, ...]
    # Bitcast inside Pallas (no copy), enabling flat DMA addressing.
    cs = updated_kv_cache_hbm_ref.shape
    cache_flat = updated_kv_cache_hbm_ref.reshape(cs[0] * cs[1], *cs[2:])

    kv_len = kv_lens_ref[seq_idx]
    # kv_lens[i] includes the new decode token → write at kv_len - 1.
    write_pos = kv_len - 1

    # Guard: only real decode seqs (seq_idx < num_decode) owned by this DCP rank.
    @pl.when((seq_idx < distribution_ref[0]) & (write_pos % cp_group_size == cp_rank))
    def do_write():
        local_pos = (write_pos + cp_group_size - 1 - cp_rank) // cp_group_size
        kv_p   = local_pos // local_page_size
        kv_off = local_pos % local_page_size

        phys_page = page_indices_ref[seq_idx * pages_per_seq + kv_p]
        token_i   = cu_q_lens_ref[seq_idx]  # first (only) token for this seq

        # Step 1: DMA load merged_kv[token_i] from HBM → VMEM staging buf.
        pltpu.make_async_copy(
            merged_kv_hbm_ref.at[pl.ds(token_i, 1)],
            kv_slot_ref,
            sem.at[0],
        ).start()
        pltpu.make_async_copy(
            merged_kv_hbm_ref.at[pl.ds(token_i, 1)],
            kv_slot_ref,
            sem.at[0],
        ).wait()

        # Step 2: DMA store VMEM staging buf → cache HBM at (phys_page, kv_off).
        flat_idx = phys_page * local_page_size + kv_off
        pltpu.make_async_copy(
            kv_slot_ref,
            cache_flat.at[pl.ds(flat_idx, 1)],
            sem.at[0],
        ).start()
        pltpu.make_async_copy(
            kv_slot_ref,
            cache_flat.at[pl.ds(flat_idx, 1)],
            sem.at[0],
        ).wait()

@jax.jit(
    static_argnames=("cp_group_size"), donate_argnames="kv_cache",
)
def write_decode_kv(
    merged_kv: jax.Array,      # [max_num_tokens, nh_x2//kp, kp, head_dim]
    kv_cache: jax.Array,       # [local_pages, local_page_size, nh_x2//kp, kp, hd]
    kv_lens: jax.Array,        # i32[max_num_seqs]
    page_indices: jax.Array,   # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,      # i32[max_num_seqs + 1]
    distribution: jax.Array,   # i32[3]
    cp_rank: jax.Array,        # i32[1]
    *,
    cp_group_size: int,
) -> jax.Array:
    """Write one decode token per sequence into the local KV cache shard.

    Designed to run inside ``jax.shard_map``; all inputs are already local
    (per-device) shards.  Uses Pallas DMA for truly in-place writes — no
    HBM-wide data-formatting op is emitted.

    Args:
        merged_kv:    Pre-merged K+V tensor (output of ``merge_kv``).
        kv_cache:     Local KV cache shard (CONTEXT-sharded along page_size).
        kv_lens:      Per-sequence total token lengths *including* the new token.
        page_indices: Physical page mapping for each sequence.
        cu_q_lens:    Cumulative query start positions (maps seq → token index).
        distribution: [decode_end, prefill_end, total].
        cp_rank:      This device's DCP rank, shape i32[1].
        cp_group_size: DCP group size (static).

    Returns:
        Updated kv_cache with the new decode tokens written in-place.
    """
    max_num_seqs  = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    _, _, num_kv_heads_x2_per_kp, kv_packing, head_dim = kv_cache.shape

    kv_slot_scratch = pltpu.VMEM(
        (1, num_kv_heads_x2_per_kp, kv_packing, head_dim),
        kv_cache.dtype,
    )
    sem_scratch = pltpu.SemaphoreType.DMA((1,))

    scalar_prefetches = (kv_lens, page_indices, cu_q_lens, distribution, cp_rank)
    num_scalar_prefetch = len(scalar_prefetches)

    out_shape = jax.ShapeDtypeStruct(shape=kv_cache.shape, dtype=kv_cache.dtype)

    kernel = pl.pallas_call(
        functools.partial(
            _write_decode_kv_kernel,
            cp_group_size=cp_group_size,
            pages_per_seq=pages_per_seq,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # merged_kv
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),  # updated_kv_cache
            # One grid cell per (padded) sequence; @pl.when guards non-owners.
            # "arbitrary" ensures sequential execution → single shared VMEM scratch.
            grid=(max_num_seqs,),
            scratch_shapes=[kv_slot_scratch, sem_scratch],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        out_shape=out_shape,
        # kv_cache (input index num_scalar_prefetch + 1) aliased to output 0.
        input_output_aliases={num_scalar_prefetch + 1: 0},
        name="write_decode_kv",
    )

    return kernel(*scalar_prefetches, merged_kv, kv_cache)
