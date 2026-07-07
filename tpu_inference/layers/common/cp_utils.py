# Copyright 2025 Google LLC
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
"""Utilities for Context Parallelism (CP) in attention.

Covers both DCP (Decode Context Parallel) and PCP (Prefill Context Parallel).

Terminology
-----------
DCP  – Decode Context Parallel: split the KV *cache* across devices so that
       each rank stores 1/N of the cached tokens.  The decode step attends
       to its local KV shard; ranks then exchange partial (output, LSE) pairs
       and merge.

PCP  – Prefill Context Parallel: split the *query sequence* across devices
       during prefill so that rank k processes tokens
       [k*L/N : (k+1)*L/N].  Because we set dcp == pcp, the KV cache
       sharding layout is reused: each device writes its own token block to
       the cache and, during decode, DCP gathers them back.

Kernel metadata
---------------
The RPA kernel is made agnostic to CP by accepting three optional arrays
that the caller (this module) prepares:

  cu_q_lens        – cumulative Q lengths inside the *local* Q buffer
                     (existing param; for PCP this covers only 1/N of tokens)
  cu_kv_lens       – cumulative new-KV lengths inside kv_hbm_ref
                     (new param; for PCP after all-gather this spans all L tokens)
  q_global_offsets – per-sequence global start position of Q in the full KV
                     sequence; used to compute the causal-mask boundary
                     (new param; replaces the kv_q_gap derivation inside the kernel)
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def local_size(total: int, rank: int, group_size: int) -> int:
    """Number of tokens owned by *rank* when *total* tokens are distributed
    round-robin across *group_size* devices.

    Token i goes to device i % group_size, so device k gets ceil((total-k)/N)
    tokens.  This equals (total + group_size - 1 - rank) // group_size.
    """
    return (total + group_size - 1 - rank) // group_size


def local_size_jnp(total: jax.Array, rank: int, group_size: int) -> jax.Array:
    """JAX version of local_size for dynamic total values."""
    return (total + group_size - 1 - rank) // group_size


def block_local_size(total: int, rank: int, group_size: int) -> int:
    """Number of tokens owned by *rank* when *total* tokens are split into
    contiguous blocks of size total // group_size.

    Device k owns tokens [k * block : (k+1) * block].
    """
    block = total // group_size
    start = rank * block
    end = (rank + 1) * block if rank < group_size - 1 else total
    return end - start


# ---------------------------------------------------------------------------
# PCP metadata computation
# ---------------------------------------------------------------------------

def compute_pcp_prefill_metadata(
    cu_q_lens: np.ndarray,      # i32[max_num_seqs + 1], global Q cumulative lens
    kv_lens: np.ndarray,        # i32[max_num_seqs], total KV length per seq
    cp_rank: int,
    cp_group_size: int,
    kv_cache_global_lens: Optional[np.ndarray] = None,  # i32[max_num_seqs]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute kernel metadata arrays for PCP prefill.

    After this call the caller should:
      1. All-gather K and V across the CP group so that kv_hbm_ref holds all L
         new tokens.
      2. Pass the returned arrays to ragged_paged_attention as cu_q_lens,
         cu_kv_lens, and q_global_offsets.

    Args:
      cu_q_lens: Cumulative Q lengths for the *full* batch (not yet split by
        CP rank).  Shape (max_num_seqs + 1,).
      kv_lens: Total KV length (cache + new) per sequence.
      cp_rank: This device's rank in the CP group (0-based).
      cp_group_size: Number of devices in the CP group.
      kv_cache_global_lens: Prior KV-cache token counts per sequence (global,
        i.e. the portion already committed to the paged cache before this
        prefill step).  If None, assumed to be zeros (first prefill turn).

    Returns:
      local_cu_q_lens: i32[max_num_seqs + 1] — cumulative Q lengths in the
        *local* Q buffer on this device.  Used for Q fetching in the kernel.
      cu_kv_lens: i32[max_num_seqs + 1] — cumulative new-KV lengths in the
        all-gathered kv_hbm_ref.  Each sequence occupies its full Q length
        (not split), because kv_hbm_ref holds all tokens after all-gather.
      q_global_offsets: i32[max_num_seqs] — global sequence position of the
        first Q token on this device, per sequence.  Used by the kernel for
        the causal-mask boundary (kv_q_gap).
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    q_lens = (cu_q_lens[1:] - cu_q_lens[:-1]).astype(np.int32)  # full Q lens

    if kv_cache_global_lens is None:
        # kv_lens[i] = prior_cache_global[i] + q_lens[i], so derive cache length.
        kv_cache_global_lens = (kv_lens - q_lens).astype(np.int32)

    # --- local_cu_q_lens ---
    # Each device processes a contiguous block of tokens.
    # Device k handles tokens [k * block : (k+1) * block] where block = q_len // N.
    local_q_lens = np.array(
        [block_local_size(int(q_lens[i]), cp_rank, cp_group_size)
         for i in range(max_num_seqs)],
        dtype=np.int32,
    )
    local_cu_q_lens = np.concatenate(
        [np.zeros(1, dtype=np.int32), np.cumsum(local_q_lens)]
    )

    # --- cu_kv_lens ---
    # After all-gather, kv_hbm_ref holds *all* new tokens for every sequence.
    # The range for sequence i is [cu_kv_lens[i] : cu_kv_lens[i+1]] = full q_len.
    cu_kv_lens = cu_q_lens.copy()  # same structure as original cu_q_lens

    # --- q_global_offsets ---
    # The global KV-sequence position of the first Q token on this device.
    # = (prior cache for this seq) + (tokens before this device's block).
    local_q_starts = np.array(
        [cp_rank * (int(q_lens[i]) // cp_group_size) for i in range(max_num_seqs)],
        dtype=np.int32,
    )
    q_global_offsets = (kv_cache_global_lens + local_q_starts).astype(np.int32)

    return local_cu_q_lens, cu_kv_lens, q_global_offsets


def compute_dcp_decode_metadata(
    cu_q_lens: np.ndarray,           # i32[max_num_seqs + 1]
    kv_lens_global: np.ndarray,      # i32[max_num_seqs], total (global) KV lengths
    cp_rank: int,
    cp_group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute kernel metadata for DCP decode (existing behavior, made explicit).

    For decode each sequence contributes exactly one new token, so cu_q_lens
    and cu_kv_lens are identical.  The global Q start per sequence equals the
    local cache length seen by this device.

    Returns the same triple (local_cu_q_lens, cu_kv_lens, q_global_offsets)
    as compute_pcp_prefill_metadata so callers can use a unified interface.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1

    # For decode: new KV = 1 token per seq, same as Q.
    local_cu_q_lens = cu_q_lens.copy()
    cu_kv_lens = cu_q_lens.copy()

    # Global KV-cache lengths seen by this device (round-robin distributed).
    kv_cache_global = kv_lens_global - 1  # subtract the 1 new decode token
    q_global_offsets = np.array(
        [local_size_jnp(int(kv_cache_global[i]), cp_rank, cp_group_size)
         for i in range(max_num_seqs)],
        dtype=np.int32,
    )

    return local_cu_q_lens, cu_kv_lens, q_global_offsets


# ---------------------------------------------------------------------------
# KV all-gather for PCP prefill
# ---------------------------------------------------------------------------

def all_gather_kv_for_prefill(
    keys: jax.Array,    # [local_tokens, num_kv_heads, head_dim]
    values: jax.Array,  # [local_tokens, num_kv_heads, head_dim]
    axis_name: str = "dcp",
) -> tuple[jax.Array, jax.Array]:
    """All-gather K and V across the CP group along the token dimension.

    Each device starts with its local slice of shape [L/N, H, D] and after
    the gather every device has shape [L, H, D].  Devices are concatenated in
    rank order so device k's tokens occupy rows [k*L/N : (k+1)*L/N].

    Args:
      keys: Local K tensor, shape [local_tokens, num_kv_heads, head_dim].
      values: Local V tensor, same shape.
      axis_name: Name of the JAX mesh axis corresponding to the CP group
        (typically "dcp" since PCP reuses the DCP mesh axis).

    Returns:
      Gathered K and V tensors, each of shape [total_tokens, num_kv_heads, head_dim].
    """
    # tiled=True concatenates shards along axis=0 rather than adding a new dim.
    gathered_k = jax.lax.all_gather(keys, axis_name=axis_name, axis=0, tiled=True)
    gathered_v = jax.lax.all_gather(values, axis_name=axis_name, axis=0, tiled=True)
    return gathered_k, gathered_v


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

def validate_pcp_metadata(
    local_cu_q_lens: np.ndarray,
    cu_kv_lens: np.ndarray,
    q_global_offsets: np.ndarray,
    cp_rank: int,
    cp_group_size: int,
) -> None:
    """Basic shape/value sanity checks for PCP metadata arrays."""
    assert local_cu_q_lens.ndim == 1
    assert cu_kv_lens.ndim == 1
    assert q_global_offsets.ndim == 1

    max_num_seqs = q_global_offsets.shape[0]
    assert local_cu_q_lens.shape == (max_num_seqs + 1,)
    assert cu_kv_lens.shape == (max_num_seqs + 1,)

    assert local_cu_q_lens[0] == 0, "local_cu_q_lens must start at 0"
    assert cu_kv_lens[0] == 0, "cu_kv_lens must start at 0"

    for i in range(max_num_seqs):
        local_q = int(local_cu_q_lens[i + 1] - local_cu_q_lens[i])
        kv_new = int(cu_kv_lens[i + 1] - cu_kv_lens[i])
        assert local_q >= 0, f"seq {i}: local_q_len={local_q} must be >= 0"
        assert kv_new >= local_q, (
            f"seq {i}: cu_kv_lens span ({kv_new}) must be >= local_q_len ({local_q})"
        )
        assert 0 <= int(q_global_offsets[i]), (
            f"seq {i}: q_global_offsets={q_global_offsets[i]} must be >= 0"
        )

    assert 0 <= cp_rank < cp_group_size, (
        f"cp_rank={cp_rank} must be in [0, {cp_group_size})"
    )
