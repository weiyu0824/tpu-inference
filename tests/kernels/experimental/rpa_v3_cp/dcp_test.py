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
"""Reference implementation and single-device simulation tests for DCP and PCP.

This file is the canonical reference showing how to drive the rpa_v3_cp kernel
for each context-parallelism mode.

DCP – Decode Context Parallel
==============================
The global KV cache is distributed round-robin across N devices.  Device k
owns global positions {k, k+N, k+2N, …}.

Single-device simulation:
  1. Build a global paged KV cache (cache-only, no new decode token).
  2. For each rank k: extract its round-robin token subset into a local paged
     cache and call cp_ragged_paged_attention with:
       - kv_cache_lens = global_cache_lens   (kernel derives local count)
       - cu_kv_lens    = zeros               (no new-KV; attend cache only)
       - q_start       = local_size(global_cache_lens, rank, N)
       - cp_rank       = rank, cp_group_size = N
       - return_lse    = True
  3. Merge partial (out_k, lse_k) pairs via online-softmax to reconstruct the
     full global attention output.
  4. Compare against cp_ref_ragged_paged_attention on the global cache.

PCP – Prefill Context Parallel
================================
Device k processes Q tokens [k*L/N : (k+1)*L/N].  K/V are all-gathered.

Single-device simulation:
  1. Compute PCP metadata: compute_pcp_prefill_metadata → local_cu_q,
     cu_kv_lens, q_start.
  2. Slice the global Q to the local portion.
  3. Call cp_ragged_paged_attention with full all-gathered K/V (simulated),
     kv_cache_lens (prior cache), cu_kv_lens, q_start, cp_group_size=1.
  4. Compare local output against the matching slice of the full reference.

Run:
    python -m pytest tests/kernels/experimental/rpa_v3_cp/dcp_test.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import (
    ragged_paged_attention as cp_ragged_paged_attention,
    ref_ragged_paged_attention as cp_ref_ragged_paged_attention,
)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention as baseline_ref,
)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from tpu_inference.layers.common.cp_utils import (
    compute_pcp_prefill_metadata,
    local_size,
)

jax.config.parse_flags_with_absl()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _gen(shape, dtype=jnp.bfloat16):
    return jnp.array(RNG.standard_normal(shape).astype(np.float32), dtype=dtype)


def _build_kv_cache(seq_lens_q_kv, num_kv_heads, head_dim, page_size, dtype,
                    num_pages, max_num_seq):
    """Build a paged KV cache for a list of (q_len, kv_len) sequences.

    Returns (cu_q_arr, kv_arr, kv_cache, page_indices) where kv_cache stores
    the generated tokens and page_indices is the flat lookup table.
    """
    kv_packing = get_dtype_packing(dtype)
    padded_hd = align_to(head_dim, 128)
    nkv2 = align_to(num_kv_heads * 2, kv_packing)

    cu_q = [0]
    kv_lens = []
    for q, kv in seq_lens_q_kv:
        cu_q.append(cu_q[-1] + q)
        kv_lens.append(kv)

    max_kv = max(kv_lens)
    ppseq = cdiv(max_kv, page_size)

    page_idx_list, kv_pages = [], []
    page_cnt = 0
    for kv_len in kv_lens:
        kv = _gen((kv_len, nkv2 // kv_packing, kv_packing, padded_hd), dtype)
        kv_padded = jnp.pad(
            kv,
            ((0, cdiv(kv_len, page_size) * page_size - kv_len), (0, 0),
             (0, 0), (0, 0)),
        ).reshape(-1, page_size, nkv2 // kv_packing, kv_packing, padded_hd)
        idxs = page_cnt + jnp.arange(kv_padded.shape[0], dtype=jnp.int32)
        idxs = jnp.pad(idxs, (0, ppseq - idxs.shape[0]))
        page_idx_list.append(idxs)
        kv_pages.append(kv_padded)
        page_cnt += kv_padded.shape[0]

    kv_cache = jnp.concatenate(kv_pages, axis=0)
    kv_cache = jnp.pad(kv_cache,
                       ((0, num_pages - kv_cache.shape[0]), (0, 0), (0, 0),
                        (0, 0), (0, 0)))
    page_indices = jnp.stack(page_idx_list)
    page_indices = jnp.pad(page_indices,
                           ((0, max_num_seq - page_indices.shape[0]), (0, 0)))
    page_indices = page_indices.reshape(-1)

    cu_q_arr = jnp.pad(jnp.array(cu_q, jnp.int32),
                       (0, max_num_seq + 1 - len(cu_q)))
    kv_arr = jnp.pad(jnp.array(kv_lens, jnp.int32),
                     (0, max_num_seq - len(kv_lens)))
    return cu_q_arr, kv_arr, kv_cache, page_indices


def _extract_global_tokens(kv_cache, page_indices, seq_idx, cache_len,
                           page_size, ppseq):
    """Read cache_len contiguous tokens for sequence seq_idx from a paged cache.

    Returns shape [cache_len, nkv2//kv_packing, kv_packing, head_dim].
    """
    idxs = page_indices.reshape(-1, ppseq)[seq_idx]
    pages_needed = cdiv(cache_len, page_size)
    pages = kv_cache[idxs[:pages_needed]]  # [pages_needed, page_size, ...]
    flat = pages.reshape(-1, *kv_cache.shape[2:])
    return flat[:cache_len]


def _build_local_cache_from_tokens(local_tokens_per_seq, local_lens, page_size,
                                   dtype, num_pages, max_num_seq):
    """Pack round-robin token slices into a local paged KV cache.

    Args:
      local_tokens_per_seq: list of arrays, each [local_cache_len_i, ...]
      local_lens: list of ints, local_cache_len_i
    Returns:
      (local_kv_cache, local_page_indices_flat)
    """
    max_local = max(local_lens)
    ppseq = cdiv(max_local, page_size)

    page_idx_list, kv_pages = [], []
    page_cnt = 0
    for local_tokens, local_len in zip(local_tokens_per_seq, local_lens):
        assert local_tokens.shape[0] == local_len
        n_pad = cdiv(local_len, page_size) * page_size - local_len
        kv_padded = jnp.pad(
            local_tokens, ((0, n_pad), (0, 0), (0, 0), (0, 0))
        ).reshape(-1, page_size, *local_tokens.shape[1:])
        idxs = page_cnt + jnp.arange(kv_padded.shape[0], dtype=jnp.int32)
        idxs = jnp.pad(idxs, (0, ppseq - idxs.shape[0]))
        page_idx_list.append(idxs)
        kv_pages.append(kv_padded)
        page_cnt += kv_padded.shape[0]

    kv_cache = jnp.concatenate(kv_pages, axis=0)
    kv_cache = jnp.pad(kv_cache,
                       ((0, num_pages - kv_cache.shape[0]), (0, 0), (0, 0),
                        (0, 0), (0, 0)))
    page_indices = jnp.stack(page_idx_list)
    page_indices = jnp.pad(page_indices,
                           ((0, max_num_seq - page_indices.shape[0]), (0, 0)))
    return kv_cache, page_indices.reshape(-1)


def _merge_partial_attn(outputs, lses, num_tokens):
    """Merge N partial Flash-Attention outputs via online-softmax.

    The kernel returns LSE already reshaped to (max_num_tokens, num_q_heads).

    Args:
      outputs: list of Array[T_padded, Q_heads, D], one per rank.
      lses:    list of Array[T_padded, Q_heads], one per rank.
      num_tokens: actual number of decode tokens (= num_seqs for q_len=1).

    Returns:
      merged output of shape [num_tokens, Q_heads, D].
    """
    # lse per rank: [T_padded, Q_heads].
    lse_stack = jnp.stack(lses, axis=0)   # [N, T_padded, Q_heads]
    out_stack = jnp.stack(outputs, axis=0) # [N, T_padded, Q_heads, D]

    # Slice to valid tokens.
    lse_stack = lse_stack[:, :num_tokens]  # [N, T, Q_heads]
    out_stack = out_stack[:, :num_tokens]  # [N, T, Q_heads, D]

    # Numerically-stable logsumexp across ranks.
    lse_max = jnp.max(lse_stack, axis=0, keepdims=True)   # [1, T, Q_heads]
    lse_total = lse_max[0] + jnp.log(
        jnp.sum(jnp.exp(lse_stack - lse_max), axis=0))    # [T, Q_heads]

    # Per-rank softmax weight: [N, T, Q_heads].
    weights = jnp.exp(lse_stack - lse_total[None])

    # Weighted sum across ranks → [T, Q_heads, D].
    merged = jnp.sum(weights[..., None] * out_stack, axis=0)
    return merged


# ---------------------------------------------------------------------------
# DCP decode tests
# ---------------------------------------------------------------------------


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class DcpDecodeSimulatedTest(jtu.JaxTestCase):
    """Simulate DCP decode on a single device and verify merged output.

    Setup:
      - L global cache tokens per sequence, no new decode token in this test
        (cu_kv_lens = zeros so kv_new_len = 0; Q still attends to all L tokens).
      - Each rank k holds tokens at global positions {k, k+N, k+2N, ...}.
      - Merge uses the standard FlashAttention online-softmax formula.

    Reference: cp_ref_ragged_paged_attention on the full global cache with
    kv_cache_lens = L and cu_kv_lens = zeros.
    """

    def _simulate_dcp_decode(self, cache_lens_per_seq, num_q_heads, num_kv_heads,
                             head_dim, page_size, dtype, cp_n):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Requires TPUv4+")

        num_seqs = len(cache_lens_per_seq)
        max_seq = max(align_to(num_seqs, 8), 8)
        num_pages = 512

        # Build global paged cache (q_len=1 is a placeholder; we don't use Q here).
        global_seqs = [(1, cl) for cl in cache_lens_per_seq]
        _, _, kv_cache_global, page_indices_global = _build_kv_cache(
            global_seqs, num_kv_heads, head_dim, page_size, dtype, num_pages,
            max_seq)

        # cu_q: 1 token per seq (decode).
        cu_q_np = np.arange(num_seqs + 1, dtype=np.int32)
        cu_q_jnp = jnp.pad(jnp.array(cu_q_np, jnp.int32),
                            (0, max_seq + 1 - len(cu_q_np)))

        kv_cache_lens_np = np.array(cache_lens_per_seq, dtype=np.int32)
        kv_cache_lens_jnp = jnp.pad(jnp.array(kv_cache_lens_np, jnp.int32),
                                    (0, max_seq - num_seqs))

        # cu_kv_lens = zeros: kv_new_len = 0 per sequence (attend cache only).
        cu_kv_zero = jnp.zeros(max_seq + 1, jnp.int32)

        # distribution: all decode.
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], jnp.int32)

        # Dummy Q and K/V (K/V not used because kv_new_len = 0).
        max_tok = max(align_to(num_seqs, 128), 512)
        q = _gen((max_tok, num_q_heads, head_dim), dtype)
        k_dummy = _gen((max_tok, num_kv_heads, head_dim), dtype)
        v_dummy = _gen((max_tok, num_kv_heads, head_dim), dtype)

        # Reference: full attention to all cache tokens on a single device.
        ref_out, _ = cp_ref_ragged_paged_attention(
            q, k_dummy, v_dummy, kv_cache_global,
            kv_cache_lens_jnp,
            page_indices_global, cu_q_jnp, distribution,
            cu_kv_lens=cu_kv_zero,
        )

        # Extract per-seq global tokens from the paged cache.
        ppseq = page_indices_global.shape[0] // max_seq
        global_tokens = [
            _extract_global_tokens(kv_cache_global, page_indices_global, i,
                                   int(cache_lens_per_seq[i]), page_size, ppseq)
            for i in range(num_seqs)
        ]

        # Simulate each DCP rank.
        partial_outputs, partial_lses = [], []
        for rank in range(cp_n):
            # Round-robin local tokens: positions {rank, rank+N, rank+2N, ...}.
            local_lens = [local_size(int(cl), rank, cp_n)
                          for cl in cache_lens_per_seq]
            local_tokens_per_seq = [
                global_tokens[i][rank::cp_n] for i in range(num_seqs)
            ]
            local_kv_cache, local_page_indices = _build_local_cache_from_tokens(
                local_tokens_per_seq, local_lens, page_size, dtype, num_pages,
                max_seq)

            # q_start: local position of Q = right after this rank's cache tokens.
            q_start_np = np.array(local_lens, dtype=np.int32)
            q_start_jnp = jnp.pad(jnp.array(q_start_np, jnp.int32),
                                  (0, max_seq - num_seqs))

            out_k, _, lse_k = cp_ragged_paged_attention(
                q, k_dummy, v_dummy,
                local_kv_cache,
                kv_cache_lens_jnp,
                local_page_indices,
                cu_q_jnp,
                distribution,
                cu_kv_lens=cu_kv_zero,
                q_start=q_start_jnp,
                cp_rank=jnp.array([rank], dtype=jnp.int32),
                cp_group_size=cp_n,
                return_lse=True,
            )

            partial_outputs.append(out_k)
            partial_lses.append(lse_k)

        merged_out = _merge_partial_attn(partial_outputs, partial_lses, num_seqs)

        tol = 0.3
        self.assertAllClose(merged_out, ref_out[:num_seqs], atol=tol, rtol=tol)

    @parameterized.product(
        cp_n=[2],
        dtype=[jnp.bfloat16],
        cache_lens=[
            [256],
            [128, 192],
            [64, 128, 256],
        ],
    )
    def test_dcp_decode_simulated(self, cp_n, dtype, cache_lens):
        self._simulate_dcp_decode(
            cache_lens,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=32,
            dtype=dtype,
            cp_n=cp_n,
        )


# ---------------------------------------------------------------------------
# PCP prefill tests
# ---------------------------------------------------------------------------


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PcpPrefillSimulatedTest(jtu.JaxTestCase):
    """Simulate PCP prefill on a single device and verify local output.

    Reference: baseline_ref on full global Q and K/V.

    Per-rank PCP call::

        local_cu_q, cu_kv_lens, q_start = compute_pcp_prefill_metadata(
            global_cu_q, kv_lens_total, rank, N)
        kv_cache_lens = kv_lens_total - global_q_lens   # prior cache only

        cp_ragged_paged_attention(
            q_local,          # local Q slice: tokens [rank*block : (rank+1)*block]
            k_global,         # all-gathered K (full L tokens; simulated here)
            v_global,         # all-gathered V
            kv_cache,         # shared paged KV cache (dcp==pcp layout)
            kv_cache_lens,    # prior cache only (explicit; not total kv_lens)
            page_indices,
            local_cu_q,       # cumulative local-Q lengths
            distribution,
            cu_kv_lens=cu_kv_lens,  # global new-KV span (full Q length after all-gather)
            q_start=q_start,        # global causal-mask offset per sequence
            cp_rank=jnp.array([rank], jnp.int32),
            cp_group_size=1,        # forces CP code path; =1 means no KV sharding
        )
    """

    def _simulate_pcp_rank(self, seq_lens_q_kv, num_q_heads, num_kv_heads,
                           head_dim, page_size, dtype, cp_n, cp_rank):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Requires TPUv4+")

        max_seq = max(align_to(len(seq_lens_q_kv), 8), 8)
        num_pages = 512

        global_cu_q = np.cumsum(
            [0] + [q for q, _ in seq_lens_q_kv]).astype(np.int32)
        global_kv_lens = np.array([kv for _, kv in seq_lens_q_kv],
                                  dtype=np.int32)
        # kv_cache_lens = prior cached tokens (total kv_len - new q_len).
        kv_cache_lens_np = (global_kv_lens -
                            (global_cu_q[1:] - global_cu_q[:-1])).astype(
                                np.int32)

        # Build paged cache sized for total kv_lens.
        cu_q_arr, kv_arr, kv_cache, page_indices = _build_kv_cache(
            seq_lens_q_kv, num_kv_heads, head_dim, page_size, dtype, num_pages,
            max_seq)

        total_global_q = int(global_cu_q[-1])
        max_tok = max(align_to(total_global_q, 128), 512)

        q_global = _gen((max_tok, num_q_heads, head_dim), dtype)
        k_global = _gen((max_tok, num_kv_heads, head_dim), dtype)
        v_global = _gen((max_tok, num_kv_heads, head_dim), dtype)
        distribution = jnp.array([0, 0, len(seq_lens_q_kv)], jnp.int32)

        # Ground-truth: full attention over all Q tokens.
        ref_out, _ = baseline_ref(q_global, k_global, v_global, kv_cache,
                                  kv_arr, page_indices, cu_q_arr, distribution)

        # Compute PCP metadata for this rank.
        local_cu_q, cu_kv_lens, q_start_np = compute_pcp_prefill_metadata(
            global_cu_q, global_kv_lens, cp_rank, cp_n)

        local_cu_q_jnp = jnp.pad(
            jnp.array(local_cu_q, jnp.int32),
            (0, max_seq + 1 - len(local_cu_q)))
        cu_kv_lens_jnp = jnp.pad(
            jnp.array(cu_kv_lens, jnp.int32),
            (0, max_seq + 1 - len(cu_kv_lens)))
        q_start_jnp = jnp.pad(
            jnp.array(q_start_np, jnp.int32),
            (0, max_seq - len(q_start_np)))
        kv_cache_lens_jnp = jnp.pad(
            jnp.array(kv_cache_lens_np, jnp.int32),
            (0, max_seq - len(kv_cache_lens_np)))

        # Local Q: contiguous block [rank*block : (rank+1)*block] per sequence.
        local_q_segs = []
        for i, (q_len, _) in enumerate(seq_lens_q_kv):
            local_q_len = q_len // cp_n
            global_start = int(global_cu_q[i]) + cp_rank * local_q_len
            local_q_segs.append(q_global[global_start:global_start + local_q_len])
        local_q_concat = jnp.concatenate(local_q_segs, axis=0)
        local_total_q = local_q_concat.shape[0]
        local_max_tok = max(align_to(local_total_q, 128), 512)
        q_local = jnp.pad(local_q_concat,
                          ((0, local_max_tok - local_total_q), (0, 0), (0, 0)))

        # K/V are the full all-gathered tensors (simulated: same global tensor).
        # cp_group_size=1 forces the CP code path without actual communication;
        # get_cp_local_size(x) = x when cp_group_size=1, so no KV sharding occurs.
        cp_out, _ = cp_ragged_paged_attention(
            q_local,
            k_global,
            v_global,
            kv_cache,
            kv_cache_lens_jnp,
            page_indices,
            local_cu_q_jnp,
            distribution,
            cu_kv_lens=cu_kv_lens_jnp,
            q_start=q_start_jnp,
            cp_rank=jnp.array([0], dtype=jnp.int32),
            cp_group_size=1,
        )

        # Compare local output against the matching reference slice.
        tol = 0.25
        ref_local_segs = []
        for i, (q_len, _) in enumerate(seq_lens_q_kv):
            local_q_len = q_len // cp_n
            global_start = int(global_cu_q[i]) + cp_rank * local_q_len
            ref_local_segs.append(ref_out[global_start:global_start + local_q_len])
        ref_local = jnp.concatenate(ref_local_segs, axis=0)

        self.assertAllClose(cp_out[:local_total_q], ref_local, atol=tol, rtol=tol)

    @parameterized.product(
        cp_rank=[0, 1],
        dtype=[jnp.bfloat16],
        seq_lens=[
            [(128, 256)],
            [(64, 128), (128, 256)],
        ],
    )
    def test_pcp_prefill_simulated(self, cp_rank, dtype, seq_lens):
        self._simulate_pcp_rank(
            seq_lens,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=32,
            dtype=dtype,
            cp_n=2,
            cp_rank=cp_rank,
        )


if __name__ == "__main__":
    absltest.main()
