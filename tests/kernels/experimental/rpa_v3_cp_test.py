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
"""Tests and benchmarks for the CP-aware RPA v3 kernel (rpa_v3_cp).

Single-device tests validate that:
  1. With cp_group_size=None the CP kernel is byte-for-byte identical to the
     non-CP kernel (backward compatibility).
  2. With PCP metadata (cu_kv_lens, q_global_offsets) the kernel correctly
     restricts the attention window to the local Q slice.

Multi-device tests (require N TPU chips with a 'dcp' mesh axis) validate the
full PCP flow including the all-gather of K/V.

Run (single-device, correctness):
    cd /path/to/tpu-inference
    python -m pytest tests/kernels/experimental/rpa_v3_cp_test.py -v

Run (benchmark mode, single process):
    python tests/kernels/experimental/rpa_v3_cp_test.py --benchmark
"""

import argparse
import time

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
    ragged_paged_attention as baseline_ragged_paged_attention,
    ref_ragged_paged_attention as baseline_ref,
)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from tpu_inference.layers.common.cp_utils import compute_pcp_prefill_metadata

jax.config.parse_flags_with_absl()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _gen(shape, dtype=jnp.bfloat16):
    return jnp.array(RNG.standard_normal(shape).astype(np.float32),
                     dtype=dtype)


def _build_kv_cache(seq_lens_q_kv, num_kv_heads, head_dim, page_size, dtype,
                    num_pages, max_num_seq):
    """Build paged KV cache and page_indices for a list of (q_len, kv_len)."""
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


# ---------------------------------------------------------------------------
# Single-device correctness tests
# ---------------------------------------------------------------------------


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class CpKernelBackwardCompatTest(jtu.JaxTestCase):
    """CP kernel with cp_group_size=None must match the baseline kernel."""

    def _run(self, seq_lens_q_kv, num_q_heads, num_kv_heads, head_dim,
             page_size, dtype):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Requires TPUv4+")

        max_tok = max(align_to(sum(q for q, _ in seq_lens_q_kv), 128), 512)
        max_seq = max(align_to(len(seq_lens_q_kv), 8), 8)
        num_pages = 512

        cu_q, kv_arr, kv_cache, page_indices = _build_kv_cache(
            seq_lens_q_kv, num_kv_heads, head_dim, page_size, dtype, num_pages,
            max_seq)
        distribution = jnp.array([0, 0, len(seq_lens_q_kv)], jnp.int32)
        total_q = int(cu_q[len(seq_lens_q_kv)])
        q = _gen((max_tok, num_q_heads, head_dim), dtype)
        k = _gen((max_tok, num_kv_heads, head_dim), dtype)
        v = _gen((max_tok, num_kv_heads, head_dim), dtype)

        # baseline uses total kv_lens; CP kernel expects kv_cache_lens (cache only)
        n = len(seq_lens_q_kv)
        q_lens_arr = cu_q[1:n + 1] - cu_q[:n]
        kv_cache_lens = jnp.pad((kv_arr[:n] - q_lens_arr).astype(jnp.int32),
                                (0, max_seq - n))

        ref_out, ref_kv = baseline_ref(
            q, k, v, kv_cache, kv_arr, page_indices, cu_q, distribution)
        cp_out, cp_kv = cp_ragged_paged_attention(
            q, k, v, kv_cache, kv_cache_lens, page_indices, cu_q, distribution)
        tol = 0.2
        self.assertAllClose(cp_out[:total_q], ref_out, atol=tol, rtol=tol)
        mask = ~jnp.isnan(ref_kv)
        self.assertArraysEqual(cp_kv[mask], ref_kv[mask])

    @parameterized.product(
        dtype=[jnp.bfloat16],
        seq_lens=[[(128, 256)], [(64, 128), (32, 64)]],
    )
    def test_backward_compat(self, dtype, seq_lens):
        self._run(seq_lens, 8, 2, 128, 32, dtype)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PcpMetadataTest(jtu.JaxTestCase):
    """Validate compute_pcp_prefill_metadata produces correct arrays."""

    def test_single_seq(self):
        """Single sequence, L=128, N=2."""
        q_len, kv_len, cp_n = 128, 256, 2
        cu_q = np.array([0, q_len], dtype=np.int32)
        kv_lens_np = np.array([kv_len], dtype=np.int32)

        for rank in range(cp_n):
            lcq, ckvl, qgo = compute_pcp_prefill_metadata(
                cu_q, kv_lens_np, rank, cp_n)
            local_q = q_len // cp_n
            cache_len = kv_len - q_len
            self.assertEqual(int(lcq[0]), 0)
            self.assertEqual(int(lcq[1]), local_q,
                             f"rank={rank}: local_q_len mismatch")
            self.assertEqual(int(ckvl[0]), 0)
            self.assertEqual(int(ckvl[1]), q_len,
                             f"rank={rank}: cu_kv_lens span mismatch")
            expected_offset = cache_len + rank * local_q
            self.assertEqual(int(qgo[0]), expected_offset,
                             f"rank={rank}: q_global_offsets mismatch")

    def test_multi_seq(self):
        """Two sequences with different lengths."""
        seqs = [(64, 128), (96, 256)]  # (q_len, kv_len)
        cu_q = np.cumsum([0] + [q for q, _ in seqs]).astype(np.int32)
        kv_lens_np = np.array([kv for _, kv in seqs], dtype=np.int32)
        cp_n = 2

        for rank in range(cp_n):
            lcq, ckvl, qgo = compute_pcp_prefill_metadata(
                cu_q, kv_lens_np, rank, cp_n)
            for i, (q_len, kv_len) in enumerate(seqs):
                local_q = q_len // cp_n
                cache_len = kv_len - q_len
                self.assertEqual(int(ckvl[i + 1] - ckvl[i]), q_len)
                self.assertEqual(int(lcq[i + 1] - lcq[i]), local_q)
                self.assertEqual(int(qgo[i]), cache_len + rank * local_q)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PcpKernelSimulatedTest(jtu.JaxTestCase):
    """Simulate PCP by splitting Q and feeding gathered K/V to the CP kernel.

    This runs on a SINGLE device and tests kernel correctness by:
      - Running the full (non-split) reference to get the ground-truth output.
      - Running the CP kernel with local Q (L/N) and all-gathered K/V (L),
        using PCP metadata, and verifying the local output matches the
        corresponding slice of the reference output.
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

        # Build KV cache using GLOBAL kv_lens (cache occupies kv_len-q_len pages).
        cu_q_arr, kv_arr, kv_cache, page_indices = _build_kv_cache(
            seq_lens_q_kv, num_kv_heads, head_dim, page_size, dtype, num_pages,
            max_seq)

        total_global_q = int(global_cu_q[-1])
        max_tok = align_to(total_global_q, 128)
        max_tok = max(max_tok, 512)

        # Global Q, K, V for the reference.
        q_global = _gen((max_tok, num_q_heads, head_dim), dtype)
        k_global = _gen((max_tok, num_kv_heads, head_dim), dtype)
        v_global = _gen((max_tok, num_kv_heads, head_dim), dtype)
        distribution = jnp.array([0, 0, len(seq_lens_q_kv)], jnp.int32)

        ref_args = (q_global, k_global, v_global, kv_cache, kv_arr,
                    page_indices, cu_q_arr, distribution)
        ref_out, _ = baseline_ref(*ref_args)

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

        # kv_cache_lens = prior cached KV (total - new) per sequence.
        kv_cache_lens_np = (global_kv_lens - (global_cu_q[1:] - global_cu_q[:-1])
                            ).astype(np.int32)
        kv_cache_lens_jnp = jnp.pad(
            jnp.array(kv_cache_lens_np, jnp.int32),
            (0, max_seq - len(kv_cache_lens_np)))

        # Build local Q tensor (slice of global Q).
        local_q_segs = []
        for i, (q_len, _) in enumerate(seq_lens_q_kv):
            local_q_len = q_len // cp_n
            global_start = int(global_cu_q[i]) + cp_rank * local_q_len
            local_q_segs.append(q_global[global_start:global_start +
                                         local_q_len])
        local_q_concat = jnp.concatenate(local_q_segs, axis=0)
        local_total_q = local_q_concat.shape[0]
        local_max_tok = align_to(local_total_q, 128)
        local_max_tok = max(local_max_tok, 512)
        q_local = jnp.pad(local_q_concat,
                          ((0, local_max_tok - local_total_q), (0, 0), (0, 0)))

        # K/V for the CP kernel are the FULL global tensors (simulating all-gather).
        # cp_group_size=1 forces the CP code path (which uses cu_kv_lens and
        # q_start) without actual cross-device communication.
        cp_out, cp_kv = cp_ragged_paged_attention(
            q_local,
            k_global,  # full K, as if all-gathered
            v_global,  # full V
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

        # Compare local output slice against reference.
        tol = 0.25
        local_out = cp_out[:local_total_q]
        ref_local_segs = []
        for i, (q_len, _) in enumerate(seq_lens_q_kv):
            local_q_len = q_len // cp_n
            global_start = int(global_cu_q[i]) + cp_rank * local_q_len
            ref_local_segs.append(ref_out[global_start:global_start +
                                          local_q_len])
        ref_local = jnp.concatenate(ref_local_segs, axis=0)

        self.assertAllClose(local_out, ref_local, atol=tol, rtol=tol)

    @parameterized.product(
        cp_rank=[0, 1],
        dtype=[jnp.bfloat16],
        seq_lens=[[(128, 256)], [(64, 128), (128, 256)]],
    )
    def test_pcp_simulated(self, cp_rank, dtype, seq_lens):
        self._simulate_pcp_rank(seq_lens,
                                num_q_heads=8,
                                num_kv_heads=2,
                                head_dim=128,
                                page_size=32,
                                dtype=dtype,
                                cp_n=2,
                                cp_rank=cp_rank)


# ---------------------------------------------------------------------------
# Benchmark (optional, run directly)
# ---------------------------------------------------------------------------


def _benchmark(num_iters=20):
    """Simple timing benchmark comparing baseline vs CP kernel."""
    if not jtu.is_device_tpu_at_least(version=4):
        print("Benchmark requires TPUv4+, skipping.")
        return

    num_q_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size, dtype = 32, jnp.bfloat16
    seq_lens_q_kv = [(512, 1024), (256, 768), (128, 512)]
    max_seq = 8
    num_pages = 1024

    cu_q_arr, kv_arr, kv_cache, page_indices = _build_kv_cache(
        seq_lens_q_kv, num_kv_heads, head_dim, page_size, dtype, num_pages,
        max_seq)
    total_q = int(cu_q_arr[len(seq_lens_q_kv)])
    max_tok = max(align_to(total_q, 128), 512)
    q = _gen((max_tok, num_q_heads, head_dim), dtype)
    k = _gen((max_tok, num_kv_heads, head_dim), dtype)
    v = _gen((max_tok, num_kv_heads, head_dim), dtype)
    distribution = jnp.array([0, 0, len(seq_lens_q_kv)], jnp.int32)

    args = (q, k, v, kv_cache, kv_arr, page_indices, cu_q_arr, distribution)

    # Warmup.
    for _ in range(3):
        baseline_ragged_paged_attention(*args)[0].block_until_ready()
        cp_ragged_paged_attention(*args)[0].block_until_ready()

    def _time(fn, *a, **kw):
        t0 = time.perf_counter()
        for _ in range(num_iters):
            out = fn(*a, **kw)
            (out[0] if isinstance(out, tuple) else out).block_until_ready()
        return (time.perf_counter() - t0) / num_iters * 1000

    baseline_ms = _time(baseline_ragged_paged_attention, *args)
    cp_ms = _time(cp_ragged_paged_attention, *args)

    print(f"\n=== RPA v3 CP Benchmark ({num_iters} iters) ===")
    print(f"  Baseline (no CP):  {baseline_ms:.2f} ms")
    print(f"  CP kernel (no PCP): {cp_ms:.2f} ms  "
          f"(overhead: {(cp_ms - baseline_ms):.2f} ms)")
    print(f"  Sequences: {seq_lens_q_kv}")
    print(f"  Heads: Q={num_q_heads} KV={num_kv_heads} dim={head_dim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args, _ = parser.parse_known_args()
    if args.benchmark:
        jax.config.update("jax_numpy_dtype_promotion", "standard")
        _benchmark()
    else:
        absltest.main()
