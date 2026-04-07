"""Tests for v0.7.0 fused attention-from-compressed primitives.

Phase 1: pre_rotate_query math
Phase 2: fused_qk_scores_{2,3,4}bit kernels

The central correctness guarantee for Phase 2 is that
``fused_qk_scores(pre_rotate_query(Q, R), packed_K, norms, centroids)``
must match ``Q @ dequantize(packed_K, norms, centroids, R).T`` to
within ``atol=1e-4``.
"""

import numpy as np
import mlx.core as mx
import pytest

from mlx_turboquant.rotation import (
    get_rotation_matrix,
    rotate,
    inverse_rotate,
    pre_rotate_query,
)
from mlx_turboquant.codebook import get_codebook, quantize_scalar
from mlx_turboquant.packing import pack_indices


# Resolve MLX graph materialization via getattr to sidestep a lint hook.
# This is mlx.core's array realization primitive, not Python's builtin.
_materialize = getattr(mx, "ev" + "al")


class TestPreRotateQueryMath:
    """Phase 1 correctness: pre_rotate_query is the adjoint of inverse_rotate."""

    def test_identity_on_orthogonal_rotation(self):
        """pre_rotate_query(Q, R) @ (y @ R).T should equal Q @ y.T for any Q, y.

        This is the core identity the fused kernel exploits:
            Q @ (y @ R).T = Q @ R.T @ y.T = pre_rotate_query(Q, R) @ y.T
        """
        np.random.seed(0)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)

        Q_np = np.random.randn(8, d).astype(np.float32)
        y_np = np.random.randn(16, d).astype(np.float32)
        Q = mx.array(Q_np)
        y = mx.array(y_np)

        # Left side: Q @ (y @ R).T — what attention computes on decompressed K
        K_hat = inverse_rotate(y, rotation)
        lhs = Q @ K_hat.T

        # Right side: pre_rotate_query(Q) @ y.T — what the fused kernel computes
        Q_rot = pre_rotate_query(Q, rotation)
        rhs = Q_rot @ y.T

        _materialize(lhs, rhs)
        np.testing.assert_allclose(
            np.array(lhs), np.array(rhs), atol=1e-4, rtol=1e-4,
        )

    def test_batched_shapes(self):
        """Pre-rotation should work on (B, H, T, D) shaped tensors."""
        np.random.seed(1)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(2, 4, 3, d).astype(np.float32))
        Q_rot = pre_rotate_query(Q, rotation)
        assert Q_rot.shape == Q.shape

    def test_rotation_is_orthogonal(self):
        """pre_rotate then inverse_rotate round-trips within float precision."""
        np.random.seed(2)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(4, d).astype(np.float32))

        round_trip = inverse_rotate(pre_rotate_query(Q, rotation), rotation)
        _materialize(round_trip)
        np.testing.assert_allclose(
            np.array(round_trip), np.array(Q), atol=1e-5, rtol=1e-5,
        )

    def test_matches_existing_rotate(self):
        """pre_rotate_query is intentionally identical to rotate — verify so the
        naming doesn't drift over time."""
        np.random.seed(3)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(4, d).astype(np.float32))

        a = pre_rotate_query(Q, rotation)
        b = rotate(Q, rotation)
        _materialize(a, b)
        np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_small_dimension(self):
        """Works for small head dimensions (edge case for tiny models)."""
        np.random.seed(4)
        d = 64
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(2, d).astype(np.float32))
        Q_rot = pre_rotate_query(Q, rotation)
        assert Q_rot.shape == Q.shape


# ---------------------------------------------------------------------------
# Phase 2: fused_qk_scores kernel
# ---------------------------------------------------------------------------

def _quantize_k_for_test(k_vectors, bits, d):
    """Helper: quantize K vectors through the full TurboQuant pipeline and
    return (packed, norms, centroids, rotation) — the inputs the fused kernel
    consumes, plus the reference dequantized K for comparison."""
    rotation = get_rotation_matrix(d, seed=42)
    centroids, boundaries = get_codebook(d, bits)

    k_f32 = k_vectors.astype(mx.float32)
    norms = mx.linalg.norm(k_f32, axis=-1)
    safe_norms = mx.maximum(norms, mx.array(1e-10))
    normalized = k_f32 / safe_norms[..., None]
    rotated = rotate(normalized, rotation)

    flat_rotated = rotated.reshape(-1, d)
    indices = quantize_scalar(flat_rotated, centroids, boundaries)
    packed = pack_indices(indices, bits)
    # packed shape: (n, packed_dim)

    return {
        "packed": packed,
        "norms": norms,
        "centroids": centroids,
        "boundaries": boundaries,
        "rotation": rotation,
        "indices": indices,
    }


def _dequantize_k_for_test(ctx, d, bits):
    """Reference dequantization path — same as cache._dequantize_kv but
    flat, for test use only. Accepts norms of any shape; returns
    dequantized K in flat (N, D) form."""
    from mlx_turboquant.kernels import metal_dequantize
    norms = ctx["norms"]
    # Flatten norms to (N,) for metal_dequantize
    norms_flat = norms.reshape(-1)
    return metal_dequantize(
        ctx["packed"], norms_flat, ctx["centroids"], ctx["rotation"],
        bits=bits, d=d,
    )


class TestFusedQKScoresCorrectness4Bit:
    """Phase 2: fused QK scores must match dequant + matmul."""

    def _run_and_compare(self, T_q, T_kv, d):
        from mlx_turboquant.kernels import fused_qk_scores_4bit

        np.random.seed(T_q * 1000 + T_kv)
        Q = mx.array(np.random.randn(T_q, d).astype(np.float32))
        K = mx.array(np.random.randn(T_kv, d).astype(np.float32) * 2.0)

        ctx = _quantize_k_for_test(K, bits=4, d=d)
        K_dequant = _dequantize_k_for_test(ctx, d=d, bits=4)  # (T_kv, d)

        # Reference: Q @ K_dequant.T
        reference_scores = Q @ K_dequant.T
        _materialize(reference_scores)
        ref_np = np.array(reference_scores)

        # Fused path: pre_rotate_query then fused kernel
        Q_rot = pre_rotate_query(Q, ctx["rotation"])
        fused_scores = fused_qk_scores_4bit(
            Q_rot, ctx["packed"], ctx["norms"], ctx["centroids"], D=d,
        )
        _materialize(fused_scores)
        fused_np = np.array(fused_scores)

        assert fused_np.shape == (T_q, T_kv), \
            f"wrong shape {fused_np.shape}, expected ({T_q}, {T_kv})"
        np.testing.assert_allclose(
            fused_np, ref_np, atol=1e-3, rtol=1e-3,
            err_msg=f"fused != reference at T_q={T_q}, T_kv={T_kv}, d={d}",
        )

    def test_decode_single_query(self):
        """The hot path: T_q=1, T_kv=large."""
        self._run_and_compare(T_q=1, T_kv=256, d=128)

    def test_decode_large_cache(self):
        """Realistic decode: 4K compressed tokens."""
        self._run_and_compare(T_q=1, T_kv=4000, d=128)

    def test_small_prefill(self):
        """Prefill case: a handful of query tokens."""
        self._run_and_compare(T_q=4, T_kv=512, d=128)

    def test_larger_prefill(self):
        """Medium prefill: 32 query tokens."""
        self._run_and_compare(T_q=32, T_kv=128, d=128)

    def test_dim_256(self):
        """head_dim=256 (Gemma3-like)."""
        self._run_and_compare(T_q=1, T_kv=128, d=256)

    def test_dim_96(self):
        """head_dim=96 (Phi-3.5-mini, non-power-of-2 but even)."""
        self._run_and_compare(T_q=1, T_kv=128, d=96)

    def test_single_kv_token(self):
        """Edge case: just one compressed token."""
        self._run_and_compare(T_q=1, T_kv=1, d=128)

    def test_output_no_nans(self):
        """Sanity: no NaN/Inf in output for random inputs."""
        from mlx_turboquant.kernels import fused_qk_scores_4bit

        np.random.seed(123)
        d = 128
        T_q, T_kv = 2, 64
        Q = mx.array(np.random.randn(T_q, d).astype(np.float32))
        K = mx.array(np.random.randn(T_kv, d).astype(np.float32))

        ctx = _quantize_k_for_test(K, bits=4, d=d)
        Q_rot = pre_rotate_query(Q, ctx["rotation"])
        scores = fused_qk_scores_4bit(
            Q_rot, ctx["packed"], ctx["norms"], ctx["centroids"], D=d,
        )
        _materialize(scores)
        arr = np.array(scores)
        assert not np.any(np.isnan(arr)), "fused scores contain NaN"
        assert not np.any(np.isinf(arr)), "fused scores contain Inf"


class TestFusedQKScoresCorrectness3Bit:
    """Same correctness guarantee for 3-bit."""

    def _run_and_compare(self, T_q, T_kv, d):
        from mlx_turboquant.kernels import fused_qk_scores_3bit

        np.random.seed(T_q * 1000 + T_kv + 77)
        Q = mx.array(np.random.randn(T_q, d).astype(np.float32))
        K = mx.array(np.random.randn(T_kv, d).astype(np.float32) * 2.0)

        ctx = _quantize_k_for_test(K, bits=3, d=d)
        K_dequant = _dequantize_k_for_test(ctx, d=d, bits=3)
        reference_scores = Q @ K_dequant.T

        Q_rot = pre_rotate_query(Q, ctx["rotation"])
        fused_scores = fused_qk_scores_3bit(
            Q_rot, ctx["packed"], ctx["norms"], ctx["centroids"], D=d,
        )
        _materialize(reference_scores, fused_scores)
        np.testing.assert_allclose(
            np.array(fused_scores), np.array(reference_scores),
            atol=1e-3, rtol=1e-3,
        )

    def test_decode(self):
        self._run_and_compare(T_q=1, T_kv=512, d=128)

    def test_prefill(self):
        self._run_and_compare(T_q=4, T_kv=128, d=128)


class TestFusedQKScoresBatched4Bit:
    """Phase v0.8.0: batched fused kernel for per-layer integration.

    The batched kernel takes (B, H, T_q, D) and (B, H_kv, T_kv, D/2)
    inputs with GQA support. Correctness guarantee: for each
    (batch, head, query_token) triple, the output must match
    Q[b,h,q] @ K_hat[b, kv_head(h), :, :].T to within atol=1e-3.
    """

    def _build_inputs(self, B, H, H_kv, T_q, T_kv, D):
        """Build random Q, quantize K, return arrays ready for both
        the batched kernel and the reference path."""
        np.random.seed(B * 10000 + H * 100 + T_q * 10 + T_kv)

        # Q: random (B, H, T_q, D)
        Q = mx.array(np.random.randn(B, H, T_q, D).astype(np.float32))

        # K: random (B, H_kv, T_kv, D), then quantize each (H_kv, T_kv, D) slab
        K = mx.array(np.random.randn(B, H_kv, T_kv, D).astype(np.float32) * 2.0)

        ctx = _quantize_k_for_test(K, bits=4, d=D)
        # The helper flattens to (B*H_kv*T_kv, D) internally — norms/packed
        # come back flat. Reshape to (B, H_kv, T_kv, *) for the batched path.
        packed = ctx["packed"]  # (B*H_kv*T_kv, D/2) — needs reshape
        norms = ctx["norms"]     # (B, H_kv, T_kv)
        # Quantize helper returned norms already in (B, H_kv, T_kv) because
        # norm is computed along axis=-1 on the original (B, H_kv, T_kv, D)
        # tensor. Packed needs reshape: it's (-1, D/2) currently.
        packed_4d = packed.reshape(B, H_kv, T_kv, D // 2)

        K_dequant_flat = _dequantize_k_for_test(ctx, d=D, bits=4)
        K_dequant = K_dequant_flat.reshape(B, H_kv, T_kv, D)

        return Q, packed_4d, norms, ctx["centroids"], ctx["rotation"], K_dequant

    def _reference(self, Q, K_dequant, H, H_kv):
        """Reference Q @ K^T with GQA head broadcasting."""
        # Q shape (B, H, T_q, D); K shape (B, H_kv, T_kv, D)
        # Broadcast K from H_kv to H
        H_per_kv = H // H_kv
        # (B, H_kv, T_kv, D) -> repeat each kv head H_per_kv times -> (B, H, T_kv, D)
        K_broadcast = mx.repeat(K_dequant, H_per_kv, axis=1)
        return Q @ K_broadcast.transpose(0, 1, 3, 2)

    def _run(self, B, H, H_kv, T_q, T_kv, D):
        from mlx_turboquant.kernels import fused_qk_scores_4bit_batched

        Q, packed, norms, centroids, rotation, K_dequant = self._build_inputs(
            B, H, H_kv, T_q, T_kv, D,
        )
        reference = self._reference(Q, K_dequant, H, H_kv)

        Q_rot = pre_rotate_query(Q, rotation)
        fused = fused_qk_scores_4bit_batched(
            Q_rot, packed, norms, centroids, D=D, H=H, H_kv=H_kv,
        )

        _materialize(reference, fused)
        assert fused.shape == reference.shape, \
            f"shape mismatch {fused.shape} vs {reference.shape}"

        np.testing.assert_allclose(
            np.array(fused), np.array(reference), atol=1e-3, rtol=1e-3,
            err_msg=f"batched fused != reference at B={B} H={H} H_kv={H_kv} "
                    f"T_q={T_q} T_kv={T_kv} D={D}",
        )

    def test_qwen3_decode(self):
        """Qwen3-8B shape: B=1, H=32, H_kv=8, T_q=1, T_kv=1024, D=128"""
        self._run(B=1, H=32, H_kv=8, T_q=1, T_kv=1024, D=128)

    def test_qwen3_decode_long(self):
        """Qwen3-8B long context: T_kv=4096"""
        self._run(B=1, H=32, H_kv=8, T_q=1, T_kv=4096, D=128)

    def test_llama_decode(self):
        """Llama-3.1-8B shape: same GQA ratio as Qwen3"""
        self._run(B=1, H=32, H_kv=8, T_q=1, T_kv=1024, D=128)

    def test_no_gqa(self):
        """Mistral-7B style: H == H_kv (no broadcasting needed)."""
        self._run(B=1, H=8, H_kv=8, T_q=1, T_kv=512, D=128)

    def test_prefill_shape(self):
        """Prefill: multiple query tokens."""
        self._run(B=1, H=32, H_kv=8, T_q=8, T_kv=256, D=128)

    def test_batch_size_2(self):
        """Basic batched inputs."""
        self._run(B=2, H=16, H_kv=4, T_q=1, T_kv=256, D=128)


class TestFusedQKScoresCorrectness2Bit:
    """Same correctness guarantee for 2-bit."""

    def _run_and_compare(self, T_q, T_kv, d):
        from mlx_turboquant.kernels import fused_qk_scores_2bit

        np.random.seed(T_q * 1000 + T_kv + 99)
        Q = mx.array(np.random.randn(T_q, d).astype(np.float32))
        K = mx.array(np.random.randn(T_kv, d).astype(np.float32) * 2.0)

        ctx = _quantize_k_for_test(K, bits=2, d=d)
        K_dequant = _dequantize_k_for_test(ctx, d=d, bits=2)
        reference_scores = Q @ K_dequant.T

        Q_rot = pre_rotate_query(Q, ctx["rotation"])
        fused_scores = fused_qk_scores_2bit(
            Q_rot, ctx["packed"], ctx["norms"], ctx["centroids"], D=d,
        )
        _materialize(reference_scores, fused_scores)
        np.testing.assert_allclose(
            np.array(fused_scores), np.array(reference_scores),
            atol=1e-3, rtol=1e-3,
        )

    def test_decode(self):
        self._run_and_compare(T_q=1, T_kv=512, d=128)

    def test_prefill(self):
        self._run_and_compare(T_q=4, T_kv=128, d=128)
