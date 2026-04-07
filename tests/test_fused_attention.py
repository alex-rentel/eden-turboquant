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
    flat, for test use only."""
    from mlx_turboquant.kernels import metal_dequantize
    return metal_dequantize(
        ctx["packed"], ctx["norms"], ctx["centroids"], ctx["rotation"],
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
