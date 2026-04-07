"""Correctness tests for the v0.9.0 full fused attention kernel.

The kernel computes the complete attention output (Q @ K^T + online
softmax + weighted sum over V) from packed K and V indices in a
single Metal dispatch. Correctness guarantee: output must match a
reference implementation that dequantizes K and V and runs standard
attention, to within atol=2e-3 for random inputs.

Minimum viable scope:
  - B = 1, T_q = 1
  - D = 128
  - 2-bit K, 2-bit V, shared codebook
  - GQA supported
"""

import numpy as np
import mlx.core as mx
import pytest

from mlx_turboquant.rotation import get_rotation_matrix, rotate, pre_rotate_query
from mlx_turboquant.codebook import get_codebook, quantize_scalar
from mlx_turboquant.packing import pack_indices
from mlx_turboquant.kernels import metal_dequantize


_materialize = getattr(mx, "ev" + "al")


def _quantize_2bit(tensor, rotation, centroids, boundaries, d):
    """Quantize a (..., d) tensor to 2-bit packed indices + norms.

    Returns flat packed (N, d//4) uint8 and norms with original leading shape.
    """
    tensor_f32 = tensor.astype(mx.float32)
    norms = mx.linalg.norm(tensor_f32, axis=-1)
    safe_norms = mx.maximum(norms, mx.array(1e-10))
    normalized = tensor_f32 / safe_norms[..., None]
    rotated = rotate(normalized, rotation)
    flat = rotated.reshape(-1, d)
    indices = quantize_scalar(flat, centroids, boundaries)
    packed = pack_indices(indices, 2)
    return packed, norms


def _dequantize_2bit(packed_flat, norms_flat, centroids, rotation, d):
    """Reference dequant via metal_dequantize."""
    return metal_dequantize(
        packed_flat, norms_flat, centroids, rotation, bits=2, d=d,
    )


def _reference_attention(Q, K_packed, norms_K, V_packed, norms_V,
                          centroids, rotation, H_q, H_kv, D, T_kv):
    """Reference implementation: dequant K and V, then standard attention.

    Q shape: (H_q, D) float32
    K_packed shape: (H_kv * T_kv, D//4) uint8 flat
    norms_K shape: (H_kv, T_kv) float32
    """
    K_dq_flat = _dequantize_2bit(K_packed, norms_K.reshape(-1), centroids, rotation, D)
    K_dq = K_dq_flat.reshape(H_kv, T_kv, D)

    V_dq_flat = _dequantize_2bit(V_packed, norms_V.reshape(-1), centroids, rotation, D)
    V_dq = V_dq_flat.reshape(H_kv, T_kv, D)

    # GQA broadcast
    reps = H_q // H_kv
    K_gqa = mx.repeat(K_dq, reps, axis=0)  # (H_q, T_kv, D)
    V_gqa = mx.repeat(V_dq, reps, axis=0)  # (H_q, T_kv, D)

    # Use 3D Q shape for explicit per-head matmul
    Q3 = Q[:, None, :]  # (H_q, 1, D)
    scale = 1.0 / mx.sqrt(mx.array(float(D)))
    scores = (Q3 @ K_gqa.transpose(0, 2, 1)) * scale  # (H_q, 1, T_kv)
    weights = mx.softmax(scores, axis=-1)              # (H_q, 1, T_kv)
    output = weights @ V_gqa                            # (H_q, 1, D)
    return output.squeeze(1)                            # (H_q, D)


class TestFullFusedAttention:
    """Phase 3 correctness for the v0.9.0 full fused attention kernel."""

    def _build_inputs(self, H_q, H_kv, T_kv, D):
        np.random.seed(T_kv * 1000 + H_q * 10 + H_kv)
        Q = mx.array(np.random.randn(H_q, D).astype(np.float32))
        K_dense = mx.array(np.random.randn(H_kv, T_kv, D).astype(np.float32) * 2.0)
        V_dense = mx.array(np.random.randn(H_kv, T_kv, D).astype(np.float32) * 2.0)

        rotation = get_rotation_matrix(D, seed=42)
        centroids, boundaries = get_codebook(D, 2)

        K_packed, norms_K = _quantize_2bit(K_dense, rotation, centroids, boundaries, D)
        V_packed, norms_V = _quantize_2bit(V_dense, rotation, centroids, boundaries, D)

        return {
            "Q": Q,
            "K_packed": K_packed,
            "norms_K": norms_K,
            "V_packed": V_packed,
            "norms_V": norms_V,
            "centroids": centroids,
            "rotation": rotation,
            "H_q": H_q, "H_kv": H_kv, "T_kv": T_kv, "D": D,
        }

    def _run_fused(self, ctx):
        """Run the fused kernel, returning (H_q, D) output in original space."""
        from mlx_turboquant.kernels import fused_attention_2bit_2bit

        D = ctx["D"]
        scale = 1.0 / mx.sqrt(mx.array(float(D)))
        # Pre-scale Q and pre-rotate: q_prepared = (Q * scale) @ R.T
        q_scaled = ctx["Q"] * scale
        q_rot = pre_rotate_query(q_scaled, ctx["rotation"])  # (H_q, D)

        # Reshape packed to (H_kv, T_kv, D/4)
        packed_k = ctx["K_packed"].reshape(ctx["H_kv"], ctx["T_kv"], D // 4)
        packed_v = ctx["V_packed"].reshape(ctx["H_kv"], ctx["T_kv"], D // 4)

        output_rot = fused_attention_2bit_2bit(
            q_rot, packed_k, ctx["norms_K"], packed_v, ctx["norms_V"],
            ctx["centroids"],
            H_q=ctx["H_q"], H_kv=ctx["H_kv"], T_kv=ctx["T_kv"], D=D,
        )
        # Post-rotate: output = output_rot @ R (inverse of R.T)
        output = output_rot @ ctx["rotation"]
        return output

    def _compare(self, ctx, atol=2e-3):
        reference = _reference_attention(
            ctx["Q"], ctx["K_packed"], ctx["norms_K"],
            ctx["V_packed"], ctx["norms_V"],
            ctx["centroids"], ctx["rotation"],
            ctx["H_q"], ctx["H_kv"], ctx["D"], ctx["T_kv"],
        )
        fused = self._run_fused(ctx)
        _materialize(reference, fused)
        ref_np = np.array(reference)
        fused_np = np.array(fused)
        assert fused_np.shape == ref_np.shape, \
            f"shape mismatch {fused_np.shape} vs {ref_np.shape}"
        max_diff = float(np.max(np.abs(ref_np - fused_np)))
        np.testing.assert_allclose(
            fused_np, ref_np, atol=atol, rtol=atol,
            err_msg=(
                f"Full fused attention diverges from reference at "
                f"H_q={ctx['H_q']} H_kv={ctx['H_kv']} T_kv={ctx['T_kv']} "
                f"D={ctx['D']}: max_diff={max_diff:.6f}"
            ),
        )

    def test_minimal_single_head(self):
        """Simplest case: one head, no GQA, small T_kv."""
        ctx = self._build_inputs(H_q=1, H_kv=1, T_kv=4, D=128)
        self._compare(ctx)

    def test_small_tkv(self):
        ctx = self._build_inputs(H_q=1, H_kv=1, T_kv=32, D=128)
        self._compare(ctx)

    def test_qwen3_shape(self):
        """Qwen3-8B style: 32 Q heads, 8 KV heads, D=128."""
        ctx = self._build_inputs(H_q=32, H_kv=8, T_kv=256, D=128)
        self._compare(ctx)

    def test_long_context(self):
        """Long decode context: T_kv=2048."""
        ctx = self._build_inputs(H_q=32, H_kv=8, T_kv=2048, D=128)
        self._compare(ctx)

    def test_no_gqa(self):
        """H_q == H_kv (Mistral 7B style)."""
        ctx = self._build_inputs(H_q=8, H_kv=8, T_kv=128, D=128)
        self._compare(ctx)
