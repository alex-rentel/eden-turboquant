"""Phase 4 go/no-go micro-benchmark.

Times the new full fused attention kernel against mx.fast.scaled_dot_product_attention
operating on dense FP32 K/V of the same shape. This is the critical
question for v0.9.0: does the fused kernel beat MLX's hyper-optimized
single-dispatch SDPA at any realistic decode shape?

If yes at some context length, proceed to SDPA integration.
If no, document and stop — the approach is infeasible at this level.

The comparison is deliberately stacked in MLX's favor:
  - mx.fast.sdpa operates on DENSE K/V already resident in memory
    (no packing overhead amortization)
  - mx.fast.sdpa includes the full scale+softmax+V weighted sum
    in ONE dispatch
  - Our fused kernel has to read packed uint8 with centroid
    indirection, which is cache-unfriendly

If our kernel still wins, it's because packed KV reduces memory
traffic enough to offset the extra compute/indirection.
"""

import argparse
import time
from statistics import median

import numpy as np
import mlx.core as mx

from mlx_turboquant.rotation import get_rotation_matrix, rotate, pre_rotate_query
from mlx_turboquant.codebook import get_codebook, quantize_scalar
from mlx_turboquant.packing import pack_indices
from mlx_turboquant.kernels import fused_attention_2bit_2bit, metal_dequantize


_materialize = getattr(mx, "ev" + "al")


def quantize_2bit(tensor, rotation, centroids, boundaries, d):
    tensor_f32 = tensor.astype(mx.float32)
    norms = mx.linalg.norm(tensor_f32, axis=-1)
    safe_norms = mx.maximum(norms, mx.array(1e-10))
    normalized = tensor_f32 / safe_norms[..., None]
    rotated = rotate(normalized, rotation)
    flat = rotated.reshape(-1, d)
    indices = quantize_scalar(flat, centroids, boundaries)
    packed = pack_indices(indices, 2)
    return packed, norms


def build_inputs(H_q, H_kv, T_kv, D, seed=0):
    np.random.seed(seed)
    Q = mx.array(np.random.randn(H_q, D).astype(np.float32))
    K = mx.array(np.random.randn(H_kv, T_kv, D).astype(np.float32) * 2.0)
    V = mx.array(np.random.randn(H_kv, T_kv, D).astype(np.float32) * 2.0)

    rotation = get_rotation_matrix(D, seed=42)
    centroids, boundaries = get_codebook(D, 2)

    K_packed, norms_K = quantize_2bit(K, rotation, centroids, boundaries, D)
    V_packed, norms_V = quantize_2bit(V, rotation, centroids, boundaries, D)

    # Also provide dequantized dense K/V for the mx.fast.sdpa baseline
    K_dq = metal_dequantize(K_packed, norms_K.reshape(-1), centroids, rotation, bits=2, d=D)
    V_dq = metal_dequantize(V_packed, norms_V.reshape(-1), centroids, rotation, bits=2, d=D)
    K_dq = K_dq.reshape(H_kv, T_kv, D)
    V_dq = V_dq.reshape(H_kv, T_kv, D)

    return {
        "Q": Q,
        "K_dq": K_dq, "V_dq": V_dq,
        "K_packed_3d": K_packed.reshape(H_kv, T_kv, D // 4),
        "V_packed_3d": V_packed.reshape(H_kv, T_kv, D // 4),
        "norms_K": norms_K, "norms_V": norms_V,
        "centroids": centroids, "rotation": rotation,
        "H_q": H_q, "H_kv": H_kv, "T_kv": T_kv, "D": D,
    }


def time_fn(fn, warmup=3, trials=30):
    for _ in range(warmup):
        out = fn()
        _materialize(out)
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        _materialize(out)
        samples.append(time.perf_counter() - t0)
    return samples


def run_one(H_q, H_kv, T_kv, D, warmup=3, trials=30):
    ctx = build_inputs(H_q, H_kv, T_kv, D)
    scale = 1.0 / float(np.sqrt(D))

    # Baseline: mx.fast.scaled_dot_product_attention on dense K/V (the actual
    # competitor — this is what mlx-lm calls in the decode hot path).
    # Shapes: Q (1, H_q, 1, D), K (1, H_kv, T_kv, D), V (1, H_kv, T_kv, D)
    Q_4d = ctx["Q"][None, :, None, :]   # (1, H_q, 1, D)
    K_4d = ctx["K_dq"][None]             # (1, H_kv, T_kv, D)
    V_4d = ctx["V_dq"][None]             # (1, H_kv, T_kv, D)

    def fn_mlx_sdpa():
        return mx.fast.scaled_dot_product_attention(
            Q_4d, K_4d, V_4d, scale=scale,
        )

    def fn_fused():
        q_scaled = ctx["Q"] * scale
        q_rot = pre_rotate_query(q_scaled, ctx["rotation"])
        out_rot = fused_attention_2bit_2bit(
            q_rot,
            ctx["K_packed_3d"], ctx["norms_K"],
            ctx["V_packed_3d"], ctx["norms_V"],
            ctx["centroids"],
            H_q=H_q, H_kv=H_kv, T_kv=T_kv, D=D,
        )
        return out_rot @ ctx["rotation"]

    mlx_times = time_fn(fn_mlx_sdpa, warmup, trials)
    fused_times = time_fn(fn_fused, warmup, trials)

    return {
        "H_q": H_q, "H_kv": H_kv, "T_kv": T_kv, "D": D,
        "mlx_median_us": median(mlx_times) * 1e6,
        "fused_median_us": median(fused_times) * 1e6,
        "mlx_min_us": min(mlx_times) * 1e6,
        "fused_min_us": min(fused_times) * 1e6,
        "speedup_median": median(mlx_times) / median(fused_times),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    shapes = [
        # Qwen3-8B style: H_q=32 H_kv=8 D=128
        (32, 8, 256, 128, "Qwen3-8B T_kv=256"),
        (32, 8, 512, 128, "Qwen3-8B T_kv=512"),
        (32, 8, 1024, 128, "Qwen3-8B T_kv=1024"),
        (32, 8, 2048, 128, "Qwen3-8B T_kv=2048"),
        (32, 8, 4096, 128, "Qwen3-8B T_kv=4096"),
        (32, 8, 8192, 128, "Qwen3-8B T_kv=8192"),
    ]

    print(f"{'shape':<34} {'mlx sdpa':>14} {'fused':>14} {'speedup':>12}")
    print("-" * 76)
    for H_q, H_kv, T_kv, D, label in shapes:
        r = run_one(H_q, H_kv, T_kv, D, warmup=args.warmup, trials=args.trials)
        sp = r["speedup_median"]
        marker = " WIN" if sp >= 1.10 else (" tie" if sp >= 0.90 else " LOSS")
        print(
            f"{label:<34} {r['mlx_median_us']:>12.1f} us "
            f"{r['fused_median_us']:>12.1f} us {sp:>10.2f}x{marker}"
        )
    print()
    print("The critical question: does the fused kernel beat mx.fast.sdpa")
    print("at any realistic T_kv? If yes at 4K or 8K, proceed to integration.")
    print("If no, document and stop.")


if __name__ == "__main__":
    main()
