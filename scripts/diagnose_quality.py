"""Diagnostic script for Bug 3: Investigate low cosine similarity.

Checks:
a) Normalization precision after divide
b) Codebook grid resolution (10000 vs 50000)
c) Pure quantizer cosine sim (no model involvement)
d) Per-layer KV cosine sim with real model
"""

import numpy as np
import mlx.core as mx


def check_normalization_precision():
    """Check a) norm precision after normalization."""
    print("=" * 60)
    print("Check A: Normalization precision")
    print("=" * 60)

    np.random.seed(42)
    for scale in [1.0, 100.0, 0.001, 500.0]:
        x = mx.array(np.random.randn(1000, 128).astype(np.float32) * scale)
        x_f32 = x.astype(mx.float32)
        norms = mx.linalg.norm(x_f32, axis=-1)
        safe_norms = mx.maximum(norms, mx.array(1e-10))
        x_norm = x_f32 / safe_norms[..., None]
        recomputed_norms = mx.linalg.norm(x_norm, axis=-1)
        rn = np.array(recomputed_norms)
        print(f"  scale={scale:>8.3f}: ||x_norm|| mean={rn.mean():.8f} "
              f"std={rn.std():.2e} min={rn.min():.8f} max={rn.max():.8f}")

    print()


def check_codebook_resolution():
    """Check b) codebook grid resolution impact on MSE."""
    print("=" * 60)
    print("Check B: Codebook grid resolution")
    print("=" * 60)

    from mlx_turboquant.codebook import lloyd_max, compute_theoretical_mse

    for d in [128, 256]:
        for grid_size in [10000, 50000]:
            centroids, boundaries = lloyd_max(d, 4, grid_size=grid_size)
            mse = compute_theoretical_mse(d, 4, centroids, boundaries, grid_size=grid_size)
            total_mse = d * mse
            print(f"  d={d} grid={grid_size:>5d}: per-coord MSE={mse:.8f} "
                  f"total MSE={total_mse:.6f}")
    print()


def check_pure_quantizer_cosine_sim():
    """Check c+d) pure quantizer cosine sim on synthetic unit vectors."""
    print("=" * 60)
    print("Check C: Pure quantizer cosine sim (synthetic unit vectors)")
    print("=" * 60)

    from mlx_turboquant.quantizer import TurboQuantMSE

    np.random.seed(0)

    for d in [128, 256]:
        for bits in [2, 3, 4]:
            tq = TurboQuantMSE(d=d, bits=bits)
            x = np.random.randn(10000, d).astype(np.float32)
            x = x / np.linalg.norm(x, axis=-1, keepdims=True)
            x_mx = mx.array(x)

            qt = tq.quantize(x_mx)
            x_hat = tq.dequantize(qt)
            x_hat_np = np.array(x_hat)

            # Per-vector cosine sim
            dots = np.sum(x * x_hat_np, axis=-1)
            norms_orig = np.linalg.norm(x, axis=-1)
            norms_recon = np.linalg.norm(x_hat_np, axis=-1)
            cos_sims = dots / (norms_orig * norms_recon + 1e-30)

            print(f"  d={d} bits={bits}: mean={np.mean(cos_sims):.6f} "
                  f"median={np.median(cos_sims):.6f} "
                  f"min={np.min(cos_sims):.6f} p5={np.percentile(cos_sims, 5):.6f}")

    print()


def check_model_kv_cosine_sim():
    """Check per-layer KV vector cosine sim with a real model."""
    print("=" * 60)
    print("Check D: Real model KV cosine sim (per-layer)")
    print("=" * 60)

    try:
        from mlx_lm import load
        from mlx_lm.models.cache import KVCache
        from mlx_turboquant.cache import TurboQuantKVCache
        from mlx_turboquant.patch import apply_turboquant, detect_outlier_layers
    except ImportError:
        print("  Skipped (mlx-lm not available)")
        return

    model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = mx.array(tokenizer.encode(prompt))[None]
    print(f"  Prompt: {inputs.shape[1]} tokens")

    # Baseline KV
    baseline_cache = [KVCache() for _ in range(len(model.model.layers))]
    _ = model(inputs, cache=baseline_cache)
    mx.eval(_)

    # Outlier layers
    outliers = detect_outlier_layers(model)
    print(f"  Outlier layers: {outliers}")

    # TurboQuant KV (force all layers, no skipping)
    apply_turboquant(model, key_bits=4, value_bits=4,
                     residual_window=4, auto_detect_outliers=False)
    tq_cache = model.make_cache()
    _ = model(inputs, cache=tq_cache)
    mx.eval(_)

    # Compare per-layer
    print(f"\n  {'Layer':>5} {'K cos_sim':>10} {'V cos_sim':>10} {'K norm_ratio':>12} {'Outlier':>8}")
    print("  " + "-" * 50)

    for i in range(min(len(baseline_cache), 10)):  # First 10 layers
        bc = baseline_cache[i]
        tc = tq_cache[i]

        if bc.keys is None or not isinstance(tc, TurboQuantKVCache):
            continue

        # Get baseline KV (only valid tokens)
        bk = np.array(bc.keys[:, :, :bc.offset, :].astype(mx.float32)).astype(np.float64)
        bv = np.array(bc.values[:, :, :bc.offset, :].astype(mx.float32)).astype(np.float64)

        # Get TQ KV (decompressed + FP16 window)
        if tc._decompressed_keys_cache is not None:
            tk = np.concatenate([
                np.array(tc._decompressed_keys_cache.astype(mx.float32)).astype(np.float64),
                np.array(tc.keys.astype(mx.float32)).astype(np.float64)
            ], axis=2)
            tv = np.concatenate([
                np.array(tc._decompressed_values_cache.astype(mx.float32)).astype(np.float64),
                np.array(tc.values.astype(mx.float32)).astype(np.float64)
            ], axis=2)
        else:
            tk = np.array(tc.keys.astype(mx.float32)).astype(np.float64)
            tv = np.array(tc.values.astype(mx.float32)).astype(np.float64)

        # Truncate to same length
        min_len = min(bk.shape[2], tk.shape[2])
        bk, tk = bk[:, :, :min_len, :], tk[:, :, :min_len, :]
        bv, tv = bv[:, :, :min_len, :], tv[:, :, :min_len, :]

        # Per-vector cosine sim (flatten batch+head dims)
        bk_flat = bk.reshape(-1, bk.shape[-1])
        tk_flat = tk.reshape(-1, tk.shape[-1])
        k_dots = np.sum(bk_flat * tk_flat, axis=-1)
        k_norms = np.linalg.norm(bk_flat, axis=-1) * np.linalg.norm(tk_flat, axis=-1) + 1e-30
        k_cos = np.median(k_dots / k_norms)

        bv_flat = bv.reshape(-1, bv.shape[-1])
        tv_flat = tv.reshape(-1, tv.shape[-1])
        v_dots = np.sum(bv_flat * tv_flat, axis=-1)
        v_norms = np.linalg.norm(bv_flat, axis=-1) * np.linalg.norm(tv_flat, axis=-1) + 1e-30
        v_cos = np.median(v_dots / v_norms)

        # Norm ratio
        k_norm_ratio = np.median(np.linalg.norm(bk_flat, axis=-1)) / (
            np.median(np.linalg.norm(tk_flat, axis=-1)) + 1e-30)

        is_outlier = "***" if i in outliers else ""
        print(f"  {i:>5d} {k_cos:>10.6f} {v_cos:>10.6f} {k_norm_ratio:>12.4f} {is_outlier:>8}")

    print()


if __name__ == "__main__":
    check_normalization_precision()
    check_codebook_resolution()
    check_pure_quantizer_cosine_sim()
    check_model_kv_cosine_sim()
