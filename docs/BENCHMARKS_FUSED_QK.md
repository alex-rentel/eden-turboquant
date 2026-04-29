# Benchmarks — v0.7.0 Fused QK Kernel

Micro-benchmark for the new fused attention-from-compressed primitive shipped in v0.7.0. Measured on **Apple M1 Max 64GB / macOS 26.4 / mlx 0.31.1**, reproducible with:

```bash
python benchmarks/micro_fused_qk.py --trials 30 --warmup 5
```

## Headline

The new `fused_qk_scores_4bit` kernel computes `Q_rot @ K^T` directly from packed 4-bit codebook indices, **eliminating the per-step dequantization** of compressed KV cache tokens that dominated v0.6.0 decode overhead.

| Shape | v0.6.0 path (dequant+matmul) | v0.7.0 path (fused) | Speedup |
|---|---|---|---|
| decode T_kv=256 | 301 μs | 266 μs | **1.13×** |
| decode T_kv=1024 | 327 μs | 231 μs | **1.42×** |
| decode T_kv=4096 (long context) | 487 μs | 230 μs | **2.12×** ★ |
| prefill T_q=32 T_kv=256 | 258 μs | 258 μs | 1.00× |
| prefill T_q=32 T_kv=1024 | 311 μs | 290 μs | 1.07× |
| decode D=256 (Gemma3-like) | 491 μs | 242 μs | **2.03×** ★ |
| decode D=96 (Phi-3.5-like) | 264 μs | 229 μs | 1.16× |

**The kernel wins biggest where it matters most — long-context decode and larger head_dims.** At D=128 T_kv=4096 the speedup is 2.12×, and at D=256 T_kv=1024 it's 2.03×. Short-context decode (T_kv=256) and prefill (T_q=32) show modest gains because the fixed dispatch overhead dominates in those regimes.

## How it works

### The old path (v0.6.0)

For each compressed token k, dequantize to reconstruct the full FP16 K vector, then run standard matmul:

```
K_hat[k] = norms[k] * (centroids[idx[k, :]] @ R)    # D² multiply-adds per token
scores   = Q @ K_hat.T                               # D × T_kv multiply-adds
```

Total per-step FLOPs: `T_kv × (D² + D)` for reconstruction plus `T_q × T_kv × D` for the matmul. The `D²` inverse rotation dominates.

### The new path (v0.7.0)

Pre-rotate Q once per decode step, then compute scores directly from packed indices:

```
Q_rot    = Q @ R.T                                                      # once per step
scores[q, k] = norms[k] * Σⱼ Q_rot[q, j] × centroids[idx[k, j]]
```

Total per-step FLOPs: `T_q × D²` for Q_rot (cheap for T_q=1) plus `T_q × T_kv × D` for the fused inner loop. The `D²` per-token cost is gone.

At T_kv=4096 D=128, this is a **~125× reduction in arithmetic ops** for the K side. The realized wall-time speedup is lower (~2×) because the workload is memory-bandwidth bound at the packed-byte read step, not compute bound. But eliminating the inverse rotation per token is the key win.

See `docs/FUSED_ATTENTION_DESIGN.md` for the full math derivation and competitive analysis.

## Kernel architecture

**2D grid** — `(T_kv, T_q, 1)`. Each thread computes one output score at position `(q_idx, kv_idx)`. Using a 2D grid means `T_kv` is read from `threads_per_grid.x` at runtime, so the kernel compiles once per head_dim `D`, not once per `(D, T_kv)` pair.

**Threadgroup shared centroids** — the 16 codebook centroids live in `threadgroup float shared_centroids[16]`, loaded once per threadgroup and reused `D` times per thread (~128 shared-memory hits instead of global-memory hits per thread). This optimization alone gave +0.17 speedup at T_kv=4096.

**No `simd_sum` yet** — the D-dimension reduction is still serial per thread. A SIMD-width reduction would split the D inner loop across lanes and combine via `simd_sum`. Deferred to v0.8.0 pending integration work.

## What's shipped in v0.7.0 (and what isn't)

### Shipped

- `mlx_turboquant.rotation.pre_rotate_query(query, rotation)` — the `Q @ R.T` helper.
- `mlx_turboquant.kernels.fused_qk_scores_4bit(q_rot, packed_k, norms_k, centroids, D)` — the headline kernel.
- `mlx_turboquant.kernels.fused_qk_scores_3bit(...)` — 3-bit variant.
- `mlx_turboquant.kernels.fused_qk_scores_2bit(...)` — 2-bit variant.
- Correctness tests covering all three bit widths, all tested head_dims (96/128/256), decode and prefill shapes, and edge cases (T_kv=0, T_kv=1, single query). 12 tests in `TestFusedQKScoresCorrectness{4,3,2}Bit`.
- Full `docs/FUSED_ATTENTION_DESIGN.md` covering the math, kernel layout, integration blocker, and the three considered integration approaches.

### Not shipped in v0.7.0

- **Full SDPA integration.** The kernels are not (yet) wired into the `update_and_fetch` → SDPA hot path. mlx-lm's attention layers call `mx.fast.scaled_dot_product_attention` on dense tensors returned from the cache, and there is no hook point in mlx-lm to substitute a custom SDPA. Integration requires per-model-family attention patches (Llama, Qwen, Mistral, Gemma, Phi, DeepSeek — each has a slightly different `self_attn.__call__`) which is a large surface area. See `docs/FUSED_ATTENTION_DESIGN.md` for the three considered approaches (A: per-family patch, B: custom cache method, C: utility only).
- **Fusing V.** `softmax(scores) @ V_compressed` requires either a two-pass kernel or online softmax. Significantly more code. Deferred.
- **`simd_sum` D-reduction.** The current kernel has one thread per score with a serial D-element inner loop. Phase 4 optimizations were scoped to threadgroup-shared centroids only; `simd_sum` is deferred to v0.8.0.

### Integration pathway to v0.8.0

Users who want the ~2× long-context decode speedup today can integrate manually:

```python
from mlx_turboquant.rotation import pre_rotate_query
from mlx_turboquant.kernels import fused_qk_scores_4bit

# In your custom attention loop, replacing the SDPA call:
Q_rot = pre_rotate_query(q, rotation)                              # (B, H, T_q, D)

# Scores for the compressed region — fused, no dequant
scores_compressed = fused_qk_scores_4bit(
    Q_rot.reshape(-1, D), packed_k, norms_k, centroids, D=D,
)  # (T_q, T_kv_compressed)

# Scores for sink + residual (already FP16) — standard matmul
scores_fp16 = q @ keys_fp16.transpose(...)

# Combine, scale, mask, softmax as usual
all_scores = concat([scores_compressed, scores_fp16])
weights = softmax(all_scores * (1 / sqrt(D)) + mask)
output = weights @ all_values
```

v0.8.0 will productize this pattern via one of the integration approaches documented in the design doc, conditional on a per-family patch budget being acceptable.

## Reproducing

```bash
# Correctness test — the kernels must match dequant+matmul to atol=1e-3
python -m pytest tests/test_fused_attention.py -v

# Micro-benchmark
python benchmarks/micro_fused_qk.py --trials 30 --warmup 5

# Current full test count: 175 passing
python -m pytest tests/ -q --ignore=tests/test_integration.py
```

Per-trial noise on the M1 Max is ~5% — the 2.12× and 2.03× wins are well above noise, the 1.00× ties are within noise of 1.0×.
