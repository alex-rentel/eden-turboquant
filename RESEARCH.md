# TurboQuant Research Notes

> Research compilation from arXiv:2504.19874, reference implementations, and community findings.

## The Algorithm

### Algorithm 1: TurboQuant_mse (MSE-Optimized) — PRIMARY

**Setup (once per head_dim and bit-width):**
1. Generate random orthogonal rotation matrix Π ∈ ℝ^(d×d) via QR decomposition of i.i.d. N(0,1) matrix
2. Compute Lloyd-Max optimal codebook centroids c₁...c_{2^b} for the distribution of rotated unit-sphere coordinates

**Quantize(x):**
1. Compute norm: s = ‖x‖₂
2. Normalize: x̃ = x / s
3. Rotate: y = Π · x̃
4. For each coordinate j: idx_j = argmin_k |y_j - c_k|
5. Store: (idx₁...idx_d, s)

**Dequantize(idx, s):**
1. Look up centroids: ŷ_j = c[idx_j]
2. Inverse rotate: x̂ = Π^T · ŷ
3. Rescale: x̂ = s · x̂

### Algorithm 2: TurboQuant_prod (Inner-Product Optimized) — OPT-IN ONLY

Uses (b-1)-bit MSE quantizer + 1-bit QJL on residual.

**Quantize(x):**
1. idx = Quant_mse(x) using (b-1) bits
2. r = x - DeQuant_mse(idx)
3. qjl = sign(S · r) where S is random ±1/√m matrix
4. Store: (idx, qjl, ‖r‖₂)

**Dequantize(idx, qjl, γ):**
1. x̃_mse = DeQuant_mse(idx)
2. x̃_qjl = (√(π/2)/d) · γ · S^T · qjl
3. Return x̃_mse + x̃_qjl

## Distribution Theory

After rotating a unit vector x ∈ S^(d-1) by random orthogonal Π, each coordinate y_j follows:

```
f(y) = [Γ(d/2) / (√π · Γ((d-1)/2))] · (1-y²)^((d-3)/2)
```

This is Beta((d-1)/2, (d-1)/2) rescaled to [-1, 1]. In high dimensions, converges to N(0, 1/d).

**Key insight:** Rotation makes coordinates approximately independent and identically distributed, allowing per-coordinate scalar quantization to be near-optimal (within 2.7x of Shannon lower bound).

## Lloyd-Max Codebook

Iterative algorithm minimizing E[(X - Q(X))²] for the Beta distribution:
1. Initialize centroids (e.g., uniform spacing or quantiles)
2. Compute boundaries: b_i = (c_i + c_{i+1}) / 2
3. Update centroids: c_i = E[X | b_{i-1} < X < b_i]
4. Repeat until convergence (~300 iterations)

Codebooks are non-uniform: dense near zero (where distribution concentrates), wider in tails.

**Optimal centroids (scaled by √d for unit variance):**
- b=1: {±√(2/π)} ≈ {±0.7979}
- b=2: {±0.453, ±1.51}
- Higher bits: computed numerically

## Theoretical Bounds

| Bits | MSE Upper Bound | MSE Lower Bound | Gap Factor |
|------|----------------|-----------------|------------|
| 1    | 0.36           | 0.25            | 1.45x      |
| 2    | 0.117          | 0.0625          | 1.87x      |
| 3    | 0.03           | 0.0156          | ~1.9x      |
| 4    | 0.009          | 0.0039          | ~2.3x      |

General: D_mse ≤ (√3·π/2) · (1/4^b), Lower bound: 1/4^b

## QJL (Quantized Johnson-Lindenstrauss)

- Projects residual r onto random matrix S, stores only signs
- Provides **unbiased** inner product estimation (unlike MSE which has multiplicative bias of 2/π at 1-bit)
- Properties: E[⟨y, Q_qjl⁻¹(Q_qjl(r))⟩] = ⟨y, r⟩
- Variance: ≤ (π/(2d)) · ‖r‖² · ‖y‖²

## Community Consensus: Critical Findings

### 1. MSE-Only is Better for Drop-in KV Cache
Six independent implementations (PyTorch, C, Rust, MLX) confirmed:
- QJL eliminates bias but **dramatically increases variance**
- Softmax **exponentially amplifies** this variance
- MSE's small multiplicative bias is tolerated by softmax
- Top-1 token consistency drops 15.6% when adding QJL at 2-bit
- **Recommendation:** Use Algorithm 1 (MSE) as default; QJL only with fused attention kernels

### 2. Asymmetric K/V Allocation is Critical
Modern LLMs show dramatic norm disparities:
- Qwen models: K/V norm ratio 52-182x
- Values only do weighted sum, so softmax averages out per-vector errors
- **Recommendation:** Keys get more bits than values (e.g., K=4, V=2)

### 3. Outlier Channels are Model-Dependent
- Qwen2.5-7B: layers 0, 27 have 16.2x median norm (keep in FP16)
- Llama models: no significant outlier layers
- Detection: measure per-channel norms, flag >3x median
- **Paper's approach:** Split channels into outlier/non-outlier sets with different bit-widths
  - 2.5-bit: 32 channels at 3-bit + 96 channels at 2-bit
  - 3.5-bit: similar mixed allocation

### 4. Residual Window is Important
Recent tokens (last ~128) stay in FP16 — they're the most attended to and compression hurts most here.

### 5. Head Dimension Affects Quality
Larger head_dim = better quantization quality:
- d=256 (Gemma): significantly better than d=128 (Llama)
- V3 2.5-bit: +7% PPL on Gemma vs +27% on Llama 3B

### 6. WHT vs QR Rotation
- Walsh-Hadamard Transform: O(d log d) but has implementation pitfalls (many tiny kernel launches)
- QR decomposition: O(d²) but single matmul, simpler
- WHT with deterministic structure may give better results than random rotation
- **Recommendation:** Start with QR, optimize to WHT later if needed

## Existing MLX Implementation Insights (sharpner/turboquant-mlx)

- Uses monkey-patch dispatch mechanism (intercepts SDPA calls)
- V3 (codebook-based) is quality-focused but ~5-6x slower than V2 (affine) due to software lookup
- MLX lacks custom codebook dequant kernels
- Pre-allocation with step=256 reduces O(T) allocations
- 58 unit tests covering codebook, rotation, pack/unpack, SDPA dispatch

## Bit-Packing Scheme

| Bits | Packing | Storage per d=128 vector |
|------|---------|------------------------|
| 2    | 4 values per byte | 32 bytes + 2 bytes (norm) = 34 bytes |
| 3    | 8 values per 3 bytes | 48 bytes + 2 bytes = 50 bytes |
| 4    | 2 values per byte | 64 bytes + 2 bytes = 66 bytes |
| FP16 | native | 256 bytes |

## Implementation Plan Decisions

1. **Default path:** Algorithm 1 (MSE-only) — safe, proven, drop-in compatible
2. **QJL:** Opt-in only, with clear warnings about quality impact
3. **Rotation:** QR decomposition initially, seeded for reproducibility
4. **Codebooks:** Precomputed at import time for common (dim, bits) pairs
5. **K/V bits:** Asymmetric default (K=4, V=2)
6. **Residual window:** 128 tokens FP16 by default
7. **Outlier detection:** Per-channel norm analysis on first forward pass
8. **Bit-packing:** Efficient sub-byte packing for 2/3/4-bit
9. **Target models:** Llama, Qwen, Mistral, Gemma (different head_dims)

## References

1. Zandieh, Daliri, Hadian, Mirrokni — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874)
2. vivekvar-dl/turboquant — PyTorch reference (pip install turbokv)
3. sharpner/turboquant-mlx — MLX port (most thorough)
4. tonbistudio/turboquant-pytorch — V3 with community findings
5. llama.cpp discussion #20969 — Implementation insights
6. DEJAN blog — Paper-to-kernel walkthrough
7. Google Research blog — High-level overview
