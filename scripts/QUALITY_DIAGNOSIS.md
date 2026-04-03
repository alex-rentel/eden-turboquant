# Quality Diagnosis (Bug 3)

## Findings

### Check A: Normalization precision
- **Result:** Perfect (1.0 +/- 3.5e-8). Not a factor.

### Check B: Codebook grid resolution
- **Result:** 10000 vs 50000 grid points: MSE difference < 0.1%. Not a factor.

### Check C: Pure quantizer cosine sim
- **Result at 4-bit, d=128:** mean=0.9954, median=0.9956, min=0.9777
- **Result at 3-bit, d=128:** mean=0.9831, median=0.9837
- **Conclusion:** The math core is working correctly and matches paper expectations.

### Check D: Real model per-layer KV cosine sim
- **Per-vector quantizer quality on real model values:** 0.9955 (same as synthetic)
- **Per-layer V cos_sim through the network:** 0.20-0.51 at deep layers
- **Root cause:** Error compounding through 28 transformer layers, not quantizer quality

### Diagnosis

The 0.95 logit cosine sim at 4-bit is NOT caused by a quantizer bug. The math core 
achieves 0.9955 per-vector cosine sim — matching the paper.

The logit-level quality drop comes from **error compounding**: each transformer layer 
sees slightly-different KV from compressed tokens, producing slightly-different 
attention outputs, which feed into the next layer. Over 28 layers, small per-vector 
errors (0.45% per vector) compound to ~5% logit divergence.

This is the fundamental tradeoff of KV cache compression and is consistent with 
other implementations (sharpner/turboquant-mlx reports similar findings).

### Mitigation strategies (already implemented)
1. **Residual window** — recent tokens stay FP16, reducing compounding in local attention
2. **Outlier layer detection** — Qwen layers 0,1,3,27 kept in FP16 (these have 16x median key norms)
3. **Asymmetric K/V** — keys get more bits since they affect attention scores directly
