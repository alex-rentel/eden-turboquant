# Needle-in-a-Haystack Results

Model: Qwen3-8B-4bit, K4/V2, residual_window=128
Machine: M1 Max 64GB, 2026-04-02
Secret: "AURORA-7749"

## Results

| Context | Pos 25% | Pos 50% | Pos 75% |
|---------|---------|---------|---------|
| 1K | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND |
| 2K | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND |
| 4K | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND |
| 8K | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND | FP16: FOUND, TQ: FOUND |

**Score: TurboQuant 12/12, FP16 baseline 12/12**

TurboQuant at K4/V2 with 128-token residual window achieves perfect retrieval
at all tested context lengths up to 8K, matching the FP16 baseline.

## Bottleneck Analysis

The remaining 33% decode overhead (v0.3.0) is entirely MLX per-operation dispatch:
- Each mx.concatenate costs ~0.25ms regardless of data size
- 35 TQ layers × 2 (K+V) = 70 concat ops per decode step = ~17.5ms
- Pre-allocated buffer saves <1.5ms (8% of overhead)
- Elimination requires fused attention-from-compressed kernel (reduces op count)
