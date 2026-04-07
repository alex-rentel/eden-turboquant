# Full Fused Attention Kernel — Results (v0.9.0 attempt)

**Status:** experimental kernel complete and correct on `feat/full-fused-attention`, **NOT merged to main**. The kernel loses to `mx.fast.scaled_dot_product_attention` at every tested decode shape. Main stays at v0.7.0 + docs.
**Date:** 2026-04-07

## TL;DR

- Built a complete single-dispatch fused attention kernel with online softmax, lane-sharded D, packed 2-bit K and V, GQA support. Modeled on sharpner's `_FUSED_ATTN_NOROT_SOURCE` but adapted for our uint8 2-bit packing layout (our packing maps even more cleanly onto the 32-lane pattern than sharpner's uint32).
- **Correctness is perfect**: all 5 correctness tests pass (minimal single-head, small T_kv, Qwen3 GQA shape, long-context T_kv=2048, no-GQA) against the dequant+standard-attention reference, to `atol=2e-3` on random inputs.
- **Speed is a regression**: 0.64×-0.86× vs `mx.fast.scaled_dot_product_attention` on dense K/V across `T_kv = 256, 512, 1024, 2048, 4096, 8192`. The 4K context is the WORST ratio (0.64×), the opposite of what the back-of-envelope predicted.
- **Root cause**: both paths are dispatch/latency/compute-bound, not memory-bound, at realistic decode shapes. The packed-KV memory advantage (~5× less data) never materializes because neither path is memory-limited. `mx.fast.sdpa` at T_kv=4K is only ~4.4× the memory-bandwidth limit; our kernel is ~57× the limit. MLX's SDPA is too well optimized to beat at these scales.
- **Decision**: follow the user's explicit rule — document and stop. Main stays at v0.7.0. The branch is preserved as a library primitive for users who want to experiment.

## What was built (on `feat/full-fused-attention`)

### The kernel

`mlx_turboquant.kernels.fused_attention_2bit_2bit` — a single Metal dispatch that computes `softmax(Q @ K.T / sqrt(D)) @ V` in one kernel, reading packed 2-bit K and 2-bit V indices. Adapted from sharpner's `_FUSED_ATTN_NOROT_SOURCE`.

**API:**

```python
fused_attention_2bit_2bit(
    q_rot,       # (H_q, D) float32  — pre-scaled and pre-rotated
    packed_k,    # (H_kv, T_kv, D/4) uint8
    norms_k,     # (H_kv, T_kv) float32
    packed_v,    # (H_kv, T_kv, D/4) uint8  — stored in rotated space
    norms_v,     # (H_kv, T_kv) float32
    centroids,   # (4,) float32  — shared K/V codebook
    H_q, H_kv, T_kv, D,
) -> (H_q, D) float32 in rotated space
```

Caller is responsible for pre/post rotation:
```python
q_scaled = Q / sqrt(D)
q_rot = pre_rotate_query(q_scaled, rotation)
output_rot = fused_attention_2bit_2bit(q_rot, ...)
output = output_rot @ rotation  # single inverse rotation, amortized over all tokens
```

### Kernel architecture

- **Grid:** `(H_q * 1024, 1, 1)` — one threadgroup per query head.
- **Threadgroup:** `(1024, 1, 1)` = 32 simdgroups × 32 lanes.
- **Work split:** each simdgroup walks a stride-32 slice of `T_kv`. Lane L permanently owns output dims `[L*4, L*4+3]`.
- **QK reduction:** `simd_sum` across the 32 lanes per kv-token.
- **V accumulator:** register-only (`local_acc[4]` per lane), never reduced — D-dim parallelism is provided by lane sharding, not by reduction.
- **Online softmax state:** register-only (`local_max`, `local_sum`) during the `T_kv` loop. Single `threadgroup_barrier` at the very end for cross-simdgroup combine via `tg_max[32]`, `tg_sum[32]`, `tg_acc[32*128]` in shared memory.
- **V rotation trick:** V is accumulated in rotated space (centroids[v_idx] directly, never inverse-rotated per token). Single `output @ rotation` at the end applies the inverse rotation ONCE per query head, not per kv-token.
- **GQA:** `kv_head = query_head / (H_q / H_kv)`, pure integer arithmetic inside the kernel.

### Our uint8 layout vs sharpner's uint32

Sharpner packs into `uint32` words (16 2-bit values per word) and uses a bit-base mapping: `word_idx = lane_id >> 2`, `bit_base = (lane_id & 3) << 3`. Each lane reads 4 bytes (1 uint32 word) but those bytes cover 16 dims, only 4 of which belong to that lane.

Our uint8 layout is simpler: `bytes_per_token = D/4` = 32 at D=128, and lane L reads byte L directly. One uint8 load per kv-token per lane, containing exactly the 4 indices for that lane's 4 output dims. No bit-base shuffling. The adaptation was essentially "replace the uint32 load + bit math with a uint8 load + shift."

### Correctness

5 new tests in `tests/test_full_fused_attention.py`, all passing on first compile (once the header newline bug was fixed, see below):

| Test | H_q | H_kv | T_kv | D | Max diff vs reference |
|---|---|---|---|---|---|
| `test_minimal_single_head` | 1 | 1 | 4 | 128 | < 2e-3 |
| `test_small_tkv` | 1 | 1 | 32 | 128 | < 2e-3 |
| `test_qwen3_shape` | 32 | 8 | 256 | 128 | < 2e-3 |
| `test_long_context` | 32 | 8 | 2048 | 128 | < 2e-3 |
| `test_no_gqa` | 8 | 8 | 128 | 128 | < 2e-3 |

Reference path: dequant K via `metal_dequantize`, dequant V via `metal_dequantize`, `mx.repeat` for GQA broadcast, standard scaled-softmax attention in MLX. Total agreement across all tested shapes.

### Implementation hiccup worth logging

MLX `metal_kernel` concatenates the user's `header` string directly before its generated `template <int D, ...>` line **without a separator**. A header of `"#include <metal_simdgroup>"` without a trailing newline produces:

```metal
#include <metal_simdgroup>template <int D, int HQ, int HKV, int TKV>
```

which fails to parse. Fix: `header="#include <metal_simdgroup>\n"`. The error message cascades into dozens of Metal compile errors that look like template-instantiation bugs but are all downstream of the single missing newline.

## The benchmark

`benchmarks/micro_full_fused_attention.py` on the branch. Times our fused kernel against `mx.fast.scaled_dot_product_attention` operating on **dense FP32** K/V of the same shape, Qwen3-8B GQA layout (H_q=32, H_kv=8, D=128). Median of 30 trials, 5 warmup.

### Results (M1 Max 64GB, mlx 0.31.1)

| shape | `mx.fast.sdpa` | full fused | speedup | verdict |
|---|---|---|---|---|
| Qwen3-8B T_kv=256 | 296.8 μs | 367.7 μs | 0.81× | **loss** |
| Qwen3-8B T_kv=512 | 276.0 μs | 366.4 μs | 0.75× | **loss** |
| Qwen3-8B T_kv=1024 | 339.4 μs | 451.3 μs | 0.75× | **loss** |
| Qwen3-8B T_kv=2048 | 307.4 μs | 356.9 μs | 0.86× | **loss** |
| Qwen3-8B T_kv=4096 | 365.7 μs | 569.6 μs | **0.64×** | worst |
| Qwen3-8B T_kv=8192 | 446.4 μs | 561.3 μs | 0.80× | **loss** |

Fused kernel loses at **every** tested T_kv. The back-of-envelope predicted the 4K case would be our best win (from memory-bandwidth analysis); it turned out to be our worst ratio.

## Why the memory-bandwidth prediction failed

### What I expected

The v0.9.0 design doc estimated:
- At T_kv=4K D=128, dense K+V memory traffic per step ≈ 33.5 MB
- M1 Max memory bandwidth = 400 GB/s → mem-bound limit ≈ 83 μs
- Packed K+V traffic ≈ 4 MB → mem-bound limit ≈ 10 μs
- Expected ~1.5-2× fused speedup at 4K, scaling up at 8K+

### What actually happened

- `mx.fast.sdpa` at T_kv=4K measured **365 μs** — that's ~4.4× the dense mem-bound limit. MLX's SDPA is already running at near-memory-bound efficiency for dense data.
- Our fused kernel at T_kv=4K measured **570 μs** — that's **57×** the packed mem-bound limit. We're nowhere near memory-bound.
- Sub-linear scaling of both paths (296 → 446 μs for T_kv 256 → 8192 on `mx.fast.sdpa`; 367 → 561 μs on fused) confirms both are dominated by fixed overheads (dispatch, kernel launch, threadgroup setup, latency chains), not by data throughput.

### Root-cause dominators in our kernel

1. **Centroid indirection.** Every K and V lookup is a 2-bit index into a 4-entry codebook. That's a random 4-way gather per lane per kv-token. The M1 Max's hardware prefetcher doesn't help here — the access pattern is data-dependent. Contrast with dense FP32 K/V, which is a pure strided read the prefetcher handles perfectly.

2. **High thread count, low work per thread.** We launch 32 query heads × 1024 threads = 32,768 threads per dispatch. Each thread does O(T_kv/32 × 16) = O(T_kv/2) multiply-adds. At T_kv=4K that's ~2000 FMAs per thread — too little to amortize the threadgroup launch and scheduling overhead.

3. **Per-token scalar reads (`norms_k`, `norms_v`).** Every kv-token requires two scalar fetches that can't be coalesced with the packed byte reads.

4. **Inefficient `metal::fast::exp`.** Called twice per kv-token per simdgroup. At T_kv=4K and 32 simdgroups, that's 256K `exp` calls per layer per step. Metal's `fast::exp` is fast but not free.

5. **Online softmax rescaling.** Each kv-token does 4 `acc *= alpha` multiplications per lane. At T_kv=4K and 32 lanes and 32 query heads, that's 16M multiply-adds per layer per step.

6. **Cross-simdgroup combine.** Done by all 1024 threads redundantly across 32 simdgroups. Cheap per-thread (32 iterations) but it's 32K iterations total per query head.

None of these are showstoppers in isolation. Added together they put us at ~57× the memory-bound limit, and `mx.fast.sdpa` is already at ~4.4× (on dense data), so the packed-data advantage is buried by the compute overhead.

## Deeper lesson — from v0.8.0 and v0.9.0 together

Both v0.8.0 (decomposed fused SDPA) and v0.9.0 (single-dispatch fused kernel) tried to beat `mx.fast.scaled_dot_product_attention` at decode-time attention, each with a different strategy, and both lost. The pattern is consistent:

- **v0.8.0 lost by decomposition.** Our fused QK kernel was 2× faster than dequant+matmul in isolation. When we decomposed the SDPA call into fused_qk + sink matmul + residual matmul + softmax + V matmul, the dispatch overhead of 6-7 separate ops buried the per-op savings.
- **v0.9.0 lost despite recomposition.** We wrote a complete single-dispatch kernel with online softmax and packed V. No dispatch overhead issue this time. It's still slower because `mx.fast.sdpa` is running at ~4× its theoretical memory-bandwidth limit and our custom kernel runs at ~57× its own (smaller) limit. The compute/latency overhead of centroid indirection + online softmax eats the memory advantage.

**The structural conclusion**: `mx.fast.scaled_dot_product_attention` is not a wall you can break by writing a better kernel around it. It is already optimized to within ~4× of the memory-bandwidth ceiling on dense data. To actually beat it you'd need one of:

1. **An upstream MLX change** that makes `mx.fast.sdpa` itself aware of packed K/V, so the same hyper-optimized Apple kernel handles our compression layout directly. This is the correct path but requires a PR into `mlx` core, not a userland library.

2. **Much longer contexts (32K+).** At some point the `mx.fast.sdpa` T_kv scaling becomes linear and memory-bandwidth becomes the dominant cost. Our benchmark topped out at T_kv=8K and didn't reach that regime. If Qwen3-8B at T_kv=32K puts `mx.fast.sdpa` at 1.5× the mem-bound limit (not 4×), and our kernel can get to 2× (not 57×), then we finally win. Left for future investigation.

3. **A compression that eliminates indirection.** Bitplane packing, affine-integer quantization with FMA-friendly inner loops, or an NF4-style lookup-free encoding. All of these are major departures from TurboQuant's algorithm.

4. **Precomputed Q @ codebook vectors.** If the centroid set is small (4 for 2-bit), you could precompute `q . c[j]` for all 4 centroids ONCE per query and then the per-kv-token work becomes a pure table gather with no multiplications. Could be 2-3× faster than our current approach. Worth trying in v1.0.0.

## What's on main after this work

Only this results document. The kernel and benchmark live on `feat/full-fused-attention` (pushed). Main stays at v0.7.0 (fused QK primitives + pre_rotate_query) plus the v0.8.0 and v0.9.0 docs under `docs/`.

## What's on `feat/full-fused-attention` (pushed, NOT merged)

- `mlx_turboquant/kernels.py` adds `fused_attention_2bit_2bit` and `_FUSED_ATTN_2BIT_SOURCE` (the Metal kernel itself).
- `mlx_turboquant/cache.py` extends `get_fused_state()` with packed V state (`packed_values`, `value_norms`, `value_centroids`, `value_bits`).
- `tests/test_full_fused_attention.py` — 5 correctness tests.
- `docs/FULL_FUSED_KERNEL_DESIGN.md` — the design doc.
- `benchmarks/micro_full_fused_attention.py` — the A/B benchmark.
- Builds on the `feat/fused-sdpa-qwen3` branch, which itself has the v0.8.0 SDPA dispatch infrastructure. Both branches stay as documented experiments. Full test count on the branch: 186/186.

## Revised roadmap

- **v0.7.0 (shipped)**: fused QK kernels as primitives, 2.12× vs dequant+matmul.
- **v0.8.0 (branch only)**: decomposed SDPA integration, correct but slower. Stays on `feat/fused-sdpa-qwen3`.
- **v0.9.0 (branch only, this work)**: single-dispatch fused attention kernel, correct but slower. Stays on `feat/full-fused-attention`.
- **v1.0.0 (not yet started)**: pursue one of the four alternative paths listed above. Most promising short-term: precomputed `q . c[j]` table to eliminate the per-token multiply. Most promising long-term: an upstream MLX PR for packed-KV `mx.fast.sdpa`.

## Files on this branch

```
docs/FULL_FUSED_KERNEL_DESIGN.md       — v0.9.0 design doc (247 lines)
mlx_turboquant/kernels.py              — kernel added at L~635 (fused_attention_2bit_2bit + _FUSED_ATTN_2BIT_SOURCE)
mlx_turboquant/cache.py                — get_fused_state() extended with packed V
tests/test_full_fused_attention.py     — 5 correctness tests (all pass)
benchmarks/micro_full_fused_attention.py — Phase 4 A/B benchmark
```

186/186 tests pass on the branch.
