# Changelog

All notable changes to mlx-turboquant.

## [0.7.0] — 2026-04-07

### Added

- **Fused QK scores Metal kernel** — three new Metal kernels that
  compute `Q_rot @ K^T` directly from packed codebook indices, without
  materializing dequantized K. Eliminates the per-token inverse
  rotation that dominates v0.6.0 decode overhead.
  - `mlx_turboquant.kernels.fused_qk_scores_4bit(q_rot, packed_k, norms_k, centroids, D)`
  - `mlx_turboquant.kernels.fused_qk_scores_3bit(...)`
  - `mlx_turboquant.kernels.fused_qk_scores_2bit(...)`
  - Micro-benchmark on M1 Max 64GB vs dequant+matmul baseline:
    - decode T_kv=4096 D=128: **2.12×** speedup
    - decode T_kv=1024 D=256 (Gemma3-like): **2.03×** speedup
    - decode T_kv=1024 D=128: 1.42× speedup
    - prefill is roughly tied (dispatch-bound regime)
  - Full benchmark methodology in `BENCHMARKS_v07.md`.
  - Correctness guarantee: identical to dequant+matmul to atol=1e-3 on
    random inputs across all tested shapes. 12 tests in
    `TestFusedQKScoresCorrectness{4,3,2}Bit`.

- **`pre_rotate_query(query, rotation)` utility** in `mlx_turboquant.rotation`.
  The key insight that makes the fused kernel possible: because we store
  `K = norms * (centroids[idx] @ R)`, the dot product `Q . K` simplifies
  to `norms * (Q @ R.T) . centroids[idx]`. Pre-rotating Q once per decode
  step eliminates the `D×D` inverse rotation per compressed token inside
  the attention inner loop. 5 tests in `TestPreRotateQueryMath`.

- **`docs/FUSED_ATTENTION_DESIGN.md`** — design doc covering:
  - Math derivation for the `Q @ R.T . centroids` identity
  - Kernel thread/threadgroup layout
  - Memory layout of inputs/outputs
  - Honest analysis of the integration blocker (mlx-lm's SDPA call
    accepts only dense tensors; intercepting it requires per-family
    attention patches)
  - Three considered integration approaches (A: per-family patch,
    B: custom cache method, C: utility-only — shipped in v0.7.0)
  - Numerical considerations (float32 inside kernel, rotation
    orientation, packing layout)
  - What's deferred to v0.8.0 (fusing V, SDPA integration, simd_sum
    D-reduction)

- **Phase 4 optimization: threadgroup shared centroids** in the 4-bit
  fused kernel. The 16 codebook centroids are loaded into
  `threadgroup float shared_centroids[16]` once per threadgroup, then
  every thread reads from shared memory instead of global memory during
  the D-element inner loop. Added +0.17 speedup at T_kv=4096.

- **`benchmarks/micro_fused_qk.py`** — go/no-go micro-benchmark that
  times the fused path against dequant+matmul on 7 realistic shapes.

- **`BENCHMARKS_v07.md`** — full speedup numbers, kernel architecture
  notes, and the honest "what isn't shipped in v0.7.0" section.

### Not in v0.7.0 (deferred to v0.8.0)

- **Full SDPA integration.** The fused kernels are shipped as
  first-class utilities but are NOT automatically used by
  `apply_turboquant`. `TurboQuantKVCache.update_and_fetch` still returns
  dense FP16 tensors that feed into standard
  `mx.fast.scaled_dot_product_attention`. Users who want the speedup
  today must call the kernels manually in a custom attention loop —
  see the integration pathway in `BENCHMARKS_v07.md`.
- **Fusing V.** The current work covers only the Q @ K^T side of
  attention; the softmax-weighted V accumulation still uses dequantized
  V. Fusing V requires either a two-pass kernel or online softmax.
- **`simd_sum` D-reduction.** The current kernel has one thread per
  output score with a serial D-element inner loop. A SIMD-width
  reduction would split D across 32 lanes per SIMD group.

### Changed

- No behavior changes in existing code paths. `TurboQuantKVCache`,
  `apply_turboquant`, and all v0.6.0 features are byte-identical to
  v0.6.0 when the fused kernel is not invoked. v0.6.0 → v0.7.0 is
  strictly additive.

### Tests

- 17 new tests in `tests/test_fused_attention.py`:
  - 5 `TestPreRotateQueryMath` (Phase 1)
  - 8 `TestFusedQKScoresCorrectness4Bit` (Phase 2)
  - 2 `TestFusedQKScoresCorrectness3Bit`
  - 2 `TestFusedQKScoresCorrectness2Bit`
- Full test count: **175** (158 previous + 17 new). All passing.

## [0.6.0] — 2026-04-07

### Added

- **Attention sink** (`fp16_sink_size`): Permanent FP16 region for the first
  N tokens of every sequence, never compressed regardless of compression
  cycles. Independent of the sliding `residual_window`. Use to preserve
  system prompt / tool schema tokens for long-context inference. Default
  `0` (disabled). Pattern follows helgklaizar's MLX implementation; see
  `docs/COMPETITIVE_AUDIT.md` for credit and details.
  - Quality wins on all 3 benchmarked models (Qwen3-1.7B, Qwen3-8B,
    Gemma3-1B): cosine similarity improvement of +0.0007 to +0.0032.
  - Zero measurable speed cost.
  - Validated against needle-in-haystack at 1K/2K/4K/8K — 12/12.

- **QJL residual correction** (`qjl_correction`, `qjl_n_proj`):
  Experimental opt-in 1-bit Quantized Johnson-Lindenstrauss residual
  sketch applied additively at compression time. Unlike sharpner's
  cache_v2.py which stores QJL signs but never reads them back, our
  implementation immediately consumes the sketch and bakes the
  correction into the cached dequantized vectors — zero memory
  overhead, ~5% extra compute per compression chunk.
  - Mixed benchmark results: +0.0017 cos_sim on Qwen3-8B, **−0.0052
    on Qwen3-1.7B**, near-noise on Gemma3-1B. Default OFF. Document
    explicitly mentions experimental status.
  - Synthetic Gaussian tests confirm the correction reduces MSE in
    isolation. Real-model variance stems from KV vector structure
    interacting with the random JL projection.

- **Chunked compression** (`chunk_size`): Optional fixed-size chunked
  drain path. When `chunk_size > 0`, the FP16 buffer is drained in whole
  blocks of `chunk_size` tokens whenever it exceeds `residual_window +
  chunk_size`. Default `0` selects the v0.5.0 batch behavior (single
  variable-size drain at `2 * residual_window` threshold) which
  benchmarks identically. Provided as opt-in for future Metal kernel
  work that benefits from stable input shapes.

- New benchmark harness `benchmarks/bench_v06.py` runs 5 configs across
  3 target models and writes structured JSON to `benchmarks/results_v06/`.

- New needle test harness `benchmarks/needle_haystack_v06.py` validates
  Qwen3-8B at 4 context lengths × 3 needle positions across 3 configs.

- New `docs/COMPETITIVE_AUDIT.md` documenting findings from 4 community
  TurboQuant MLX implementations (sharpner, arozanov, rachittshah,
  helgklaizar). Honest call-outs on what's real vs marketing in each
  repo, with file:line citations.

### Fixed

- **Latent state-reload bugs in `TurboQuantKVCache`** (present in v0.5.0
  but unexercised by tests):
  - `state` property now includes `_compressed_keys_lo` and
    `_compressed_values_lo` (positions 9, 10). Without these, fractional
    bit configs (e.g. `key_bits=3.5`) could not survive a state restore.
  - `meta_state` now includes `fp16_len` and `fp16_capacity`. Previously
    a state reload lost the residual buffer count, causing subsequent
    decode calls to return wrong total token counts.
  - Lazy re-dequantization after state restore is now dispatched through
    a new `_rebuild_decompressed_cache()` helper that handles fractional
    and non-fractional cases correctly. Old code called the
    non-fractional dequant helper with `None` centroids on fractional
    state reloads, which would crash.
  - State setter is backward-compatible: 6-tuple (pre-v0.6.0), 8-tuple
    (interim sink-only), and 10-tuple (current) all load correctly.

- Removed dead `_dequant_calls` counter that was incremented but never
  read.

### Changed

- `TurboQuantKVCache.__init__` and `apply_turboquant()` gain four new
  parameters: `fp16_sink_size`, `chunk_size`, `qjl_correction`,
  `qjl_n_proj`. All default to values that exactly preserve v0.5.0
  behavior — **no behavioral change unless you explicitly opt in.**

- `BENCHMARKS.md` rewritten with v0.6.0 numbers, methodology section,
  per-feature pass/fail analysis, and v0.5.0 comparison.

- `README.md` updated with v0.6.0 quality numbers, sink configuration,
  needle table, and v0.6.0 roadmap entry.

### Tests

- Added `TestAttentionSink` (8 tests) covering sink default-off,
  single-prefill fill, multi-call fill, partial overlap with residual,
  survival across compression cycles, KV head ordering, offset tracking,
  and meta_state round-trip.
- Added `TestChunkedCompression` (5 tests) for the opt-in chunked path.
- Added `TestQJLCorrection` (4 tests) including a synthetic MSE-reduction
  regression test.
- Added `TestStateReload` (3 tests) covering integer roundtrip,
  fractional roundtrip (regression for the latent bugs above), and
  legacy 6-tuple state backward compatibility.
- Total test count: **157** (up from 137 in v0.5.0). All passing.

### Deferred to v0.7.0

- Rewrite `_quant_4bit_pack_kernel` to use shared-memory pre-normalized
  vectors instead of recomputing the normalized x inside the per-byte
  inner loop. Currently O(D) redundant work per output byte; flagged in
  the competitive audit as the largest single Metal kernel win.
- Adopt `simd_sum` and SIMD-group reductions in dequant kernels per
  sharpner's pattern (`/tmp/sharpner-tq/turboquant/kernels.py:597`).
- Build a single fused attention-from-packed kernel modeled on
  sharpner's `_FUSED_ATTN_NOROT_SOURCE` to eliminate the per-step
  concat overhead that drives the residual ~11-22% decode penalty.

## [0.5.0] — 2026-04-02

- Batch compression at 2x residual window threshold
- Pre-allocated FP16 window with slice-write
- 33% → 11% decode overhead on Qwen3-8B at 2K context
- 137 tests passing

## [0.4.0] — 2026-03

- 3.5-bit fractional quantization
- Needle-in-a-haystack at 1K-8K (12/12)
- PyPI packaging

## [0.3.0] — 2026-03

- Fused Metal dequantize kernels for 2/3/4-bit
- Fused Metal quantize kernel for 4-bit
- 57% → 33% decode overhead

## [0.2.0] — 2026-02

- Real model testing across 6 model families
- Vectorized quantization
- 57% decode overhead (Python-only)
