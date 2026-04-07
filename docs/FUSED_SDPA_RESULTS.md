# Fused SDPA Integration Results (v0.8.0 attempt)

**Status:** experimental integration complete on `feat/fused-sdpa-qwen3`, **NOT merged to main**. End-to-end decode is slower than the standard path in current form. The v0.7.0 kernel primitives on main are unaffected and still correct.
**Date:** 2026-04-07

## TL;DR

- We built a complete per-family fused SDPA integration (patched `mlx_lm.models.base.scaled_dot_product_attention` covering qwen3, llama, qwen2, phi3, deepseek_v2, deepseek_v3 in one shot — no per-family patches needed).
- **Correctness is perfect**: real-model Qwen3-1.7B-4bit produces bit-identical last-token logits (`cos_sim = 1.000000`) between fused and standard paths, with and without the attention sink. 3 integration tests pass.
- **End-to-end speed is a regression**: 0.61×-0.99× vs standard path on decode at 1K-4K context. The fused kernel wins in isolation (v0.7.0: 2.12× at T_kv=4096) but loses once it's competing against `mx.fast.scaled_dot_product_attention`, not against `dequant + matmul`.
- **Integration branch pushed** at `origin/feat/fused-sdpa-qwen3` as a documented correctness-proof-of-concept. Main stays at v0.7.0.

## What we built

### The patch point discovery

The v0.8.0 plan called for per-family attention patches (Qwen3 branch, Llama branch, etc.). During Phase 1 research we found that all five candidate families (qwen3, llama, qwen2, phi3, deepseek_v2, deepseek_v3) import `scaled_dot_product_attention` from `mlx_lm.models.base`, and their attention `__call__` methods are nearly identical — they all just call this shared wrapper. So we replaced 6 per-family patches with a single module-level monkey-patch that swaps each module's local reference to our fused version. One loop in `install_patch()` covers all six families.

Gemma3 does not use this wrapper (it has its own alternating-window attention) and is excluded. Mistral is not a separate module in mlx-lm; it reuses `mlx_lm.models.llama.Attention`, so the Llama patch implicitly covered it.

This makes the Phase 2/3 split in the original plan redundant — the work that would have been "Qwen3 branch then Llama branch" collapsed into a single branch.

### The integration pieces (on `feat/fused-sdpa-qwen3`)

1. **Batched fused QK kernel** — `fused_qk_scores_4bit_batched(q_rot, packed_k, norms_k, centroids, D, H, H_kv)`. A 3D-grid variant of the v0.7.0 kernel that processes all `(batch, head, query, kv)` combinations in a single Metal dispatch, with GQA support (`kv_head = query_head / (H / H_kv)`). Template parameters `(D, H, H_KV)` mean one compile per model.

2. **Cache state accessor** — `TurboQuantKVCache.get_fused_state()` returns the packed state dict without copying anything: `packed_keys`, `key_norms`, `key_centroids`, `rotation`, `key_bits`, `sink_len`, `compressed_len`, `fp16_len`, `head_dim`. Plus `_use_fused_attention` flag and `has_compressed` property.

3. **`update_and_fetch` fused mode** — when `_use_fused_attention is True AND T_new == 1`, skip materializing the decompressed middle for keys. Sink + residual is returned. Values are still fully dense (we do not fuse V). For prefill (`T_new > 1`), returns dense keys as normal — the fused SDPA path only handles T_q=1 decode in v0.8.0.

4. **Patched SDPA** (`mlx_turboquant/fused_sdpa.py`) — replaces `scaled_dot_product_attention` in six supported model modules. Dispatch check: `isinstance(cache, TurboQuantKVCache) and cache._use_fused_attention and cache.has_compressed and queries.shape[2] == 1 and cache.key_bits == 4`. If any check fails, falls through to the original wrapper (which in turn handles the mlx-lm quantized cache path and falls through to `mx.fast.scaled_dot_product_attention` for FP16).

5. **`apply_turboquant(use_fused_attention=True)`** — new parameter. When True, sets `_use_fused_attention` on each cache instance and calls `install_patch()`. Idempotent; safe to re-apply. Default False.

6. **Correctness tests** — `tests/test_fused_sdpa_integration.py` loads Qwen3-1.7B-4bit and A/B's the standard vs fused path on a 500-token prompt with K4/V2 + residual_window=128:
   - With `fp16_sink_size=128`: `cos_sim = 1.000000`, top-1 match.
   - Without sink: `cos_sim = 1.000000`.
   - Short prompt (50 tokens, no compression): bit-identical fall-through.

## The benchmark

Test: A/B decode tok/s, median of 3 runs, 1 warmup, 30 decode tokens per run, K4/V2 + `fp16_sink_size=128`, `residual_window=128`. Reproduce with `python benchmarks/bench_fused_sdpa.py` on the feature branch.

### Qwen3-8B-4bit

| context | standard | fused | speedup | verdict |
|---|---|---|---|---|
| 256 | 41.5 tok/s | 41.0 tok/s | 0.99× | tie |
| 1024 | 35.8 tok/s | 30.2 tok/s | **0.84×** | loss |
| 2048 | 33.0 tok/s | 25.4 tok/s | **0.77×** | loss |
| 4096 | 27.8 tok/s | 16.9 tok/s | **0.61×** | loss |

### Qwen3-1.7B-4bit

| context | standard | fused | speedup | verdict |
|---|---|---|---|---|
| 256 | 106.8 tok/s | 123.2 tok/s | 1.15× | win |
| 1024 | 94.8 tok/s | 72.7 tok/s | **0.77×** | loss |
| 2048 | 78.0 tok/s | 60.6 tok/s | **0.78×** | loss |
| 4096 | 57.1 tok/s | 42.9 tok/s | **0.75×** | loss |

## Why it's slower (the honest post-mortem)

The v0.7.0 micro-benchmark at `BENCHMARKS_v07.md` measured `fused_qk_scores` vs `metal_dequantize + matmul` and reported a 2.12× speedup. That measurement was technically correct but **not the right comparison for the real hot path**.

In the actual decode loop, the standard path does:

```
cache.update_and_fetch(k, v)      # concat sink + decompressed_middle + residual (already materialized)
mx.fast.scaled_dot_product_attention(q, k, v, scale, mask)  # ONE hyper-optimized fused kernel
```

The decompressed middle is built INCREMENTALLY — it's only (re)built when a new chunk is compressed, not every step. At steady-state decode, the per-step cost is dominated by `mx.fast.scaled_dot_product_attention`, which computes `Q @ K^T + softmax + weighted sum over V` in a single fused kernel. It's one of the most heavily optimized ops in MLX.

Our fused SDPA decomposes this monolithic kernel back into pieces:

```
q_rot = pre_rotate_query(q, rotation)
scores_sink     = q @ k_sink.T                          # matmul 1
scores_compressed = fused_qk_scores_4bit_batched(...)   # our kernel
scores_residual = q @ k_residual.T                      # matmul 2
all_scores = concat([scores_sink, scores_compressed, scores_residual])  # concat
all_scores = all_scores * scale + mask                  # elementwise
weights = mx.softmax(all_scores, axis=-1)               # softmax
values_bc = broadcast(values, H_kv -> H)                # GQA broadcast
output = weights @ values_bc                            # matmul 3
```

Each step here is a separate Metal dispatch. Our fused_qk is ~2× faster than dequant+matmul in isolation, but the rest of the pipeline (sink matmul, residual matmul, concat, softmax, GQA broadcast, V matmul) is 6-7 additional dispatches that the standard path fuses into one call. The wins from `fused_qk` are eaten by dispatch overhead + losing access to MLX's in-house SDPA optimizations (fused softmax, fused scaling, memory-bandwidth-optimized reductions).

**The architectural takeaway**: competing against `mx.fast.scaled_dot_product_attention` means writing a *complete* fused attention kernel — not just a Q@K^T kernel — that consumes packed K (and ideally packed V too) and does scores + softmax + weighted sum in ONE dispatch. That's what sharpner's `_FUSED_ATTN_NOROT_SOURCE` does (see `docs/COMPETITIVE_AUDIT.md`): `simd_sum` reductions, threadgroup-shared softmax normalization across 32 simdgroups, online softmax state, and accumulating the V-weighted sum inline.

## What's on main after this work

Only the Phase 1 research commit (this file + `docs/attention_source_dump.txt`). The integration code is **not merged** per the user's escape-hatch rule "if the kernel doesn't outperform, don't ship it." The v0.7.0 kernels (`fused_qk_scores_4bit`, etc.) stay on main as proven-faster primitives in isolation.

## What's on `feat/fused-sdpa-qwen3`

The full integration branch, pushed to `origin/feat/fused-sdpa-qwen3`:

- `mlx_turboquant/kernels.py` — adds `fused_qk_scores_4bit_batched` (GQA-aware batched kernel, 3D grid, shared_centroids)
- `mlx_turboquant/cache.py` — adds `_use_fused_attention` flag, `get_fused_state()`, `has_compressed` property, T_new==1 gated decompressed-middle skip
- `mlx_turboquant/fused_sdpa.py` — new module with the patched `scaled_dot_product_attention` and `install_patch()` / `uninstall_patch()`
- `mlx_turboquant/patch.py` — adds `use_fused_attention=False` parameter
- `tests/test_fused_attention.py` — adds `TestFusedQKScoresBatched4Bit` (6 tests, GQA correctness)
- `tests/test_fused_sdpa_integration.py` — adds 3 real-model end-to-end tests (all pass, cos_sim == 1.000000)
- `benchmarks/bench_fused_sdpa.py` — A/B harness that produced the numbers above

Tests on the branch: 185/185 pass (175 v0.7.0 + 6 batched kernel + 3 integration + 1 previous integration).

## What would actually win (deferred to v0.9.0 or later)

A single Metal kernel that implements a complete fused attention over packed KV, modeled on sharpner's `_FUSED_ATTN_NOROT_SOURCE`:

```
fused_attention_from_packed(
    q_rot,           # (B, H, T_q, D)
    packed_k,        # (B, H_kv, T_kv, D/2)
    norms_k,         # (B, H_kv, T_kv)
    packed_v,        # (B, H_kv, T_kv, D/2)  -- V must also be packed!
    norms_v,         # (B, H_kv, T_kv)
    k_centroids,
    v_centroids,
    fp16_sink_k, fp16_sink_v,
    fp16_residual_k, fp16_residual_v,
    scale, mask,
) -> output          # (B, H, T_q, D)
```

Engineering cost:
- One kernel per (D, H, H_kv) combination, doing all the work: Q@K score + sink/residual score + online softmax + weighted sum over V
- Needs `simd_sum` and threadgroup memory for the softmax normalization (32 threads per simdgroup, 1 simdgroup per head-batch)
- Needs online softmax state (running max and running sum) because softmax normalization requires all scores first
- Needs to fuse V, which means V must also be packed (v0.6.0 leaves V dequantized for speed; this becomes a chicken-and-egg problem)

Estimated effort: 2-4 days of Metal kernel work, with risk of correctness bugs that are hard to debug. Not fit for this session. **Deferred**.

## Lessons learned

1. **Micro-benchmarks must match the real hot path's comparison.** The v0.7.0 micro-benchmark showed our kernel beat dequant+matmul 2.12×. That's true in isolation, but the real decode path does NOT do dequant+matmul every step — it uses `mx.fast.scaled_dot_product_attention` on an already-incrementally-built dense cache. The right question was never "can we beat dequant+matmul" but "can we beat MLX's fused SDPA." The answer is no, not by decomposing it.

2. **Always run the end-to-end A/B before committing to integration.** The branch exists to contain the negative result. Main stays clean.

3. **`mx.fast.scaled_dot_product_attention` is the baseline to beat.** Any competing attention implementation must either be a full fused kernel of its own, or target contexts so large that mlx's SDPA becomes memory-bound (which is where sharpner's kernel is reported to win — but we didn't reproduce that here).

4. **The patched-SDPA approach is still the right shape for future work.** When we eventually write a full fused attention kernel in v0.9.0+, it can slot into the same `install_patch()` mechanism. All the infrastructure on `feat/fused-sdpa-qwen3` (dispatch, cache state accessor, T_new==1 gating, 6-module patch installer) is reusable.
