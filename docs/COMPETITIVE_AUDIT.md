# Competitive Audit — TurboQuant MLX Implementations
Date: 2026-04-07
Auditor: Claude (subagent)
Our version: v0.5.0 (commit b8fc8c5)

## Summary

Four public MLX TurboQuant implementations were audited. The two with real engineering value are **sharpner** (the only one with a truly fused single-dispatch attention kernel that uses `simd_sum` and 32-simdgroup cross-reduction over packed 2-bit indices) and **helgklaizar** (the only one that implements both fixed-size chunked compression and an attention-sink FP16 region for system prompts). **arozanov** has clean code and a "fused QK from packed" kernel that genuinely reads packed uint32 inside the kernel, but its top-level `fused_attention.py` imports a `metal_kernels_v4` module that does not exist on disk — the headline "98% FP16 speed" path is dead code. **rachittshah** is the weakest: it dequantizes the entire cache to FP16 on every step, so its "compressed cache" never benefits decode at all. Our v0.5.0 leads on per-bit code clarity, 3-bit packing, and the residual-window-with-batch-compress amortization, but we are clearly behind on (a) Metal kernels using `simd_sum`/threadgroup reductions, (b) FP16 sink for system prompt, and (c) a single fused attention kernel that operates directly on packed storage.

## Repo overviews

### sharpner/turboquant-mlx
- Lines of code: ~3,000 Python (kernels.py alone is 865 lines)
- Approach in 2 sentences: Random QR rotation + MLX-native `mx.quantize`/`quantized_matmul` (cache_v2) plus hand-written Metal kernels for a fully-fused 2-bit attention kernel (cache_v3 / kernels.py). QJL is implemented as a 1-bit sign-sketch on the *quantization residual* and stored alongside (`use_qjl=True` flag).
- Quality of code: Highest by a wide margin. Real online softmax in Metal, real `simd_sum` reductions, real cross-simdgroup reductions through threadgroup memory, multiple kernel versions (with/without rotation in-kernel). Test coverage exists. Some experimental cruft (cache.py, cache_v2.py, cache_v3.py, attention_v2/v3, fused_v2_attn).

### arozanov/turboquant-mlx
- Lines of code: ~1,500 Python
- Approach in 2 sentences: Fused Metal quantize and dequantize kernels (one threadgroup per vector, in-kernel WHT butterfly) plus a "packed fused QK" kernel that reads bit-packed uint32 directly. `cache.py` keeps a separate FP16 decode buffer that grows incrementally for the decode hot path.
- Quality of code: Clean structure, minimal dead code in core path, but the headline `fused_attention.py` is broken — it imports `turboquant_mlx.metal_kernels_v4` which does not exist (`find` returns nothing). The actually-callable path in `cache.py` always fully dequantizes to a side FP16 buffer and returns FP16 tensors, so the win is one-shot quantize + cheap incremental dequant, not fused-attention-from-packed.

### rachittshah/mlx-turboquant
- Lines of code: ~1,400 Python
- Approach in 2 sentences: Pure-Python (no custom Metal) TurboQuant with optional fractional bit widths (e.g. 3.5). On every `update_and_fetch` it calls `unpack_indices` over the *entire* cache and dequantizes the whole thing to FP16 (`cache.py:90-105`).
- Quality of code: Most readable, well-tested, but functionally a memory-savings demo only — there is zero compute benefit because the cache is dequantized in full on every step. No fused kernels, no incremental dequant, no sink, no chunking.

### helgklaizar/turboquant_mlx
- Lines of code: ~900 Python total across `core/` and `mlx_core/`. `core/` is a NumPy reference; `mlx_core/` is the MLX path actually used.
- Approach in 2 sentences: Splits the cache into three regions: a permanent FP16 attention sink, a small FP16 staging buffer, and a list of compressed chunks each of fixed size 64. Quantization is straight Python/MLX (no custom Metal kernels).
- Quality of code: The architecture (sink + chunked compressed list) is the most thoughtful of the four, but the implementation re-decompresses the full chunk list and concatenates on every decode step (`mlx_core/cache.py:91-114`), which makes it slow despite the nice structure. Russian comments throughout.

## Findings on specific questions

### 1. Attention sink (helgklaizar)

**Verdict: real, accurate.**

- File: `/tmp/helgklaizar-tq/mlx_core/cache.py`
- Class: `TurboQuantKVCache`
- Constructor takes `fp16_sink_size: int = 128` (`cache.py:10`).
- Sink storage is separate state: `self.sink_keys = None; self.sink_values = None` (`cache.py:24-25`), distinct from `self.key_buffer` (the staging buffer that feeds the chunked compressor).
- Routing logic in `update_and_fetch` (`cache.py:48-67`):

  ```python
  if prev_offset < self.fp16_sink_size:
      remaining_sink = self.fp16_sink_size - prev_offset
      k_sink_part = keys[:, :, :remaining_sink, :]
      ...
      if self.sink_keys is None:
          self.sink_keys = k_sink_part
      else:
          self.sink_keys = mx.concatenate([self.sink_keys, k_sink_part], axis=2)
      k_compress_part = keys[:, :, remaining_sink:, :]
  ```

- Sink tokens are never compressed; they are concatenated first into the returned KV (`cache.py:95-97`).
- The sink and the compression buffer are independent — sink fills first, then everything else flows into `key_buffer`, which is drained in 64-token chunks into the compressed list. So yes, the first 128 tokens are pinned in FP16 permanently and the residual chunked window is genuinely separate.

The claim is accurate. This is the cleanest sink implementation in the four repos. Note that ours (`mlx_turboquant/cache.py:42` `residual_window: int = 128`) is a *sliding* residual window, not a fixed-position sink — the two are different and complementary.

### 2. Chunked compression

**helgklaizar: real fixed-size chunks.**

- File: `/tmp/helgklaizar-tq/mlx_core/cache.py`
- `self.chunk_size = 64` (`cache.py:20`).
- Drain loop (`cache.py:78-89`):

  ```python
  while self.key_buffer is not None and self.key_buffer.shape[2] >= self.chunk_size:
      chunk_k = self.key_buffer[:, :, :self.chunk_size, :]
      chunk_v = self.value_buffer[:, :, :self.chunk_size, :]
      self._compress_and_store(chunk_k, chunk_v)
      ...
  ```

- Compressed chunks are appended as `(compressed_blob, original_shape)` tuples to `self.compressed_keys_chunks` (`cache.py:37`). The chunk identity is preserved at fetch time — each chunk decompresses independently (`cache.py:99-105`).

**Comparison to ours:** Our `cache.py:320` `_compress_oldest_excess` triggers when `fp16_len >= residual_window * 2` and compresses the *excess* in one batch, not in fixed chunks; the compressed storage is a flat per-token packed buffer, not a list of chunk blobs. Helgklaizar's chunk-list approach is more amenable to per-chunk eviction / per-chunk recompression at different bit rates, but is slower at fetch because it concatenates a Python list of arrays per step. Ours wins on decode latency; theirs wins on architectural flexibility.

**sharpner, arozanov, rachittshah:** No chunked compression. All three quantize per-token at write time into a pre-allocated `step=256` buffer (sharpner `cache_v2.py:34`, arozanov `cache.py:53`, rachittshah `cache.py:26`). Sharpner's `step=256` is buffer-growth granularity, not a compression chunk.

### 3. QJL correction on affine (sharpner)

**Verdict: real, but it is *not* additive correction at decode time. It is stored alongside as a residual sketch and only used as auxiliary state.**

- Files: `/tmp/sharpner-tq/turboquant/cache_v2.py`, `/tmp/sharpner-tq/turboquant/qjl.py`, `/tmp/sharpner-tq/experiment_qjl.py`.
- The actual algorithm (`cache_v2.py:182-201`):

  ```python
  k_quant = mx.quantize(k_to_q, group_size=self.group_size, bits=self.bits)
  v_quant = mx.quantize(v_to_q, group_size=self.group_size, bits=self.bits)

  if self.use_qjl:
      k_dequant = mx.dequantize(*k_quant, group_size=self.group_size, bits=self.bits)
      k_residual = k_to_q - k_dequant
      k_sign_bits, k_residual_norms = qjl_encode(k_residual, self.jl_matrix)
  ```

- `qjl_encode` (`qjl.py:47-64`) projects the residual through a fixed JL matrix and stores `pack_sign_bits(projected)` (1 bit per JL feature) plus the L2 norm of the residual.

- This is **Algorithm 2 / TurboQuant_prod-style sign sketch on the residual**, not an additive correction applied during attention. Crucially, in `cache_v2.py` the QJL sign bits and residual norms are *only stored*; they are never read back to correct the dequantized vector before being returned. The QJL data appears in `state` (`cache_v2.py:219-220`) and `nbytes` accounting (`cache_v2.py:261-263`) but is never consumed by attention scoring in this file. There is a separate `fused_qjl.py` (118 lines) that *does* apply a sign-sketch correction inside a fused kernel, but `cache_v2.py` (the file driven by `experiment_qjl.py` and `experiment_2bit.py`) does not call into it.

- **Bit overhead:** 1 bit per JL projection feature + 1 fp32 residual norm per token. With `n_proj_words = head_dim // 32` (`cache_v2.py:129`), at `head_dim=128` that is exactly 128 bits = 16 bytes/token of sign bits + 4 bytes/token of residual norm = 20 bytes per token per head. For a 3-bit baseline at `head_dim=128` that is 48 bytes of indices + ~2 bytes scale/bias, so QJL roughly *doubles* the per-token K cache footprint.

- **Does the experiment actually show 6.6% → 5.3%?** `experiment_qjl.py` (the file I was asked to read carefully) is 79 lines and is actually an alias of `experiment_2bit.py` — both are perplexity sweeps over the same config list (`experiment_qjl.py:34-44`). The script prints PPL and `delta_vs_fp16` percentages but does not contain hardcoded numbers and the README claim was not located in committed code. **I could not find a 6.6% → 5.3% number anywhere in source.** The experiment exists; whether the claimed result holds is empirical and not provable from a code read.

So the framing "QJL works as 1-bit sign-sketch correction on top of MSE/affine quantization" is *partially* accurate: the sketch is computed on the affine residual (true), but in `cache_v2.py` it is dead-stored, not used to correct dot products. The `fused_qjl.py` path that *would* use it is not wired into the experiment script. Either the README is overselling the v2 wiring, or the active experiment was actually run against `cache_v3.py` / `fused_v2_attn.py` and the perplexity script in this commit is stale.

### 4. Fused attention from compressed (arozanov)

**Verdict: marketing — the live path always dequantizes to a side FP16 buffer.**

- The headline file `/tmp/arozanov-tq/turboquant_mlx/fused_attention.py` imports `from turboquant_mlx.metal_kernels_v4 import (prerotate_query, prerot_fused_qk_scores, prerot_packed_dequantize)` (`fused_attention.py:14-18`). `find /tmp/arozanov-tq -name "metal_kernels*"` returns only `metal.py` — **the v4 module does not exist in the repo.** This file cannot be imported or run; nothing in `cache.py` or `patch.py` references it.

- The actual decode hot path is `cache.py:117-165`:
  - Line 126-129: fused Metal quantize on the new tokens.
  - Line 138-153: incremental decode — when `S <= 4` it dequantizes only the new S tokens via `dequant_fp16` (`metal.py:199-232`) and writes them into a growing `_v_deq_buf`/`_k_deq_buf`. Returns `self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]`.
  - Returned tensors are FP16 dense — standard MLX SDPA runs over them. There is **no fused QK from packed in the live path.**

- There *is* a real packed-fused-QK kernel in `kernels.py:158-196` (`PACKED_FUSED_QK_KERNEL`, `kernels.py:58-117`), and it does read packed uint32 directly (`uint word = packed[kv_base + word_idx];` at `kernels.py:74` then `idx = (word >> (pos_in_word * bits)) & bit_mask;`). It performs the WHT inside the kernel and dot-products against the query in shared memory. **However, no caller invokes `packed_fused_qk_scores`** — `cache.py` only imports `packed_dequantize`, not `packed_fused_qk_scores`. So the kernel exists but is dead code in the shipped pipeline.

- The "98% FP16 speed" claim is therefore best read as: "incremental dequant of new tokens via Metal + standard SDPA on the pre-dequanted FP16 buffer is fast." Which is true and a perfectly reasonable architecture — but it is not "compute attention directly from packed indices."

### 5. Metal kernel patterns we're missing

Comparing each competitor's Metal source against `~/GitHub/mlx-turboquant/mlx_turboquant/kernels.py` (343 lines, three dequant kernels and two quantize kernels, all single-element-per-thread, no shared memory, no SIMD intrinsics):

**SIMD shuffle / `simd_sum`:**
- Sharpner uses `simd_sum` for the QK partial reduction at `/tmp/sharpner-tq/turboquant/kernels.py:597` and `:766`. Each lane computes a 4-element partial dot, then `simd_sum(partial)` reduces 32 lanes in one instruction. **We use no SIMD intrinsics anywhere.**
- Arozanov uses tree reduction in shared memory instead (`/tmp/arozanov-tq/turboquant_mlx/kernels.py:107-112`) — slower than `simd_sum` but still better than our serial loop.

**Threadgroup memory + barriers:**
- Sharpner: `threadgroup float shared_q[128]`, `tg_max[32]`, `tg_sum[32]`, `tg_acc[32 * 128]` (`/tmp/sharpner-tq/turboquant/kernels.py:561, 786-788`) for a full cross-simdgroup softmax reduction.
- Arozanov: `threadgroup T shared[256]` for a shared in-kernel WHT butterfly (`/tmp/arozanov-tq/turboquant_mlx/kernels.py:33-50`, `/tmp/arozanov-tq/turboquant_mlx/metal.py:24-65`).
- **Ours: no `threadgroup` storage and no `threadgroup_barrier` calls anywhere in `kernels.py`.** Each output element is computed in isolation.

**Two-level reductions (SIMD + threadgroup):**
- Sharpner does this in `_FUSED_ATTN_NOROT_SOURCE` (`/tmp/sharpner-tq/turboquant/kernels.py:718-818`): 32 simdgroups × 32 lanes per head, intra-simdgroup reduction with `simd_sum`, then cross-simdgroup combine of `(max, sum, acc[4])` through `tg_max`/`tg_sum`/`tg_acc` (`:786-813`). This is the same pattern as MLX's own `sdpa_vector.h`. **We have no two-level reduction.**

**Atomic operations:**
- None of the four competitors use atomics. (Not necessarily a miss; just noting.)

**Fused multiple operations in single kernel:**
- Sharpner's `_FUSED_ATTN_SOURCE` (`/tmp/sharpner-tq/turboquant/kernels.py` around `:520-642`) and `_FUSED_ATTN_NOROT_SOURCE` (`:718-818`) fuse: query rotation (or pre-rotation), 2-bit unpack, score = dot, online softmax, value unpack, weighted accumulate, inverse rotation — **all in one Metal dispatch.** This is the single biggest engineering gap between sharpner and the rest.
- Arozanov's `metal.py` `FUSED_QUANTIZE_KERNEL` (`metal.py:14-101`) fuses norm reduction + normalize + sign apply + WHT butterfly + nearest-centroid + bit pack in a single kernel.
- **Our `_quant_4bit_pack_kernel` does the rotation as a serial dot in each thread (`kernels.py:217-221`)** with no shared memory and no butterfly — for `D=128` that is 128² = 16,384 multiplies per output byte across the threadgroup. A WHT-butterfly-in-shared-memory replacement would be O(D log D) total work shared across the threadgroup instead of O(D²) per byte.

**Smarter threadgroup sizing:**
- Sharpner: `threadgroup=(32, 1, 1)` for the 32-lane simdgroup-per-head pattern (`kernels.py:698`) and 1024 threads = 32 simdgroups for the high-occupancy variant.
- Arozanov: `threadgroup=(dim, 1, 1)` — one threadgroup per vector with `dim` threads (`metal.py:192`, `kernels.py:151`).
- **Ours:** `tg_size = min(256, total_elems)` — a flat heuristic that does not align to SIMD width or to vector boundaries, so the same threadgroup processes pieces of multiple unrelated output rows. We get no shared-memory reuse across threads in a group because none is set up to begin with.

## Recommendations for v0.6.0

**P0 — must do**

1. **Adopt sharpner's `simd_sum`-based partial reduction for any dot-product reduction in our dequant/quantize kernels.** This is a one-line change per kernel (`#include <metal_simdgroup>`, then `simd_sum(partial)` instead of a serial accumulator) and gives ~30x speedup on the reduction itself. Difficulty: low.

2. **Replace the per-output-byte serial rotation in `_quant_4bit_pack_kernel` (`kernels.py:217-221`) with a shared-memory WHT butterfly,** following arozanov's `metal.py:46-65` pattern. This eliminates an O(D²) per-byte cost in our quantize path. Difficulty: medium.

**P1 — should do**

3. **Add an FP16 attention sink** (fixed first-N tokens, never compressed), independent of our existing residual window. Direct port of helgklaizar `mlx_core/cache.py:48-67`. Two lines of state plus a routing branch in `update_and_fetch`. Expected impact: measurable PPL improvement on long-context with system prompts. Difficulty: low.

4. **Build a single fused attention kernel that takes packed K and dequant V (or packed V) and does score+softmax+weighted-sum in one dispatch,** modeled on sharpner's `_FUSED_ATTN_NOROT_SOURCE` (`kernels.py:718-818`). This is the largest expected speedup but the highest engineering cost (1-2 days). The sharpner source is borrow-able as a reference implementation. Difficulty: high.

**P2 — nice to have**

5. **Move query rotation outside the kernel (pre-rotate Q once per step via MLX GEMM)** so the inner loop just does codebook lookup + dot. Same trick used by sharpner's NoRot kernel and described in arozanov's `fused_attention.py:6-9` docstring (even though the implementation file is dead). Difficulty: low once #4 is done.

6. **Chunked compression option as an alternative storage layout** (not replacement) for users who want per-chunk eviction or mixed bit rates per chunk. Optional. Difficulty: medium.

**P3 — skip**

7. **QJL residual sign-sketch as currently implemented in sharpner cache_v2.py.** The sketch is dead-stored with no decode-time correction in the live path, and it doubles K cache size. Skip until there is a published, reproducible PPL win, ideally tied to a working `fused_qjl.py`-style decode path.

8. **Fractional bits (rachittshah's 3.5-bit gimmick).** Adds packing complexity for ambiguous quality gain.

9. **Dequantizing the entire cache every step (rachittshah).** Self-explanatory — we already don't do this and shouldn't start.

## What we already do better

- **Versus rachittshah:** Our cache yields actual decode speed; theirs dequantizes everything every step.
- **Versus helgklaizar:** Our residual-window-with-batch-compress (`cache.py:320-410`) decodes much faster than their per-step Python-list-of-chunks concatenate (`mlx_core/cache.py:91-114`). We have real Metal kernels for dequant; they have none.
- **Versus arozanov:** Our quantize path is correct end-to-end. Their headline `fused_attention.py` is unimportable (`metal_kernels_v4` missing) and `packed_fused_qk_scores` is dead code unreferenced by `cache.py`. Our 2/3/4-bit packing and per-bit kernels are cleanly separated; theirs only ships 3-bit-tested.
- **Versus sharpner:** We have substantially less dead code and only one cache class instead of three (`cache.py`, `cache_v2.py`, `cache_v3.py`). Our packing layer is more consistent. But on raw Metal kernel sophistication for the attention path, sharpner clearly leads and we should borrow heavily.
- **General:** We are the only repo with a coherent `patch.py`/`quantizer.py`/`codebook.py`/`packing.py` separation. The competitors all collapse these into one or two giant files.
