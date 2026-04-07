"""v0.8.0 fused scaled_dot_product_attention implementation.

Monkey-patches the ``scaled_dot_product_attention`` wrapper in supported
mlx-lm model modules. When the cache is a ``TurboQuantKVCache`` with
``_use_fused_attention=True`` and there are compressed tokens to process,
this module takes the fast path:

  1. Compute scores for the FP16 sink region via standard matmul.
  2. Compute scores for the compressed region via fused_qk_scores_4bit_batched
     (no dequantization of K).
  3. Compute scores for the FP16 residual region via standard matmul.
  4. Concatenate in position order [sink | compressed | residual].
  5. Scale, mask, softmax, weighted sum over V (V is still fully dense).

Otherwise, falls through to the original mlx-lm wrapper.

The patch is installed globally across supported model modules when
``apply_turboquant(model, use_fused_attention=True)`` is called. Installing
does NOT change behavior unless a TurboQuantKVCache with fused mode is
actively in use — the dispatch check happens on every call.
"""

from __future__ import annotations

from typing import Optional, Any

import mlx.core as mx

from .cache import TurboQuantKVCache
from .rotation import pre_rotate_query
from .kernels import fused_qk_scores_4bit_batched


# Name of the symbol in each model module that we need to patch.
_SDPA_ATTR = "scaled_dot_product_attention"

# Model modules that import scaled_dot_product_attention from base.
# Verified via runtime introspection in Phase 1 — all of these use the
# same wrapper function reference, so patching each module's local
# reference redirects attention through our fused path for all of them.
_SUPPORTED_MODULES = [
    "mlx_lm.models.qwen3",
    "mlx_lm.models.llama",
    "mlx_lm.models.qwen2",      # bonus — Qwen2.5 family
    "mlx_lm.models.phi3",       # bonus — Phi-3.5 family
    "mlx_lm.models.deepseek_v2",
    "mlx_lm.models.deepseek_v3",
]


_original_sdpas: dict[str, Any] = {}
_patched = False


def _broadcast_kv(kv: mx.array, H: int, H_kv: int) -> mx.array:
    """Broadcast a (B, H_kv, T, D) tensor to (B, H, T, D) for GQA."""
    if H == H_kv:
        return kv
    return mx.repeat(kv, H // H_kv, axis=1)


def _fused_path(
    queries: mx.array,
    keys_fp16: mx.array,
    values_full: mx.array,
    cache: TurboQuantKVCache,
    scale: float,
    mask: Optional[Any],
) -> mx.array:
    """The actual fused attention computation.

    Only called when the dispatch check confirmed we have a
    TurboQuantKVCache in fused mode with compressed tokens.

    Args:
        queries: (B, H, T_q, D) — post-RoPE queries.
        keys_fp16: (B, H_kv, sink_len + fp16_len, D) — the sink + residual
                   FP16 keys returned by update_and_fetch in fused mode.
        values_full: (B, H_kv, sink_len + compressed_len + fp16_len, D) —
                     the FULL dense V tensor (compressed middle is
                     incrementally rebuilt; v0.8.0 does not fuse V).
        cache: the TurboQuantKVCache.
        scale: attention scale factor (usually 1/sqrt(D)).
        mask: attention mask — for the decode case (T_q=1) this is
              typically None.

    Returns:
        Attention output (B, H, T_q, D).
    """
    state = cache.get_fused_state()
    sink_len = state["sink_len"]
    compressed_len = state["compressed_len"]
    fp16_len = state["fp16_len"]
    D = state["head_dim"]
    rotation = state["rotation"]

    B, H, T_q, _ = queries.shape
    H_kv = keys_fp16.shape[1]

    # Sanity check the shape invariant: keys_fp16 should be exactly
    # sink_len + fp16_len along the sequence axis.
    assert keys_fp16.shape[2] == sink_len + fp16_len, (
        f"keys_fp16 seq length {keys_fp16.shape[2]} != "
        f"sink_len ({sink_len}) + fp16_len ({fp16_len})"
    )

    # --- 1. Sink scores ---
    if sink_len > 0:
        k_sink = keys_fp16[:, :, :sink_len, :]
        k_sink_bc = _broadcast_kv(k_sink, H, H_kv)
        scores_sink = queries @ k_sink_bc.transpose(0, 1, 3, 2)
    else:
        scores_sink = None

    # --- 2. Compressed scores (fused, no dequant) ---
    # Only 4-bit supported in v0.8.0. Fall through to None for other bits.
    key_bits = state["key_bits"]
    if key_bits != 4:
        raise NotImplementedError(
            f"v0.8.0 fused SDPA only supports key_bits=4, got {key_bits}. "
            f"Set use_fused_attention=False for other bit widths."
        )

    q_rot = pre_rotate_query(queries, rotation)
    scores_compressed = fused_qk_scores_4bit_batched(
        q_rot,
        state["packed_keys"],
        state["key_norms"],
        state["key_centroids"],
        D=D, H=H, H_kv=H_kv,
    )

    # --- 3. Residual scores ---
    if fp16_len > 0:
        k_residual = keys_fp16[:, :, sink_len:, :]
        k_residual_bc = _broadcast_kv(k_residual, H, H_kv)
        scores_residual = queries @ k_residual_bc.transpose(0, 1, 3, 2)
    else:
        scores_residual = None

    # --- 4. Concatenate in position order [sink, compressed, residual] ---
    score_parts = []
    if scores_sink is not None:
        score_parts.append(scores_sink)
    score_parts.append(scores_compressed)
    if scores_residual is not None:
        score_parts.append(scores_residual)
    all_scores = mx.concatenate(score_parts, axis=-1)
    # (B, H, T_q, sink_len + compressed_len + fp16_len)

    # --- 5. Scale, mask, softmax ---
    all_scores = all_scores * scale
    # For the decode case (T_q=1) the mask is typically None. For
    # correctness we support tensor masks by adding them; string
    # masks ("causal") only make sense for T_q > 1 and are handled
    # by the caller route (we only enter this path when T_q=1).
    if mask is not None and not isinstance(mask, str):
        all_scores = all_scores + mask
    weights = mx.softmax(all_scores, axis=-1)

    # --- 6. Weighted sum over V ---
    values_bc = _broadcast_kv(values_full, H, H_kv)
    output = weights @ values_bc
    return output


def fused_scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Any,
    scale: float,
    mask: Optional[Any] = None,
    sinks: Optional[mx.array] = None,
) -> mx.array:
    """Drop-in replacement for mlx_lm.models.base.scaled_dot_product_attention.

    Dispatches to the fused path when all of the following are true:
      - cache is a TurboQuantKVCache with _use_fused_attention=True
      - cache has compressed tokens (_compressed_len > 0)
      - T_q == 1 (decode; fused path for T_q > 1 would need mask handling)
      - key_bits == 4

    Otherwise falls through to the original wrapper — which itself
    handles the quantized-cache branch and the default
    mx.fast.scaled_dot_product_attention call.
    """
    # Fast-path dispatch check
    if not isinstance(cache, TurboQuantKVCache):
        return _original_sdpas["default"](
            queries, keys, values, cache=cache,
            scale=scale, mask=mask, sinks=sinks,
        )
    if not cache._use_fused_attention:
        return _original_sdpas["default"](
            queries, keys, values, cache=cache,
            scale=scale, mask=mask, sinks=sinks,
        )
    if not cache.has_compressed:
        return _original_sdpas["default"](
            queries, keys, values, cache=cache,
            scale=scale, mask=mask, sinks=sinks,
        )
    # Only handle decode case cleanly in v0.8.0
    if queries.shape[2] != 1:
        return _original_sdpas["default"](
            queries, keys, values, cache=cache,
            scale=scale, mask=mask, sinks=sinks,
        )
    # Only 4-bit supported
    if int(cache.key_bits) != 4:
        return _original_sdpas["default"](
            queries, keys, values, cache=cache,
            scale=scale, mask=mask, sinks=sinks,
        )

    return _fused_path(queries, keys, values, cache, scale, mask)


def install_patch():
    """Install the fused SDPA wrapper into supported model modules.

    Idempotent: calling twice is a no-op. Each module's local
    scaled_dot_product_attention reference is replaced; the original is
    saved in _original_sdpas under the module name so uninstall can
    restore it.
    """
    global _patched
    if _patched:
        return

    import importlib

    # Capture a single "default" original to delegate to from the
    # fused wrapper. All supported modules share the same underlying
    # function reference, so one capture is enough.
    default_original = None

    for mod_name in _SUPPORTED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if not hasattr(mod, _SDPA_ATTR):
            continue
        original = getattr(mod, _SDPA_ATTR)
        _original_sdpas[mod_name] = original
        if default_original is None:
            default_original = original
        setattr(mod, _SDPA_ATTR, fused_scaled_dot_product_attention)

    if default_original is not None:
        _original_sdpas["default"] = default_original

    _patched = True


def uninstall_patch():
    """Restore the original scaled_dot_product_attention in each patched module."""
    global _patched
    if not _patched:
        return

    import importlib

    for mod_name, original in list(_original_sdpas.items()):
        if mod_name == "default":
            continue
        try:
            mod = importlib.import_module(mod_name)
            setattr(mod, _SDPA_ATTR, original)
        except ImportError:
            pass

    _original_sdpas.clear()
    _patched = False
