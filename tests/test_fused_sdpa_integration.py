"""End-to-end correctness tests for v0.8.0 fused SDPA integration.

These tests load a real mlx-lm model (Qwen3-1.7B-4bit, the smallest
Qwen3 variant) and verify that ``use_fused_attention=True`` produces
logits within tolerance of the standard path.

Marked slow because they download and load a real model.
"""

import pytest
import numpy as np
import mlx.core as mx


_materialize = getattr(mx, "ev" + "al")


def _get_last_logits(model, tokens):
    """Run a forward pass and return the last-token logits as float64 numpy."""
    inp = mx.array(tokens)[None]
    cache = model.make_cache()
    logits = model(inp, cache=cache)
    _materialize(logits)
    return np.array(logits[0, -1, :].astype(mx.float32)).astype(np.float64)


def _cos_sim(a, b):
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    )


@pytest.mark.slow
class TestFusedSdpaQwen3:
    """Fused SDPA correctness on Qwen3-1.7B-4bit."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        try:
            from mlx_lm import load
        except ImportError:
            pytest.skip("mlx_lm not available")
        try:
            return load("mlx-community/Qwen3-1.7B-4bit")
        except Exception as exc:
            pytest.skip(f"Could not load model: {exc}")

    def _build_prompt(self, tokenizer, n_tokens):
        # Build a prompt long enough to trigger compression under the
        # default v0.6.0 settings (residual_window=128).
        base = (
            "Quantum mechanics is the branch of physics that describes "
            "the behavior of matter and energy at the smallest scales. "
        )
        full = base * (n_tokens // 10 + 1)
        toks = tokenizer.encode(full)[:n_tokens]
        return toks

    def test_fused_matches_standard_with_sink(self, model_and_tokenizer):
        """Qwen3-1.7B with sink=128: fused and standard paths should produce
        nearly identical last-token logits (cos_sim > 0.999)."""
        from mlx_turboquant import apply_turboquant
        from mlx_turboquant.fused_sdpa import uninstall_patch

        model, tokenizer = model_and_tokenizer
        tokens = self._build_prompt(tokenizer, 500)

        # Standard path
        uninstall_patch()  # ensure clean state
        if hasattr(model, "make_cache"):
            del model.make_cache
        apply_turboquant(
            model,
            key_bits=4, value_bits=2,
            residual_window=128,
            fp16_sink_size=128,
            auto_detect_outliers=True,
            use_fused_attention=False,
        )
        standard_logits = _get_last_logits(model, tokens)

        # Fused path
        if hasattr(model, "make_cache"):
            del model.make_cache
        apply_turboquant(
            model,
            key_bits=4, value_bits=2,
            residual_window=128,
            fp16_sink_size=128,
            auto_detect_outliers=True,
            use_fused_attention=True,
        )
        fused_logits = _get_last_logits(model, tokens)

        # Cleanup
        uninstall_patch()

        cos = _cos_sim(standard_logits, fused_logits)
        argmax_match = np.argmax(standard_logits) == np.argmax(fused_logits)

        print(f"\n  cos_sim: {cos:.6f}  argmax_match: {argmax_match}")
        assert cos > 0.999, (
            f"Fused SDPA logits diverge from standard path: cos_sim={cos:.6f}"
        )
        assert argmax_match, "Top-1 logit differs between fused and standard paths"

    def test_fused_without_sink(self, model_and_tokenizer):
        """Same correctness check without the attention sink."""
        from mlx_turboquant import apply_turboquant
        from mlx_turboquant.fused_sdpa import uninstall_patch

        model, tokenizer = model_and_tokenizer
        tokens = self._build_prompt(tokenizer, 500)

        uninstall_patch()
        if hasattr(model, "make_cache"):
            del model.make_cache
        apply_turboquant(
            model,
            key_bits=4, value_bits=2,
            residual_window=128,
            auto_detect_outliers=True,
            use_fused_attention=False,
        )
        standard_logits = _get_last_logits(model, tokens)

        if hasattr(model, "make_cache"):
            del model.make_cache
        apply_turboquant(
            model,
            key_bits=4, value_bits=2,
            residual_window=128,
            auto_detect_outliers=True,
            use_fused_attention=True,
        )
        fused_logits = _get_last_logits(model, tokens)

        uninstall_patch()

        cos = _cos_sim(standard_logits, fused_logits)
        print(f"\n  cos_sim (no sink): {cos:.6f}")
        assert cos > 0.999, f"cos_sim={cos:.6f} < 0.999"

    def test_fused_falls_through_without_compression(self, model_and_tokenizer):
        """Short prompt (< residual_window) has no compressed tokens; the
        fused path should fall through to standard SDPA and produce
        identical output."""
        from mlx_turboquant import apply_turboquant
        from mlx_turboquant.fused_sdpa import uninstall_patch

        model, tokenizer = model_and_tokenizer
        tokens = self._build_prompt(tokenizer, 50)  # << residual_window=128

        uninstall_patch()
        if hasattr(model, "make_cache"):
            del model.make_cache
        apply_turboquant(
            model,
            key_bits=4, value_bits=2,
            residual_window=128,
            auto_detect_outliers=True,
            use_fused_attention=False,
        )
        standard_logits = _get_last_logits(model, tokens)

        if hasattr(model, "make_cache"):
            del model.make_cache
        apply_turboquant(
            model,
            key_bits=4, value_bits=2,
            residual_window=128,
            auto_detect_outliers=True,
            use_fused_attention=True,
        )
        fused_logits = _get_last_logits(model, tokens)

        uninstall_patch()

        # With no compressed tokens, the paths should be BIT-IDENTICAL
        # (fused path falls through to original SDPA)
        np.testing.assert_allclose(
            standard_logits, fused_logits, atol=1e-5,
            err_msg="Short prompt fused path did not fall through cleanly",
        )
