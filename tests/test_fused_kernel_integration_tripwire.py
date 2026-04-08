"""Tripwire: detect when the v0.7.0 fused QK kernel gets wired into the
end-to-end decode path.

As of v0.8.1 the fused QK kernels (`fused_qk_scores_{2,3,4}bit`) and the
`pre_rotate_query` utility ship as research-grade primitives only —
`mlx_turboquant.patch` does NOT import them, and `apply_turboquant` runs
the dequant+matmul path through `mx.fast.scaled_dot_product_attention`.
The two attempts to integrate them (v0.8.0 decomposed SDPA, v0.9.0 full
fused kernel) live on branches as documented negative results.

When someone finally lands a successful integration on `main`, this test
will fail. That failure is the cue to:

  1. Delete this tripwire file.
  2. Add a real end-to-end integration test that exercises the fused
     kernel inside a forward pass on a small model and checks logits
     against the dequant+matmul path.
  3. Update README "Next Steps" / "What Could Actually Win" to reflect
     which approach shipped.
"""

import inspect

import mlx_turboquant.patch as patch_module


def test_patch_module_does_not_import_fused_kernels():
    """Source-level check: patch.py must not reference fused_qk_scores
    or pre_rotate_query. If this fails, see the module docstring."""
    source = inspect.getsource(patch_module)
    assert "fused_qk_scores" not in source, (
        "patch.py now references fused_qk_scores — the v0.7.0 kernel has "
        "been wired into the decode path. Delete this tripwire and add a "
        "real end-to-end integration test (see module docstring)."
    )
    assert "pre_rotate_query" not in source, (
        "patch.py now references pre_rotate_query — the v0.7.0 kernel has "
        "been wired into the decode path. Delete this tripwire and add a "
        "real end-to-end integration test (see module docstring)."
    )
