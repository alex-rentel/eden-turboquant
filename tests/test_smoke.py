"""Smoke tests — fail loudly when the v1.0.0 public contract breaks.

These tests run first in CI (alphabetical order: test_smoke comes before
everything else) and cover the minimal surface a library user depends on:

1. The package imports cleanly.
2. `__version__` and `__all__` are defined.
3. Every symbol in `__all__` is actually reachable.
4. The CLI entry point is importable and its `main` function exists.
5. The fused-kernel module can be imported without crashing even though
   it's research-only — downstream code may still touch it directly.

If any of these fail, the rest of the test suite is almost certainly
also failing — but a dedicated smoke file makes the root cause obvious
in CI output.
"""


def test_package_imports():
    import mlx_turboquant  # noqa: F401


def test_version_is_present_and_plausible():
    import mlx_turboquant
    v = mlx_turboquant.__version__
    assert isinstance(v, str)
    assert v[:1].isdigit(), f"__version__ looks wrong: {v!r}"


def test_all_exports_are_reachable():
    """Every symbol in __all__ must actually resolve on the module."""
    import mlx_turboquant
    expected = {
        "__version__",
        "apply_turboquant",
        "enable_turboquant",
        "TurboQuantKVCache",
        "TurboQuantMSE",
        "TurboQuantProd",
    }
    assert set(mlx_turboquant.__all__) == expected, (
        f"public API drifted: {set(mlx_turboquant.__all__) ^ expected}"
    )
    for name in expected:
        assert hasattr(mlx_turboquant, name), f"missing export: {name}"


def test_apply_turboquant_is_callable():
    from mlx_turboquant import apply_turboquant
    assert callable(apply_turboquant)


def test_cache_class_contract():
    """TurboQuantKVCache should expose the cache protocol attributes."""
    from mlx_turboquant import TurboQuantKVCache
    assert hasattr(TurboQuantKVCache, "update_and_fetch")
    assert hasattr(TurboQuantKVCache, "state")
    assert hasattr(TurboQuantKVCache, "nbytes")


def test_cli_entry_point_importable():
    """The mlx-turboquant console script should resolve to cli.main."""
    from mlx_turboquant.cli import main
    assert callable(main)


def test_fused_kernel_primitives_importable():
    """Research-only but must stay importable — downstream users may
    experiment with them directly."""
    from mlx_turboquant.kernels import (
        fused_qk_scores_2bit,
        fused_qk_scores_3bit,
        fused_qk_scores_4bit,
    )
    assert callable(fused_qk_scores_4bit)
    assert callable(fused_qk_scores_3bit)
    assert callable(fused_qk_scores_2bit)

    from mlx_turboquant.rotation import pre_rotate_query
    assert callable(pre_rotate_query)
