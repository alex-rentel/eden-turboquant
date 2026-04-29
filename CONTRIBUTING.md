# Contributing to mlx-turboquant

Thanks for your interest. This file is a quick orientation — what's where, how to set up locally, and a few "read this before reattempting" notes about prior research dead-ends.

## Dev environment

Apple Silicon required (MLX is Apple-Silicon-only). Python 3.10–3.13 supported.

```bash
git clone https://github.com/alex-rentel/eden-turboquant.git
cd eden-turboquant
pip install -e ".[dev]"
```

That installs the package editable plus the dev tools: `pytest`, `pytest-cov`, `scipy`, `ruff`, `pyright`.

## Running checks locally

The full set CI runs:

```bash
ruff check mlx_turboquant/ tests/        # style / imports / unused
pyright                                   # type-check (basic mode)
pytest tests/ -q --ignore=tests/test_integration.py
```

Coverage:

```bash
pytest tests/ -q --ignore=tests/test_integration.py --cov=mlx_turboquant --cov-report=term-missing
```

`tests/test_integration.py` is the `@pytest.mark.slow` real-model test (downloads multi-GB checkpoints). It's CI-skipped; run it manually before cutting a release.

## Layout

| Path | What lives here |
|---|---|
| `mlx_turboquant/` | The package. Public API in `__init__.py`. |
| `mlx_turboquant/cache.py` | `TurboQuantKVCache` — the production hot path. |
| `mlx_turboquant/patch.py` | `apply_turboquant` — model integration. |
| `mlx_turboquant/codebook.py` | Lloyd-Max scalar quantizer. |
| `mlx_turboquant/rotation.py` | Random orthogonal rotation matrices. |
| `mlx_turboquant/packing.py` | uint8 ↔ {2,3,4}-bit pack/unpack. |
| `mlx_turboquant/kernels.py` | Metal kernels. Read the docstring — only `metal_dequantize` is on the supported decode path; the rest are research-only primitives. |
| `mlx_turboquant/qjl.py` | Quantized Johnson–Lindenstrauss helpers. |
| `tests/` | All tests. `test_smoke.py` runs first and gates the public API contract. |
| `docs/` | Internals, post-mortems, benchmark results. |
| `benchmarks/` | Standalone scripts. Run manually; not part of CI. |

## Configuration

All tool config is in `pyproject.toml`:
- `[tool.ruff]` / `[tool.ruff.lint]` — linter rules
- `[tool.pyright]` — type-check rules (basic mode, with the noisy Optional-narrowing rules off; `reportArgumentType` stays at `error`)
- `[tool.pytest.ini_options]` — pytest config + the `slow` marker
- `[tool.coverage.run]` / `[tool.coverage.report]` — coverage config

There's no separate `pyrightconfig.json` or `.coveragerc`; everything lives in pyproject.

## Tripwires (read before reattempting)

`tests/test_fused_kernel_integration_tripwire.py` is a source-level guard against a class of work that has already been tried and concluded as a negative result. If you see this test fail, you're (intentionally or not) wiring the v0.7.0 fused QK kernels or the v0.8.0 fused-SDPA hooks into the supported decode path.

Before deleting the tripwire and replacing it with a real integration test, please read:

- `docs/FUSED_SDPA_RESULTS.md` — the v0.8.0 attempt. Bit-identical correctness, 0.61×–0.99× decode speed (a regression).
- `docs/FULL_FUSED_ATTENTION_RESULTS.md` — the v0.9.0 attempt. Same shape of result.

Both lost to `mx.fast.scaled_dot_product_attention` because realistic decode is dispatch/latency-bound, not memory-bandwidth-bound — the packed-KV memory advantage doesn't materialize. New attempts need a structural argument for what changed, not just "let's try fusing again."

The kernels themselves stay in `kernels.py` as tested research primitives so the integration is one wire-up away if someone finds a new angle.

## Style

- Match existing surrounding code. Lloyd-Max math is intentionally a bit dense; cache state machinery is intentionally explicit.
- Type-narrowing asserts (`assert self._compressed_keys is not None`) are deliberate — pyright can't see runtime length-guards. Don't replace with `# type: ignore` unless an `assert` doesn't make sense.
- Don't add docstrings that just restate the function name; do write them when there's a non-obvious invariant or a "why this and not the obvious thing" answer.
- Ruff's `select = ["E", "F", "W", "I", "B", "UP"]` — keep changes within those rules. If a new rule is too noisy, document why in the `ignore` list.

## Releases

1. Bump `__version__` in `mlx_turboquant/__init__.py` and `version` in `pyproject.toml`.
2. Add a CHANGELOG entry under a new `## [vX.Y.Z] — YYYY-MM-DD` heading.
3. Run the full slow suite once: `pytest tests/`.
4. `python -m build` produces sdist + wheel under `dist/`.
5. `git tag -a vX.Y.Z -m "..."` and push the tag.
6. `gh release create vX.Y.Z --notes-file <changelog-block> dist/mlx_turboquant-X.Y.Z.tar.gz dist/mlx_turboquant-X.Y.Z-py3-none-any.whl`.
7. PyPI upload (`twine upload dist/...`) is a separate, irreversible step — only run it after the GitHub release looks correct.

## Issues / questions

Open an issue at https://github.com/alex-rentel/eden-turboquant/issues. For research-direction questions, the post-mortems in `docs/` answer most of them.
