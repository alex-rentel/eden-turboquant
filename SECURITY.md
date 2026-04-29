# Security policy

## Reporting a vulnerability

Please email **alex@renaissanceintelligence.ai** with the details. Avoid filing a public GitHub issue for anything you believe could be exploited — open a private channel first.

A useful report includes:

- The version and commit SHA you reproduced against (e.g. `v1.0.4` / `f84dfee`).
- A minimal repro: input shapes, model, the exact `apply_turboquant` call, the failure mode.
- Any relevant `mlx` / `mlx-lm` / `numpy` versions (`pip freeze | grep -i mlx`).

You should expect a first reply within a few days. Once the issue is confirmed, fixes typically ship in the next patch release; coordinated disclosure timing is negotiable.

## Supported versions

Only the latest minor line (`1.0.x`) gets security fixes. Pre-`1.0` versions are unsupported.

## Scope

In scope: anything that lets crafted input crash the runtime, leak memory across cache boundaries, or produce attacker-controlled outputs from the cache compression layer.

Out of scope:

- Quality regressions (cosine-similarity drops on legitimate inputs) — these are bugs, not security issues. Open a public GitHub issue.
- Issues in MLX, mlx-lm, or model weights themselves — report upstream.
- Anything that requires the attacker to already have local code execution or pip-install rights.
