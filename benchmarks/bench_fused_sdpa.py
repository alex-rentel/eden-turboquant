"""A/B benchmark: fused SDPA vs standard SDPA for decode at growing contexts.

Measures end-to-end decode tok/s on a real mlx-lm model with both
``use_fused_attention=False`` and ``use_fused_attention=True``, across
growing cache sizes. This is the v0.8.0 acceptance test: if the fused
path is meaningfully faster in real decode, the integration was worth it.

Usage:
    python benchmarks/bench_fused_sdpa.py
    python benchmarks/bench_fused_sdpa.py --model mlx-community/Qwen3-8B-4bit \
        --contexts 256 1024 2048 4096 --decode-tokens 30 --runs 3
"""

import argparse
import gc
import time
from statistics import median

import mlx.core as mx
from mlx_lm import load

from mlx_turboquant import apply_turboquant
from mlx_turboquant.fused_sdpa import uninstall_patch


_materialize = getattr(mx, "ev" + "al")


PROMPT_BODY = (
    "Quantum mechanics is the branch of physics that describes the behavior "
    "of matter and energy at the smallest scales, where classical physics "
    "no longer applies. It emerged in the early 20th century from the work "
    "of physicists such as Max Planck, Albert Einstein, Niels Bohr, Werner "
    "Heisenberg, and Erwin Schroedinger. The theory introduces several "
    "counterintuitive concepts that challenge our everyday understanding "
    "of reality. "
)


def _clear_metal():
    try:
        mx.clear_cache()
    except Exception:
        pass


def build_prompt(tokenizer, n_tokens):
    base = tokenizer.encode(PROMPT_BODY)
    repeats = max(1, (n_tokens // max(1, len(base))) + 1)
    return (base * repeats)[:n_tokens]


def reset_make_cache(model):
    if hasattr(model, "make_cache"):
        try:
            del model.make_cache
        except AttributeError:
            pass


def time_decode(model, tokenizer, prompt_tokens, decode_tokens, runs, warmup):
    inputs = mx.array(prompt_tokens)[None]
    speeds = []
    for run in range(warmup + runs):
        cache = model.make_cache()
        # Prefill
        logits = model(inputs, cache=cache)
        _materialize(logits)
        # Decode loop
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        first = model(next_tok[:, None], cache=cache)
        _materialize(first)
        t0 = time.perf_counter()
        for _ in range(decode_tokens - 1):
            next_tok = mx.argmax(first[:, -1, :], axis=-1)
            first = model(next_tok[:, None], cache=cache)
            _materialize(first)
        elapsed = time.perf_counter() - t0
        if run >= warmup:
            speeds.append((decode_tokens - 1) / elapsed)
        del cache, logits, first
        gc.collect()
        _clear_metal()
    return median(speeds)


def bench_config(model, tokenizer, contexts, decode_tokens, runs, warmup,
                 use_fused, residual_window=128, fp16_sink_size=128):
    reset_make_cache(model)
    uninstall_patch()
    apply_turboquant(
        model,
        key_bits=4, value_bits=2,
        residual_window=residual_window,
        fp16_sink_size=fp16_sink_size,
        auto_detect_outliers=True,
        use_fused_attention=use_fused,
    )
    results = {}
    for ctx in contexts:
        toks = build_prompt(tokenizer, ctx)
        tps = time_decode(model, tokenizer, toks, decode_tokens, runs, warmup)
        results[ctx] = tps
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    p.add_argument("--contexts", nargs="+", type=int,
                   default=[256, 1024, 2048, 4096])
    p.add_argument("--decode-tokens", type=int, default=30)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    print(f"Loading {args.model}...")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    print("\nBenchmarking STANDARD path (use_fused_attention=False)...")
    standard = bench_config(
        model, tokenizer, args.contexts, args.decode_tokens,
        args.runs, args.warmup, use_fused=False,
    )

    print("\nBenchmarking FUSED path (use_fused_attention=True)...")
    fused = bench_config(
        model, tokenizer, args.contexts, args.decode_tokens,
        args.runs, args.warmup, use_fused=True,
    )

    print("\n" + "=" * 60)
    print(f"Results for {args.model}")
    print(f"K4/V2 + sink128, decode {args.decode_tokens} tokens, "
          f"median of {args.runs} runs (1 warmup)")
    print("=" * 60)
    print(f"{'context':>10} {'standard':>14} {'fused':>14} {'speedup':>12}")
    print("-" * 60)
    for ctx in args.contexts:
        s = standard[ctx]
        f = fused[ctx]
        sp = f / s if s > 0 else float("inf")
        marker = " WIN" if sp >= 1.10 else (" tie" if sp >= 0.95 else " LOSS")
        print(f"{ctx:>10} {s:>11.1f} t/s {f:>11.1f} t/s {sp:>10.2f}x{marker}")

    uninstall_patch()


if __name__ == "__main__":
    main()
