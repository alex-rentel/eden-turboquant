#!/usr/bin/env python3
"""Benchmark TurboQuant KV cache compression on Apple Silicon.

Measures generation speed, memory usage, TTFT, and perplexity across
baseline, kv_quant (4-bit), and turboquant (3.5-bit) modes.

Usage:
    python benchmarks/run_benchmark.py \
        --model mlx-community/gemma-4-26B-A4B-it-4bit \
        --modes baseline,kv_quant,turboquant \
        --context-lengths 2048,8192,16384,32768 \
        --output results/gemma4-26b.json
"""

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx

try:
    import mlx_lm
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False

try:
    import mlx_vlm
    HAS_MLX_VLM = True
except ImportError:
    HAS_MLX_VLM = False


WIKITEXT_SAMPLE = (
    "Robert Boulter is an English film, television and theatre actor. He had a "
    "guest-starring role on the television series The Bill in 2000. This was "
    "followed by a starring role in the short film The Survey. He had a guest "
    "role in the episode 'Weep' of the television series 'Judge John Deed' in "
    "2002. In 2003 Boulter landed a role as 'Michael' in the Supply and Demand "
    "sketch written by Supply and Demand. He also appeared in the 2004 film "
    "Layer Cake. Boulter is married to Olivia Sheringham and has two children. "
    "He attended the London Academy of Music and Dramatic Art. He is known for "
    "his work on Hollyoaks, EastEnders, and Casualty. His theatre credits "
    "include work at the Royal National Theatre, the Donmar Warehouse, and the "
    "Young Vic. He has been nominated for several awards including the Ian "
    "Charleson Award. In 2010, he starred in the BBC drama 'Five Daughters'. "
    "The show received critical acclaim and Boulter was praised for his "
    "performance. He continued to work in television throughout the 2010s with "
    "roles in Silent Witness, Vera, and Midsomer Murders. In 2015, he appeared "
    "in the film 'Suffragette' alongside Carey Mulligan and Helena Bonham "
    "Carter. The film depicts the early feminist movement in Britain and the "
    "struggle for women's suffrage. Boulter played a supporting role that was "
    "well received by critics. He returned to theatre in 2017 with a production "
    "of 'The Tempest' at the Globe Theatre. His portrayal of Ariel was described "
    "as 'mesmerizing' by The Guardian. He has since appeared in numerous stage "
    "productions across London's West End. In addition to acting, Boulter has "
    "done voice work for several animated series and video games. He voiced a "
    "character in the popular game 'Assassin's Creed Valhalla' in 2020. His "
    "voice acting has been praised for its range and emotional depth. Boulter "
    "continues to be active in film, television, and theatre, with several "
    "projects announced for the upcoming year."
)


@dataclass
class BenchmarkResult:
    model: str
    mode: str
    context_length: int
    generation_speed_tok_s: float = 0.0
    time_to_first_token_ms: float = 0.0
    peak_memory_mb: float = 0.0
    perplexity: float = 0.0
    error: str = ""
    runs: list = field(default_factory=list)


def get_peak_memory_mb():
    """Get peak RSS in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)  # macOS reports in bytes


def get_hardware_info():
    """Collect hardware metadata."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["total_memory_gb"] = int(result.stdout.strip()) / (1024**3)
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["cpu"] = result.stdout.strip()
    except Exception:
        pass
    return info


def load_model_baseline(model_path):
    """Load model with mlx-lm for baseline and kv_quant modes."""
    if not HAS_MLX_LM:
        raise ImportError("mlx-lm is required. Install with: pip install mlx-lm")
    model, tokenizer = mlx_lm.load(model_path)
    return model, tokenizer


def generate_baseline(model, tokenizer, prompt, max_tokens=100):
    """Generate with no KV cache quantization."""
    return mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
    )


def generate_kv_quant(model, tokenizer, prompt, max_tokens=100):
    """Generate with 4-bit KV cache quantization (mlx-lm built-in)."""
    return mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        kv_bits=4, verbose=False
    )


def generate_turboquant(model_path, prompt, max_tokens=100):
    """Generate with TurboQuant 3.5-bit KV cache compression via mlx-vlm."""
    if not HAS_MLX_VLM:
        raise ImportError(
            "mlx-vlm is required for TurboQuant mode. "
            "Install with: pip install mlx-vlm"
        )
    from mlx_vlm import load as vlm_load, generate as vlm_generate
    model, processor = vlm_load(model_path)
    return vlm_generate(
        model, processor, prompt=prompt, max_tokens=max_tokens,
        kv_bits=3.5, kv_quant_scheme="turboquant", verbose=False
    )


def build_context_prompt(target_length, tokenizer):
    """Build a prompt that fills approximately target_length tokens."""
    base = WIKITEXT_SAMPLE
    tokens = tokenizer.encode(base)
    if len(tokens) >= target_length:
        truncated = tokenizer.decode(tokens[:target_length - 20])
        return truncated + "\n\nSummarize the above text in one sentence:"

    repeats = (target_length // len(tokens)) + 1
    long_text = (base + " ") * repeats
    tokens = tokenizer.encode(long_text)
    truncated = tokenizer.decode(tokens[:target_length - 20])
    return truncated + "\n\nSummarize the above text in one sentence:"


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity on a text sample."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float("inf")

    tokens = tokens[:1000]
    input_ids = mx.array([tokens[:-1]])
    targets = mx.array([tokens[1:]])

    logits = model(input_ids)
    mx.eval(logits)  # MLX lazy eval — evaluates computation graph

    log_probs = mx.softmax(logits, axis=-1)
    target_probs = mx.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)

    neg_log_likelihood = -mx.mean(mx.log(target_probs + 1e-10))
    mx.eval(neg_log_likelihood)  # MLX lazy eval

    return math.exp(float(neg_log_likelihood))


def measure_ttft(model, tokenizer, prompt):
    """Measure time to first token."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    start = time.perf_counter()
    logits = model(input_ids)
    mx.eval(logits)  # MLX lazy eval
    ttft = (time.perf_counter() - start) * 1000  # ms

    return ttft


def run_single_benchmark(model_path, mode, context_length, model=None,
                         tokenizer=None, warmup=False):
    """Run a single benchmark configuration."""
    result = BenchmarkResult(
        model=model_path, mode=mode, context_length=context_length
    )

    try:
        if mode == "turboquant" and not HAS_MLX_VLM:
            result.error = "mlx-vlm not installed, skipping turboquant mode"
            print(f"  WARNING: {result.error}")
            return result

        if model is None or tokenizer is None:
            model, tokenizer = load_model_baseline(model_path)

        prompt = build_context_prompt(context_length, tokenizer)
        max_tokens = 50

        # TTFT
        ttft = measure_ttft(model, tokenizer, prompt)
        result.time_to_first_token_ms = round(ttft, 2)

        # Generation speed
        if mode == "baseline":
            gen_fn = lambda: generate_baseline(model, tokenizer, prompt, max_tokens)
        elif mode == "kv_quant":
            gen_fn = lambda: generate_kv_quant(model, tokenizer, prompt, max_tokens)
        elif mode == "turboquant":
            gen_fn = lambda: generate_turboquant(model_path, prompt, max_tokens)
        else:
            result.error = f"Unknown mode: {mode}"
            return result

        # Warmup
        if warmup:
            gen_fn()

        # Timed runs
        start = time.perf_counter()
        output = gen_fn()
        elapsed = time.perf_counter() - start

        output_tokens = len(tokenizer.encode(output)) if isinstance(output, str) else max_tokens
        tok_s = output_tokens / elapsed if elapsed > 0 else 0
        result.generation_speed_tok_s = round(tok_s, 2)

        # Memory
        result.peak_memory_mb = round(get_peak_memory_mb(), 2)

        # Perplexity
        ppl = compute_perplexity(model, tokenizer, WIKITEXT_SAMPLE)
        result.perplexity = round(ppl, 4)

    except Exception as e:
        result.error = str(e)
        print(f"  ERROR: {e}")

    return result


def run_benchmark(model_path, modes, context_lengths, runs_per_test=3,
                  warmup_runs=1):
    """Run full benchmark suite for a model."""
    results = []

    model = None
    tokenizer = None
    if HAS_MLX_LM:
        print(f"Loading model: {model_path}")
        model, tokenizer = load_model_baseline(model_path)

    for mode in modes:
        for ctx_len in context_lengths:
            print(f"  Benchmarking: mode={mode}, context={ctx_len}")
            run_results = []

            for run_idx in range(warmup_runs + runs_per_test):
                is_warmup = run_idx < warmup_runs
                if is_warmup:
                    print(f"    Warmup run {run_idx + 1}/{warmup_runs}")
                else:
                    print(f"    Run {run_idx - warmup_runs + 1}/{runs_per_test}")

                r = run_single_benchmark(
                    model_path, mode, ctx_len,
                    model=model, tokenizer=tokenizer,
                    warmup=is_warmup
                )

                if not is_warmup:
                    run_results.append(asdict(r))

            # Average across runs
            if run_results:
                avg = BenchmarkResult(
                    model=model_path, mode=mode, context_length=ctx_len
                )
                valid = [r for r in run_results if not r.get("error")]
                if valid:
                    avg.generation_speed_tok_s = round(
                        sum(r["generation_speed_tok_s"] for r in valid) / len(valid), 2
                    )
                    avg.time_to_first_token_ms = round(
                        sum(r["time_to_first_token_ms"] for r in valid) / len(valid), 2
                    )
                    avg.peak_memory_mb = round(
                        max(r["peak_memory_mb"] for r in valid), 2
                    )
                    avg.perplexity = round(
                        sum(r["perplexity"] for r in valid) / len(valid), 4
                    )
                else:
                    avg.error = run_results[0].get("error", "All runs failed")
                avg.runs = run_results
                results.append(asdict(avg))

    return results


def generate_summary_table(results):
    """Generate a markdown summary table from results."""
    lines = []
    lines.append("## Benchmark Results\n")

    lines.append("### Generation Speed (tok/s)\n")
    lines.append("| Model | Context | Mode | tok/s | TTFT (ms) | Peak Memory (MB) | Perplexity |")
    lines.append("|-------|---------|------|-------|-----------|-------------------|------------|")

    for r in results:
        if r.get("error"):
            lines.append(
                f"| {r['model'].split('/')[-1]} | {r['context_length']} | "
                f"{r['mode']} | ERROR: {r['error'][:30]} | — | — | — |"
            )
        else:
            lines.append(
                f"| {r['model'].split('/')[-1]} | {r['context_length']} | "
                f"{r['mode']} | {r['generation_speed_tok_s']} | "
                f"{r['time_to_first_token_ms']} | {r['peak_memory_mb']} | "
                f"{r['perplexity']} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TurboQuant KV cache compression"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--modes", default="baseline,kv_quant,turboquant",
        help="Comma-separated modes: baseline,kv_quant,turboquant"
    )
    parser.add_argument(
        "--context-lengths", default="2048,8192",
        help="Comma-separated context lengths"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed runs per configuration"
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--output", default="results/benchmark.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    context_lengths = [int(c.strip()) for c in args.context_lengths.split(",")]

    print(f"MLX TurboQuant Benchmark")
    print(f"========================")
    print(f"Model: {args.model}")
    print(f"Modes: {modes}")
    print(f"Context lengths: {context_lengths}")
    print(f"Runs per test: {args.runs} (+ {args.warmup} warmup)")
    print()

    hardware = get_hardware_info()
    results = run_benchmark(
        args.model, modes, context_lengths,
        runs_per_test=args.runs, warmup_runs=args.warmup
    )

    output = {
        "hardware": hardware,
        "config": {
            "model": args.model,
            "modes": modes,
            "context_lengths": context_lengths,
            "runs_per_test": args.runs,
            "warmup_runs": args.warmup,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    summary = generate_summary_table(results)
    print(f"\n{summary}")

    summary_path = output_path.with_suffix(".md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
