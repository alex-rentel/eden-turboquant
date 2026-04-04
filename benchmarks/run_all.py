#!/usr/bin/env python3
"""Run full benchmark suite from a hardware config file.

Usage:
    python benchmarks/run_all.py --config configs/m1_max_64gb.yaml
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from benchmarks.run_benchmark import run_benchmark, get_hardware_info, generate_summary_table
from benchmarks.tool_call_accuracy import run_accuracy_test


def main():
    parser = argparse.ArgumentParser(
        description="Run full benchmark suite from config"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to hardware config YAML"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-accuracy", action="store_true",
        help="Skip tool-calling accuracy tests"
    )
    parser.add_argument(
        "--test-set", default="benchmarks/tool_call_tests.jsonl",
        help="Path to tool-call test set"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    hardware_name = config.get("hardware", "unknown")
    models = config.get("models", [])
    context_lengths = config.get("context_lengths", [2048])
    modes = config.get("modes", ["baseline"])
    runs_per_test = config.get("runs_per_test", 3)
    warmup_runs = config.get("warmup_runs", 1)

    print(f"MLX TurboQuant — Full Benchmark Suite")
    print(f"======================================")
    print(f"Hardware: {hardware_name}")
    print(f"Models: {len(models)}")
    print(f"Context lengths: {context_lengths}")
    print(f"Modes: {modes}")
    print(f"Runs per test: {runs_per_test} (+ {warmup_runs} warmup)")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware_info = get_hardware_info()
    hardware_info["config_hardware"] = hardware_name

    all_benchmark_results = []
    all_accuracy_results = []

    for model_cfg in models:
        model_id = model_cfg["id"]
        model_name = model_cfg.get("name", model_id.split("/")[-1])

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})")
        print(f"{'='*60}")

        # Performance benchmarks
        safe_name = model_name.lower().replace(" ", "_").replace(".", "")
        bench_output = output_dir / f"bench_{safe_name}.json"

        print(f"\n--- Performance Benchmarks ---")
        try:
            results = run_benchmark(
                model_id, modes, context_lengths,
                runs_per_test=runs_per_test, warmup_runs=warmup_runs
            )
            bench_data = {
                "hardware": hardware_info,
                "model": {"id": model_id, "name": model_name},
                "config": {
                    "modes": modes,
                    "context_lengths": context_lengths,
                    "runs_per_test": runs_per_test,
                    "warmup_runs": warmup_runs,
                },
                "results": results,
            }
            with open(bench_output, "w") as f:
                json.dump(bench_data, f, indent=2)
            print(f"  Saved to {bench_output}")
            all_benchmark_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")

        # Tool-calling accuracy
        if not args.skip_accuracy:
            accuracy_output = output_dir / f"accuracy_{safe_name}.json"
            print(f"\n--- Tool-Calling Accuracy ---")
            try:
                accuracy_data = run_accuracy_test(
                    model_id, modes, args.test_set, str(accuracy_output)
                )
                all_accuracy_results.append(accuracy_data)
            except Exception as e:
                print(f"  ERROR: {e}")

    # Combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_output = output_dir / f"combined_{timestamp}.json"
    combined = {
        "hardware": hardware_info,
        "config_file": args.config,
        "timestamp": timestamp,
        "benchmark_results": all_benchmark_results,
        "accuracy_results": all_accuracy_results,
    }
    with open(combined_output, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to {output_dir}/")
    print(f"Combined results: {combined_output}")

    if all_benchmark_results:
        summary = generate_summary_table(all_benchmark_results)
        summary_path = output_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Summary: {summary_path}")
        print(f"\n{summary}")


if __name__ == "__main__":
    main()
