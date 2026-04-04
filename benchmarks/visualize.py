#!/usr/bin/env python3
"""Visualize benchmark results as markdown tables and HTML reports.

Usage:
    python benchmarks/visualize.py --input results/ --output reports/
"""

import argparse
import json
import os
from pathlib import Path


def load_results(input_dir):
    """Load all JSON result files from a directory."""
    results = []
    input_path = Path(input_dir)
    for f in sorted(input_path.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            data["_source_file"] = f.name
            results.append(data)
    return results


def generate_speed_table(results):
    """Generate markdown table for generation speed."""
    lines = []
    lines.append("### Generation Speed (tok/s)\n")
    lines.append("| Model | Context | Baseline | KV Quant (4-bit) | TurboQuant (3.5-bit) |")
    lines.append("|-------|---------|----------|-------------------|----------------------|")

    # Group by model + context
    rows = {}
    for data in results:
        for r in data.get("results", data.get("benchmark_results", [])):
            model = r.get("model", "").split("/")[-1]
            ctx = r.get("context_length", "")
            mode = r.get("mode", "")
            speed = r.get("generation_speed_tok_s", 0)
            error = r.get("error", "")

            key = (model, ctx)
            if key not in rows:
                rows[key] = {"baseline": "—", "kv_quant": "—", "turboquant": "—"}
            if error:
                rows[key][mode] = "ERR"
            else:
                rows[key][mode] = f"{speed}"

    for (model, ctx), modes in sorted(rows.items()):
        lines.append(
            f"| {model} | {ctx} | {modes['baseline']} | "
            f"{modes['kv_quant']} | {modes['turboquant']} |"
        )

    return "\n".join(lines)


def generate_memory_table(results):
    """Generate markdown table for memory usage."""
    lines = []
    lines.append("### Peak Memory Usage (MB)\n")
    lines.append("| Model | Context | Baseline | KV Quant (4-bit) | TurboQuant (3.5-bit) |")
    lines.append("|-------|---------|----------|-------------------|----------------------|")

    rows = {}
    for data in results:
        for r in data.get("results", data.get("benchmark_results", [])):
            model = r.get("model", "").split("/")[-1]
            ctx = r.get("context_length", "")
            mode = r.get("mode", "")
            mem = r.get("peak_memory_mb", 0)
            error = r.get("error", "")

            key = (model, ctx)
            if key not in rows:
                rows[key] = {"baseline": "—", "kv_quant": "—", "turboquant": "—"}
            if error:
                rows[key][mode] = "ERR"
            else:
                rows[key][mode] = f"{mem}"

    for (model, ctx), modes in sorted(rows.items()):
        lines.append(
            f"| {model} | {ctx} | {modes['baseline']} | "
            f"{modes['kv_quant']} | {modes['turboquant']} |"
        )

    return "\n".join(lines)


def generate_ttft_table(results):
    """Generate markdown table for time to first token."""
    lines = []
    lines.append("### Time to First Token (ms)\n")
    lines.append("| Model | Context | Baseline | KV Quant (4-bit) | TurboQuant (3.5-bit) |")
    lines.append("|-------|---------|----------|-------------------|----------------------|")

    rows = {}
    for data in results:
        for r in data.get("results", data.get("benchmark_results", [])):
            model = r.get("model", "").split("/")[-1]
            ctx = r.get("context_length", "")
            mode = r.get("mode", "")
            ttft = r.get("time_to_first_token_ms", 0)
            error = r.get("error", "")

            key = (model, ctx)
            if key not in rows:
                rows[key] = {"baseline": "—", "kv_quant": "—", "turboquant": "—"}
            if error:
                rows[key][mode] = "ERR"
            else:
                rows[key][mode] = f"{ttft}"

    for (model, ctx), modes in sorted(rows.items()):
        lines.append(
            f"| {model} | {ctx} | {modes['baseline']} | "
            f"{modes['kv_quant']} | {modes['turboquant']} |"
        )

    return "\n".join(lines)


def generate_accuracy_table(results):
    """Generate markdown table for tool-calling accuracy."""
    lines = []
    lines.append("### Tool-Calling Accuracy\n")
    lines.append("| Model | Metric | Baseline | KV Quant (4-bit) | TurboQuant (3.5-bit) |")
    lines.append("|-------|--------|----------|-------------------|----------------------|")

    for data in results:
        summaries = data.get("summaries", {})
        model = data.get("model", "").split("/")[-1]
        if not summaries:
            continue

        for metric in ["exact_match", "function_match", "parseable", "format_correct", "total_score"]:
            row = f"| {model} | {metric} |"
            for mode in ["baseline", "kv_quant", "turboquant"]:
                if mode in summaries:
                    val = summaries[mode].get(metric, "—")
                    total = summaries[mode].get("total", 0)
                    if metric == "total_score":
                        row += f" {val}% |"
                    else:
                        row += f" {val}/{total} |"
                else:
                    row += " — |"
            lines.append(row)

    return "\n".join(lines)


def generate_html_report(results, accuracy_results):
    """Generate an HTML report with charts."""
    # Collect speed data for chart
    speed_data = {}
    for data in results:
        for r in data.get("results", data.get("benchmark_results", [])):
            model = r.get("model", "").split("/")[-1]
            mode = r.get("mode", "")
            speed = r.get("generation_speed_tok_s", 0)
            ctx = r.get("context_length", "")
            if not r.get("error") and speed > 0:
                label = f"{model} ({ctx})"
                if label not in speed_data:
                    speed_data[label] = {}
                speed_data[label][mode] = speed

    memory_data = {}
    for data in results:
        for r in data.get("results", data.get("benchmark_results", [])):
            model = r.get("model", "").split("/")[-1]
            mode = r.get("mode", "")
            mem = r.get("peak_memory_mb", 0)
            ctx = r.get("context_length", "")
            if not r.get("error") and mem > 0:
                label = f"{model} ({ctx})"
                if label not in memory_data:
                    memory_data[label] = {}
                memory_data[label][mode] = mem

    labels_speed = json.dumps(list(speed_data.keys()))
    baseline_speed = json.dumps([d.get("baseline", 0) for d in speed_data.values()])
    kv_quant_speed = json.dumps([d.get("kv_quant", 0) for d in speed_data.values()])
    turbo_speed = json.dumps([d.get("turboquant", 0) for d in speed_data.values()])

    labels_mem = json.dumps(list(memory_data.keys()))
    baseline_mem = json.dumps([d.get("baseline", 0) for d in memory_data.values()])
    kv_quant_mem = json.dumps([d.get("kv_quant", 0) for d in memory_data.values()])
    turbo_mem = json.dumps([d.get("turboquant", 0) for d in memory_data.values()])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLX TurboQuant Benchmark Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f5f5f5; color: #333; }}
        h1 {{ color: #1a1a2e; }}
        h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 0.5rem; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        canvas {{ max-width: 100%; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background: #1a1a2e; color: white; padding: 0.75rem 1rem; text-align: left; }}
        td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #eee; }}
        tr:hover td {{ background: #f0f0f0; }}
        .note {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 1rem; margin: 1rem 0; }}
    </style>
</head>
<body>
    <h1>MLX TurboQuant Benchmark Results</h1>
    <p>Comparing baseline, KV Quant (4-bit), and TurboQuant (3.5-bit) modes on Apple Silicon.</p>

    <h2>Generation Speed (tok/s)</h2>
    <div class="chart-container">
        <canvas id="speedChart" height="300"></canvas>
    </div>

    <h2>Peak Memory Usage (MB)</h2>
    <div class="chart-container">
        <canvas id="memoryChart" height="300"></canvas>
    </div>

    <div class="note">
        <strong>Note:</strong> Charts require an internet connection to load Chart.js CDN.
        If charts don't render, check the markdown tables below.
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <script>
        const speedCtx = document.getElementById('speedChart').getContext('2d');
        new Chart(speedCtx, {{
            type: 'bar',
            data: {{
                labels: {labels_speed},
                datasets: [
                    {{ label: 'Baseline', data: {baseline_speed}, backgroundColor: 'rgba(54, 162, 235, 0.7)' }},
                    {{ label: 'KV Quant (4-bit)', data: {kv_quant_speed}, backgroundColor: 'rgba(255, 206, 86, 0.7)' }},
                    {{ label: 'TurboQuant (3.5-bit)', data: {turbo_speed}, backgroundColor: 'rgba(75, 192, 192, 0.7)' }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Generation Speed (tok/s) — higher is better' }} }},
                scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'tok/s' }} }} }}
            }}
        }});

        const memCtx = document.getElementById('memoryChart').getContext('2d');
        new Chart(memCtx, {{
            type: 'bar',
            data: {{
                labels: {labels_mem},
                datasets: [
                    {{ label: 'Baseline', data: {baseline_mem}, backgroundColor: 'rgba(54, 162, 235, 0.7)' }},
                    {{ label: 'KV Quant (4-bit)', data: {kv_quant_mem}, backgroundColor: 'rgba(255, 206, 86, 0.7)' }},
                    {{ label: 'TurboQuant (3.5-bit)', data: {turbo_mem}, backgroundColor: 'rgba(75, 192, 192, 0.7)' }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Peak Memory Usage (MB) — lower is better' }} }},
                scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'MB' }} }} }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results"
    )
    parser.add_argument(
        "--input", default="results",
        help="Input directory with JSON result files"
    )
    parser.add_argument(
        "--output", default="reports",
        help="Output directory for reports"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input directory {input_path} does not exist")
        sys.exit(1)

    # Load benchmark results
    bench_files = list(input_path.glob("bench_*.json")) + list(input_path.glob("combined_*.json"))
    bench_results = []
    for f in bench_files:
        with open(f) as fh:
            bench_results.append(json.load(fh))

    # Load accuracy results
    accuracy_files = list(input_path.glob("accuracy_*.json"))
    accuracy_results = []
    for f in accuracy_files:
        with open(f) as fh:
            accuracy_results.append(json.load(fh))

    if not bench_results and not accuracy_results:
        print("No result files found. Run benchmarks first.")
        print(f"  python benchmarks/run_all.py --config configs/m1_max_64gb.yaml")
        sys.exit(1)

    # Generate markdown report
    md_lines = ["# MLX TurboQuant Benchmark Results\n"]

    if bench_results:
        md_lines.append(generate_speed_table(bench_results))
        md_lines.append("")
        md_lines.append(generate_memory_table(bench_results))
        md_lines.append("")
        md_lines.append(generate_ttft_table(bench_results))

    if accuracy_results:
        md_lines.append("")
        md_lines.append(generate_accuracy_table(accuracy_results))

    md_report = "\n".join(md_lines)
    md_path = output_path / "report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Markdown report: {md_path}")
    print(md_report)

    # Generate HTML report
    html = generate_html_report(bench_results, accuracy_results)
    html_path = output_path / "report.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML report: {html_path}")


if __name__ == "__main__":
    main()
