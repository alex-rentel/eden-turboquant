#!/usr/bin/env python3
"""Test whether tool-calling accuracy is preserved under KV cache compression.

This is our unique contribution — nobody else benchmarks tool-call JSON
generation accuracy under TurboQuant compression.

Usage:
    python benchmarks/tool_call_accuracy.py \
        --model mlx-community/gemma-4-26B-A4B-it-4bit \
        --modes baseline,turboquant \
        --test-set benchmarks/tool_call_tests.jsonl \
        --output results/tool_accuracy_gemma4.json
"""

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
JSON_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


SYSTEM_PROMPT = """You are a helpful assistant with access to the following tools. When the user asks you to perform an action, respond with a tool call in the following format:

<tool_call>
{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}
</tool_call>

Available tools:

- bash: Execute a shell command. Parameters: command (string, required)
- file_read: Read a file's contents. Parameters: path (string, required)
- file_write: Write content to a file. Parameters: path (string, required), content (string, required)
- grep: Search for a pattern in files. Parameters: pattern (string, required), path (string, optional), include (string, optional)
- http_request: Make an HTTP request. Parameters: method (string, required), url (string, required), headers (object, optional), body (string, optional)
- database_query: Execute a database query. Parameters: query (string, required), database (string, optional)
- send_email: Send an email. Parameters: to (string, required), subject (string, required), body (string, required)
- create_calendar_event: Create a calendar event. Parameters: title (string, required), start_time (string, required), end_time (string, required), attendees (array of strings, optional)
- web_search: Search the web. Parameters: query (string, required), num_results (integer, optional)
- image_generate: Generate an image from a prompt. Parameters: prompt (string, required), width (integer, optional), height (integer, optional)

Respond ONLY with the tool call, no other text."""


@dataclass
class AccuracyResult:
    test_id: int
    description: str
    mode: str
    exact_match: bool = False
    function_match: bool = False
    parseable: bool = False
    format_correct: bool = False
    expected_function: str = ""
    generated_function: str = ""
    generated_raw: str = ""
    error: str = ""


@dataclass
class AccuracySummary:
    mode: str
    total: int = 0
    exact_match: int = 0
    function_match: int = 0
    parseable: int = 0
    format_correct: int = 0
    total_score: float = 0.0
    errors: int = 0


def parse_tool_call(text):
    """Extract tool call JSON from model output."""
    # Try <tool_call> tags first
    match = TOOL_CALL_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group(1)), True
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    match = JSON_PATTERN.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "name" in parsed or "function" in parsed:
                return parsed, False
        except json.JSONDecodeError:
            pass

    return None, False


def normalize_tool_call(tc):
    """Normalize tool call to standard format."""
    if tc is None:
        return None

    name = tc.get("name") or tc.get("function") or tc.get("function_name", "")
    args = tc.get("arguments") or tc.get("parameters") or tc.get("params", {})

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}

    return {"name": name, "arguments": args}


def compare_tool_calls(expected, generated):
    """Compare expected vs generated tool calls."""
    if generated is None:
        return False, False

    exp_norm = normalize_tool_call(expected)
    gen_norm = normalize_tool_call(generated)

    if exp_norm is None or gen_norm is None:
        return False, False

    function_match = exp_norm["name"].lower() == gen_norm["name"].lower()
    exact_match = (
        function_match
        and json.dumps(exp_norm["arguments"], sort_keys=True)
        == json.dumps(gen_norm["arguments"], sort_keys=True)
    )

    return exact_match, function_match


def load_test_set(path):
    """Load test cases from JSONL file."""
    tests = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))
    return tests


def generate_for_test(model, tokenizer, test_case, mode, model_path=None):
    """Generate model output for a test case."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_case["input"]},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {test_case['input']}\n\nAssistant:"

    max_tokens = 200

    if mode == "baseline":
        output = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
    elif mode == "kv_quant":
        output = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
            kv_bits=4, verbose=False
        )
    elif mode == "turboquant":
        if not HAS_MLX_VLM:
            raise ImportError("mlx-vlm required for turboquant mode")
        from mlx_vlm import load as vlm_load, generate as vlm_generate
        vlm_model, processor = vlm_load(model_path)
        output = vlm_generate(
            vlm_model, processor, prompt=prompt, max_tokens=max_tokens,
            kv_bits=3.5, kv_quant_scheme="turboquant", verbose=False
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return output


def evaluate_single(test_case, output, mode, test_id):
    """Evaluate a single test case."""
    result = AccuracyResult(
        test_id=test_id,
        description=test_case.get("description", ""),
        mode=mode,
        expected_function=test_case["expected"].get("name", ""),
        generated_raw=output,
    )

    parsed, has_tags = parse_tool_call(output)
    result.format_correct = has_tags
    result.parseable = parsed is not None

    if parsed:
        normalized = normalize_tool_call(parsed)
        result.generated_function = normalized["name"] if normalized else ""

        exact, func = compare_tool_calls(test_case["expected"], parsed)
        result.exact_match = exact
        result.function_match = func

    return result


def run_accuracy_test(model_path, modes, test_set_path, output_path):
    """Run the full accuracy test suite."""
    tests = load_test_set(test_set_path)
    print(f"Loaded {len(tests)} test cases from {test_set_path}")

    model = None
    tokenizer = None
    if HAS_MLX_LM:
        print(f"Loading model: {model_path}")
        model, tokenizer = mlx_lm.load(model_path)

    all_results = []
    summaries = {}

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        if mode == "turboquant" and not HAS_MLX_VLM:
            print("  WARNING: mlx-vlm not installed, skipping turboquant mode")
            continue

        summary = AccuracySummary(mode=mode, total=len(tests))
        mode_results = []

        for i, test in enumerate(tests):
            print(f"  Test {i+1}/{len(tests)}: {test.get('description', '')[:50]}")

            try:
                output = generate_for_test(
                    model, tokenizer, test, mode, model_path=model_path
                )
                result = evaluate_single(test, output, mode, i)
            except Exception as e:
                result = AccuracyResult(
                    test_id=i,
                    description=test.get("description", ""),
                    mode=mode,
                    error=str(e),
                )
                summary.errors += 1

            if result.exact_match:
                summary.exact_match += 1
            if result.function_match:
                summary.function_match += 1
            if result.parseable:
                summary.parseable += 1
            if result.format_correct:
                summary.format_correct += 1

            mode_results.append(asdict(result))
            all_results.append(asdict(result))

        # Weighted total score
        if summary.total > 0:
            summary.total_score = round(
                (
                    summary.exact_match * 0.4
                    + summary.function_match * 0.3
                    + summary.parseable * 0.2
                    + summary.format_correct * 0.1
                )
                / summary.total
                * 100,
                2,
            )

        summaries[mode] = asdict(summary)
        print(f"\n  Results for {mode}:")
        print(f"    Exact match:    {summary.exact_match}/{summary.total}")
        print(f"    Function match: {summary.function_match}/{summary.total}")
        print(f"    Parseable:      {summary.parseable}/{summary.total}")
        print(f"    Format correct: {summary.format_correct}/{summary.total}")
        print(f"    Total score:    {summary.total_score}%")

    output_data = {
        "model": model_path,
        "test_set": str(test_set_path),
        "modes": modes,
        "summaries": summaries,
        "results": all_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate comparison table
    print("\n## Tool-Calling Accuracy Comparison\n")
    print("| Metric | " + " | ".join(modes) + " |")
    print("|--------|" + "|".join(["---"] * len(modes)) + "|")
    for metric in ["exact_match", "function_match", "parseable", "format_correct", "total_score"]:
        row = f"| {metric} |"
        for mode in modes:
            if mode in summaries:
                val = summaries[mode][metric]
                if metric == "total_score":
                    row += f" {val}% |"
                else:
                    row += f" {val}/{summaries[mode]['total']} |"
            else:
                row += " — |"
        print(row)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Test tool-calling accuracy under KV cache compression"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--modes", default="baseline,turboquant",
        help="Comma-separated modes to test"
    )
    parser.add_argument(
        "--test-set", default="benchmarks/tool_call_tests.jsonl",
        help="Path to JSONL test set"
    )
    parser.add_argument(
        "--output", default="results/tool_accuracy.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]

    print("MLX TurboQuant — Tool-Calling Accuracy Test")
    print("=============================================")
    print(f"Model: {args.model}")
    print(f"Modes: {modes}")
    print(f"Test set: {args.test_set}")
    print()

    run_accuracy_test(args.model, modes, args.test_set, args.output)


if __name__ == "__main__":
    main()
