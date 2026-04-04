# mlx-turboquant

Benchmarking and integration layer for TurboQuant KV cache compression on Apple Silicon.

## What This Is

TurboQuant (Google Research, ICLR 2026) compresses LLM KV caches to 3-3.5 bits per channel with zero accuracy loss. The community has already built working MLX implementations. This repo does NOT reimplement TurboQuant — instead it:

1. **Benchmarks** TurboQuant across Apple Silicon hardware (M1 Max, M4) on models we actually use
2. **Validates** that tool-calling accuracy is preserved under compression
3. **Provides** integration configs for Ollama and mlx-lm

## Why Tool-Calling Validation Matters

Most TurboQuant benchmarks test perplexity and needle-in-haystack. But if you're deploying fine-tuned tool-calling models (like we do with eden-models), you need to know: does the model still generate valid `<tool_call>` JSON under 3.5-bit KV compression? We test that.

## Community Implementations (credit)
- [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) — Custom Metal kernels, V2+V3 paths
- [flovflo/turboquant-mlx-qwen35-kv](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv) — Qwen 3.5 specific
- mlx-vlm native support: `--kv-bits 3.5 --kv-quant-scheme turboquant`

## Installation

```bash
git clone https://github.com/alex-rentel/mlx-turboquant.git
cd mlx-turboquant
pip install -e .

# Optional: install mlx-vlm for TurboQuant mode
pip install mlx-vlm
```

## Usage

### Run benchmarks on a single model

```bash
python benchmarks/run_benchmark.py \
  --model mlx-community/gemma-4-26B-A4B-it-4bit \
  --modes baseline,kv_quant,turboquant \
  --context-lengths 2048,8192,16384,32768 \
  --output results/gemma4-26b.json
```

### Run full hardware suite

```bash
python benchmarks/run_all.py --config configs/m1_max_64gb.yaml
```

### Run tool-calling accuracy test

```bash
python benchmarks/tool_call_accuracy.py \
  --model mlx-community/gemma-4-26B-A4B-it-4bit \
  --modes baseline,turboquant \
  --test-set benchmarks/tool_call_tests.jsonl \
  --output results/tool_accuracy_gemma4.json
```

### Visualize results

```bash
python benchmarks/visualize.py --input results/ --output reports/
```

## Results

> Benchmarks pending — run `python benchmarks/run_all.py --config configs/m1_max_64gb.yaml` on your hardware and submit a PR!

### Tool-Calling Accuracy Under Compression

| Model | Baseline | KV Quant (4-bit) | TurboQuant (3.5-bit) |
|-------|----------|-------------------|----------------------|
| Gemma 4 26B MoE | — | — | — |
| Qwen 3.5 35B | — | — | — |
| Qwen 2.5 7B | — | — | — |

### Generation Speed (tok/s)

| Model | Context | Baseline | KV Quant (4-bit) | TurboQuant (3.5-bit) |
|-------|---------|----------|-------------------|----------------------|
| Gemma 4 26B MoE | 2048 | — | — | — |
| Gemma 4 26B MoE | 32768 | — | — | — |

### Peak Memory Usage (GB)

| Model | Context | Baseline | KV Quant (4-bit) | TurboQuant (3.5-bit) |
|-------|---------|----------|-------------------|----------------------|
| Gemma 4 26B MoE | 2048 | — | — | — |
| Gemma 4 26B MoE | 32768 | — | — | — |

## Sister Repos

- **eden-models** — Training configs for tool-calling fine-tunes
- **training-flywheel** — Data capture pipeline
- **mlx-nanochat** — Local training on Apple Silicon

Results from this repo validate that models trained by those repos will work under KV cache compression.

## Paper

[TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)

## License

MIT — see [LICENSE](LICENSE). Credits to Google Research for the TurboQuant paper and community implementors listed above.
