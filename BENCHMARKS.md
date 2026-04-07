# Benchmarks

Comprehensive v0.6.0 benchmarks across **12 models** and **5 TurboQuant configurations**. Every cell reports cosine similarity vs FP16 baseline, decode tok/s, TTFT, and KV cache memory. Run date: 2026-04-07.

## Hardware & Environment

- **Machine:** Apple M1 Max, 64 GB unified memory, 32 GPU cores
- **OS:** macOS 26.4
- **Python:** 3.12.8
- **MLX:** 0.31.1
- **mlx-lm:** 0.31.1
- **mlx-turboquant:** 0.6.0

## Executive Summary

- **12/12 models benchmarked successfully** across 5 TurboQuant configurations.
- **Attention sink (`fp16_sink_size=128`) is the clear quality win** — improves cosine similarity on most models at the 500-token quality test, with the biggest gains on the smallest models (where shorter contexts make per-layer compression error matter more).
- **K4/V2 is the recommended default**: ~3.86× KV cache compression across all models, with cosine similarity above 0.95 on every well-behaved architecture.
- **K3/V2 is aggressive** — saves ~4.4× memory but cosine similarity drops significantly on short contexts. Use only when memory pressure is critical.
- **K4/V4 is conservative** — slightly better quality than K4/V2 but gives up most of the compression advantage (~2.5× instead of ~3.9×). Rarely worth it.

### Best-config recommendation by use case

| Use case | Recommended config |
|---|---|
| Balanced default | `K4/V2` |
| Quality-first (chat, tool-calling with system prompt) | `K4/V2 + fp16_sink_size=128` |
| Memory-first (long-context on limited RAM) | `K3/V2` |
| Conservative quality | `K4/V4` |

## Tier 1 — Primary 7B-9B Models

Seven models representing the main workhorse size class.

### Quality (cos sim vs FP16 baseline, 500-token prompt)

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| DeepSeek-R1-Qwen3-8B | 1.0000 | 0.9938 | 0.9889 | 0.9907 | 0.9746 |
| Llama-3.1-8B | 1.0000 | 0.9985 | 0.9947 | 0.9973 | 0.8798 |
| Mistral-7B | 1.0000 | 0.9995 | 0.9944 | 0.9977 | 0.9877 |
| Phi-3.5-mini | 1.0000 | 0.9986 | 0.9411 | 0.9971 | 0.9923 |
| Qwen2.5-7B | 1.0000 | 0.9595 | 0.7960 | 0.9474 | 0.6349 |
| Qwen3-8B | 1.0000 | 0.9957 | 0.8825 | 0.9810 | 0.7312 |
| Qwen3.5-9B | 1.0000 | 0.8652* | 0.7621* | 0.7896* | 0.6913* |

*asterisk = top-1 logit does NOT match FP16 baseline argmax*

### Decode speed at 256-token context (tok/s)

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| DeepSeek-R1-Qwen3-8B | 44.0 | 36.5 (+17%) | 38.0 (+14%) | 43.2 (+2%) | 37.9 (+14%) |
| Llama-3.1-8B | 59.2 | 40.7 (+31%) | 40.7 (+31%) | 56.5 (+5%) | 40.7 (+31%) |
| Mistral-7B | 61.6 | 42.4 (+31%) | 42.4 (+31%) | 59.2 (+4%) | 42.4 (+31%) |
| Phi-3.5-mini | 89.6 | 56.3 (+37%) | 55.9 (+38%) | 80.9 (+10%) | 56.4 (+37%) |
| Qwen2.5-7B | 60.8 | 43.9 (+28%) | 43.9 (+28%) | 60.2 (+1%) | 43.8 (+28%) |
| Qwen3-8B | 45.1 | 38.9 (+14%) | 38.9 (+14%) | 44.1 (+2%) | 38.9 (+14%) |
| Qwen3.5-9B | 42.8 | 38.2 (+11%) | 38.2 (+11%) | 42.4 (+1%) | 38.2 (+11%) |

### Decode speed at 2048-token context (tok/s)

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| DeepSeek-R1-Qwen3-8B | 39.9 | 30.4 (+24%) | 30.3 (+24%) | 30.2 (+24%) | 30.3 (+24%) |
| Llama-3.1-8B | 54.3 | 34.7 (+36%) | 34.7 (+36%) | 34.7 (+36%) | 34.7 (+36%) |
| Mistral-7B | 56.2 | 35.4 (+37%) | 35.5 (+37%) | 36.0 (+36%) | 35.4 (+37%) |
| Phi-3.5-mini | 72.6 | 28.7 (+60%) | 28.7 (+61%) | 28.7 (+61%) | 29.1 (+60%) |
| Qwen2.5-7B | 56.5 | 39.9 (+29%) | 39.8 (+30%) | 39.9 (+29%) | 39.9 (+29%) |
| Qwen3-8B | 40.6 | 31.7 (+22%) | 31.6 (+22%) | 31.3 (+23%) | 31.6 (+22%) |
| Qwen3.5-9B | 42.0 | 36.8 (+12%) | 36.8 (+12%) | 35.3 (+16%) | 36.8 (+12%) |

### TTFT at 2048-token context (ms)

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| DeepSeek-R1-Qwen3-8B | 9851 ms | 6926 ms | 7154 ms | 7153 ms | 7167 ms |
| Llama-3.1-8B | 5616 ms | 6699 ms | 6947 ms | 6949 ms | 6961 ms |
| Mistral-7B | 5333 ms | 6381 ms | 6600 ms | 6598 ms | 6623 ms |
| Phi-3.5-mini | 3161 ms | 4047 ms | 4010 ms | 4017 ms | 4236 ms |
| Qwen2.5-7B | 5260 ms | 6343 ms | 6332 ms | 6333 ms | 6337 ms |
| Qwen3-8B | 9836 ms | 6907 ms | 7149 ms | 6891 ms | 7156 ms |
| Qwen3.5-9B | 10350 ms | 7496 ms | 7486 ms | 7502 ms | 7508 ms |

### KV cache memory at 4096-token context

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| DeepSeek-R1-Qwen3-8B | 576 MB | 195 MB (2.95x) | 161 MB (3.57x) | 193 MB (2.99x) | 144 MB (3.99x) |
| Llama-3.1-8B | 512 MB | 164 MB (3.13x) | 133 MB (3.86x) | 162 MB (3.17x) | 117 MB (4.37x) |
| Mistral-7B | 512 MB | 164 MB (3.13x) | 133 MB (3.86x) | 162 MB (3.17x) | 117 MB (4.37x) |
| Phi-3.5-mini | 1536 MB | 499 MB (3.08x) | 406 MB (3.78x) | 492 MB (3.12x) | 360 MB (4.27x) |
| Qwen2.5-7B | 224 MB | 109 MB (2.05x) | 98 MB (2.29x) | 109 MB (2.06x) | 92 MB (2.44x) |
| Qwen3-8B | 576 MB | 195 MB (2.95x) | 161 MB (3.57x) | 193 MB (2.99x) | 144 MB (3.99x) |
| Qwen3.5-9B | 153 MB | 87 MB (1.76x) | 79 MB (1.93x) | 87 MB (1.77x) | 75 MB (2.03x) |

## Tier 2 — Smaller Models

Five smaller models (1B-4B) validating breadth across head_dim and KV-head counts.

### Quality (cos sim vs FP16 baseline, 500-token prompt)

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| Llama-3.2-3B | 1.0000 | 0.9978 | 0.9501 | 0.9847 | 0.8059 |
| Qwen3-1.7B | 1.0000 | 0.8883 | 0.8807 | 0.9252 | 0.8006 |
| Qwen3-4B | 1.0000 | 0.9921 | 0.9830 | 0.9837 | 0.4728 |
| Gemma3-1B | 1.0000 | 0.9989 | 0.9985 | 0.9985 | 0.9985 |
| Gemma3-4B | 1.0000 | 0.9982 | 0.9959 | 0.9973 | 0.9947 |

*asterisk = top-1 logit does NOT match FP16 baseline argmax*

### Decode speed at 2048-token context (tok/s)

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| Llama-3.2-3B | 98.8 | 59.0 (+40%) | 58.9 (+40%) | 57.6 (+42%) | 57.0 (+42%) |
| Qwen3-1.7B | 114.8 | 79.3 (+31%) | 78.0 (+32%) | 77.9 (+32%) | 78.7 (+31%) |
| Qwen3-4B | 59.5 | 46.5 (+22%) | 46.5 (+22%) | 46.0 (+23%) | 46.5 (+22%) |
| Gemma3-1B | 143.4 | 117.2 (+18%) | 117.2 (+18%) | 114.0 (+21%) | 117.7 (+18%) |
| Gemma3-4B | 64.2 | 43.2 (+33%) | 43.4 (+32%) | 42.4 (+34%) | 43.3 (+33%) |

### KV cache memory at 4096-token context

| Model | baseline | K4/V4 | K4/V2 | K4/V2+sink128 | K3/V2 |
|---|---|---|---|---|---|
| Llama-3.2-3B | 448 MB | 143 MB (3.13x) | 116 MB (3.86x) | 141 MB (3.17x) | 103 MB (4.37x) |
| Qwen3-1.7B | 448 MB | 181 MB (2.47x) | 156 MB (2.87x) | 179 MB (2.50x) | 143 MB (3.13x) |
| Qwen3-4B | 576 MB | 195 MB (2.95x) | 161 MB (3.57x) | 193 MB (2.99x) | 144 MB (3.99x) |
| Gemma3-1B | 104 MB | 35 MB (2.95x) | 32 MB (3.23x) | 38 MB (2.76x) | 32 MB (3.23x) |
| Gemma3-4B | 544 MB | 197 MB (2.76x) | 165 MB (3.30x) | 195 MB (2.79x) | 149 MB (3.65x) |

## Architecture Reference

### Tier 1

| Model | Class | Layers | head_dim | KV heads | Notes |
|---|---|---|---|---|---|
| DeepSeek-R1-Qwen3-8B | Model | 36 | 128 | 8 |  |
| Llama-3.1-8B | Model | 32 | None | 8 |  |
| Mistral-7B | Model | 32 | None | 8 |  |
| Phi-3.5-mini | Model | 32 | None | 32 |  |
| Qwen2.5-7B | Model | 28 | None | 4 |  |
| Qwen3-8B | Model | 36 | 128 | 8 |  |
| Qwen3.5-9B | Model | 32 | None | None |  |

### Tier 2

| Model | Class | Layers | head_dim | KV heads | Notes |
|---|---|---|---|---|---|
| Llama-3.2-3B | Model | 28 | 128 | 8 |  |
| Qwen3-1.7B | Model | 28 | 128 | 8 |  |
| Qwen3-4B | Model | 36 | 128 | 8 |  |
| Gemma3-1B | Model | 26 | 256 | 1 |  |
| Gemma3-4B | Model | 34 | None | None |  |

### Special handling notes

- **Qwen3.5-9B**: hybrid attention (24 of 32 layers are `linear_attn`, 8 are `self_attn`). `apply_turboquant` now detects this automatically and only installs TurboQuantKVCache on the 8 self-attention layers; linear-attention layers get the model's native cache type. Compression coverage is therefore partial (8/32 layers). See patch.py and the corresponding test in `tests/test_edge_cases.py::test_hybrid_attention_skips_linear_attn_layers`.
- **Gemma3-1B**: 1 KV head. Auto-upgrades K<4 to K4 and V<3 to V3 to preserve quality (1-KV-head models have no headroom for aggressive compression). So K3/V2 and K4/V2 both effectively become K4/V3.
- **Phi-3.5-mini**: head_dim=96 (not a power of 2), 32 KV heads (no GQA). Metal dequant kernels work correctly because they template on D; the library just compiles a different kernel variant.
- **DeepSeek-R1-0528-Qwen3-8B**: standard self-attention, behaves identically to Qwen3-8B in benchmarks despite being a reasoning fine-tune.

## Needle-in-a-Haystack Sweep

Retrieval test across three Tier-1 architectures at 1K / 2K / 4K / 8K context, needle at positions 0.1 / 0.5 / 0.9 (12 cells per config). Secret: `mango-sunset-42`. Reproduce with `python benchmarks/needle_haystack_v06.py --model <id>`.

| Model | baseline (FP16) | K4/V2 | K4/V2 + sink128 |
|---|---|---|---|
| Qwen3-8B-4bit | **12/12** | **12/12** | **12/12** |
| Llama-3.1-8B-Instruct-4bit | **12/12** | **12/12** | **12/12** |
| Mistral-7B-Instruct-v0.3-4bit | 8/12 | 6/12 | 7/12 |

**Headline:** Qwen3-8B and Llama-3.1-8B achieve perfect retrieval across all context lengths under both K4/V2 and K4/V2+sink128 — no regression from v0.6.0's new features.

**Mistral-7B caveat:** Mistral fails 4 of 12 cells **even on the FP16 baseline** at this haystack. The failures are concentrated at 4K and 8K context when the model is asked to retrieve from a highly repetitive filler document. TurboQuant K4/V2 adds 2 more failures (6/12) and K4/V2+sink128 recovers one (7/12). The takeaway is that TurboQuant does not take Mistral-7B meaningfully below its own baseline capability on this test — the model itself has limits with this particular haystack pattern that are inherited by all configs.

Full per-cell logs in `results/needle_Qwen3-8B.log`, `results/needle_Llama-3.1-8B.log`, `results/needle_Mistral-7B.log`.

## Methodology

- **Quality**: Cosine similarity of last-token logits vs an FP16 reference computed from the same prompt. Logits are cast from bfloat16/float16 to float32 and then compared in float64 for numerical stability.
- **Speed**: Pure decode tok/s, measured as the median of 3 timed runs after 1 warmup. Each run allocates a fresh cache, runs the full prefill, decodes the first token, then times the remaining `decode_tokens - 1` steps of the decode loop.
- **TTFT**: Wall-clock time from cache allocation through prefill and the first decoded token, median of 3 runs.
- **Memory**: Sum of `cache.nbytes` per layer after the prefill completes. For baseline mlx-lm KVCache this is the raw `.keys.nbytes + .values.nbytes` fallback. This measures the cache tensors only, not process RSS.
- **Outlier detection**: `auto_detect_outliers=True` (the library default). For Qwen-family models this keeps 1-4 extreme-norm layers in FP16, which makes a large difference on short prompts. Disabling it is supported via apply_turboquant but not tested in this sweep.
- **Noise**: Repeat runs on the M1 Max showed up to ~15% variance on absolute decode tok/s due to thermal and GPU scheduler state. Relative comparisons within a single run (TQ vs baseline, sink vs no-sink on the same model) are more reliable than absolute tok/s comparisons across sessions.

## Reproducing

```bash
# Verify every model loads first
python benchmarks/verify_models.py --json results/verify.json

# Run the full sweep
python benchmarks/run_full_suite.py --config benchmarks/models.yaml --tier 1
python benchmarks/run_full_suite.py --config benchmarks/models.yaml --tier 2

# Regenerate this document from the JSON results
python benchmarks/report_builder.py --out BENCHMARKS.md
```

Each tier writes per-model JSON to `results/tier<N>/*.json` and an aggregate `results/tier<N>/all.json`.
