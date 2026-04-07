# mlx-turboquant

Near-optimal KV cache quantization for Apple Silicon. Faithful implementation of [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026).

Compresses the KV cache during LLM inference by 3x with <0.3% quality loss. No training, no calibration data, fully data-oblivious. Drop-in for any [mlx-lm](https://github.com/ml-explore/mlx-lm) model.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Quick Start

```bash
git clone https://github.com/alex-rentel/mlx-turboquant.git
cd mlx-turboquant
pip install -e .
```

```python
from mlx_lm import load
from mlx_turboquant import apply_turboquant

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
apply_turboquant(model, key_bits=4, value_bits=2)

cache = model.make_cache()
# Use cache normally — older tokens are compressed automatically
logits = model(inputs, cache=cache)
```

## Benchmarks

**12 models × 5 configs**, M1 Max 64GB / mlx 0.31.1. Full tables and methodology in [BENCHMARKS.md](BENCHMARKS.md). Highlights below.

### Quality (cos sim vs FP16, 500-token prompt, K4/V2 default)

| Model | K4/V2 | **K4/V2 + sink128** | Sink boost |
|---|---|---|---|
| Qwen3-8B-4bit | 0.8825 | **0.9810** | **+0.098** |
| Qwen2.5-7B-Instruct-4bit | 0.7960 | **0.9474** | **+0.151** |
| Phi-3.5-mini-instruct-4bit | 0.9411 | **0.9971** | **+0.056** |
| Llama-3.1-8B-Instruct-4bit | 0.9947 | 0.9973 | +0.003 |
| Mistral-7B-Instruct-v0.3-4bit | 0.9944 | 0.9977 | +0.003 |
| DeepSeek-R1-0528-Qwen3-8B-4bit | 0.9889 | 0.9907 | +0.002 |

**Attention sink (`fp16_sink_size=128`) is the headline quality win of v0.6.0** — massive improvements on Qwen family and Phi-3.5-mini, negligible cost on models that already score >0.99 at K4/V2.

### KV cache memory compression (K4/V2, 4K context)

| Model | Baseline | K4/V2 | Ratio |
|---|---|---|---|
| Phi-3.5-mini | 1536 MB | 406 MB | **3.78x** |
| Qwen3-8B / DeepSeek-R1-Qwen3-8B | 576 MB | 161 MB | **3.57x** |
| Llama-3.1-8B / Mistral-7B / Llama-3.2-3B | 512/448 MB | 133/116 MB | **3.86x** |
| Gemma3-4B | 544 MB | 165 MB | **3.30x** |

### Needle-in-a-Haystack (K4/V2 and K4/V2+sink128)

| Model | FP16 baseline | K4/V2 | K4/V2 + sink128 |
|---|---|---|---|
| Qwen3-8B-4bit | **12/12** | **12/12** | **12/12** |
| Llama-3.1-8B-Instruct-4bit | **12/12** | **12/12** | **12/12** |
| Mistral-7B-Instruct-v0.3-4bit | 8/12 | 6/12 | 7/12 |

Qwen3 and Llama-3.1 achieve perfect retrieval across all context lengths up to 8K under K4/V2 and K4/V2+sink128. Mistral-7B fails 4/12 cells even on FP16 baseline with this haystack — TurboQuant tracks but does not exceed the model's inherent limit.

### Architecture breadth

Validated on Qwen3, Qwen3.5 (hybrid attention), Qwen2.5, Llama 3.1/3.2, Mistral, Gemma 3, Phi 3.5, and DeepSeek R1 — **13 models across 8 architecture families**. The new hybrid attention support path automatically detects and skips `linear_attn` layers (Qwen3.5 has 24 of 32 layers that would otherwise crash compression).

## How It Works

TurboQuant implements both algorithms from the paper:

**Algorithm 1 — TurboQuant_mse (default).** A fixed random orthogonal matrix (QR decomposition) rotates each KV vector so that its coordinates become approximately independent and Beta-distributed. Each coordinate is then quantized using a precomputed Lloyd-Max codebook optimized for that distribution. Only the quantization indices and the vector's L2 norm are stored. To dequantize: look up centroids, inverse-rotate, rescale by norm. The per-vector MSE is within 2.7x of the Shannon information-theoretic lower bound.

**Algorithm 2 — TurboQuant_prod (opt-in).** Applies (b-1)-bit MSE quantization, computes the residual, then applies a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform on the residual. This produces an unbiased inner product estimator. However, 6+ independent community implementations confirmed that QJL hurts quality as a drop-in KV cache replacement — it's only beneficial with fused attention kernels that consume the two-part representation directly. Algorithm 1 (MSE-only) is the default for this reason.

**Key design decisions informed by community findings:**
- Asymmetric K/V allocation — keys need more bits than values (K4/V2 recommended)
- Residual window — recent tokens stay in FP16 for quality (default: 128 tokens)
- Outlier layer detection — some models (Qwen3) have layers with extreme key norms that must stay uncompressed
- Few-KV-head safety — models with ≤2 KV heads (Gemma3-1B) auto-upgrade to K4/V3 minimum

## Configuration

```python
apply_turboquant(
    model,
    key_bits=4,                # 2, 3, 3.5, or 4 bits for keys
    value_bits=2,              # 2, 3, 3.5, or 4 bits for values
    residual_window=128,       # recent tokens stay in FP16 (sliding window)
    auto_detect_outliers=True, # skip layers with extreme key norms
    skip_layers=[0, 27],       # manually specify FP16 layers
    # New in v0.6.0:
    fp16_sink_size=128,        # permanent FP16 sink for first N tokens
                               # (system prompt). Default 0 = disabled.
                               # Improves cosine sim by +0.002 to +0.003
                               # on Qwen3 family with no speed cost.
    chunk_size=0,              # 0 (default) = v0.5.0 batch compression.
                               # >0 enables fixed-size chunked draining.
                               # Opt in for future kernel work; benchmark-
                               # neutral with current Metal kernels.
    qjl_correction=False,      # EXPERIMENTAL. 1-bit QJL sign-sketch
                               # residual correction. Helps Qwen3-8B
                               # (+0.0017 cos sim) but HURTS Qwen3-1.7B
                               # (-0.0052). Default OFF.
    qjl_n_proj=32,             # QJL projection rank (when correction on).
)
```

### When to enable `fp16_sink_size`

Enable for tool-calling, JSON-mode extraction, or any workflow where the
first 64-256 tokens carry critical schema or instruction tokens you want
to preserve bit-exactly. The cost is `fp16_sink_size * 2 * head_dim *
num_kv_heads * 4 bytes` per layer of additional FP16 memory — typically
under 5 MB total even for large models.

You can also use the low-level quantizer directly:

```python
from mlx_turboquant import TurboQuantMSE, TurboQuantProd

# Algorithm 1: MSE-optimized (recommended for KV cache)
tq = TurboQuantMSE(d=128, bits=4)
qt = tq.quantize(vectors)
vectors_hat = tq.dequantize(qt)

# Algorithm 2: Inner-product optimized (for custom attention kernels)
tq_prod = TurboQuantProd(d=128, bits=4)
qt = tq_prod.quantize(vectors)
```

## CLI

```bash
# Generate text with TurboQuant compression
mlx-turboquant generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Explain quantum computing" \
  --key-bits 4 --value-bits 2

# Run benchmarks
mlx-turboquant benchmark \
  --model mlx-community/Qwen3-8B-4bit \
  --benchmarks quality memory speed
```

## Supported Models

Any model loaded via `mlx_lm.load()`. Tested on:

| Family | Models Tested | head_dim | Notes |
|--------|--------------|----------|-------|
| Qwen3 | 1.7B, 8B | 128 | Excellent quality, has outlier layers |
| Gemma3 | 1B, 4B | 256 | Works well up to ~1K context |
| Llama 3.2 | 3B | 128 | Tested, coherent generation |
| Mistral | 7B v0.3 | 128 | Tested, excellent quality (0.998 cos_sim) |

## Project Structure

```
mlx_turboquant/
  codebook.py     # Lloyd-Max optimal scalar quantizer for Beta distributions
  rotation.py     # Random orthogonal rotation (QR) + Walsh-Hadamard
  quantizer.py    # TurboQuantMSE (Algorithm 1) + TurboQuantProd (Algorithm 2)
  qjl.py          # Quantized Johnson-Lindenstrauss transform
  packing.py      # 1/2/3/4-bit index packing into uint8
  kernels.py      # Fused Metal compute kernels (dequantize, quantize)
  cache.py        # TurboQuantKVCache — drop-in for mlx-lm's KVCache
  patch.py        # Model patching — apply_turboquant() monkey-patch
  cli.py          # Command-line interface
  codebooks/      # Precomputed Lloyd-Max codebooks (.npz)
benchmarks/       # Quality, memory, speed, needle-in-haystack benchmarks
                  # bench_v06.py + needle_haystack_v06.py for v0.6.0 sweep
tests/            # 157 unit tests + 1 integration test with real model
```

## Running Tests

```bash
# Unit tests (no model download, fast)
python -m pytest tests/ -q --ignore=tests/test_integration.py

# Integration test (downloads ~1GB model)
python -m pytest tests/test_integration.py -v -m slow

# Benchmarks (downloads models, takes several minutes)
python benchmarks/bench_quality.py
python benchmarks/bench_memory.py
python benchmarks/bench_speed.py
```

## Limitations

1. **Decode speed overhead (~11%):** Fused Metal kernels + batch compression + pre-allocated windows reduced overhead from 57% to 11%. The remaining overhead is MLX per-operation dispatch cost (~0.25ms per concatenate). A fused attention-from-compressed kernel would eliminate this entirely.
2. **Error compounding:** Per-vector reconstruction error (~0.5%) compounds through transformer layers. Models with more KV heads (Qwen3: 8 heads) are more robust than those with fewer (Gemma3-1B: 1 head).
3. **Context-dependent compression:** At contexts shorter than `residual_window`, no compression occurs. Memory savings grow with context length.

## Roadmap

### Completed

| Version | Feature | Result |
|---------|---------|--------|
| v0.2.0 | Real model testing, vectorized quantization | 6 model families validated |
| v0.3.0 | Fused Metal dequantize kernels | 57% → 33% overhead |
| v0.4.0 | 3.5-bit fractional, needle-in-haystack, PyPI packaging | 12/12 retrieval at 1K-8K |
| v0.5.0 | Batch compression + pre-allocated FP16 window | 33% → **11%** overhead |
| **v0.6.0** | Attention sink + QJL correction + state-reload fixes + hybrid attention | **+0.098 cos_sim on Qwen3-8B** from sink, 12/12 needle preserved, 12 models validated |

### Next: Fused Attention-from-Compressed Kernel

The remaining 11% overhead is MLX dispatch cost from concatenating decompressed + FP16 tensors. The path to sub-5%:

1. **Fused attention-from-compressed kernel:** Compute Q @ K^T directly from packed indices without materializing the full dequantized K tensor. Another MLX TurboQuant implementation demonstrated 0.98x native speed with this approach.
2. **Walsh-Hadamard fast path:** WHT is O(d log d) vs O(d²) for QR. Works for Qwen3 but degrades Gemma3 by 1.6%. A dimension-adaptive strategy could use WHT where safe.

### Future

- Needle-in-a-haystack validation at 16K-32K context
- Integration with [eden-fleet](https://github.com/alex-rentel/eden-fleet) for distributed inference with compressed KV cache transfer between nodes
- Upstream contribution to mlx-lm as an optional KV cache backend

## Part of the Eden Ecosystem

mlx-turboquant is part of a local-first AI development ecosystem for Apple Silicon:

- [eden](https://github.com/alex-rentel/eden) — Local AI agent framework
- [eden-flywheel](https://github.com/alex-rentel/eden-flywheel) — Capture Claude Code sessions → training data → fine-tune → deploy
- [eden-fleet](https://github.com/alex-rentel/eden-fleet) — Distribute AI workloads across a Mac homelab via SSH
- [eden-models](https://github.com/alex-rentel/eden-models) — Training pipeline for 1-bit tool-calling LLMs on HPC
- [mlx-nanochat](https://github.com/alex-rentel/mlx-nanochat) — Train ChatGPT-class models on Mac (port of Karpathy's nanochat)
- **mlx-turboquant** — KV cache compression for longer contexts on Apple Silicon

## Community Implementations

The TurboQuant MLX space moved fast after the paper dropped. Notable implementations worth checking out:

- [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) — V2 (affine, fast) + V3 (Lloyd-Max, correct) dual-path architecture. Thorough analysis of QJL as correction vs replacement.
- [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) — Fused Metal kernels with parallel threadgroup WHT, layer-adaptive FP16/TurboQuant mixing. Reports 4.6x compression at 98% FP16 speed.
- [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) — Clean drop-in KVCache replacement with precomputed Lloyd-Max codebooks for N(0,1).
- [helgklaizar/turboquant_mlx](https://github.com/helgklaizar/turboquant_mlx) — Asymmetric PolarQuant with attention sink preservation and dynamic chunking. Two-line monkey-patch into mlx_lm.
- [flovflo/turboquant-mlx-qwen35-kv](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv) — Qwen 3.5 35B benchmarks showing 26% faster generation and 44% cache reduction.

This repo differentiates by combining fused Metal dequant kernels, asymmetric K/V bit allocation, per-model outlier layer detection, and fractional (3.5-bit) quantization in a single drop-in package — validated across 6 model families with needle-in-haystack testing.

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

MIT — crediting original authors (Zandieh, Daliri, Hadian, Mirrokni). See [LICENSE](LICENSE).
