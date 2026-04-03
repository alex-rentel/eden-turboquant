# Real Model Integration Test Results

Date: 2026-04-02
Machine: M1 Max 64GB

## Models Tested (6 families)

| Model | Repo | head_dim | KV heads | Layers | Smoke | Quality |
|-------|------|----------|----------|--------|-------|---------|
| Qwen3-1.7B | mlx-community/Qwen3-1.7B-4bit | 128 | 8 | 28 | PASS | Good |
| Qwen3-8B | mlx-community/Qwen3-8B-4bit | 128 | 8 | 36 | PASS | Excellent |
| Gemma3-1B | mlx-community/gemma-3-1b-it-4bit | 256 | 1 | 26 | PASS | Good |
| Gemma3-4B | mlx-community/gemma-3-4b-it-4bit | 256 | 4 | 34 | PASS | Degrades >1K |
| Llama-3.2-3B | mlx-community/Llama-3.2-3B-Instruct-4bit | 128 | 8 | 28 | PASS | Good |
| Mistral-7B | mlx-community/Mistral-7B-Instruct-v0.3-4bit | 128 | 8 | 32 | PASS | Excellent |

## Quality at ~500 tokens (cosine sim vs FP16)

| Model | K4/V4 | K4/V2 | K3.5/V2 | K3/V2 |
|-------|-------|-------|---------|-------|
| Qwen3-1.7B | 0.9914 | 0.9853 | — | 0.9687 |
| Qwen3-8B | 0.9994 | 0.9976 | 0.9963 | 0.9872 |
| Gemma3-1B | 0.9953 | 0.9802 | — | 0.9619 |
| Gemma3-4B | 0.9925 | 0.9848 | — | 0.9753 |
| Llama-3.2-3B | 0.9871 | 0.9478 | 0.9342 | — |
| Mistral-7B | 0.9980 | 0.9921 | 0.9852 | — |

## Needle-in-a-Haystack (Qwen3-8B, K4/V2, w128)

| Context | 25% | 50% | 75% |
|---------|-----|-----|-----|
| 1K | FOUND | FOUND | FOUND |
| 2K | FOUND | FOUND | FOUND |
| 4K | FOUND | FOUND | FOUND |
| 8K | FOUND | FOUND | FOUND |

**Score: 12/12 — matches FP16 baseline perfectly.**

## Fixes Applied During Testing

1. Fixed pyproject.toml build-backend (`setuptools.build_meta`)
2. Fixed `_get_model_config` for Gemma3 conditional models (nested `language_model.model.args` path)
3. Auto-upgrade bits for few-KV-head models (Gemma3-1B)
