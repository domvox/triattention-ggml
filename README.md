# TriAttention — KV Cache Pruning for llama.cpp

Frequency-based KV cache pruning with GPU compaction. Removes low-importance KV rows based on attention frequency scores, then physically compacts the cache on GPU.

Based on the same pre-RoPE Q/K concentration principle as [TriAttention (Mao et al., 2026)](https://arxiv.org/abs/2604.04921) — independent implementation for llama.cpp with HIP/ROCm GPU compaction kernel.

## Results (Qwen3.5-27B, RX 7900 XTX)

| Context | Retention | PPL | Δ vs baseline | KV rows kept |
|---|---|---|---|---|
| 4K | 100% | 8.1524 | baseline | 4096 |
| 4K | 75% | 8.1735 | +0.3% | ~3072 |
| 4K | 50% | 8.2290 | +0.9% | ~2048 |
| 16K | 100% | 8.1213 | baseline | 16384 |
| 16K | 75% | 8.1550 | +0.4% | ~12288 |
| 16K | 50% | 8.2488 | +1.8% | ~8192 |

75% retention holds under +1% PPL degradation. 50% retention gives 28% wall-time speedup (FA operates on shorter contiguous cache).

## How it works

1. **Calibration**: Run model on representative text, record per-head attention frequency scores using RoPE-aware phase prediction
2. **Runtime**: Every N tokens, score KV rows by predicted attention frequency, evict lowest-scoring rows, physically compact remaining rows on GPU
3. **Sink protection**: First N tokens always kept (attention sinks, StreamingLLM-style)

## Usage

```bash
# Integrated into llama.cpp fork:
llama-server -m model.gguf -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --triattention stats.bin --tri-budget 75 --tri-window 512
```

## Features

- GPU compaction kernel (HIP) — no CPU roundtrip
- Composable with TurboQuant KV compression
- Attention sink protection (`--tri-sink N`)
- Per-head frequency scoring with correct rope_theta

## Integration

TriAttention is integrated into the [TurboQuant HIP fork](https://github.com/domvox/llama.cpp-turboquant-hip) (branch: `feature/triattention-scoring`).

This repo contains the standalone calibration tools and research notes.

## Hardware

Tested on: AMD RX 7900 XTX 24GB, ROCm 6.4, openSUSE Tumbleweed.

## TriAttention + TurboQuant Combo (Qwen3-8B, 16K context)

| Config | PPL | Δ vs turbo3 |
|---|---|---|
| f16 baseline | 7.68 | — |
| turbo3 only | 22.71 | — |
| turbo3 + TriAttention 75% | **20.62** | **-9.2%** |

TriAttention pruning improves turbo3 quality by removing low-importance KV rows that contribute noise.
