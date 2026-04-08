# TriAttention ggml — KV Cache Pruning for llama.cpp

Standalone prototype + **working llama.cpp integration** of [TriAttention](https://arxiv.org/abs/2604.04921) KV cache pruning. First implementation in llama.cpp / ggml.

TriAttention exploits pre-RoPE Q/K concentration in frequency space to predict which KV cache entries will be important — without computing actual attention. This allows physically compacting the KV cache at runtime, reducing memory and speeding up Flash Attention.

## llama.cpp Integration Results

Tested on **Qwen3-8B Q4_K_M** / RX 7900 XTX (24GB) / ROCm 6.4 / WikiText-2 / 20 chunks / ctx=4096.

### Retention vs Quality (attention masking)

| Retention | Evicted | PPL | Δ vs baseline |
|-----------|---------|-----|---------------|
| 100% (baseline) | 0% | 8.1524 | — |
| 75% | 25% | 8.2129 | +0.7% |
| 50% | 50% | 8.4907 | +4.1% |
| 25% | 75% | 8.6415 | +6.0% |
| 10% | 90% | 8.7901 | +7.8% |

### Physical Compaction (GPU kernel)

| Metric | Baseline | TriAttention 50% |
|--------|----------|-----------------|
| Total time (5 chunks) | ~58s | **41.9s** |
| PPL | 8.1524 | 8.5623 (+5.0%) |

Physical compaction makes the total wall time **faster than baseline** because Flash Attention operates on a shorter, contiguous cache after pruning.

### Optimization History

| Version | Total time | Speedup |
|---------|-----------|---------|
| Initial (CPU compaction) | 94.7s | — |
| + quickselect, batch copy, no malloc | 65.6s | 1.4× |
| + HIP GPU gather kernel | 41.9s | 2.3× |

## Architecture

### Python Prototype (this repo)
```
triattention_common.py        — Shared scoring, I/O, keep-set computation
triattention_calibrate.py     — Capture pre-RoPE Q, compute MRL stats → TRIA binary
triattention_score.py         — Score keys, simulate pruning
triattention_validate*.py     — Recall, attention mass, NLL validation
```

### C Scoring Library (this repo)
```
triattention.h / .c           — TRIA binary loader, per-KV-head scoring (21× optimized)
triattention_score_ref.c      — Reference implementation for cross-validation
bench_tria.c                  — Scoring throughput benchmark
```

### llama.cpp Integration ([separate branch](https://github.com/domvox/llama.cpp-turboquant-hip/tree/feature/triattention-scoring))
```
src/triattention-runtime.c    — Scoring hook in llama_decode, global score aggregation
src/triattention-bridge.cpp   — C++ bridge to KV cache internals
src/triattention-hip.hip      — GPU gather kernel for physical compaction
src/llama-kv-cache.cpp        — Cache compaction + metadata rebuild
```

**Pipeline:** calibrate offline → load TRIA stats → score every N tokens → z-normalize + max-aggregate across heads → global top-K → physically compact K/V on GPU → FA runs on shorter cache.

## Python Prototype Results

Tested on Qwen3-8B, 50% retention:

| Phase | Metric | Result |
|---|---|---|
| Calibration | Heads with MRL > 0.95 | 93.1% |
| Attention mass | Mass retained at 50% | 90.2% |
| Decode NLL | Top-1 agreement | 83.9% |
| Decode NLL | ΔNLL vs full attention | +0.28 |

## Usage

### Calibrate (Python)
```bash
HIP_VISIBLE_DEVICES=0 python3 triattention_calibrate.py \
  --model Qwen/Qwen3-8B \
  --input wikitext-2-raw/wiki.test.raw \
  --output stats/qwen3_8b.bin
```

### Run with llama.cpp
```bash
# Build with HIP
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1100
cmake --build . -j16

# Perplexity with TriAttention pruning
HIP_VISIBLE_DEVICES=0 ./bin/llama-perplexity \
  -m model.gguf -f wiki.test.raw \
  -c 4096 -ngl 99 --chunks 20 \
  --triattention stats/qwen3_8b_v2.bin \
  --tri-budget 50 --tri-window 128 --tri-interval 256
```

## TRIA v2 Binary Format

See [TRIA_FORMAT.md](TRIA_FORMAT.md) for byte-level specification.

## Related

- [TriAttention paper](https://arxiv.org/abs/2604.04921) (Song Han, Yukang Chen, Bohan Zhuang)
- [TriAttention code](https://github.com/WeianMao/triattention) (Python/vLLM)
- [TurboQuant HIP port](https://github.com/ggml-org/llama.cpp/discussions/21526) — complementary KV cache quantization
- [llama.cpp integration branch](https://github.com/domvox/llama.cpp-turboquant-hip/tree/feature/triattention-scoring)

## Status

**Working prototype with llama.cpp integration.** Physical KV compaction on GPU, validated on Qwen3-8B. Single-stream KV cache only. Scoring is CPU-side (GPU scoring kernel planned).

## License

Apache 2.0
