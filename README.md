# TriAttention — KV Cache Pruning for llama.cpp

Decode-only KV cache eviction with GPU scoring kernel. Scores KV rows by trigonometric frequency prediction, evicts lowest-scoring rows during decode, physically compacts the cache on GPU.

Based on [TriAttention (Mao et al., 2026)](https://arxiv.org/abs/2604.04921) — independent C/HIP implementation for llama.cpp.

## Decode Speedup (Qwen3-8B Q4_K_M, KV q8_0, v2 12% budget)

| Depth | Dense t/s | TriAttention t/s | Speedup |
|-------|-----------|------------------|---------|
| 8k | 81.38 | 90.74 | **+11.5%** |
| 16k | 70.56 | 89.64 | **+27.0%** |
| 32k | 55.60 | 88.51 | **+59.2%** |
| 65k | 39.04 | 86.45 | **+121%** |

Measured with `llama-bench -n 1024 -r 2 -fa 1`, RX 7900 XTX 24GB, ROCm 7.2.1.

## Quality (MATH500, Qwen3-8B Q4_K_M, budget=512 tokens)

| Config | Accuracy | Δ |
|--------|----------|---|
| Dense | 35/50 (70.0%) | baseline |
| TriAttention budget=512 | 29/50 (58.0%) | -12.0pp |
| Paper (Qwen3-8B bf16) | 69.6% → 56.0% | -13.6pp |

Our degradation (-12.0pp) matches the paper (-13.6pp). Eval: `llama-server` + MATH500, temp=0.6, top_p=0.95.

## How it works

1. **Calibration** (offline): Collect pre-RoPE Q/K statistics per head → binary stats file
2. **Scoring** (runtime, GPU): Every 128 decode tokens, score all KV rows using trigonometric series from Q/K centers
3. **Eviction**: Remove lowest-scoring rows, compact KV cache on GPU
4. **Exponential ramp**: Gradual eviction (10→20→40→80→100%) prevents semantic shock

Key design: **decode-only** — prefill runs dense, eviction only during token generation. This matches the paper's approach and preserves prefill quality.

## Usage

Integrated into [TurboQuant HIP fork](https://github.com/domvox/llama.cpp-turboquant-hip) (branch: `feature/triattention-scoring`).

```bash
# Via environment variables:
TRIA_STATS_PATH=stats/qwen3_8b_v4.bin \
TRIA_BUDGET_TOKENS=512 \
llama-server -m model.gguf -ngl 99 -np 1 \
  -ctk q8_0 -ctv q8_0 -fa on -c 32768

# Or via Docker Compose:
docker compose up
```

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRIA_STATS_PATH` | Path to calibration stats binary | — (disabled) |
| `TRIA_BUDGET_TOKENS` | Absolute KV budget in tokens | — (uses percentage) |
| `TRIA_BUDGET_PCT` | KV budget as % of context | 50 |
| `TRIA_RAMP_START_PCT` | Exponential ramp start | 10 |

## Version history

| Version | Speedup @65k | Notes |
|---------|-------------|-------|
| v1 (prefill+decode) | +143% | Repetition, quality issues |
| **v2 (decode-only, GPU kernel)** | **+121%** | **Production version** |
| v3 (full GQA 32 heads) | -22% | Kernel too slow |
| v2.5 (corrected RoPE) | +10% | Overhead not worth it, rejected |

## Comparison with paper

| | Paper (NVIDIA) | This repo |
|---|---|---|
| Implementation | Python (vLLM/Triton) | C + HIP kernel |
| Runtime | vLLM, HuggingFace | llama.cpp (ggml) |
| Scoring | Full eq 6-13 | Simplified v2 (KV-head level) |
| GPU | CUDA (A100/H100) | HIP/ROCm (RDNA2/3/4) |
| TurboQuant combo | No | Yes |
| MATH500 quality | -13.6pp @budget=512 | -12.0pp @budget=512 |

## Confirmed hardware

| GPU | Architecture | Status |
|-----|-------------|--------|
| RX 7900 XTX | gfx1100 (RDNA3) | ✅ Primary test platform |
| RX 6700 XT | gfx1030 (RDNA2) | ✅ Community tested |
| RX 9060 XT | gfx1200 (RDNA4) | ✅ Community tested |
| Strix Halo | gfx1151 | ✅ Community tested |

## Known limitations

- `tria_get_n_kv` uses max-over-sequences — correct for `-np 1`, approximate for multi-slot server
- Scoring doesn't trigger during prefill (by design)
- `llama-perplexity` cannot measure v2 quality (batch prefill only)
- Quality eval requires task-based benchmarks (MATH500, AIME25)

## Calibration

```bash
cd triattention-ggml
python3 triattention_calibrate.py \
  --model Qwen/Qwen3-8B \
  --output stats/qwen3_8b_v4.bin
```

Pre-computed stats: `~/triattention-stats/`

## Contributing

PRs welcome. Priority areas:
- Stats for more models (Llama 3, Mistral, Gemma 4)
- CUDA GPU scoring kernel
- Per-sequence tracking for multi-slot server
- Longer context quality benchmarks (AIME25, 32k decode)
