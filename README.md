# TriAttention ggml Prototype

Standalone Python prototype of [TriAttention](https://arxiv.org/abs/2604.04921) KV cache pruning, targeting eventual [llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml integration.

TriAttention exploits the fact that pre-RoPE Q/K vectors concentrate around fixed centers in frequency space. This allows predicting which KV cache entries will be important for future queries using a trigonometric series — without computing actual attention. The paper reports **10.7× KV memory reduction** while matching Full Attention accuracy on AIME25.

## Results

Tested on **Qwen3-8B** / RX 7900 XTX (24GB) / ROCm 6.4. All at 50% cache retention.

| Phase | Metric | 512 ctx | 1024 ctx | 1536 ctx |
|---|---|---|---|---|
| 1 Calibration | Heads with MRL > 0.95 | 93.1% | — | — |
| 2c Mass | Attention mass retained | 90.2% | 90.6% | 90.5% |
| 2c Mass | Multi-query recall | 0.601 | 0.621 | 0.603 |
| 2d Decode | Top-1 agreement | 83.9% | — | — |
| 2d Decode | ΔNLL vs full attention | +0.28 | — | — |

Key findings:
- **90% attention mass retained** at 50% cache size — stable across context lengths
- Scoring math validated against paper equations 6-13 (3 independent LLM reviews)
- U-shaped per-layer pattern: early layers hardest to prune (67% mass), late layers easiest (98%)
- Critical bug found in naive validation: zeroing evicted K/V ≠ true eviction (must use -inf attention mask)

## Architecture

```
triattention_common.py      — Shared scoring, I/O, keep-set computation
triattention_calibrate.py   — Phase 1: capture pre-RoPE Q, compute MRL stats
triattention_score.py       — Phase 2: score keys, simulate pruning every 128 tokens
triattention_validate.py    — Phase 2b: single-query recall validation
triattention_validate_mass.py — Phase 2c: multi-query attention mass validation
triattention_validate_nll.py  — Phase 2d: decode NLL / top-1 / KL validation
```

## TRIA v2 Binary Format

Compact calibration stats file (~1.1MB for 8B model):
- 64-byte header: magic, version, model dims, RoPE theta
- Per-layer budget scales (derived from MRL — adaptive token allocation)
- Per-head stats: complex Q centers, expected norms, MRL per frequency band
- Backward compatible: v1 readers skip budget section

## Usage

```bash
# Phase 1: Calibrate
HIP_VISIBLE_DEVICES=0 python3 triattention_calibrate.py \
  --model Qwen/Qwen3-8B \
  --input wikitext-2-raw/wiki.test.raw \
  --output stats/qwen3_8b.bin

# Phase 2c: Validate attention mass retention
HIP_VISIBLE_DEVICES=0 python3 triattention_validate_mass.py \
  --model Qwen/Qwen3-8B \
  --stats stats/qwen3_8b.bin \
  --input wikitext-2-raw/wiki.test.raw \
  --max-length 1024 --budget 256
```

## Roadmap

This prototype validates the math. The C++ ggml port is next:

1. **TRIA loader** — mmap binary at model init
2. **Pre-RoPE K fork** — tap K_cur before ggml_rope node
3. **Scoring** — ggml compute graph for eq 6-13
4. **Indirection mask** — retained-index list per KV head (correctness first)
5. **Physical compaction** — block-level cache compaction (performance)
6. **TurboQuant combo** — eviction + quantization = ~50× KV reduction

## Related

- [TriAttention paper](https://arxiv.org/abs/2604.04921) (Song Han, Yukang Chen, Bohan Zhuang)
- [TriAttention code](https://github.com/WeianMao/triattention) (Python/vLLM)
- [TurboQuant HIP port](https://github.com/ggml-org/llama.cpp/discussions/21526) — complementary KV cache quantization

## Status

**Prototype / research code.** Not a llama.cpp port yet. Validated on Qwen3-8B only.

## License

Apache 2.0 (matching upstream TriAttention)
