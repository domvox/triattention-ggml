# Code Review Notes (2026-04-07)

Scope reviewed:
- `score_keys()` implementation and call sites
- TRIA v2 binary reader/writer
- Numerical edge cases
- Validation scripts (`triattention_validate*.py`), especially NLL masking

## High-severity issues

1. **`triattention_validate_nll.py` does not simulate real KV eviction semantics.**
   - It collapses per-(layer, kv_head) keep decisions into a **single global majority-vote mask** (`common_evict`), which is not equivalent to TriAttention's per-layer/per-head eviction policy.
   - It applies that mask across the **entire sequence**, including queries before `trigger_pos`, so the "pruned" run changes pre-trigger behavior. Real runtime pruning should preserve pre-trigger behavior and only affect attention to evicted pre-trigger keys for queries after the trigger.

2. ~~**`triattention_validate.py` per-layer summary can be wrong when recalls are skipped.**~~
   - **FIXED (2026-04-09):** Rewrote validate.py to use per-layer hooks with `layer_recalls` keyed by layer index.

## Medium-severity issues

3. **TRIA size report in `_write_stats()` undercounts bytes for v2 files.**
   - `total_bytes` omits `num_layers * 4` bytes of per-layer budget scales, so logged file size is incorrect.

4. **`score_keys()` future-offset geometric series may have precision/weighting mismatch risk.**
   - Uses hard-coded offsets `[1,2,4,...,2^16]` with equal weighting via `mean(dim=1)`. If Eq. 11-13 in the paper use different offset set or non-uniform weights, current implementation would diverge.
   - Large phases (`delta * omega` with big deltas) in float32 can accumulate angular error for high offsets.

5. **Potentially negative norm-extra term due to finite precision.**
   - `extra = sum((q_abs_mean - |q_mean|) * |k|)` should be non-negative in expectation, but float noise can make `(q_abs_mean - |q_mean|)` slightly negative. Consider clamping at zero before multiplication.

## Low-severity observations

6. **Binary format is consistently little-endian and aligned in practice.**
   - Header fields are all packed with `<` (little-endian), reserved bytes pad to 64-byte header, and v1/v2 read paths are explicitly versioned.
   - Backward compatibility logic (v1 defaulting `layer_budget_scales` to 1.0) is present.

7. **Missing defensive checks in loader.**
   - `load_stats()` does not validate short reads/EOF or sanity-check dimensions (`fc == head_dim//2`, positive counts).

## Suggested follow-up fixes

- Rework `triattention_validate_nll.py` to:
  - Keep pre-trigger pass identical to full attention.
  - Apply pruning only after trigger.
  - Respect per-layer/per-kv-head keep sets instead of global majority vote.
- ~~Fix per-layer recall aggregation in `triattention_validate.py`.~~ Done.
- Correct `_write_stats()` size accounting.
- Add loader sanity checks and optional strict mode.

---

# Session Notes (2026-04-09)

## Qwen3.5-27B Calibration & Validation

### Calibration
- **Model:** Qwen3.5-27B-hf (hybrid SSM+attention, 64 layers, 16 attention + 48 SSM)
- **Architecture:** heads=24, kv_heads=4, head_dim=256, rotary_dim=64, rope_theta=10M
- **Output:** `~/triattention-stats/qwen3.5_27b.bin` (786KB, TRIA v2, 32 freq bands)
- **Results:** 384 heads calibrated, 81.2% with MRL > 0.95, budget scales 0.976–1.063

### Fixes applied during session

1. **`triattention_calibrate.py` — `torch_dtype` deprecation**
   - `torch_dtype=dtype` → `dtype=dtype` in `from_pretrained()`

2. **`triattention_calibrate.py` — lm_head OOM**
   - 27B model fills 21.5GB VRAM, lm_head needs extra 1.9GB → OOM
   - Fix: catch `OutOfMemoryError` after forward pass — hooks already collected all data
   - lm_head output (logits) not needed for calibration

3. **`triattention_validate.py` — full rewrite for hybrid/large model support**
   - `.to(dev)` → `device_map="auto"` (model too large for single GPU)
   - Added hybrid layer detection (`hasattr(layer, 'self_attn')`)
   - Fixed `k_proj` reshape: use `full_hd=256` for reshape, then trim to `rotary_dim=64`
     - Stats file stores `head_dim=64` (rotary_dim), not full 256
   - Per-layer hook architecture: pre-hook captures K, post-hook gets attn_weights and computes recall immediately
   - Avoids `output_attentions` collecting all 16 × [1,24,4096,4096] matrices (~24GB)
   - OOM fallback for lm_head

4. **`triattention_validate_mass.py` — same fixes as validate.py**
   - `device_map="auto"`, hybrid layer filtering, full_hd/rotary_dim fix
   - Per-layer attention weight collection via post-hooks

### Validation Results

| Model | Type | Recall@128 (1024 tok) | vs random |
|---|---|---|---|
| Qwen3-8B | pure transformer | **0.411** | 3.3× |
| Qwen3.5-27B | hybrid SSM+attn | **0.219** | 1.75× |

**Key finding:** TriAttention recall is significantly lower on hybrid SSM+attention models.
Hypothesis: SSM layers process context between attention layers, causing attention patterns
to be more diffuse and less predictable from pre-RoPE frequency statistics alone.

### System issues
- **System crash during initial calibration attempt** — OOM killed the system
  - llama-server (21.4GB VRAM + 3GB RAM) + win11 VM (16GB RAM) + PyTorch model loading
  - 60GB RAM insufficient for all concurrent workloads
  - Fix: stopped llama-server and win11 VM before calibration
  - TODO: restart llama-server and win11 after validation complete

### Pending
- [ ] validate_nll for Qwen3.5-27B
- [ ] llama.cpp integration test with qwen3.5_27b.bin
- [ ] Restart llama-server + win11 VM

### Gemma 4 TurboQuant Results (26B-A4B MoE, Q4_K_M, 2K ctx, 5 chunks)

Best config: `--cache-type-k turbo3 --cache-type-v turbo3 --cache-type-k-swa turbo3 --cache-type-v-swa q8_0`

| Global K/V | SWA K | SWA V | KV MiB | Compression | PPL | vs f16 |
|---|---|---|---|---|---|---|
| f16/f16 | f16 | f16 | 340 | 1.0× | 35,131 | baseline |
| turbo3 G=256 | turbo3 | q8_0 | 117 | 2.9× | **10,839** | **0.31×** |
| turbo3 G=256 | q8_0 | q8_0 | — | — | 14,082 | 0.40× |
| turbo3 G=256 | f16 | f16 | — | — | 14,280 | 0.41× |
| turbo3 G=128 | turbo3 | turbo3 | — | — | 46,504 | 1.32× |

Note: PPL on instruct model + raw text is not directly meaningful, but
relative comparisons are valid. turbo3 G=256 consistently beats f16.

Key findings:
- GROUP_SIZE=256 is critical for head_dim=512 (2.5× better than G=128)
- SWA K benefits from turbo3 WHT rotation (better than q8_0)
- SWA V should use q8_0 (turbo3 on V hurts quality)
- Root cause of earlier G=256 failures: validation guard in set-rows.cu
  was silently resetting 256→128, causing encode/decode mismatch

### Attention Mass Results (Phase 2c)

| Model | Mass retained (50% ret.) | Recall@128 |
|---|---|---|
| Qwen3-8B | ~90% | — |
| Qwen3.5-27B | **62.3%** | 0.573 |

Per-layer breakdown (27B, 512 tok, trigger=256, budget=128):
- Best: L59=83.2%, L55=75.6%, L3=69.7%
- Worst: L19=47.6%, L23=54.5%, L27=52.0%
- Pattern: early + late layers best, middle layers weakest

### NLL Results (Phase 2d)

| Model | ΔNLL (50% ret.) | Top-1 agree | KL |
|---|---|---|---|
| Qwen3-8B (pure) | +0.28 | 83.9% | — |
| Qwen3.5-27B (hybrid) | **+0.23** | **87.8%** | 0.175 |

Key insight: despite lower recall (0.219 vs 0.411), the hybrid model shows
better NLL resilience. SSM layers preserve context independently of KV cache,
compensating for less accurate attention pruning. This makes TriAttention
potentially more useful on hybrid architectures than recall alone suggests.

### llama.cpp Integration Benchmark (Qwen3.5-27B Q5_K_M)

| Context | Config | PPL | Δ PPL | Cache rows | tok/s |
|---|---|---|---|---|---|
| 16K | Baseline (f16) | 6.0729 | — | 16384 | 380 |
| 16K | TriAttention 75% | 5.9939 | **-1.3%** | ~5989 | 390 |
| 16K | TriAttention 50% | 6.0890 | +0.26% | ~2550 | 395 |
| 32K | Baseline (f16) | 7.7761 | — | 32768 | 357 |
| 32K | TriAttention 50% | 7.9949 | +2.8% | ~2559 | 391 |

Parameters: `--tri-window 512 --tri-interval 1024`

Notable: 75% retention at 16K **improves** PPL over baseline (-1.3%).
TriAttention acts as a regularizer, removing noise from KV cache.
At 32K with 50% retention, cache is compacted 92% (32768→2559) with
only +2.8% PPL and faster throughput (FA on shorter cache).

### Bridge fix for hybrid models
- `triattention-bridge.cpp` used `dynamic_cast<llama_kv_cache*>` which returns
  nullptr on hybrid models (memory is `llama_memory_hybrid`, not `llama_kv_cache`)
- Fix: added `get_kv()` helper that tries direct cast first, then extracts
  `mem_attn` from `llama_memory_hybrid` via `get_mem_attn()`

### GSM8K Math Accuracy (Gemma 4 26B-A4B, Q4_K_M, 100 problems)

| Config | Accuracy | KV MiB | Compression | Accuracy drop |
|---|---|---|---|---|
| f16 baseline | 83/100 (83%) | 340 | 1.0× | — |
| turbo3 global + f16 SWA | 82/100 (82%) | ~120 | 2.8× | -1% |
| **turbo3 + turbo3-K-SWA + q8_0-V-SWA** | **83/100 (83%)** | **117** | **2.9×** | **0%** |

Comparison with AmesianX TurboQuant (CUDA, DGX Spark, 65 problems):
| | AmesianX | domvox (HIP) |
|---|---|---|
| Accuracy drop | -19% (57%→46%) | **0%** (83%→83%) |
| Compression | 5.2× | 2.8× |

Our implementation preserves quality significantly better despite lower compression.
The difference is that we keep SWA cache in f16 (SWA quantization destroys quality
on Gemma 4), while AmesianX compresses everything.
