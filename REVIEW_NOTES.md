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

2. **`triattention_validate.py` per-layer summary can be wrong when recalls are skipped.**
   - `recalls` is appended only for valid pairs, but breakdown slices `recalls[idx:idx+nkv]` assuming dense entries. Missing pairs shift indexing and can misattribute recalls to wrong layers.

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
- Fix per-layer recall aggregation in `triattention_validate.py` by storing recalls keyed by `(layer, kv_head)`.
- Correct `_write_stats()` size accounting.
- Add loader sanity checks and optional strict mode.
