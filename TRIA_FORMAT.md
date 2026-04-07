# TRIA Binary Format Reference (v1/v2)

This document specifies the on-disk binary format written by `triattention_calibrate.py` and read by `triattention_common.py`.

- **Magic**: ASCII `TRIA` as little-endian `u32` (`0x54524941`)
- **Endianness**: **little-endian** for all integer and float fields
- **Current writer version**: **v2**
- **Backward compatibility**: loader accepts **v1** and **v2**

---

## 1) File layout overview

```text
+---------------------------+
| Header (fixed 64 bytes)   |
+---------------------------+
| v2+ layer scales block    |  (num_layers * 4 bytes)
+---------------------------+
| Per-head stats block      |  (num_layers * num_heads entries)
+---------------------------+
```

For **v1**, the layer-scales block is absent and readers should assume scale `1.0` for each layer.

---

## 2) Fixed header (64 bytes)

| Byte offset | Size | Type | Field | Description |
|---:|---:|---|---|---|
| 0  | 4 | `u32` | `magic` | Must be `0x54524941` (`TRIA`) |
| 4  | 4 | `u32` | `version` | `1` or `2` |
| 8  | 4 | `u32` | `num_layers` | Transformer layer count |
| 12 | 4 | `u32` | `num_heads` | Attention query-head count |
| 16 | 4 | `u32` | `num_kv_heads` | KV-head count |
| 20 | 4 | `u32` | `head_dim` | Per-head hidden dimension |
| 24 | 4 | `u32` | `freq_count` | `head_dim / 2` |
| 28 | 4 | `f32` | `rope_theta` | RoPE base theta |
| 32 | 4 | `f32` | `attn_scale` | Rotary/attention scaling metadata |
| 36 | 28 | `u32[7]` / bytes | `reserved` | Must be ignored by readers |

> Header size is always 64 bytes; future versions should preserve this for easier C/C++ parsing.

---

## 3) v2+ layer budget scales block

Present only when `version >= 2`.

| Byte offset (relative) | Size | Type | Field |
|---:|---:|---|---|
| 0 | `num_layers * 4` | `f32[num_layers]` | `layer_budget_scales` |

Reader behavior:
- **v2+**: read exact `num_layers` floats.
- **v1**: do not read; synthesize `[1.0] * num_layers`.

---

## 4) Per-head stats block

Entries are written in **layer-major** order:

```text
for layer in [0..num_layers-1]:
  for head in [0..num_heads-1]:
    write HeadStats
```

Each `HeadStats` entry has 4 contiguous float arrays of length `freq_count`:

| Entry-relative offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | `freq_count * 4` | `f32[freq_count]` | `q_mean_real` |
| `freq_count*4` | `freq_count * 4` | `f32[freq_count]` | `q_mean_imag` |
| `freq_count*8` | `freq_count * 4` | `f32[freq_count]` | `q_abs_mean` |
| `freq_count*12` | `freq_count * 4` | `f32[freq_count]` | `mrl` |

So each head entry size is:

```text
head_entry_bytes = 4 arrays * freq_count * 4 bytes = freq_count * 16
```

Total stats block size:

```text
total_head_stats_bytes = num_layers * num_heads * freq_count * 16
```

---

## 5) Total file size formulas

Let:
- `H = 64` (header)
- `L = num_layers`
- `N = num_heads`
- `F = freq_count`

Then:
- **v1** size: `H + L * N * F * 16`
- **v2** size: `H + L*4 + L * N * F * 16`

---

## 6) Version handling guidance for C++ readers

1. Read first 8 bytes: `magic`, `version`.
2. Validate `magic == 0x54524941`.
3. Accept known versions (`1`, `2`), reject others.
4. Read remaining fixed header fields.
5. Seek/skip to byte 64 regardless of any struct packing assumptions.
6. If `version >= 2`, read `layer_budget_scales[num_layers]`; else initialize to `1.0`.
7. Read per-head arrays exactly in layer-major/head-major order.
8. Avoid assuming nonzero values for missing heads; writer may zero-fill.

---

## 7) Robustness notes

- All floating-point payloads are IEEE-754 single precision (`f32`).
- No checksums are currently included.
- `reserved` bytes are for forward-compatible metadata; ignore on read, write zeros on emit.
- `attn_scale` is metadata for calibration/runtime parity; it is not currently used by all consumers.
