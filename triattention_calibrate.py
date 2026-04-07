#!/usr/bin/env python3
"""
TriAttention Calibration Tool — Phase 1

Computes per-head frequency statistics (Q/K centers, norms, MRL) from a
HuggingFace model.  Emits a compact binary stats file for use in ggml-based
scoring (Phase 2).

Based on the reference implementation at
https://github.com/WeianMao/triattention (Apache 2.0).

Usage:
    # Stop Hermes first to free VRAM:
    #   sudo systemctl stop llama-server.service
    HIP_VISIBLE_DEVICES=0 python3 triattention_calibrate.py \
        --model Qwen/Qwen3-8B \
        --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
        --output ~/triattention-stats/qwen3_8b.bin \
        --max-length 4096 \
        --device cuda
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# RoPE helpers (from reference, simplified)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard half-rotation for RoPE inversion (Qwen/Llama style)."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Invert RoPE rotation to recover pre-RoPE vectors."""
    s = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / s
    cos_u = cos / s
    sin_u = sin / s
    return base * cos_u - _rotate_half(base) * sin_u


def _to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [*, head_dim] to [*, head_dim//2] complex pairs (half style)."""
    t = tensor.to(dtype=torch.float32)
    fc = t.shape[-1] // 2
    return torch.complex(t[..., :fc].contiguous(), t[..., fc:].contiguous())


# ---------------------------------------------------------------------------
# Binary stats format — TRIA v2
# ---------------------------------------------------------------------------
# Header (64 bytes):
#   magic       u32  = 0x54524941  ("TRIA")
#   version     u32  = 2
#   num_layers  u32
#   num_heads   u32  (attention heads, not KV heads)
#   num_kv_heads u32
#   head_dim    u32
#   freq_count  u32  (= head_dim // 2)
#   rope_theta  f32
#   attn_scale  f32
#   reserved    u32[7]  (pad to 64 bytes)
#
# Per-layer budget scales (v2+):
#   layer_budget_scale  f32[num_layers]
#   (1.0 = paper-faithful global budget; >1.0 = more tokens for this layer)
#   Derived from MRL: low avg dominant MRL → higher scale
#
# Per head (num_layers * num_heads entries, layer-major order):
#   q_mean_real  f32[freq_count]
#   q_mean_imag  f32[freq_count]
#   q_abs_mean   f32[freq_count]
#   mrl          f32[freq_count]   (Mean Resultant Length per band)

MAGIC = 0x54524941  # "TRIA"
VERSION = 2
HEADER_SIZE = 64  # bytes


def _write_stats(
    path: Path,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    attn_scale: float,
    stats: Dict[Tuple[int, int], dict],
    layer_budget_scales: list[float] | None = None,
) -> None:
    """Write calibration stats in compact binary format (TRIA v2)."""
    freq_count = head_dim // 2
    path.parent.mkdir(parents=True, exist_ok=True)
    if layer_budget_scales is None:
        layer_budget_scales = [1.0] * num_layers

    with open(path, "wb") as f:
        # Header (64 bytes)
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_layers))
        f.write(struct.pack("<I", num_heads))
        f.write(struct.pack("<I", num_kv_heads))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", freq_count))
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<f", attn_scale))
        f.write(b"\x00" * (HEADER_SIZE - 9 * 4))  # reserved

        # Per-layer budget scales (v2)
        f.write(struct.pack(f"<{num_layers}f", *layer_budget_scales))

        # Per-head stats
        written = 0
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = (layer_idx, head_idx)
                if key in stats:
                    s = stats[key]
                    f.write(struct.pack(f"<{freq_count}f", *s["q_mean_real"]))
                    f.write(struct.pack(f"<{freq_count}f", *s["q_mean_imag"]))
                    f.write(struct.pack(f"<{freq_count}f", *s["q_abs_mean"]))
                    f.write(struct.pack(f"<{freq_count}f", *s["mrl"]))
                else:
                    # Zero-fill missing heads
                    f.write(b"\x00" * (freq_count * 4 * 4))
                written += 1

    total_bytes = HEADER_SIZE + num_layers * 4 + written * freq_count * 4 * 4
    print(f"Wrote {path} ({total_bytes} bytes, {written} heads)", file=sys.stderr)


def _read_stats(path: Path) -> dict:
    """Read back stats file for verification. Handles v1 and v2."""
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        assert magic == MAGIC, f"Bad magic: {magic:#x}"
        assert version in (1, 2), f"Bad version: {version}"
        nl, nh, nkv, hd, fc = struct.unpack("<5I", f.read(20))
        rope_theta, attn_scale = struct.unpack("<2f", f.read(8))
        f.read(HEADER_SIZE - 9 * 4)  # skip reserved

        # v2: per-layer budget scales
        if version >= 2:
            layer_budget_scales = list(struct.unpack(f"<{nl}f", f.read(nl * 4)))
        else:
            layer_budget_scales = [1.0] * nl

        stats = {}
        for li in range(nl):
            for hi in range(nh):
                qmr = struct.unpack(f"<{fc}f", f.read(fc * 4))
                qmi = struct.unpack(f"<{fc}f", f.read(fc * 4))
                qam = struct.unpack(f"<{fc}f", f.read(fc * 4))
                mrl = struct.unpack(f"<{fc}f", f.read(fc * 4))
                stats[(li, hi)] = {
                    "q_mean_real": list(qmr),
                    "q_mean_imag": list(qmi),
                    "q_abs_mean": list(qam),
                    "mrl": list(mrl),
                }

    return {
        "num_layers": nl, "num_heads": nh, "num_kv_heads": nkv,
        "head_dim": hd, "freq_count": fc,
        "rope_theta": rope_theta, "attn_scale": attn_scale,
        "layer_budget_scales": layer_budget_scales,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    model_name: str,
    input_path: str,
    output_path: str,
    max_length: int = 4096,
    device: str = "cuda",
) -> None:
    dev = torch.device(device)
    dtype = torch.bfloat16

    # --- Load model ---
    print(f"Loading {model_name} ...", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype,
        attn_implementation="sdpa",  # memory-efficient attention
        trust_remote_code=True,
    ).to(dev)
    model.eval()

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    freq_count = head_dim // 2

    print(f"  layers={num_layers} heads={num_heads} kv_heads={num_kv_heads} "
          f"head_dim={head_dim} rope_theta={rope_theta}", file=sys.stderr)

    # --- Find rotary embedding ---
    backbone = getattr(model, "model", model)
    layers = backbone.layers
    attn0 = layers[0].self_attn
    rotary = getattr(backbone, "rotary_emb", None) or attn0.rotary_emb
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # --- Tokenize input ---
    print(f"Reading {input_path} ...", file=sys.stderr)
    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(dev)
    seq_len = input_ids.shape[1]
    print(f"  tokens={seq_len}", file=sys.stderr)

    # --- Register hooks to capture pre-RoPE Q ---
    captured_q: Dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return
            bsz, qlen, _ = hidden.shape
            q = module.q_proj(hidden)
            q = q.view(bsz, qlen, num_heads, head_dim).transpose(1, 2)
            if hasattr(module, 'q_norm'):
                q = module.q_norm(q)
            # This IS the pre-RoPE Q — save directly, no RoPE needed
            captured_q[layer_idx] = q.detach()
        return hook_fn

    handles = []
    for li, layer in enumerate(layers):
        h = layer.self_attn.register_forward_pre_hook(
            _make_hook(li), with_kwargs=True
        )
        handles.append(h)

    # --- Forward pass ---
    print("Forward pass ...", file=sys.stderr)
    t0 = time.time()
    with torch.no_grad():
        model(input_ids)
    print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)

    for h in handles:
        h.remove()

    # --- Compute statistics ---
    print("Computing per-head frequency statistics ...", file=sys.stderr)

    all_stats: Dict[Tuple[int, int], dict] = {}
    high_mrl_count = 0
    total_heads = 0

    for layer_idx in range(num_layers):
        q_pre = captured_q.get(layer_idx)
        if q_pre is None:
            print(f"  [warn] layer {layer_idx}: no Q captured", file=sys.stderr)
            continue

        for head_idx in range(num_heads):
            q_head = q_pre[0, head_idx]  # [seq_len, head_dim]
            q_complex = _to_complex(q_head)  # [seq_len, freq_count]

            # E[q_f] — complex mean
            q_mean = q_complex.mean(dim=0)  # [freq_count]
            # E[||q_f||] — mean of absolute values
            q_abs_mean = q_complex.abs().mean(dim=0)  # [freq_count]
            # MRL = ||E[q_f]|| / E[||q_f||]
            q_mean_abs = q_mean.abs()
            mrl = q_mean_abs / q_abs_mean.clamp_min(1e-12)

            all_stats[(layer_idx, head_idx)] = {
                "q_mean_real": q_mean.real.cpu().tolist(),
                "q_mean_imag": q_mean.imag.cpu().tolist(),
                "q_abs_mean": q_abs_mean.cpu().tolist(),
                "mrl": mrl.cpu().tolist(),
            }

            # Check dominant bands (top-2 by contribution = q_abs_mean)
            topk = torch.topk(q_abs_mean, k=min(2, freq_count))
            dominant_mrl = mrl[topk.indices].mean().item()

            total_heads += 1
            if dominant_mrl > 0.95:
                high_mrl_count += 1

        # Free memory
        del captured_q[layer_idx]

    pct = 100.0 * high_mrl_count / max(total_heads, 1)
    print(f"  {total_heads} heads, {high_mrl_count} ({pct:.1f}%) with mean MRL > 0.95",
          file=sys.stderr)

    # --- Compute per-layer budget scales from MRL ---
    # Low avg dominant MRL → layer needs more tokens → higher scale
    # Scale = 1 + alpha * (1 - avg_dominant_mrl), normalized so mean = 1.0
    layer_dominant_mrls = []
    for li in range(num_layers):
        mrls = []
        for hi in range(num_heads):
            s = all_stats.get((li, hi))
            if s is None: continue
            qam = torch.tensor(s["q_abs_mean"])
            mrl_t = torch.tensor(s["mrl"])
            topk = torch.topk(qam, k=min(2, freq_count))
            mrls.append(mrl_t[topk.indices].mean().item())
        layer_dominant_mrls.append(sum(mrls) / len(mrls) if mrls else 0.5)

    alpha = 1.0  # sensitivity: 0=global, 1=moderate, 2=aggressive
    raw_scales = [1.0 + alpha * (1.0 - m) for m in layer_dominant_mrls]
    mean_scale = sum(raw_scales) / len(raw_scales)
    layer_budget_scales = [s / mean_scale for s in raw_scales]  # normalize mean=1.0

    print(f"  Budget scales: min={min(layer_budget_scales):.3f} max={max(layer_budget_scales):.3f}",
          file=sys.stderr)

    # --- Write binary stats ---
    out = Path(output_path)
    _write_stats(out, num_layers, num_heads, num_kv_heads, head_dim,
                 rope_theta, attn_scale, all_stats, layer_budget_scales)

    # --- Verify round-trip ---
    readback = _read_stats(out)
    assert readback["num_layers"] == num_layers
    assert readback["num_heads"] == num_heads
    assert len(readback["stats"]) == total_heads
    assert len(readback["layer_budget_scales"]) == num_layers
    print("Round-trip verification OK", file=sys.stderr)

    # --- Summary ---
    print(f"\n=== Calibration Summary ===", file=sys.stderr)
    print(f"Model:      {model_name}", file=sys.stderr)
    print(f"Tokens:     {seq_len}", file=sys.stderr)
    print(f"Heads:      {total_heads} ({num_layers}L × {num_heads}H)", file=sys.stderr)
    print(f"High MRL:   {high_mrl_count}/{total_heads} ({pct:.1f}%)", file=sys.stderr)
    print(f"Budget:     min={min(layer_budget_scales):.3f} max={max(layer_budget_scales):.3f}", file=sys.stderr)
    print(f"Output:     {out} ({out.stat().st_size} bytes)", file=sys.stderr)
    print(f"Format:     TRIA v{VERSION}, {freq_count} freq bands per head", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TriAttention calibration — compute Q/K frequency statistics"
    )
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--input", required=True, help="Plain text file for calibration")
    parser.add_argument("--output", required=True, help="Output binary stats file")
    parser.add_argument("--max-length", type=int, default=4096, help="Max tokens")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    calibrate(args.model, args.input, args.output, args.max_length, args.device)


if __name__ == "__main__":
    main()
