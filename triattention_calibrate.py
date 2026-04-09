#!/usr/bin/env python3
"""
TriAttention Calibration Tool — Phase 1

Computes per-head frequency statistics (Q/K centers, norms, MRL) from a
HuggingFace model.  Emits a compact binary stats file for use in ggml-based
scoring (Phase 2).

Supports:
  - Pure transformer models (Qwen3-8B, Llama, etc.)
  - Hybrid SSM+attention models (Qwen3.5-27B) — only calibrates full_attention layers
  - Multi-resolution RoPE (mrope) with partial_rotary_factor

Usage:
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
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from triattention_common import _to_complex


# ---------------------------------------------------------------------------
# Binary stats format — TRIA v2
# ---------------------------------------------------------------------------
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
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_layers))
        f.write(struct.pack("<I", num_heads))
        f.write(struct.pack("<I", num_kv_heads))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", freq_count))
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<f", attn_scale))
        f.write(b"\x00" * (HEADER_SIZE - 9 * 4))

        f.write(struct.pack(f"<{num_layers}f", *layer_budget_scales))

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
        f.read(HEADER_SIZE - 9 * 4)

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
# Config helpers for hybrid / nested configs
# ---------------------------------------------------------------------------

def _get_text_config(config):
    """Extract the text sub-config for multimodal wrappers (Qwen3.5, etc.)."""
    return getattr(config, "text_config", config)


def _get_attention_layer_indices(text_config) -> Set[int]:
    """Return set of layer indices that have full attention (KV cache).
    For pure transformers, returns all layers."""
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        n = getattr(text_config, "num_hidden_layers",
                     getattr(text_config, "n_layer", 0))
        return set(range(n))
    return {i for i, t in enumerate(layer_types) if t == "full_attention"}


def _get_rope_theta(text_config) -> float:
    """Extract rope_theta from potentially nested config."""
    # Direct attribute
    theta = getattr(text_config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    # Nested in rope_parameters (Qwen3.5)
    rp = getattr(text_config, "rope_parameters", None)
    if rp is not None:
        if isinstance(rp, dict):
            theta = rp.get("rope_theta")
        else:
            theta = getattr(rp, "rope_theta", None)
        if theta is not None:
            return float(theta)
    return 10000.0


def _get_partial_rotary_factor(text_config) -> float:
    """Get fraction of head_dim that is rotated. 1.0 for standard RoPE."""
    prf = getattr(text_config, "partial_rotary_factor", None)
    if prf is not None:
        return float(prf)
    rp = getattr(text_config, "rope_parameters", None)
    if rp is not None:
        if isinstance(rp, dict):
            prf = rp.get("partial_rotary_factor")
        else:
            prf = getattr(rp, "partial_rotary_factor", None)
        if prf is not None:
            return float(prf)
    return 1.0


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
    dtype = torch.bfloat16

    # --- Load model ---
    print(f"Loading {model_name} ...", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # --- Extract config (handle nested/multimodal) ---
    text_config = _get_text_config(config)
    num_layers = getattr(text_config, "num_hidden_layers",
                          getattr(text_config, "n_layer", 0))
    num_heads = getattr(text_config, "num_attention_heads", 32)
    head_dim = getattr(text_config, "head_dim",
                        getattr(text_config, "hidden_size", 4096) // num_heads)
    num_kv_heads = getattr(text_config, "num_key_value_heads", num_heads)
    rope_theta = _get_rope_theta(text_config)
    partial_rotary = _get_partial_rotary_factor(text_config)
    attn_layer_indices = _get_attention_layer_indices(text_config)

    # For TriAttention scoring we use the rotated portion of head_dim.
    # The stats file stores full head_dim so the runtime can index correctly,
    # but freq_count is based on the rotated dims only.
    rotary_dim = int(head_dim * partial_rotary)
    freq_count = rotary_dim // 2

    if rope_theta == 10000.0:
        print(f"  WARNING: rope_theta=10000 (default). Verify this matches your model!\n"
              f"  Qwen3 uses 1000000, Llama 3.1 uses 500000. Wrong theta = broken scoring at long context.",
              file=sys.stderr)

    is_hybrid = len(attn_layer_indices) < num_layers
    print(f"  layers={num_layers} (attention={len(attn_layer_indices)}"
          f"{' hybrid' if is_hybrid else ''}) heads={num_heads} kv_heads={num_kv_heads} "
          f"head_dim={head_dim} rotary_dim={rotary_dim} rope_theta={rope_theta}",
          file=sys.stderr)

    # --- Find model backbone ---
    backbone = getattr(model, "model", model)
    # Qwen3.5 multimodal: model.model is the VL wrapper, text model is deeper
    if hasattr(backbone, "text_model"):
        backbone = backbone.text_model
    elif hasattr(backbone, "language_model"):
        lm = backbone.language_model
        backbone = getattr(lm, "model", lm)

    layers = backbone.layers

    # --- Find rotary embedding ---
    # Try backbone-level first, then first attention layer
    rotary = getattr(backbone, "rotary_emb", None)
    if rotary is None:
        for li in attn_layer_indices:
            attn = layers[li].self_attn
            rotary = getattr(attn, "rotary_emb", None)
            if rotary is not None:
                break
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0)) if rotary else 1.0

    # --- Tokenize input ---
    print(f"Reading {input_path} ...", file=sys.stderr)
    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)
    seq_len = input_ids.shape[1]
    print(f"  tokens={seq_len}", file=sys.stderr)

    # --- Register hooks only on attention layers ---
    captured_q: Dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return
            bsz, qlen, _ = hidden.shape
            q_raw = module.q_proj(hidden)
            # Qwen3.5 gated attention: q_proj outputs head_dim*2, split into Q and gate
            proj_dim = q_raw.shape[-1] // num_heads
            if proj_dim == head_dim * 2:
                q_raw = q_raw.view(bsz, qlen, num_heads, head_dim * 2)
                q = q_raw[..., :head_dim].transpose(1, 2)
            else:
                q = q_raw.view(bsz, qlen, num_heads, head_dim).transpose(1, 2)
            if hasattr(module, 'q_norm'):
                q = module.q_norm(q)
            if partial_rotary < 1.0:
                q = q[..., :rotary_dim]
            captured_q[layer_idx] = q.detach().cpu()
        return hook_fn

    handles = []
    for li in attn_layer_indices:
        h = layers[li].self_attn.register_forward_pre_hook(
            _make_hook(li), with_kwargs=True
        )
        handles.append(h)

    print(f"  Hooked {len(handles)} attention layers, skipping {num_layers - len(handles)} SSM layers",
          file=sys.stderr)

    # --- Forward pass ---
    print("Forward pass ...", file=sys.stderr)
    t0 = time.time()
    with torch.no_grad():
        try:
            model(input_ids)
        except torch.cuda.OutOfMemoryError:
            # lm_head OOM is fine — we only need attention hook data
            print("  (lm_head OOM ignored, hooks collected)", file=sys.stderr)
    print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)

    for h in handles:
        h.remove()

    # --- Compute statistics ---
    print("Computing per-head frequency statistics ...", file=sys.stderr)

    all_stats: Dict[Tuple[int, int], dict] = {}
    high_mrl_count = 0
    total_heads = 0

    for layer_idx in attn_layer_indices:
        q_pre = captured_q.get(layer_idx)
        if q_pre is None:
            print(f"  [warn] layer {layer_idx}: no Q captured", file=sys.stderr)
            continue

        for head_idx in range(num_heads):
            q_head = q_pre[0, head_idx]  # [seq_len, rotary_dim]
            q_complex = _to_complex(q_head)  # [seq_len, freq_count]

            q_mean = q_complex.mean(dim=0)
            q_abs_mean = q_complex.abs().mean(dim=0)
            q_mean_abs = q_mean.abs()
            mrl = q_mean_abs / q_abs_mean.clamp_min(1e-12)

            all_stats[(layer_idx, head_idx)] = {
                "q_mean_real": q_mean.real.cpu().tolist(),
                "q_mean_imag": q_mean.imag.cpu().tolist(),
                "q_abs_mean": q_abs_mean.cpu().tolist(),
                "mrl": mrl.cpu().tolist(),
            }

            topk = torch.topk(q_abs_mean, k=min(2, freq_count))
            dominant_mrl = mrl[topk.indices].mean().item()
            total_heads += 1
            if dominant_mrl > 0.95:
                high_mrl_count += 1

        del captured_q[layer_idx]

    pct = 100.0 * high_mrl_count / max(total_heads, 1)
    print(f"  {total_heads} heads, {high_mrl_count} ({pct:.1f}%) with mean MRL > 0.95",
          file=sys.stderr)

    # --- Per-layer budget scales (only meaningful for attention layers) ---
    layer_dominant_mrls = []
    for li in range(num_layers):
        if li not in attn_layer_indices:
            layer_dominant_mrls.append(0.5)  # neutral for SSM layers
            continue
        mrls = []
        for hi in range(num_heads):
            s = all_stats.get((li, hi))
            if s is None:
                continue
            qam = torch.tensor(s["q_abs_mean"])
            mrl_t = torch.tensor(s["mrl"])
            topk = torch.topk(qam, k=min(2, freq_count))
            mrls.append(mrl_t[topk.indices].mean().item())
        layer_dominant_mrls.append(sum(mrls) / len(mrls) if mrls else 0.5)

    # Compute scales only from attention layers, then assign
    att_mrls = [layer_dominant_mrls[i] for i in sorted(attn_layer_indices)]
    alpha = 1.0
    att_raw = [1.0 + alpha * (1.0 - m) for m in att_mrls]
    att_mean = sum(att_raw) / len(att_raw)
    att_scales = [s / att_mean for s in att_raw]

    layer_budget_scales = [1.0] * num_layers  # SSM layers get 1.0 (neutral)
    for i, li in enumerate(sorted(attn_layer_indices)):
        layer_budget_scales[li] = att_scales[i]

    print(f"  Budget scales (attention only): min={min(att_scales):.3f} max={max(att_scales):.3f}",
          file=sys.stderr)

    # --- Write binary stats ---
    # Store full num_layers slots. SSM layers get zero-filled stats.
    # head_dim in header = rotary_dim (what the runtime uses for freq scoring)
    out = Path(output_path)
    _write_stats(out, num_layers, num_heads, num_kv_heads, rotary_dim,
                 rope_theta, attn_scale, all_stats, layer_budget_scales)

    # --- Verify round-trip ---
    readback = _read_stats(out)
    assert readback["num_layers"] == num_layers
    assert readback["num_heads"] == num_heads
    print("Round-trip verification OK", file=sys.stderr)

    # --- Summary ---
    print(f"\n=== Calibration Summary ===", file=sys.stderr)
    print(f"Model:      {model_name}", file=sys.stderr)
    print(f"Type:       {'hybrid (SSM+attention)' if is_hybrid else 'pure transformer'}", file=sys.stderr)
    print(f"Tokens:     {seq_len}", file=sys.stderr)
    print(f"Layers:     {num_layers} total, {len(attn_layer_indices)} attention", file=sys.stderr)
    if is_hybrid:
        print(f"Attention:  {sorted(attn_layer_indices)}", file=sys.stderr)
    print(f"Heads:      {total_heads} calibrated ({num_heads}H × {len(attn_layer_indices)}L)", file=sys.stderr)
    print(f"Head dim:   {head_dim} (rotary: {rotary_dim}, factor: {partial_rotary})", file=sys.stderr)
    print(f"RoPE theta: {rope_theta}", file=sys.stderr)
    print(f"High MRL:   {high_mrl_count}/{total_heads} ({pct:.1f}%)", file=sys.stderr)
    print(f"Budget:     min={min(att_scales):.3f} max={max(att_scales):.3f}", file=sys.stderr)
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
