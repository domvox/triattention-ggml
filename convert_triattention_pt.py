#!/usr/bin/env python3
"""Convert official TriAttention .pt calibration files to TRIA v2 binary.

Usage:
    python convert_triattention_pt.py <input.pt> <output.bin> [--model-meta <model_id>]

Input:  WeianMao/triattention .pt format (PyTorch pickle)
Output: TRIA v2 binary (triattention-ggml native format)
"""
from __future__ import annotations
import argparse, re, struct, sys
from pathlib import Path
import torch

MAGIC = 0x54524941  # "TRIA"
VERSION = 2
HEADER_SIZE = 64


def parse_key(key: str):
    m = re.match(r"layer(\d+)_head(\d+)", key)
    if not m:
        raise ValueError(f"Unexpected key format: {key}")
    return int(m.group(1)), int(m.group(2))


def convert(pt_path: str, out_path: str, rope_theta: float = 0.0):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    meta = data["metadata"]
    stats = data["stats"]

    head_dim = meta["head_dim"]
    freq_count = head_dim // 2

    # Discover layers/heads from keys
    layers, heads = set(), set()
    for k in stats:
        li, hi = parse_key(k)
        layers.add(li)
        heads.add(hi)
    num_layers = max(layers) + 1
    num_heads = max(heads) + 1

    # num_kv_heads not in WeianMao format — try to infer from model config
    num_kv_heads = num_heads  # default: no GQA
    model_name = meta.get("model_name", "")
    if model_name:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            tcfg = getattr(cfg, "text_config", cfg)
            num_kv_heads = getattr(tcfg, "num_key_value_heads", num_heads)
            print(f"Inferred num_kv_heads={num_kv_heads} from {model_name}")
        except Exception:
            pass

    # Override rope_theta from meta if available
    rt = rope_theta or 0.0

    print(f"Converting: {num_layers} layers, {num_heads} heads, head_dim={head_dim}, freq_count={freq_count}")

    with open(out_path, "wb") as f:
        # Header (64 bytes)
        hdr = struct.pack("<II", MAGIC, VERSION)
        hdr += struct.pack("<5I", num_layers, num_heads, num_kv_heads, head_dim, freq_count)
        hdr += struct.pack("<2f", rt, 0.0)  # rope_theta, reserved
        hdr += b"\x00" * (HEADER_SIZE - len(hdr))
        f.write(hdr)

        # Per-layer budget scales (all 1.0 — WeianMao doesn't store budgets)
        f.write(struct.pack(f"<{num_layers}f", *([1.0] * num_layers)))

        # Per-head stats: q_mean_real, q_mean_imag, q_abs_mean, mrl
        for li in range(num_layers):
            for hi in range(num_heads):
                key = f"layer{li:02d}_head{hi:02d}"
                s = stats.get(key)
                if s is None:
                    # Zero-fill missing heads
                    f.write(b"\x00" * (freq_count * 4 * 4))
                    continue
                qmr = s["q_mean_real"].float().contiguous()
                qmi = s["q_mean_imag"].float().contiguous()
                qam = s["q_abs_mean"].float().contiguous()
                assert qmr.shape[0] == freq_count, f"{key}: expected {freq_count}, got {qmr.shape[0]}"
                # Compute MRL = q_abs_mean - |complex(q_mean_real, q_mean_imag)|
                mrl = qam - torch.abs(torch.complex(qmr, qmi))
                f.write(struct.pack(f"<{freq_count}f", *qmr.tolist()))
                f.write(struct.pack(f"<{freq_count}f", *qmi.tolist()))
                f.write(struct.pack(f"<{freq_count}f", *qam.tolist()))
                f.write(struct.pack(f"<{freq_count}f", *mrl.tolist()))

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"Written: {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert TriAttention .pt to TRIA v2 binary")
    p.add_argument("input", help="Input .pt file (WeianMao format)")
    p.add_argument("output", help="Output .bin file (TRIA v2)")
    p.add_argument("--rope-theta", type=float, default=0.0, help="RoPE theta (not in .pt, set manually)")
    args = p.parse_args()
    convert(args.input, args.output, args.rope_theta)
