#!/usr/bin/env python3
"""Validate inverse-RoPE scoring parity using cached post-RoPE keys."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from triattention_common import build_omega, score_keys


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _invert_rope(rotated: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale: float) -> torch.Tensor:
    s = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / s
    cos_u = cos / s
    sin_u = sin / s
    return base * cos_u - _rotate_half(base) * sin_u


def _spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = torch.sqrt((ar * ar).sum() * (br * br).sum()).clamp_min(1e-12)
    return float((ar * br).sum() / denom)


def _extract_post_rope_keys(past_key_values) -> List[torch.Tensor]:
    if past_key_values is None:
        raise RuntimeError("Model output did not include past_key_values")

    if hasattr(past_key_values, "key_cache"):
        return [k.detach() for k in past_key_values.key_cache]

    layers = []
    for entry in past_key_values:
        if isinstance(entry, (tuple, list)):
            layers.append(entry[0].detach())
        elif hasattr(entry, "key"):
            layers.append(entry.key.detach())
        else:
            raise TypeError(f"Unsupported KV entry type: {type(entry)}")
    return layers


def run(model_name: str, input_path: str, max_length: int, device: str) -> None:
    dev = torch.device(device)
    dtype = torch.bfloat16

    print(f"Loading {model_name} ...", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(dev)
    model.eval()

    nl = config.num_hidden_layers
    nh = config.num_attention_heads
    nkv = getattr(config, "num_key_value_heads", nh)
    hd = getattr(config, "head_dim", config.hidden_size // nh)
    fc = hd // 2
    gqa = nh // nkv
    rope_theta = float(getattr(config, "rope_theta", 10000.0))

    backbone = getattr(model, "model", model)
    rotary = getattr(backbone, "rotary_emb", None) or backbone.layers[0].self_attn.rotary_emb
    rope_scale = float(getattr(rotary, "attention_scaling", 1.0))
    omega = build_omega(rope_theta, hd, fc, dev)

    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(dev)
    seq_len = input_ids.shape[1]
    if seq_len < 3:
        raise ValueError("Need at least 3 tokens for rope validation")

    captured_q: Dict[int, torch.Tensor] = {}
    captured_k_pre: Dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def fn(mod, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return
            bsz, qlen, _ = hidden.shape
            q = mod.q_proj(hidden).view(bsz, qlen, nh, hd).transpose(1, 2)
            k = mod.k_proj(hidden).view(bsz, qlen, nkv, hd).transpose(1, 2)
            if hasattr(mod, "q_norm"):
                q = mod.q_norm(q)
            if hasattr(mod, "k_norm"):
                k = mod.k_norm(k)
            captured_q[layer_idx] = q.detach()
            captured_k_pre[layer_idx] = k.detach()
        return fn

    handles = [
        layer.self_attn.register_forward_pre_hook(_make_hook(li), with_kwargs=True)
        for li, layer in enumerate(backbone.layers)
    ]

    pos_ids = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    for h in handles:
        h.remove()

    post_k = _extract_post_rope_keys(out.past_key_values)
    key_positions = torch.arange(seq_len, device=dev, dtype=torch.long)
    trigger_pos = seq_len - 1

    # rotary forward gives cos/sin matching current rope scaling/config
    probe = torch.zeros((1, nkv, seq_len, hd), device=dev, dtype=torch.float32)
    cos, sin = rotary(probe, pos_ids)
    if cos.dim() == 4:
        cos = cos.squeeze(1)
        sin = sin.squeeze(1)
    cos = cos.to(dtype=torch.float32)
    sin = sin.to(dtype=torch.float32)

    layer_corrs: List[float] = []
    layer_maxdiff: List[float] = []

    for li in range(min(nl, len(post_k))):
        q_pre = captured_q[li][0].float()      # [heads, seq, hd]
        k_pre = captured_k_pre[li][0].float()  # [kv, seq, hd]
        k_post = post_k[li][0].float()         # [kv, seq, hd]

        cos_u = cos.unsqueeze(0)
        sin_u = sin.unsqueeze(0)
        k_recovered = _invert_rope(k_post, cos_u, sin_u, rope_scale)

        pair_corrs: List[float] = []
        pair_diffs: List[float] = []

        for kvi in range(nkv):
            for g in range(gqa):
                ah = kvi * gqa + g
                q_head = q_pre[ah]
                q_complex = torch.complex(q_head[:, :fc], q_head[:, fc:])
                q_mean = q_complex.mean(dim=0)
                q_abs_mean = q_complex.abs().mean(dim=0)

                direct = score_keys(
                    k_pre[kvi, :trigger_pos],
                    key_positions[:trigger_pos],
                    q_mean,
                    q_abs_mean,
                    omega,
                    trigger_pos,
                )
                recovered = score_keys(
                    k_recovered[kvi, :trigger_pos],
                    key_positions[:trigger_pos],
                    q_mean,
                    q_abs_mean,
                    omega,
                    trigger_pos,
                )

                pair_corrs.append(_spearman_corr(direct, recovered))
                pair_diffs.append(float(torch.max(torch.abs(direct - recovered)).item()))

        layer_corrs.append(sum(pair_corrs) / len(pair_corrs))
        layer_maxdiff.append(max(pair_diffs) if pair_diffs else math.nan)

    print("\n=== Inverse-RoPE Score Validation ===")
    print(f"Model: {model_name}")
    print(f"Tokens: {seq_len}, max_length={max_length}, device={device}")
    print(f"Layers compared: {len(layer_corrs)}")
    if layer_corrs:
        print(f"Mean Spearman rank corr: {sum(layer_corrs)/len(layer_corrs):.6f}")
        print(f"Worst-layer Spearman corr: {min(layer_corrs):.6f}")
        print(f"Max absolute score diff: {max(layer_maxdiff):.6e}")

    print("\nPer-layer summary:")
    for li, (corr, md) in enumerate(zip(layer_corrs, layer_maxdiff)):
        print(f"  L{li:02d}: spearman={corr:.6f}  max|Δscore|={md:.6e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate inverse-RoPE score parity from cached post-RoPE K")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--input", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.model, args.input, args.max_length, args.device)


if __name__ == "__main__":
    main()
