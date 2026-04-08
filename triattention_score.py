#!/usr/bin/env python3
"""
TriAttention Scoring Prototype — Phase 2

Loads calibration stats, runs a model forward pass, captures KV cache,
and computes TriAttention key importance scores every 128 tokens.
Reports which tokens each KV-head would keep under a given budget.

Usage:
    # Stop Hermes first:
    #   sudo systemctl stop llama-server.service
    HIP_VISIBLE_DEVICES=0 python3 triattention_score.py \
        --model Qwen/Qwen3-8B \
        --stats ~/triattention-stats/qwen3_8b_v3.bin \
        --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
        --max-length 2048 \
        --budget 512 \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from triattention_common import load_stats, build_omega, compute_keep_sets


def run_scoring(
    model_name: str,
    stats_path: str,
    input_path: str,
    max_length: int = 2048,
    budget: int = 512,
    device: str = "cuda",
) -> None:
    dev = torch.device(device)
    dtype = torch.bfloat16

    # Load stats
    print(f"Loading stats from {stats_path} ...", file=sys.stderr)
    cal = load_stats(Path(stats_path), dev)
    num_layers = cal["num_layers"]
    num_heads = cal["num_heads"]
    num_kv_heads = cal["num_kv_heads"]
    head_dim = cal["head_dim"]
    freq_count = cal["freq_count"]
    gqa_groups = num_heads // num_kv_heads

    omega = build_omega(cal["rope_theta"], head_dim, freq_count, dev)

    # Load model
    print(f"Loading {model_name} ...", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(dev)
    model.eval()

    # Capture pre-RoPE K per layer
    captured_k: Dict[int, torch.Tensor] = {}
    backbone = getattr(model, "model", model)

    def _make_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return
            bsz, qlen, _ = hidden.shape
            k = module.k_proj(hidden)
            k = k.view(bsz, qlen, num_kv_heads, head_dim).transpose(1, 2)
            if hasattr(module, 'k_norm'):
                k = module.k_norm(k)
            captured_k[layer_idx] = k.detach()
        return hook_fn

    handles = []
    for li, layer in enumerate(backbone.layers):
        h = layer.self_attn.register_forward_pre_hook(
            _make_hook(li), with_kwargs=True
        )
        handles.append(h)

    # Tokenize and forward
    print(f"Reading {input_path} ...", file=sys.stderr)
    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(dev)
    seq_len = input_ids.shape[1]
    print(f"  tokens={seq_len}", file=sys.stderr)

    print("Forward pass ...", file=sys.stderr)
    t0 = time.time()
    with torch.no_grad():
        model(input_ids)
    print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)

    for h in handles:
        h.remove()

    # Score keys at simulated trigger points (every 128 tokens)
    print(f"\nScoring (budget={budget}, trigger every 128 tokens):", file=sys.stderr)
    positions = torch.arange(seq_len, device=dev, dtype=torch.long)

    trigger_points = list(range(128, seq_len, 128))
    if not trigger_points:
        trigger_points = [seq_len - 1]

    for trigger_pos in trigger_points:
        if trigger_pos <= budget:
            continue

        keep_sets = compute_keep_sets(
            captured_k, positions, cal, omega, trigger_pos,
            budget, num_kv_heads, gqa_groups, num_layers,
        )
        total_kept = sum(len(v) for v in keep_sets.values())
        total_heads = len(keep_sets)
        total_evicted = total_heads * trigger_pos - total_kept

        evict_pct = 100.0 * total_evicted / (total_kept + total_evicted) if (total_kept + total_evicted) > 0 else 0
        print(f"  pos={trigger_pos:5d}: kept={total_kept} evicted={total_evicted} ({evict_pct:.1f}% pruned)", file=sys.stderr)

    # Summary
    print(f"\n=== Scoring Summary ===", file=sys.stderr)
    print(f"Model:    {model_name}", file=sys.stderr)
    print(f"Tokens:   {seq_len}", file=sys.stderr)
    print(f"Budget:   {budget} per KV-head per layer", file=sys.stderr)
    print(f"Layers:   {num_layers}", file=sys.stderr)
    print(f"KV heads: {num_kv_heads}", file=sys.stderr)
    print(f"Triggers: {len([t for t in trigger_points if t > budget])}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="TriAttention scoring prototype")
    parser.add_argument("--model", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_scoring(args.model, args.stats, args.input, args.max_length, args.budget, args.device)


if __name__ == "__main__":
    main()
