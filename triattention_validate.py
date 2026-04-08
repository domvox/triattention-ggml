#!/usr/bin/env python3
"""
TriAttention Scoring Validation — Phase 2b

Compares TriAttention's predicted keep-sets against actual attention weights.
Measures "recall@B": what fraction of the true top-B attended tokens
are also in TriAttention's predicted top-B.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 triattention_validate.py \
        --model Qwen/Qwen3-8B \
        --stats ~/triattention-stats/qwen3_8b_v3.bin \
        --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
        --max-length 1024 --budget 256 --device cuda
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from triattention_common import load_stats, score_keys, build_omega

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = torch.device(args.device)
    dtype = torch.bfloat16

    cal = load_stats(Path(args.stats), dev)
    nl, nh, nkv = cal["num_layers"], cal["num_heads"], cal["num_kv_heads"]
    hd, fc = cal["head_dim"], cal["freq_count"]
    gqa = nh // nkv
    omega = build_omega(cal["rope_theta"], hd, fc, dev)

    print(f"Loading {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Use eager attention to capture attention weights
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager", trust_remote_code=True
    ).to(dev)
    model.eval()

    # Capture pre-RoPE K
    captured_k: Dict[int, torch.Tensor] = {}
    backbone = getattr(model, "model", model)
    def _make_k_hook(li):
        def fn(mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is None: return
            bsz, qlen, _ = h.shape
            k = mod.k_proj(h).view(bsz, qlen, nkv, hd).transpose(1,2)
            if hasattr(mod, 'k_norm'): k = mod.k_norm(k)
            captured_k[li] = k.detach()
        return fn

    handles = []
    for li, layer in enumerate(backbone.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(_make_k_hook(li), with_kwargs=True))

    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(dev)
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        raise ValueError("Need at least 2 tokens to run validation")
    print(f"  tokens={seq_len}", file=sys.stderr)

    print("Forward pass (eager, with attention weights) ...", file=sys.stderr)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    for h in handles: h.remove()

    attentions = out.attentions  # tuple of [batch, heads, seq, seq] per layer
    positions = torch.arange(seq_len, device=dev, dtype=torch.long)
    budget = args.budget

    # Validate at last position: which tokens does the last query attend to most?
    query_pos = seq_len - 1
    print(f"\nValidation at query position {query_pos}, budget={budget}:", file=sys.stderr)

    recalls = []
    layer_recalls = {li: [] for li in range(nl)}
    for li in range(nl):
        k_pre = captured_k.get(li)
        attn_layer = attentions[li]  # [1, num_heads, seq, seq]
        if k_pre is None: continue

        for kv_hi in range(nkv):
            # True attention: average across GQA query heads for this KV head
            attn_scores = []
            for g in range(gqa):
                ah = kv_hi * gqa + g
                attn_scores.append(attn_layer[0, ah, query_pos, :query_pos])  # [query_pos]
            true_attn = torch.stack(attn_scores).mean(dim=0)  # [query_pos]
            true_topk = torch.topk(true_attn, k=min(budget, query_pos)).indices
            true_set = set(true_topk.cpu().tolist())

            # TriAttention predicted scores
            k_head = k_pre[0, kv_hi, :query_pos]
            pred_scores_list = []
            for g in range(gqa):
                ah = kv_hi * gqa + g
                s = cal["stats"].get((li, ah))
                if s is None: continue
                sc = score_keys(k_head, positions[:query_pos], s["q_mean"], s["q_abs_mean"], omega, query_pos)
                pred_scores_list.append(sc)
            if not pred_scores_list: continue
            stacked = torch.stack(pred_scores_list)
            mean = stacked.mean(dim=1, keepdim=True)
            std = stacked.std(dim=1, keepdim=True).clamp_min(1e-6)
            norm = (stacked - mean) / std
            final = norm.max(dim=0).values
            pred_topk = torch.topk(final, k=min(budget, query_pos)).indices
            pred_set = set(pred_topk.cpu().tolist())

            overlap = len(true_set & pred_set)
            recall = overlap / len(true_set) if true_set else 0
            recalls.append(recall)
            layer_recalls[li].append(recall)

    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    print(f"\n=== Validation Results ===", file=sys.stderr)
    print(f"Recall@{budget}: {mean_recall:.3f} ({len(recalls)} layer×kv_head pairs)", file=sys.stderr)
    print(f"  (fraction of true top-{budget} attended tokens captured by TriAttention)", file=sys.stderr)

    # Breakdown by layer
    per_layer = {}
    for li in range(nl):
        if layer_recalls[li]:
            per_layer[li] = sum(layer_recalls[li])/len(layer_recalls[li])
    
    best_layers = sorted(per_layer.items(), key=lambda x: -x[1])[:5]
    worst_layers = sorted(per_layer.items(), key=lambda x: x[1])[:5]
    print(f"\nBest layers:  {', '.join(f'L{l}={r:.3f}' for l,r in best_layers)}", file=sys.stderr)
    print(f"Worst layers: {', '.join(f'L{l}={r:.3f}' for l,r in worst_layers)}", file=sys.stderr)

if __name__ == "__main__":
    main()
