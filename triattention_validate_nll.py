#!/usr/bin/env python3
"""
TriAttention Phase 2d — Decode NLL Validation

Compares next-token loss with full KV cache vs pruned KV cache.
Simulates real TriAttention deployment: prune at trigger, then measure
how much generation quality degrades over subsequent tokens.

Metrics:
  - NLL_full: avg negative log-likelihood with full cache
  - NLL_pruned: avg NLL with TriAttention-pruned cache
  - delta_NLL: NLL_pruned - NLL_full (lower = better pruning)
  - KL divergence between full and pruned logit distributions

Usage:
    sudo systemctl stop llama-server.service && sleep 2 && \
    cd ~/triattention-ggml && HIP_VISIBLE_DEVICES=0 python3 triattention_validate_nll.py \
      --model Qwen/Qwen3-8B --stats ~/triattention-stats/qwen3_8b_v3.bin \
      --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
      --max-length 512 --budget 128 --device cuda
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from triattention_common import load_stats, build_omega, compute_keep_sets

def get_keep_mask(captured_k, positions, cal, omega, trigger_pos, budget, nkv, gqa, nl, device):
    """Convert keep-sets to per-(layer,kv_head) boolean masks."""
    keep_sets = compute_keep_sets(captured_k, positions, cal, omega, trigger_pos, budget, nkv, gqa, nl)
    masks = {}
    for (li, kvi), kept in keep_sets.items():
        mask = torch.zeros(trigger_pos, dtype=torch.bool, device=device)
        mask[torch.tensor(sorted(kept), device=device, dtype=torch.long)] = True
        masks[(li, kvi)] = mask
    return masks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--trigger-pos", type=int, default=0)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager", trust_remote_code=True
    ).to(dev)
    model.eval()

    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(dev)
    seq_len = input_ids.shape[1]
    trigger_pos = args.trigger_pos if args.trigger_pos > 0 else seq_len // 2
    budget = args.budget
    print(f"  tokens={seq_len}, trigger={trigger_pos}, budget={budget}", file=sys.stderr)

    # --- Step 1: Full forward pass to get baseline logits + capture pre-RoPE K ---
    captured_k: Dict[int, torch.Tensor] = {}
    backbone = getattr(model, "model", model)
    def _make_k_hook(li):
        def fn(mod, args_, kwargs):
            h = args_[0] if args_ else kwargs.get("hidden_states")
            if h is None: return
            bsz, qlen, _ = h.shape
            k = mod.k_proj(h).view(bsz, qlen, nkv, hd).transpose(1,2)
            if hasattr(mod, 'k_norm'): k = mod.k_norm(k)
            captured_k[li] = k.detach()
        return fn
    handles = []
    for li, layer in enumerate(backbone.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(_make_k_hook(li), with_kwargs=True))

    print("Full forward pass (eager) ...", file=sys.stderr)
    with torch.no_grad():
        out_full = model(input_ids)
    for h in handles: h.remove()

    full_logits = out_full.logits[0]  # [seq, vocab]

    # --- Step 2: Compute keep masks ---
    print("Computing keep masks ...", file=sys.stderr)
    positions = torch.arange(seq_len, device=dev, dtype=torch.long)
    masks = get_keep_mask(captured_k, positions, cal, omega, trigger_pos, budget, nkv, gqa, nl, dev)
    del captured_k

    # --- Step 3: Re-run with eager attention and custom 4D mask ---
    # HF 2D mask removes tokens from embedding level (wrong for KV eviction)
    # We need 4D mask that only affects attention scores, not embeddings
    # Switch to eager attention for the pruned pass
    print("Building eviction mask ...", file=sys.stderr)
    evict_votes = torch.zeros(trigger_pos, dtype=torch.float32, device=dev)
    total_pairs = 0
    for li in range(nl):
        for kvi in range(nkv):
            mask = masks.get((li, kvi))
            if mask is None: continue
            total_pairs += 1
            evict_votes += (~mask).float()
    common_evict = evict_votes > (total_pairs * 0.5)
    n_evicted = common_evict.sum().item()
    print(f"  {n_evicted}/{trigger_pos} positions evicted (majority vote)", file=sys.stderr)

    # Build 4D causal mask with eviction: [1, 1, seq, seq]
    # Start from causal mask, then set evicted columns to -inf
    causal = torch.triu(torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=dev, dtype=dtype), diagonal=1)
    for pos_idx in range(trigger_pos):
        if common_evict[pos_idx]:
            causal[:, pos_idx] = torch.finfo(dtype).min
    attn_mask_4d = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

    print("Pruned forward pass (eager + 4D eviction mask) ...", file=sys.stderr)
    with torch.no_grad():
        out_pruned = model(input_ids, attention_mask=attn_mask_4d)

    # --- Step 5: Compare NLL and KL ---
    # pruned pass is full sequence, so pruned_logits covers all positions
    pruned_logits = out_pruned.logits[0].cpu()
    full_logits = full_logits.cpu()
    n_eval = seq_len - trigger_pos - 1
    targets = input_ids[0, trigger_pos+1:].cpu().long()

    full_log = full_logits[trigger_pos:seq_len-1]       # [n_eval, vocab]
    prun_log = pruned_logits[trigger_pos:seq_len-1]      # [n_eval, vocab]

    # NLL
    nll_full = F.cross_entropy(full_log, targets, reduction='none')
    nll_prun = F.cross_entropy(prun_log, targets, reduction='none')

    # KL divergence: KL(full || pruned)
    full_p = F.log_softmax(full_log.float(), dim=-1)
    prun_p = F.log_softmax(prun_log.float(), dim=-1)
    kl = F.kl_div(prun_p, full_p.exp(), reduction='none', log_target=False).sum(dim=-1)

    # Top-1 agreement
    top1_full = full_log.argmax(dim=-1)
    top1_prun = prun_log.argmax(dim=-1)
    agree = (top1_full == top1_prun).float().mean().item()

    # Print results over time windows
    print(f"\n{'Window':>12} {'NLL_full':>9} {'NLL_prun':>9} {'ΔNLL':>7} {'KL':>8} {'Top1%':>6}", file=sys.stderr)
    print("-" * 58, file=sys.stderr)
    windows = [(0, min(32, n_eval)), (0, min(64, n_eval)), (0, min(128, n_eval)), (0, n_eval)]
    for start, end in windows:
        if end <= start: continue
        sl = slice(start, end)
        nf = nll_full[sl].mean().item()
        np_ = nll_prun[sl].mean().item()
        k = kl[sl].mean().item()
        a = (top1_full[sl] == top1_prun[sl]).float().mean().item()
        label = f"+{start}..+{end}"
        print(f"  {label:>10} {nf:9.4f} {np_:9.4f} {np_-nf:+7.4f} {k:8.5f} {a*100:5.1f}%", file=sys.stderr)

    print(f"\n=== Phase 2d Results ===", file=sys.stderr)
    print(f"NLL full:    {nll_full.mean().item():.4f}", file=sys.stderr)
    print(f"NLL pruned:  {nll_prun.mean().item():.4f}", file=sys.stderr)
    print(f"ΔNLL:        {(nll_prun - nll_full).mean().item():+.4f}", file=sys.stderr)
    print(f"KL(full||pr): {kl.mean().item():.5f}", file=sys.stderr)
    print(f"Top-1 agree: {agree*100:.1f}%", file=sys.stderr)
    print(f"  ({n_eval} tokens after trigger={trigger_pos}, budget={budget}, retention={budget/trigger_pos*100:.0f}%)", file=sys.stderr)

if __name__ == "__main__":
    main()
