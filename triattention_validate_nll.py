#!/usr/bin/env python3
"""
TriAttention Phase 2d — Decode NLL Validation (large model support)

Compares next-token loss with full KV cache vs pruned KV cache.
Supports device_map="auto" and hybrid SSM+attention models.

Usage:
    cd ~/triattention-ggml && HIP_VISIBLE_DEVICES=0 python3 triattention_validate_nll.py \
      --model ~/models/Qwen3.5-27B-hf \
      --stats ~/triattention-stats/qwen3.5_27b.bin \
      --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
      --max-length 512 --budget 128 --device cuda
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    hd, fc = cal["head_dim"], cal["freq_count"]  # hd = rotary_dim
    gqa = nh // nkv
    omega = build_omega(cal["rope_theta"], hd, fc, dev)

    print(f"Loading {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    full_hd = getattr(text_cfg, "head_dim",
                       getattr(text_cfg, "hidden_size", 4096) // nh)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager", trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True,
                                  max_length=args.max_length).to(dev)
    seq_len = input_ids.shape[1]
    trigger_pos = args.trigger_pos if args.trigger_pos > 0 else seq_len // 2
    trigger_pos = max(1, min(trigger_pos, seq_len - 2))
    budget = args.budget
    print(f"  tokens={seq_len}, trigger={trigger_pos}, budget={budget}", file=sys.stderr)

    # --- Step 1: Full forward pass + capture pre-RoPE K ---
    captured_k: Dict[int, torch.Tensor] = {}
    backbone = getattr(model, "model", model)

    attn_layer_indices = []
    for li, layer in enumerate(backbone.layers):
        if hasattr(layer, 'self_attn'):
            attn_layer_indices.append(li)
    print(f"  {len(attn_layer_indices)} attention layers, "
          f"{len(backbone.layers) - len(attn_layer_indices)} SSM layers",
          file=sys.stderr)

    def _make_k_hook(li):
        def fn(mod, args_, kwargs):
            h = args_[0] if args_ else kwargs.get("hidden_states")
            if h is None: return
            bsz, qlen, _ = h.shape
            k = mod.k_proj(h).view(bsz, qlen, nkv, full_hd).transpose(1, 2)
            if hasattr(mod, 'k_norm'): k = mod.k_norm(k)
            if hd < full_hd:
                k = k[..., :hd]
            captured_k[li] = k.detach()
        return fn

    handles = []
    for li in attn_layer_indices:
        handles.append(backbone.layers[li].self_attn.register_forward_pre_hook(
            _make_k_hook(li), with_kwargs=True))

    print("Full forward pass ...", file=sys.stderr)
    with torch.no_grad():
        try:
            out_full = model(input_ids)
        except torch.cuda.OutOfMemoryError:
            print("  (OOM on lm_head, retrying)", file=sys.stderr)
            torch.cuda.empty_cache()
            model.lm_head = torch.nn.Identity()
            out_full = model(input_ids)
    for h in handles: h.remove()

    full_logits = out_full.logits[0].cpu()  # [seq, vocab]
    del out_full
    torch.cuda.empty_cache()

    # --- Step 2: Compute keep masks ---
    print("Computing keep masks ...", file=sys.stderr)
    positions = torch.arange(seq_len, device=dev, dtype=torch.long)
    masks = get_keep_mask(captured_k, positions, cal, omega, trigger_pos, budget,
                          nkv, gqa, nl, dev)
    del captured_k

    # --- Step 3: Build eviction mask and re-run ---
    print("Building eviction mask ...", file=sys.stderr)
    evict_votes = torch.zeros(trigger_pos, dtype=torch.float32, device=dev)
    total_pairs = 0
    for li in attn_layer_indices:
        for kvi in range(nkv):
            mask = masks.get((li, kvi))
            if mask is None: continue
            total_pairs += 1
            evict_votes += (~mask).float()
    common_evict = evict_votes > (total_pairs * 0.5)
    n_evicted = common_evict.sum().item()
    print(f"  {n_evicted}/{trigger_pos} positions evicted (majority vote)", file=sys.stderr)

    causal = torch.triu(torch.full((seq_len, seq_len), torch.finfo(dtype).min,
                                    device=dev, dtype=dtype), diagonal=1)
    for pos_idx in range(trigger_pos):
        if common_evict[pos_idx]:
            causal[:, pos_idx] = torch.finfo(dtype).min
    attn_mask_4d = causal.unsqueeze(0).unsqueeze(0)

    # Reload model for pruned pass if lm_head was replaced
    if isinstance(model.lm_head, torch.nn.Identity):
        print("Reloading model for pruned pass (lm_head was replaced) ...", file=sys.stderr)
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="eager", trust_remote_code=True,
            device_map="auto",
        )
        model.eval()

    print("Pruned forward pass ...", file=sys.stderr)
    with torch.no_grad():
        out_pruned = model(input_ids, attention_mask=attn_mask_4d)

    pruned_logits = out_pruned.logits[0].cpu()
    n_eval = seq_len - trigger_pos - 1
    targets = input_ids[0, trigger_pos+1:].cpu().long()

    full_log = full_logits[trigger_pos:seq_len-1]
    prun_log = pruned_logits[trigger_pos:seq_len-1]

    # NLL
    nll_full = F.cross_entropy(full_log, targets, reduction='none')
    nll_prun = F.cross_entropy(prun_log, targets, reduction='none')

    # KL divergence
    full_p = F.log_softmax(full_log.float(), dim=-1)
    prun_p = F.log_softmax(prun_log.float(), dim=-1)
    kl = F.kl_div(prun_p, full_p.exp(), reduction='none', log_target=False).sum(dim=-1)

    # Top-1 agreement
    top1_full = full_log.argmax(dim=-1)
    top1_prun = prun_log.argmax(dim=-1)
    agree = (top1_full == top1_prun).float().mean().item()

    print(f"\n{'Window':>12} {'NLL_full':>9} {'NLL_prun':>9} {'ΔNLL':>7} {'KL':>8} {'Top1%':>6}",
          file=sys.stderr)
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
        print(f"  {label:>10} {nf:9.4f} {np_:9.4f} {np_-nf:+7.4f} {k:8.5f} {a*100:5.1f}%",
              file=sys.stderr)

    print(f"\n=== Phase 2d Results ===", file=sys.stderr)
    print(f"NLL full:    {nll_full.mean().item():.4f}", file=sys.stderr)
    print(f"NLL pruned:  {nll_prun.mean().item():.4f}", file=sys.stderr)
    print(f"ΔNLL:        {(nll_prun - nll_full).mean().item():+.4f}", file=sys.stderr)
    print(f"KL(full||pr): {kl.mean().item():.5f}", file=sys.stderr)
    print(f"Top-1 agree: {agree*100:.1f}%", file=sys.stderr)
    print(f"  ({n_eval} tokens after trigger={trigger_pos}, budget={budget}, "
          f"retention={budget/trigger_pos*100:.0f}%)", file=sys.stderr)

if __name__ == "__main__":
    main()
