#!/usr/bin/env python3
"""
TriAttention Phase 2c — Attention-Mass Validation

Prunes KV cache at a trigger point, then measures what fraction of
attention mass future queries place on retained keys.

Metrics:
  - attention_mass_retained: avg fraction of attn weight on kept tokens
  - recall@B: set overlap (from Phase 2b, for comparison)

Usage:
    sudo systemctl stop llama-server.service && sleep 2 && \
    cd ~/triattention-ggml && HIP_VISIBLE_DEVICES=0 python3 triattention_validate_mass.py \
      --model Qwen/Qwen3-8B --stats ~/triattention-stats/qwen3_8b_v3.bin \
      --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
      --max-length 512 --budget 128 --device cuda
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from triattention_common import load_stats, score_keys, build_omega, compute_keep_sets

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--trigger-pos", type=int, default=0, help="Prune at this position (0=midpoint)")
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
    trigger_pos = args.trigger_pos if args.trigger_pos > 0 else seq_len // 2
    budget = args.budget
    print(f"  tokens={seq_len}, trigger={trigger_pos}, budget={budget}", file=sys.stderr)

    print("Forward pass (eager + attention weights) ...", file=sys.stderr)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)
    for h in handles: h.remove()

    attentions = out.attentions  # tuple of [1, num_heads, seq, seq]
    positions = torch.arange(seq_len, device=dev, dtype=torch.long)

    # Get TriAttention keep-sets at trigger point
    print("Computing keep-sets ...", file=sys.stderr)
    keep = compute_keep_sets(captured_k, positions, cal, omega, trigger_pos, budget, nkv, gqa, nl)

    # Evaluate: for each future query position after trigger, measure attention mass on kept tokens
    future_positions = list(range(trigger_pos + 1, seq_len))
    if not future_positions:
        print("ERROR: no future positions after trigger", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(future_positions)} future queries (pos {future_positions[0]}-{future_positions[-1]}) ...", file=sys.stderr)

    # Collect per-layer stats
    layer_mass = {li: [] for li in range(nl)}
    layer_recall = {li: [] for li in range(nl)}

    for li in range(nl):
        attn_layer = attentions[li]  # [1, num_heads, seq, seq]
        for kvi in range(nkv):
            kept = keep.get((li, kvi))
            if kept is None: continue
            kept_t = torch.tensor(sorted(kept), device=dev, dtype=torch.long)

            for qpos in future_positions:
                # Aggregate attention across GQA heads for this KV head
                attn_sum = torch.zeros(qpos, device=dev, dtype=torch.float32)
                for g in range(gqa):
                    ah = kvi * gqa + g
                    attn_sum += attn_layer[0, ah, qpos, :qpos].float()
                attn_sum /= gqa

                # Only consider keys before trigger (the ones we pruned)
                pre_trigger = attn_sum[:trigger_pos]
                total_pre = pre_trigger.sum().item()
                if total_pre < 1e-8: continue

                # Mass on kept tokens
                valid_kept = kept_t[kept_t < trigger_pos]
                mass_kept = pre_trigger[valid_kept].sum().item()
                layer_mass[li].append(mass_kept / total_pre)

                # Recall (set overlap with true top-B)
                true_topk = torch.topk(pre_trigger, k=min(budget, trigger_pos)).indices
                true_set = set(true_topk.cpu().tolist())
                recall = len(true_set & kept) / len(true_set) if true_set else 0
                layer_recall[li].append(recall)

    # Aggregate
    all_mass = []
    all_recall = []
    print(f"\n{'Layer':>6} {'Mass%':>7} {'Recall':>7} {'N':>5}", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    for li in range(nl):
        m = layer_mass[li]
        r = layer_recall[li]
        if not m: continue
        avg_m = sum(m)/len(m)
        avg_r = sum(r)/len(r)
        all_mass.extend(m)
        all_recall.extend(r)
        print(f"  L{li:<3d} {avg_m*100:6.1f}% {avg_r:.3f} {len(m):5d}", file=sys.stderr)

    print("-" * 30, file=sys.stderr)
    mean_mass = sum(all_mass)/len(all_mass) if all_mass else 0
    mean_recall = sum(all_recall)/len(all_recall) if all_recall else 0
    print(f"  {'AVG':>4} {mean_mass*100:6.1f}% {mean_recall:.3f} {len(all_mass):5d}", file=sys.stderr)

    print(f"\n=== Phase 2c Results ===", file=sys.stderr)
    print(f"Attention mass retained: {mean_mass*100:.1f}%", file=sys.stderr)
    print(f"Multi-query recall@{budget}: {mean_recall:.3f}", file=sys.stderr)
    print(f"  (averaged over {len(future_positions)} future queries × {nl} layers × {nkv} KV heads)", file=sys.stderr)
    print(f"  trigger={trigger_pos}, budget={budget}, retention={budget/trigger_pos*100:.0f}%", file=sys.stderr)

if __name__ == "__main__":
    main()
