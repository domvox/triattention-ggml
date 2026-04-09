#!/usr/bin/env python3
"""
TriAttention Phase 2c — Attention-Mass Validation (memory-efficient)

Prunes KV cache at a trigger point, then measures what fraction of
attention mass future queries place on retained keys.

Uses per-layer hooks — works with device_map="auto" and hybrid SSM+attention.

Usage:
    cd ~/triattention-ggml && HIP_VISIBLE_DEVICES=0 python3 triattention_validate_mass.py \
      --model ~/models/Qwen3.5-27B-hf \
      --stats ~/triattention-stats/qwen3.5_27b.bin \
      --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
      --max-length 512 --budget 128 --device cuda
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

    backbone = getattr(model, "model", model)

    # Detect attention layers
    attn_layer_indices = []
    for li, layer in enumerate(backbone.layers):
        if hasattr(layer, 'self_attn'):
            attn_layer_indices.append(li)
    print(f"  {len(attn_layer_indices)} attention layers, "
          f"{len(backbone.layers) - len(attn_layer_indices)} SSM layers",
          file=sys.stderr)

    # Capture pre-RoPE K
    captured_k: Dict[int, torch.Tensor] = {}

    def _make_k_hook(li):
        def fn(mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is None: return
            bsz, qlen, _ = h.shape
            k = mod.k_proj(h).view(bsz, qlen, nkv, full_hd).transpose(1, 2)
            if hasattr(mod, 'k_norm'): k = mod.k_norm(k)
            if hd < full_hd:
                k = k[..., :hd]
            captured_k[li] = k.detach()
        return fn

    # Collect attention weights per-layer in post-hook
    layer_attn_data: Dict[int, torch.Tensor] = {}

    def _make_post_hook(li):
        def fn(mod, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                layer_attn_data[li] = output[1].detach().cpu()
        return fn

    handles = []
    for li in attn_layer_indices:
        attn_mod = backbone.layers[li].self_attn
        handles.append(attn_mod.register_forward_pre_hook(_make_k_hook(li), with_kwargs=True))
        handles.append(attn_mod.register_forward_hook(_make_post_hook(li)))

    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True,
                                  max_length=args.max_length).to(dev)
    seq_len = input_ids.shape[1]
    trigger_pos = args.trigger_pos if args.trigger_pos > 0 else seq_len // 2
    trigger_pos = max(1, min(trigger_pos, seq_len - 1))
    budget = args.budget
    print(f"  tokens={seq_len}, trigger={trigger_pos}, budget={budget}", file=sys.stderr)

    print("Forward pass (eager + attention weights) ...", file=sys.stderr)
    t0 = time.time()
    with torch.no_grad():
        try:
            model(input_ids, output_attentions=True)
        except torch.cuda.OutOfMemoryError:
            print("  (OOM caught, using collected hooks)", file=sys.stderr)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)
    for h in handles: h.remove()

    positions = torch.arange(seq_len, device=dev, dtype=torch.long)

    # Get TriAttention keep-sets at trigger point
    print("Computing keep-sets ...", file=sys.stderr)
    keep = compute_keep_sets(captured_k, positions, cal, omega, trigger_pos, budget, nkv, gqa, nl)

    # Evaluate future queries
    future_positions = list(range(trigger_pos + 1, seq_len))
    if not future_positions:
        print("ERROR: no future positions after trigger", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(future_positions)} future queries "
          f"(pos {future_positions[0]}-{future_positions[-1]}) ...", file=sys.stderr)

    layer_mass: Dict[int, List[float]] = {li: [] for li in attn_layer_indices}
    layer_recall: Dict[int, List[float]] = {li: [] for li in attn_layer_indices}

    for li in attn_layer_indices:
        attn_layer = layer_attn_data.get(li)
        if attn_layer is None:
            continue
        attn_layer = attn_layer.to(dev)

        for kvi in range(nkv):
            kept = keep.get((li, kvi))
            if kept is None: continue
            kept_t = torch.tensor(sorted(kept), device=dev, dtype=torch.long)

            for qpos in future_positions:
                attn_sum = torch.zeros(qpos, device=dev, dtype=torch.float32)
                for g in range(gqa):
                    ah = kvi * gqa + g
                    attn_sum += attn_layer[0, ah, qpos, :qpos].float()
                attn_sum /= gqa

                pre_trigger = attn_sum[:trigger_pos]
                total_pre = pre_trigger.sum().item()
                if total_pre < 1e-8: continue

                valid_kept = kept_t[kept_t < trigger_pos]
                mass_kept = pre_trigger[valid_kept].sum().item()
                layer_mass[li].append(mass_kept / total_pre)

                true_topk = torch.topk(pre_trigger, k=min(budget, trigger_pos)).indices
                true_set = set(true_topk.cpu().tolist())
                recall = len(true_set & kept) / len(true_set) if true_set else 0
                layer_recall[li].append(recall)

        del attn_layer  # zwolnij po przetworzeniu warstwy

    # Aggregate
    all_mass: List[float] = []
    all_recall: List[float] = []
    print(f"\n{'Layer':>6} {'Mass%':>7} {'Recall':>7} {'N':>5}", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    for li in attn_layer_indices:
        m = layer_mass[li]
        r = layer_recall[li]
        if not m: continue
        avg_m = sum(m) / len(m)
        avg_r = sum(r) / len(r)
        all_mass.extend(m)
        all_recall.extend(r)
        print(f"  L{li:<3d} {avg_m*100:6.1f}% {avg_r:.3f} {len(m):5d}", file=sys.stderr)

    print("-" * 30, file=sys.stderr)
    mean_mass = sum(all_mass) / len(all_mass) if all_mass else 0
    mean_recall = sum(all_recall) / len(all_recall) if all_recall else 0
    print(f"  {'AVG':>4} {mean_mass*100:6.1f}% {mean_recall:.3f} {len(all_mass):5d}", file=sys.stderr)

    print(f"\n=== Phase 2c Results ===", file=sys.stderr)
    print(f"Attention mass retained: {mean_mass*100:.1f}%", file=sys.stderr)
    print(f"Multi-query recall@{budget}: {mean_recall:.3f}", file=sys.stderr)
    print(f"  (averaged over {len(future_positions)} future queries × "
          f"{len(attn_layer_indices)} layers × {nkv} KV heads)", file=sys.stderr)
    print(f"  trigger={trigger_pos}, budget={budget}, retention={budget/trigger_pos*100:.0f}%",
          file=sys.stderr)


if __name__ == "__main__":
    main()
