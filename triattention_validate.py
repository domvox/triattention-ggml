#!/usr/bin/env python3
"""
TriAttention Scoring Validation — Phase 2b (memory-efficient)

Computes recall@B by capturing Q,K post-RoPE in a forward hook on each
attention layer's output, computing attention weights on-the-fly per layer.
No output_attentions needed — works with device_map="auto" and accelerate.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 triattention_validate.py \
        --model ~/models/Qwen3.5-27B-hf \
        --stats ~/triattention-stats/qwen3.5_27b.bin \
        --input ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
        --max-length 4096 --budget 256 --device cuda
"""
from __future__ import annotations
import argparse, sys, time, math
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True,
                                  max_length=args.max_length).to(dev)
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        raise ValueError("Need at least 2 tokens")
    print(f"  tokens={seq_len}", file=sys.stderr)

    budget = args.budget
    query_pos = seq_len - 1
    positions = torch.arange(seq_len, device=dev, dtype=torch.long)

    recalls: List[float] = []
    layer_recalls: Dict[int, List[float]] = {li: [] for li in attn_layer_indices}

    # Przechwytujemy pre-RoPE K w pre-hooku na self_attn
    captured_k_pre: Dict[int, torch.Tensor] = {}

    def _make_pre_hook(li):
        """Capture pre-RoPE K (trimmed to rotary dims) for TriAttention scoring."""
        def fn(mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is None: return
            bsz, qlen, _ = h.shape
            k = mod.k_proj(h).view(bsz, qlen, nkv, full_hd).transpose(1, 2)
            if hasattr(mod, 'k_norm'): k = mod.k_norm(k)
            if hd < full_hd:
                k = k[..., :hd]
            captured_k_pre[li] = k.detach()
        return fn

    def _make_post_hook(li):
        """After self_attn forward, compute attention from Q,K post-RoPE
        and compare with TriAttention predictions. Frees memory immediately."""
        def fn(mod, input, output):
            # eager returns (attn_output, attn_weights)
            attn_weights = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
            k_pre = captured_k_pre.pop(li, None)
            if attn_weights is None or k_pre is None:
                if k_pre is None:
                    print(f"  L{li}: no k_pre captured", file=sys.stderr)
                if attn_weights is None:
                    print(f"  L{li}: no attn_weights in output", file=sys.stderr)
                return

            # attn_weights: [batch, num_heads, seq, seq]
            for kv_hi in range(nkv):
                attn_scores = []
                for g in range(gqa):
                    ah = kv_hi * gqa + g
                    attn_scores.append(attn_weights[0, ah, query_pos, :query_pos])
                true_attn = torch.stack(attn_scores).mean(dim=0)
                true_topk = torch.topk(true_attn, k=min(budget, query_pos)).indices
                true_set = set(true_topk.cpu().tolist())

                # TriAttention predicted scores
                k_head = k_pre[0, kv_hi, :query_pos]
                pred_scores_list = []
                for g in range(gqa):
                    ah = kv_hi * gqa + g
                    s = cal["stats"].get((li, ah))
                    if s is None: continue
                    sc = score_keys(k_head, positions[:query_pos],
                                    s["q_mean"], s["q_abs_mean"], omega, query_pos)
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

            del attn_weights
        return fn

    handles = []
    for li in attn_layer_indices:
        attn_mod = backbone.layers[li].self_attn
        handles.append(attn_mod.register_forward_pre_hook(_make_pre_hook(li), with_kwargs=True))
        handles.append(attn_mod.register_forward_hook(_make_post_hook(li)))

    # Forward pass — output_attentions=True so eager returns weights
    print("Forward pass ...", file=sys.stderr)
    t0 = time.time()
    with torch.no_grad():
        try:
            model(input_ids, output_attentions=True)
        except torch.cuda.OutOfMemoryError:
            print("  (OOM caught — hooks may have partial data)", file=sys.stderr)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s", file=sys.stderr)
    for h in handles: h.remove()

    if not recalls:
        print("\nERROR: No recall data collected. Post-hooks did not fire.", file=sys.stderr)
        print("Debugging: checking if output_attentions propagates...", file=sys.stderr)
        # Fallback: mniejszy max-length
        print("Try reducing --max-length (e.g. 512) or check accelerate hook interference.",
              file=sys.stderr)
        sys.exit(1)

    mean_recall = sum(recalls) / len(recalls)
    print(f"\nValidation at query position {query_pos}, budget={budget}:", file=sys.stderr)
    print(f"\n=== Validation Results ===", file=sys.stderr)
    print(f"Recall@{budget}: {mean_recall:.3f} ({len(recalls)} layer×kv_head pairs)", file=sys.stderr)
    print(f"  (fraction of true top-{budget} attended tokens captured by TriAttention)", file=sys.stderr)

    per_layer = {}
    for li in attn_layer_indices:
        if layer_recalls[li]:
            per_layer[li] = sum(layer_recalls[li]) / len(layer_recalls[li])

    best_layers = sorted(per_layer.items(), key=lambda x: -x[1])[:5]
    worst_layers = sorted(per_layer.items(), key=lambda x: x[1])[:5]
    print(f"\nBest layers:  {', '.join(f'L{l}={r:.3f}' for l,r in best_layers)}", file=sys.stderr)
    print(f"Worst layers: {', '.join(f'L{l}={r:.3f}' for l,r in worst_layers)}", file=sys.stderr)


if __name__ == "__main__":
    main()
