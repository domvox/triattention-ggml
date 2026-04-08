#!/usr/bin/env python3
"""Export GQA-aggregated test vectors for C library validation."""
import struct, sys
import torch
from triattention_common import load_stats, score_keys, build_omega

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <stats.bin> <output.bin> [layer] [kv_head]", file=sys.stderr)
        sys.exit(1)

    stats_path, out_path = sys.argv[1], sys.argv[2]
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    kv_head = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    cal = load_stats(stats_path, torch.device("cpu"))
    fc, hd = cal["freq_count"], cal["head_dim"]
    nkv, nh = cal["num_kv_heads"], cal["num_heads"]
    gqa = nh // nkv
    omega = build_omega(cal["rope_theta"], hd, fc, torch.device("cpu"))

    seq_len, cur_pos = 128, 128
    torch.manual_seed(42)
    k_pre = torch.randn(seq_len, hd)
    key_pos = torch.arange(seq_len)

    # GQA: z-normalize per query head, max-aggregate
    all_scores = []
    for g in range(gqa):
        ah = kv_head * gqa + g
        s = cal["stats"][(layer, ah)]
        sc = score_keys(k_pre, key_pos, s["q_mean"], s["q_abs_mean"], omega, cur_pos)
        all_scores.append(sc)
    stacked = torch.stack(all_scores)
    mean = stacked.mean(dim=1, keepdim=True)
    std = stacked.std(dim=1, keepdim=True).clamp_min(1e-6)
    norm = (stacked - mean) / std
    final = norm.max(dim=0).values

    k_real = k_pre[:, :fc].contiguous()
    k_imag = k_pre[:, fc:].contiguous()

    with open(out_path, "wb") as f:
        f.write(struct.pack("<iiii", fc, seq_len, cur_pos, gqa))
        f.write(struct.pack(f"<{seq_len}i", *key_pos.tolist()))
        f.write(struct.pack(f"<{seq_len*fc}f", *k_real.flatten().tolist()))
        f.write(struct.pack(f"<{seq_len*fc}f", *k_imag.flatten().tolist()))
        f.write(struct.pack(f"<{seq_len}f", *final.tolist()))

    print(f"GQA test: L{layer} KV{kv_head} gqa={gqa} seq={seq_len}", file=sys.stderr)
    print(f"  Score range: [{final.min():.4f}, {final.max():.4f}]", file=sys.stderr)

if __name__ == "__main__":
    main()
