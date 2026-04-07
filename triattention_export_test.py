#!/usr/bin/env python3
"""Export test vectors for C reference scoring validation."""
import struct, sys
import torch
from triattention_common import load_stats, score_keys, build_omega

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <stats.bin> <output.bin> [layer] [head]", file=sys.stderr)
        sys.exit(1)

    stats_path, out_path = sys.argv[1], sys.argv[2]
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    head = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    cal = load_stats(stats_path, torch.device("cpu"))
    fc = cal["freq_count"]
    hd = cal["head_dim"]
    omega = build_omega(cal["rope_theta"], hd, fc, torch.device("cpu"))

    s = cal["stats"][(layer, head)]
    q_mean = s["q_mean"]       # [fc] complex
    q_abs_mean = s["q_abs_mean"]  # [fc]

    # Generate synthetic keys (random pre-RoPE K)
    seq_len = 128
    cur_pos = seq_len
    torch.manual_seed(42)
    k_pre = torch.randn(seq_len, hd)  # [seq, head_dim]
    key_pos = torch.arange(seq_len)

    # Compute Python scores
    scores = score_keys(k_pre, key_pos, q_mean, q_abs_mean, omega, cur_pos)

    # Split k into real/imag halves
    k_real = k_pre[:, :fc].contiguous()
    k_imag = k_pre[:, fc:].contiguous()

    # Write binary
    with open(out_path, "wb") as f:
        f.write(struct.pack("<iii", fc, seq_len, cur_pos))
        f.write(struct.pack(f"<{fc}f", *q_mean.real.tolist()))
        f.write(struct.pack(f"<{fc}f", *q_mean.imag.tolist()))
        f.write(struct.pack(f"<{fc}f", *q_abs_mean.tolist()))
        f.write(struct.pack(f"<{fc}f", *omega.tolist()))
        f.write(struct.pack(f"<{seq_len}i", *key_pos.tolist()))
        f.write(struct.pack(f"<{seq_len*fc}f", *k_real.flatten().tolist()))
        f.write(struct.pack(f"<{seq_len*fc}f", *k_imag.flatten().tolist()))
        f.write(struct.pack(f"<{seq_len}f", *scores.tolist()))

    print(f"Wrote {out_path}: fc={fc} seq={seq_len} cur_pos={cur_pos}", file=sys.stderr)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]", file=sys.stderr)

if __name__ == "__main__":
    main()
