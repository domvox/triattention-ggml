"""Shared TriAttention scoring and I/O utilities."""
from __future__ import annotations
import struct
from pathlib import Path
from typing import Dict, Tuple
import torch

MAGIC = 0x54524941  # "TRIA"
HEADER_SIZE = 64


def _to_complex(t: torch.Tensor) -> torch.Tensor:
    t = t.to(dtype=torch.float32)
    fc = t.shape[-1] // 2
    return torch.complex(t[..., :fc].contiguous(), t[..., fc:].contiguous())


def load_stats(path: Path | str, device: torch.device) -> dict:
    """Load TRIA v1/v2 binary stats file."""
    with open(path, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        assert magic == MAGIC, f"Bad magic: {magic:#x}"
        assert ver in (1, 2), f"Unsupported version: {ver}"
        nl, nh, nkv, hd, fc = struct.unpack("<5I", f.read(20))
        rt, asc = struct.unpack("<2f", f.read(8))
        f.read(HEADER_SIZE - 9 * 4)
        if ver >= 2:
            layer_budget_scales = list(struct.unpack(f"<{nl}f", f.read(nl * 4)))
        else:
            layer_budget_scales = [1.0] * nl
        stats: Dict[Tuple[int, int], dict] = {}
        for li in range(nl):
            for hi in range(nh):
                qmr = torch.tensor(struct.unpack(f"<{fc}f", f.read(fc * 4)), device=device)
                qmi = torch.tensor(struct.unpack(f"<{fc}f", f.read(fc * 4)), device=device)
                qam = torch.tensor(struct.unpack(f"<{fc}f", f.read(fc * 4)), device=device)
                f.read(fc * 4)  # skip mrl
                stats[(li, hi)] = {"q_mean": torch.complex(qmr, qmi), "q_abs_mean": qam}
    return {
        "num_layers": nl, "num_heads": nh, "num_kv_heads": nkv,
        "head_dim": hd, "freq_count": fc, "rope_theta": rt,
        "layer_budget_scales": layer_budget_scales, "stats": stats,
    }


def score_keys(
    k_pre: torch.Tensor,        # [seq_len, head_dim]
    key_pos: torch.Tensor,       # [seq_len]
    q_mean: torch.Tensor,        # [fc] complex
    q_abs_mean: torch.Tensor,    # [fc]
    omega: torch.Tensor,         # [fc]
    cur_pos: int,
) -> torch.Tensor:
    """TriAttention importance score per key (paper eq 6-13)."""
    k_c = _to_complex(k_pre)
    qma = q_mean.abs()
    ka = k_c.abs()
    # Phase: angle(q_mean) - angle(k) via conjugate multiply
    rel = q_mean.unsqueeze(0) * torch.conj(k_c)
    phi = torch.atan2(rel.imag, rel.real)  # NOTE: C++ can bypass atan2 with trig identities
    amp = qma.unsqueeze(0) * ka
    # Norm extra term: (E[||q||] - ||E[q]||) * ||k|| — eq 8-10
    extra = ((q_abs_mean - qma).unsqueeze(0) * ka).sum(dim=1)
    # Geometric future offsets: 1, 2, 4, ..., 2^16
    offsets = torch.tensor([2.0 ** i for i in range(17)], device=k_pre.device)
    bd = cur_pos - key_pos.float()
    dg = bd.unsqueeze(1) + offsets.unsqueeze(0)  # [seq, 17] = base_delta + future_offset
    # Trig score: sum_f amp_f * cos(delta * omega_f + phi_f)
    phase = dg.unsqueeze(2) * omega.unsqueeze(0).unsqueeze(0) + phi.unsqueeze(1)
    trig = (amp.unsqueeze(1) * torch.cos(phase)).sum(dim=2)
    return trig.mean(dim=1) + extra


def build_omega(rope_theta: float, head_dim: int, freq_count: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([rope_theta ** (-2.0 * i / head_dim) for i in range(freq_count)],
                        device=device, dtype=torch.float32)


def compute_keep_sets(
    captured_k: dict, positions: torch.Tensor, cal: dict,
    omega: torch.Tensor, trigger_pos: int, budget: int,
    nkv: int, gqa: int, nl: int,
) -> Dict[Tuple[int, int], set]:
    """Returns (layer, kv_head) -> set of kept position indices."""
    keep = {}
    for li in range(nl):
        k = captured_k.get(li)
        if k is None:
            continue
        layer_scale = cal["layer_budget_scales"][li]
        layer_budget = max(1, int(budget * layer_scale))
        for kvi in range(nkv):
            k_head = k[0, kvi, :trigger_pos]
            kp = positions[:trigger_pos]
            scores_list = []
            for g in range(gqa):
                ah = kvi * gqa + g
                s = cal["stats"].get((li, ah))
                if s is None:
                    continue
                scores_list.append(score_keys(k_head, kp, s["q_mean"], s["q_abs_mean"], omega, trigger_pos))
            if not scores_list:
                continue
            stacked = torch.stack(scores_list)
            mean = stacked.mean(dim=1, keepdim=True)
            std = stacked.std(dim=1, keepdim=True).clamp_min(1e-6)
            norm = (stacked - mean) / std
            final = norm.max(dim=0).values
            topk = torch.topk(final, k=min(layer_budget, trigger_pos)).indices
            keep[(li, kvi)] = set(topk.cpu().tolist())
    return keep
