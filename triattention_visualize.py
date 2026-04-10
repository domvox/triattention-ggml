#!/usr/bin/env python3
"""Visualize TriAttention calibration stats — heatmap of per-head frequency scores."""
import argparse, struct, sys
import numpy as np

def read_stats(path):
    """Read binary stats file."""
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == 0x54524941, f"Bad magic: {magic:#x}"
        version = struct.unpack('<I', f.read(4))[0]
        n_layer = struct.unpack('<I', f.read(4))[0]
        n_head = struct.unpack('<I', f.read(4))[0]
        head_dim = struct.unpack('<I', f.read(4))[0]
        rope_theta = struct.unpack('<f', f.read(4))[0]
        
        print(f"Stats: v{version}, {n_layer} layers, {n_head} heads, dim={head_dim}, theta={rope_theta:.0f}")
        
        layers = []
        for il in range(n_layer):
            heads = []
            for ih in range(n_head):
                # Read per-head stats
                q_center = np.frombuffer(f.read(head_dim * 4), dtype=np.float32)
                k_center = np.frombuffer(f.read(head_dim * 4), dtype=np.float32)
                q_norm = struct.unpack('<f', f.read(4))[0]
                k_norm = struct.unpack('<f', f.read(4))[0]
                mrl = struct.unpack('<f', f.read(4))[0]
                heads.append({'q_norm': q_norm, 'k_norm': k_norm, 'mrl': mrl})
            layers.append(heads)
        
        return n_layer, n_head, head_dim, layers

def plot_heatmap(n_layer, n_head, layers, output):
    """Generate heatmap of attention frequency scores."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        # Fallback: text heatmap
        print(f"\n{'':>6}", end='')
        for h in range(n_head):
            print(f"H{h:>3}", end='')
        print()
        for il in range(n_layer):
            print(f"L{il:>4} ", end='')
            for ih in range(n_head):
                mrl = layers[il][ih]['mrl']
                print(f"{mrl:>4.1f}", end='')
            print()
        return
    
    # Build MRL matrix
    mrl_matrix = np.zeros((n_layer, n_head))
    knorm_matrix = np.zeros((n_layer, n_head))
    for il in range(n_layer):
        for ih in range(n_head):
            mrl_matrix[il][ih] = layers[il][ih]['mrl']
            knorm_matrix[il][ih] = layers[il][ih]['k_norm']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_layer * 0.3)))
    
    # MRL heatmap
    im1 = axes[0].imshow(mrl_matrix, aspect='auto', cmap='viridis')
    axes[0].set_title('Mean Resultant Length (MRL)\nHigher = more concentrated attention')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # K-norm heatmap
    im2 = axes[1].imshow(knorm_matrix, aspect='auto', cmap='magma')
    axes[1].set_title('K-vector Norm\nHigher = stronger key signal')
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Layer')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output}")

def main():
    p = argparse.ArgumentParser(description='Visualize TriAttention calibration stats')
    p.add_argument('stats', help='Path to .bin stats file')
    p.add_argument('--output', '-o', default='triattention_heatmap.png', help='Output image path')
    args = p.parse_args()
    
    n_layer, n_head, head_dim, layers = read_stats(args.stats)
    plot_heatmap(n_layer, n_head, layers, args.output)

if __name__ == '__main__':
    main()
