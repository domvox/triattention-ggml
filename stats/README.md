# Pre-built Calibration Stats

Ready-to-use stats files for TriAttention KV pruning. Calibrated on WikiText-2 with 4096 context.

## Usage

```bash
# Copy to your llama.cpp directory and use directly:
llama-server -m qwen3-8b.gguf -ngl 99 \
  --triattention stats/qwen3-8b.bin --tri-budget 75
```

## Available Models

| File | Model | Layers | Heads | head_dim | Size |
|---|---|---|---|---|---|
| `qwen3-1.7b.bin` | Qwen3-1.7B | 28 | 16 | 8 | 449K |
| `qwen3-8b.bin` | Qwen3-8B | 36 | 32 | 8 | 1.2M |
| `qwen3.5-27b.bin` | Qwen3.5-27B (hybrid) | 16 KV layers | 4 | 8 | 769K |

## Generate Your Own

```bash
./calibrate.sh <huggingface_model_name>
```
