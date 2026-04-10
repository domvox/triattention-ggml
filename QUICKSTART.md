# Quick Start: TriAttention + TurboQuant

## 1. Build llama.cpp with TurboQuant

```bash
git clone https://github.com/domvox/llama.cpp-turboquant-hip
cd llama.cpp-turboquant-hip
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release  # or -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

## 2. Download pre-built stats

```bash
# From this repo:
wget https://github.com/domvox/triattention-ggml/raw/master/stats/qwen3-8b.bin
wget https://github.com/domvox/triattention-ggml/raw/master/stats/qwen3.5-27b.bin
```

Or generate your own: `./calibrate.sh Qwen/Qwen3-8B`

## 3. Run

```bash
# TurboQuant only (5x KV compression):
./build/bin/llama-server -m model.gguf -ngl 99 -ctk turbo -ctv turbo

# TurboQuant + TriAttention (~6.8x KV compression):
./build/bin/llama-server -m model.gguf -ngl 99 -ctk turbo -ctv turbo \
  --triattention qwen3-8b.bin --tri-budget 75 --tri-window 512

# Aggressive (turbo2 + TriAttention, ~10x compression):
./build/bin/llama-server -m model.gguf -ngl 99 -ctk turbo2 -ctv turbo2 \
  --triattention qwen3-8b.bin --tri-budget 50 --tri-window 256
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--triattention PATH` | disabled | Path to calibration stats |
| `--tri-budget N` | 0 | Retention % (75 = keep 75%, prune 25%) |
| `--tri-window N` | 512 | Recent tokens always kept |
| `--tri-interval N` | 128 | Score/prune every N tokens |
| `--tri-sink N` | 0 | First N tokens always kept (attention sinks) |

## Model Compatibility

| Model | Stats file | Tested |
|---|---|---|
| Qwen3-8B | `qwen3-8b.bin` | ✅ PPL validated |
| Qwen3.5-27B | `qwen3.5-27b.bin` | ✅ PPL validated |
| Qwen3-1.7B | `qwen3-1.7b.bin` | ✅ loads |
| Llama 3, Mistral | generate with `calibrate.sh` | untested |
