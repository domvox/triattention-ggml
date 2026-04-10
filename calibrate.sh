#!/bin/bash
# TriAttention Calibration — Quick Start
# Generates stats.bin for use with llama.cpp --triattention flag
#
# Requirements: pip install torch transformers
# Usage: ./calibrate.sh <model_name_or_path> [output_path]
#
# Examples:
#   ./calibrate.sh Qwen/Qwen3-8B
#   ./calibrate.sh /path/to/local/model ./my_stats.bin

set -e

MODEL="${1:?Usage: $0 <model_name_or_path> [output_path]}"
OUTPUT="${2:-triattention_stats.bin}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check dependencies
python3 -c "import torch, transformers" 2>/dev/null || {
    echo "Missing dependencies. Install with:"
    echo "  pip install torch transformers"
    exit 1
}

# Check for wikitext calibration data
WIKITEXT=""
for p in \
    "$SCRIPT_DIR/../llama.cpp/wikitext-2-raw/wiki.test.raw" \
    "$HOME/llama.cpp/wikitext-2-raw/wiki.test.raw" \
    "./wikitext-2-raw/wiki.test.raw"; do
    if [ -f "$p" ]; then WIKITEXT="$p"; break; fi
done

if [ -z "$WIKITEXT" ]; then
    echo "WikiText-2 not found. Downloading..."
    mkdir -p wikitext-2-raw
    curl -sL "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" -o /tmp/wikitext.zip
    unzip -qo /tmp/wikitext.zip -d .
    WIKITEXT="wikitext-2-raw/wiki.test.raw"
fi

echo "Model:  $MODEL"
echo "Input:  $WIKITEXT"
echo "Output: $OUTPUT"
echo ""

HIP_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/triattention_calibrate.py" \
    --model "$MODEL" \
    --input "$WIKITEXT" \
    --output "$OUTPUT" \
    --max-length 4096 \
    --device cuda

echo ""
echo "Done! Use with llama.cpp:"
echo "  llama-server -m model.gguf --triattention $OUTPUT --tri-budget 75"
