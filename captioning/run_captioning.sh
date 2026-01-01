#!/bin/bash
#
# ContraStyles VLM Captioning Script
# Generates captions for images using Gemma 3 VLM
#

# =============================================================================
# Configuration - Modify these variables as needed
# =============================================================================

# Model configuration
MODEL="google/gemma-3-27b-it"  # Options: google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it
MAX_TOKENS=256                  # Maximum tokens for caption generation

# HuggingFace configuration
HF_REPO="ContraStylesRecap"
HF_USERNAME="somepago"

# Processing configuration
BATCH_SIZE=1                    # Increase for faster processing on larger GPUs
PUSH_EVERY=500                  # Push to HuggingFace every N samples
MAX_IMAGES=""                   # Leave empty for full dataset, or set a number for testing

# Checkpoint configuration
CHECKPOINT_DIR="./checkpoints"

# Flags
RESUME=false                    # Set to true to resume from checkpoint
NO_PUSH=false                   # Set to true to skip HuggingFace upload

# =============================================================================
# Parse command line arguments (override defaults)
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --push-every)
            PUSH_EVERY="$2"
            shift 2
            ;;
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --hf-repo)
            HF_REPO="$2"
            shift 2
            ;;
        --hf-username)
            HF_USERNAME="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL         Gemma model to use (default: google/gemma-3-27b-it)"
            echo "  --batch-size N        Batch size for processing (default: 1)"
            echo "  --push-every N        Push to HF every N samples (default: 500)"
            echo "  --max-images N        Limit number of images (default: all)"
            echo "  --hf-repo REPO        HuggingFace repo name (default: ContraStylesRecap)"
            echo "  --hf-username USER    HuggingFace username (default: somepago)"
            echo "  --checkpoint-dir DIR  Checkpoint directory (default: ./checkpoints)"
            echo "  --max-tokens N        Max tokens for caption (default: 256)"
            echo "  --resume              Resume from checkpoint"
            echo "  --no-push             Don't push to HuggingFace"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 --model google/gemma-3-4b-it       # Use smaller model"
            echo "  $0 --max-images 100 --no-push         # Test on 100 images"
            echo "  $0 --resume                           # Resume interrupted job"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Build command
# =============================================================================

CMD="python batched_image_inference.py"
CMD+=" --model $MODEL"
CMD+=" --batch-size $BATCH_SIZE"
CMD+=" --push-every $PUSH_EVERY"
CMD+=" --hf-repo $HF_REPO"
CMD+=" --hf-username $HF_USERNAME"
CMD+=" --checkpoint-dir $CHECKPOINT_DIR"
CMD+=" --max-tokens $MAX_TOKENS"

if [ -n "$MAX_IMAGES" ]; then
    CMD+=" --max-images $MAX_IMAGES"
fi

if [ "$RESUME" = true ]; then
    CMD+=" --resume"
fi

if [ "$NO_PUSH" = true ]; then
    CMD+=" --no-push"
fi

# =============================================================================
# Run
# =============================================================================

echo "=============================================="
echo "ContraStyles VLM Captioning"
echo "=============================================="
echo "Model:          $MODEL"
echo "Batch size:     $BATCH_SIZE"
echo "Push every:     $PUSH_EVERY samples"
echo "Max images:     ${MAX_IMAGES:-all}"
echo "HF Repo:        $HF_USERNAME/$HF_REPO"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Resume:         $RESUME"
echo "Push to HF:     $([ "$NO_PUSH" = true ] && echo "no" || echo "yes")"
echo "=============================================="
echo ""
echo "Running: $CMD"
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Run the captioning script
$CMD
