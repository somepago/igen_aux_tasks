#!/bin/bash
#
# ContraStyles VLM Captioning - Setup Script
#
# This script sets up the environment and optionally downloads the dataset.
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

VENV_NAME="${VENV_NAME:-vlm-caption}"
PYTHON_VERSION="${PYTHON_VERSION:-python3}"
DATA_DIR="${DATA_DIR:-./data/contrastyles_full}"
SKIP_DATA="${SKIP_DATA:-false}"

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            VENV_NAME="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --venv NAME       Virtual environment name (default: vlm-caption)"
            echo "  --data-dir DIR    Directory for dataset (default: ./data/contrastyles_full)"
            echo "  --skip-data       Skip dataset download"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

echo "=============================================="
echo "ContraStyles VLM Captioning - Setup"
echo "=============================================="
echo ""

# Check Python
echo "Checking Python..."
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "Error: $PYTHON_VERSION not found"
    exit 1
fi
$PYTHON_VERSION --version

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up virtual environment: $VENV_NAME"
if [ ! -d "$VENV_NAME" ]; then
    $PYTHON_VERSION -m venv "$VENV_NAME"
    echo "  Created new virtual environment"
else
    echo "  Using existing virtual environment"
fi

# Activate virtual environment
source "$VENV_NAME/bin/activate"
echo "  Activated: $(which python)"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt -q
echo "  Done!"

# Login to HuggingFace
echo ""
echo "Checking HuggingFace login..."
if python -c "import huggingface_hub; huggingface_hub.whoami()" 2>/dev/null; then
    HF_USER=$(python -c "import huggingface_hub; print(huggingface_hub.whoami()['name'])")
    echo "  Logged in as: $HF_USER"
else
    echo "  Not logged in. Run 'huggingface-cli login' to enable HF uploads."
fi

# Download dataset
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "=============================================="
    echo "Dataset Download"
    echo "=============================================="
    echo ""

    if [ -d "$DATA_DIR/images" ]; then
        echo "Dataset directory exists: $DATA_DIR"
        read -p "Download anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping download."
            SKIP_DATA=true
        fi
    fi

    if [ "$SKIP_DATA" = false ]; then
        echo "Downloading dataset to: $DATA_DIR"
        echo "This will take a while for 500k images..."
        python download_data.py --output "$DATA_DIR"
    fi
fi

# Summary
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To run captioning:"
echo "  ./run_captioning.sh --max-images 100  # Test on 100 images"
echo "  ./run_captioning.sh                   # Full dataset"
echo ""
echo "To resume an interrupted job:"
echo "  ./run_captioning.sh --resume"
echo ""
