# ContraStyles VLM Captioning

Generate detailed VLM captions for the ContraStyles dataset using Google's Gemma 3 multimodal model and push annotated metadata to HuggingFace.

## Features

- **VLM Captioning**: Uses Gemma 3 (4B/12B/27B) for detailed image descriptions
- **Full Metadata Preservation**: Keeps all original columns (key, md5, tags, merged_tags)
- **URL-Only Output**: Stores image URLs, not actual images (saves storage)
- **Resume Support**: Automatically skips already-processed images
- **Incremental HF Push**: Push to HuggingFace periodically (configurable)
- **Checkpoint System**: Local checkpoints for crash recovery
- **Self-Contained**: Includes data download, setup, and captioning scripts

## Quick Start

```bash
# 1. Setup environment
./setup.sh

# 2. Activate environment
source vlm-caption/bin/activate

# 3. Download dataset (if needed)
python download_data.py --output ./data/contrastyles_full

# 4. Test on small subset
./run_captioning.sh --max-images 10 --no-push

# 5. Run full captioning
./run_captioning.sh
```

## Files

| File | Description |
|------|-------------|
| `batched_image_inference.py` | Main captioning script |
| `download_data.py` | Download ContraStyles dataset with all metadata |
| `test_pipeline.py` | Test script for validation |
| `run_captioning.sh` | Convenience bash script |
| `setup.sh` | Environment setup script |

## Data Download

The download script fetches images from the ContraStyles dataset and **preserves all original metadata columns**.

### Download Commands

```bash
# Download to default location (./data/contrastyles_full)
python download_data.py

# Custom output directory
python download_data.py --output /path/to/data

# Fewer parallel downloads (for slower connections)
python download_data.py --threads 32

# Longer timeout for slow servers
python download_data.py --timeout 60
```

### What Gets Downloaded

```
data/contrastyles_full/
├── images/
│   ├── 00000/           # Shard directory
│   │   ├── 000000000.jpg
│   │   ├── 000000000.json
│   │   ├── 000000000.txt
│   │   └── ...
│   ├── 00000.parquet    # Shard metadata (includes original_key, md5, tags, merged_tags)
│   ├── 00001/
│   ├── 00001.parquet
│   └── ...
├── urls.parquet         # Input URLs with all metadata
├── full_metadata.parquet # Combined metadata from all shards
└── metadata.csv         # Simple CSV for training
```

### Preserved Metadata Columns

The download script preserves these columns from the original HF dataset:

| Column | Description |
|--------|-------------|
| `original_key` | Original key from tomg-group-umd/ContraStyles |
| `md5` | MD5 hash of original image |
| `tags` | Artist/style tags |
| `merged_tags` | Combined tags with styles |

**Note**:
- Full dataset is ~150-300GB
- ~395K images successfully download (some URLs fail)
- Download is resumable (uses `--incremental_mode`)

## Usage

### Using the Bash Script (Recommended)

```bash
# Test on 100 images, don't push to HF
./run_captioning.sh --max-images 100 --no-push

# Full run with default settings
./run_captioning.sh

# Use smaller model (4B)
./run_captioning.sh --model google/gemma-3-4b-it

# Resume interrupted run
./run_captioning.sh --resume

# Custom settings
./run_captioning.sh --batch-size 4 --push-every 1000 --model google/gemma-3-12b-it
```

### Using Python Directly

```bash
# Basic usage
python batched_image_inference.py

# With options
python batched_image_inference.py \
    --model google/gemma-3-27b-it \
    --max-images 1000 \
    --push-every 500 \
    --hf-repo ContraStylesRecap \
    --hf-username somepago

# Resume from checkpoint
python batched_image_inference.py --resume

# Local only (no HF push)
python batched_image_inference.py --no-push
```

### Testing

```bash
# Quick test (data loading only)
python test_pipeline.py --skip-model

# Test with 3 images
python test_pipeline.py --num-images 3

# Full end-to-end test with HF push
python test_pipeline.py --full --push --num-images 5
```

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `google/gemma-3-27b-it` | Model to use (4b/12b/27b) |
| `--batch-size` | `1` | Processing batch size |
| `--push-every` | `500` | Push to HF every N samples |
| `--max-images` | `None` | Limit images (for testing) |
| `--max-tokens` | `256` | Max tokens for caption |
| `--hf-repo` | `ContraStylesRecap` | HuggingFace repo name |
| `--hf-username` | `somepago` | HuggingFace username |
| `--checkpoint-dir` | `./checkpoints` | Checkpoint directory |
| `--resume` | `False` | Resume from checkpoint |
| `--no-push` | `False` | Skip HuggingFace upload |

### Model Options

| Model | VRAM Needed | Speed | Quality |
|-------|-------------|-------|---------|
| `google/gemma-3-4b-it` | ~8GB | Fast | Good |
| `google/gemma-3-12b-it` | ~24GB | Medium | Better |
| `google/gemma-3-27b-it` | ~54GB | Slow | Best |

## Output Format

The output parquet files contain all original metadata plus VLM captions:

| Column | Type | Description |
|--------|------|-------------|
| `key` | string | Local image identifier (from img2dataset) |
| `original_key` | string | Original key from ContraStyles |
| `url` | string | Original image URL |
| `original_caption` | string | Caption from ContraStyles |
| `vlm_caption` | string | Generated VLM caption |
| `width` | int | Image width |
| `height` | int | Image height |
| `original_width` | int | Original image width |
| `original_height` | int | Original image height |
| `sha256` | string | SHA256 hash of downloaded image |
| `md5` | string | MD5 hash from original dataset |
| `tags` | string | Artist/style tags |
| `merged_tags` | string | Combined tags |

## Resume & Checkpointing

The system automatically:
1. Saves checkpoints after every `--push-every` samples
2. Scans existing parquet files to find already-processed images
3. Skips already-processed images on restart

To resume after interruption:
```bash
./run_captioning.sh --resume
```

## HuggingFace Upload

Dataset is pushed to: `somepago/ContraStylesRecap`

The script:
- Creates the repo if it doesn't exist
- Pushes incremental parquet files to `data/` folder
- Final results are also saved locally to `checkpoints/all_captions.parquet`

## Hardware Requirements

| Model | Minimum VRAM | Recommended VRAM |
|-------|--------------|------------------|
| 4B | 8GB | 16GB |
| 12B | 24GB | 32GB |
| 27B | 54GB | 80GB |

For machines with less VRAM, the model will use CPU offloading (slower).

## Time Estimates

Approximate time for full dataset (395K images):

| Model | Per Image | Total (single GPU) |
|-------|-----------|-------------------|
| 4B | ~5 sec | ~23 days |
| 12B | ~10 sec | ~46 days |
| 27B | ~15 sec | ~69 days |

To speed up: run on multiple machines with different shard ranges.

## Dataset Information

- **Source**: [tomg-group-umd/ContraStyles](https://huggingface.co/datasets/tomg-group-umd/ContraStyles)
- **Size**: ~395K images (after download failures)
- **Shards**: 50 directories (00000-00049)
- **Organization**: Each shard has images, JSON metadata, and parquet files
