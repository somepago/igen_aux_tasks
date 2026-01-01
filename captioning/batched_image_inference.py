#!/usr/bin/env python3
"""
Batched Image Captioning Pipeline using Gemma 3

This script processes images from the ContraStyles dataset in batches,
generates captions using Google Gemma 3 model, and pushes annotated data
to HuggingFace intermittently.

Key features:
- Uses shard-level parquet files for metadata
- Stores only URLs (no actual images) in HF dataset
- Supports resume from checkpoint
- Periodic HF pushes
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from tqdm import tqdm
import pandas as pd

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from datasets import Dataset, Features, Value
import huggingface_hub
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    device: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    model_id: str = "google/gemma-3-27b-it"


@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    data_root: str = "/home/duality/projects/DiffSynth-Studio/data/contrastyles_full"
    images_dir: str = "images"
    hf_repo_name: str = "ContraStylesRecap"
    hf_username: Optional[str] = "somepago"
    push_every_n_samples: int = 500  # Push to HF every N samples
    max_images: Optional[int] = None  # Limit for testing
    checkpoint_dir: str = "./checkpoints"


class GemmaCaptioner:
    """Handles image captioning with Gemma 3 model"""

    CAPTION_PROMPT = """Describe this image in detail for training an image generation model. Include:
1. Main subjects and their appearance
2. Art style, medium, and technique (e.g., oil painting, digital art, photograph)
3. Color palette and lighting
4. Composition and perspective
5. Mood and atmosphere
6. Any text, signatures, or watermarks visible

Be specific and descriptive. Output only the description, no preamble."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the Gemma 3 model and processor"""
        logger.info(f"Loading model: {self.config.model_id}")

        self.processor = AutoProcessor.from_pretrained(self.config.model_id, use_fast=True)

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.config.model_id,
            device_map=self.config.device,
            dtype=self.config.torch_dtype,
        ).eval()

        logger.info("Model loaded successfully")

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for a single image"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.CAPTION_PROMPT}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
            )

        caption = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return caption.strip()

    def generate_captions_batch(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for a batch of images (sequential for now)"""
        captions = []
        for image in images:
            try:
                caption = self.generate_caption(image)
                captions.append(caption)
            except Exception as e:
                logger.error(f"Error generating caption: {e}")
                captions.append("")
        return captions


class ContraStylesProcessor:
    """Handles processing of ContraStyles dataset using shard parquets"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.images_root = Path(config.data_root) / config.images_dir
        self.shard_data: Dict[str, pd.DataFrame] = {}
        self._discover_shards()

    def _discover_shards(self):
        """Discover all shard directories and their parquet files"""
        self.shard_dirs = sorted([
            d for d in self.images_root.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        logger.info(f"Found {len(self.shard_dirs)} shards")

    def _load_shard_parquet(self, shard_name: str) -> pd.DataFrame:
        """Load parquet for a specific shard (cached)"""
        if shard_name not in self.shard_data:
            parquet_path = self.images_root / f"{shard_name}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                # Index by key for fast lookup
                df = df.set_index('key')
                self.shard_data[shard_name] = df
                logger.debug(f"Loaded parquet for shard {shard_name}")
            else:
                logger.warning(f"No parquet found for shard {shard_name}")
                self.shard_data[shard_name] = pd.DataFrame()
        return self.shard_data[shard_name]

    def get_image_paths(self) -> List[str]:
        """Get all image paths from the dataset"""
        image_paths = []

        for shard_dir in self.shard_dirs:
            jpg_files = list(shard_dir.glob("*.jpg"))
            image_paths.extend([str(p) for p in jpg_files])

        image_paths.sort()

        if self.config.max_images:
            image_paths = image_paths[:self.config.max_images]
            logger.info(f"Limited to {len(image_paths)} images for testing")

        logger.info(f"Found {len(image_paths)} images to process")
        return image_paths

    def get_metadata_for_image(self, image_path: str) -> Dict[str, Any]:
        """Get metadata for an image from the shard parquet"""
        path = Path(image_path)
        shard_name = path.parent.name
        key = path.stem

        df = self._load_shard_parquet(shard_name)

        if df.empty or key not in df.index:
            # Fallback to JSON file
            json_path = path.parent / f"{key}.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    return json.load(f)
            return {"key": key, "url": "", "caption": ""}

        row = df.loc[key]
        return {
            "key": key,
            "original_key": row.get("original_key", ""),  # Original key from HF dataset
            "url": row.get("url", ""),
            "caption": row.get("caption", ""),
            "width": row.get("width"),
            "height": row.get("height"),
            "original_width": row.get("original_width"),
            "original_height": row.get("original_height"),
            "sha256": row.get("sha256", ""),
            "md5": row.get("md5", ""),  # From preserved columns
            "tags": row.get("tags", ""),  # From preserved columns
            "merged_tags": row.get("merged_tags", ""),  # From preserved columns
        }


class HuggingFaceUploader:
    """Handles uploading to HuggingFace"""

    def __init__(self, repo_name: str, username: Optional[str] = None):
        self.repo_name = repo_name
        self.username = username or huggingface_hub.whoami()["name"]
        self.full_repo_name = f"{self.username}/{self.repo_name}"
        self.api = HfApi()
        self._ensure_repo()

    def _ensure_repo(self):
        """Create the repository if it doesn't exist"""
        try:
            self.api.create_repo(
                repo_id=self.full_repo_name,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            logger.info(f"Repository ready: {self.full_repo_name}")
        except Exception as e:
            logger.warning(f"Repo creation note: {e}")

    def push_parquet(self, df: pd.DataFrame, filename: str = "data.parquet"):
        """Push a parquet file to HuggingFace"""
        try:
            # Save locally first
            local_path = Path(filename)
            df.to_parquet(local_path, index=False)

            # Upload to HF
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"data/{filename}",
                repo_id=self.full_repo_name,
                repo_type="dataset",
            )
            logger.info(f"Pushed {len(df)} rows to {self.full_repo_name}")

            # Clean up local file
            local_path.unlink()
        except Exception as e:
            logger.error(f"Failed to push parquet: {e}")
            raise


def create_url_dataset(samples: List[Dict[str, Any]]) -> Dataset:
    """Create a HuggingFace dataset with URLs only (no images)"""
    features = Features({
        "key": Value("string"),
        "url": Value("string"),
        "original_caption": Value("string"),
        "vlm_caption": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
    })

    # Clean up samples for dataset creation
    clean_samples = []
    for s in samples:
        clean_samples.append({
            "key": s.get("key", ""),
            "url": s.get("url", ""),
            "original_caption": s.get("original_caption", ""),
            "vlm_caption": s.get("vlm_caption", ""),
            "width": int(s.get("width") or 0),
            "height": int(s.get("height") or 0),
        })

    return Dataset.from_list(clean_samples, features=features)


def load_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    """Load processing checkpoint if exists"""
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.json"
    processed_keys = set()

    # Load checkpoint JSON
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            processed_keys = set(data.get("processed_keys", []))

    # Also scan for existing parquet files to get processed keys
    checkpoint_dir_path = Path(checkpoint_dir)
    if checkpoint_dir_path.exists():
        for parquet_file in checkpoint_dir_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                if "key" in df.columns:
                    processed_keys.update(df["key"].tolist())
            except Exception as e:
                logger.warning(f"Could not read {parquet_file}: {e}")

    return {"processed_keys": processed_keys}


def save_checkpoint(checkpoint_dir: str, processed_keys: List[str], last_index: int, samples: List[Dict] = None):
    """Save processing checkpoint and intermediate results"""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save checkpoint JSON
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump({
            "processed_keys": list(processed_keys) if isinstance(processed_keys, set) else processed_keys,
            "last_index": last_index
        }, f)

    # Save intermediate results as parquet
    if samples:
        intermediate_path = Path(checkpoint_dir) / f"intermediate_{last_index:08d}.parquet"
        pd.DataFrame(samples).to_parquet(intermediate_path, index=False)
        logger.info(f"Saved intermediate results to {intermediate_path}")


def process_images(
    image_paths: List[str],
    captioner: GemmaCaptioner,
    processor: ContraStylesProcessor,
    config: DatasetConfig,
    uploader: Optional[HuggingFaceUploader] = None,
    already_processed: set = None,
) -> List[Dict[str, Any]]:
    """Process images and generate captions, skipping already-processed ones"""

    already_processed = already_processed or set()
    all_samples = []
    batch_samples = []  # Samples since last checkpoint
    processed_keys = set(already_processed)

    # Filter out already processed images
    images_to_process = []
    for image_path in image_paths:
        key = Path(image_path).stem
        if key not in already_processed:
            images_to_process.append(image_path)

    skipped = len(image_paths) - len(images_to_process)
    if skipped > 0:
        logger.info(f"Skipping {skipped} already-processed images")

    logger.info(f"Processing {len(images_to_process)} images")

    pbar = tqdm(images_to_process, desc="Captioning")

    for idx, image_path in enumerate(pbar):
        try:
            key = Path(image_path).stem

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Get metadata
            metadata = processor.get_metadata_for_image(image_path)

            # Generate VLM caption
            vlm_caption = captioner.generate_caption(image)

            # Create sample (URL only, no image data)
            sample = {
                "key": key,
                "original_key": metadata.get("original_key", ""),
                "url": metadata.get("url", ""),
                "original_caption": metadata.get("caption", ""),
                "vlm_caption": vlm_caption,
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "original_width": metadata.get("original_width"),
                "original_height": metadata.get("original_height"),
                "sha256": metadata.get("sha256", ""),
                "md5": metadata.get("md5", ""),
                "tags": metadata.get("tags", ""),
                "merged_tags": metadata.get("merged_tags", ""),
            }

            all_samples.append(sample)
            batch_samples.append(sample)
            processed_keys.add(key)

            pbar.set_description(f"Processed {key}")

            # Periodic checkpoint and HF push
            if len(batch_samples) >= config.push_every_n_samples:
                # Save intermediate checkpoint
                save_checkpoint(config.checkpoint_dir, processed_keys, idx, batch_samples)

                # Push to HF
                if uploader:
                    logger.info(f"Pushing {len(batch_samples)} samples to HuggingFace...")
                    df = pd.DataFrame(batch_samples)
                    batch_num = len(all_samples) // config.push_every_n_samples
                    uploader.push_parquet(df, f"batch_{batch_num:05d}.parquet")

                batch_samples = []  # Reset batch

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue

    # Save final batch if any remaining
    if batch_samples:
        save_checkpoint(config.checkpoint_dir, processed_keys, len(images_to_process), batch_samples)

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Batch image captioning with Gemma 3")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (currently sequential)")
    parser.add_argument("--push-every", type=int, default=500, help="Push to HF every N samples")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images for testing")
    parser.add_argument("--hf-repo", type=str, default="ContraStylesRecap", help="HuggingFace repo name")
    parser.add_argument("--hf-username", type=str, default="somepago", help="HuggingFace username")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        choices=["google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"],
                        help="Gemma 3 model to use")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HuggingFace")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens for caption")

    args = parser.parse_args()

    # Configuration
    batch_config = BatchConfig(
        batch_size=args.batch_size,
        model_id=args.model,
        max_new_tokens=args.max_tokens,
    )

    dataset_config = DatasetConfig(
        hf_repo_name=args.hf_repo,
        hf_username=args.hf_username,
        push_every_n_samples=args.push_every,
        max_images=args.max_images,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize components
    logger.info("Initializing captioner...")
    captioner = GemmaCaptioner(batch_config)

    logger.info("Initializing dataset processor...")
    processor = ContraStylesProcessor(dataset_config)

    uploader = None
    if not args.no_push:
        logger.info("Initializing HuggingFace uploader...")
        uploader = HuggingFaceUploader(dataset_config.hf_repo_name, dataset_config.hf_username)

    # Get image paths
    image_paths = processor.get_image_paths()

    # Check for resume - load already processed keys
    already_processed = set()
    if args.resume:
        checkpoint = load_checkpoint(dataset_config.checkpoint_dir)
        already_processed = checkpoint.get("processed_keys", set())
        if already_processed:
            logger.info(f"Found {len(already_processed)} already-processed images")

    # Process images
    logger.info(f"Starting processing...")
    all_samples = process_images(
        image_paths,
        captioner,
        processor,
        dataset_config,
        uploader,
        already_processed,
    )

    # Final push
    if uploader and all_samples:
        logger.info(f"Final push: {len(all_samples)} samples to HuggingFace...")
        df = pd.DataFrame(all_samples)
        uploader.push_parquet(df, "final.parquet")

    # Also save locally
    output_path = Path(dataset_config.checkpoint_dir) / "all_captions.parquet"
    pd.DataFrame(all_samples).to_parquet(output_path, index=False)
    logger.info(f"Saved all captions to {output_path}")

    logger.info(f"Processing complete! Processed {len(all_samples)} images")


if __name__ == "__main__":
    main()
