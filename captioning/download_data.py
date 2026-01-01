#!/usr/bin/env python3
"""
Download full ContraStyles dataset (500k images) using img2dataset.

This downloads all images locally and preserves ALL original metadata columns
including: key, md5, tags, merged_tags.

Estimated size: ~150-300GB depending on image sizes.

Usage:
    python download_data.py                     # Download to default location
    python download_data.py --output ./data     # Custom output directory
    python download_data.py --threads 32        # Fewer parallel downloads
"""

import os
import argparse
import subprocess
import glob
import json
from datasets import load_dataset
import pandas as pd


def download_dataset(output_dir: str, threads: int = 64, timeout: int = 30):
    """Download the ContraStyles dataset with all metadata preserved."""

    parquet_path = os.path.join(output_dir, "urls.parquet")
    images_dir = os.path.join(output_dir, "images")
    metadata_path = os.path.join(output_dir, "metadata.csv")
    full_metadata_path = os.path.join(output_dir, "full_metadata.parquet")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("Step 1: Loading dataset metadata from HuggingFace...")
    ds = load_dataset("tomg-group-umd/ContraStyles", split="train")
    print(f"  Found {len(ds)} samples")

    print("\nStep 2: Creating URL parquet with ALL metadata columns...")
    # Include all original columns for preservation
    df = pd.DataFrame({
        "url": ds["url"],
        "caption": ds["caption"],
        "original_key": ds["key"],  # Renamed to avoid confusion with img2dataset key
        "md5": ds["md5"],
        "tags": ds["tags"],
        "merged_tags": ds["merged_tags"],
    })
    df.to_parquet(parquet_path)
    print(f"  Saved to {parquet_path}")
    print(f"  Columns: {df.columns.tolist()}")

    print("\nStep 3: Downloading images with img2dataset...")
    print("  This will take a while for 500k images...")
    print("  Additional columns (original_key, md5, tags, merged_tags) will be preserved.")

    # img2dataset command with additional columns preserved
    cmd = [
        "img2dataset",
        "--url_list", parquet_path,
        "--output_folder", images_dir,
        "--output_format", "files",
        "--input_format", "parquet",
        "--url_col", "url",
        "--caption_col", "caption",
        "--save_additional_columns", '["original_key", "md5", "tags", "merged_tags"]',
        "--disable_all_reencoding", "True",
        "--thread_count", str(threads),
        "--retries", "3",
        "--incremental_mode", "incremental",
        "--timeout", str(timeout),
    ]

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print("\nStep 4: Creating unified metadata parquet...")
    create_unified_metadata(output_dir, images_dir, full_metadata_path)

    print("\nStep 5: Generating metadata.csv for training...")
    generate_metadata(output_dir, images_dir, metadata_path)

    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"  - Images: {images_dir}")
    print(f"  - Full metadata: {full_metadata_path}")
    print(f"  - Training metadata: {metadata_path}")


def create_unified_metadata(output_dir: str, images_dir: str, output_path: str):
    """Create a unified parquet with all metadata from shard parquets."""

    shard_parquets = sorted(glob.glob(os.path.join(images_dir, "*.parquet")))
    print(f"  Found {len(shard_parquets)} shard parquets")

    if not shard_parquets:
        print("  No shard parquets found, skipping unified metadata")
        return

    # Combine all shard parquets
    dfs = []
    for pq in shard_parquets:
        try:
            df = pd.read_parquet(pq)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {pq}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output_path, index=False)
        print(f"  Saved unified metadata: {len(combined)} rows")
        print(f"  Columns: {combined.columns.tolist()}")


def generate_metadata(output_dir: str, images_dir: str, metadata_path: str):
    """Generate metadata.csv from downloaded images."""

    # Find all downloaded images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))

    print(f"  Found {len(image_files)} downloaded images")

    # Load metadata from shard parquets (includes original_key now)
    shard_parquets = sorted(glob.glob(os.path.join(images_dir, "*.parquet")))

    key_to_data = {}
    for pq in shard_parquets:
        try:
            df = pd.read_parquet(pq)
            for _, row in df.iterrows():
                key_to_data[row["key"]] = {
                    "caption": row.get("caption", ""),
                    "original_key": row.get("original_key", ""),
                    "tags": row.get("tags", ""),
                }
        except Exception as e:
            print(f"  Warning: Could not read {pq}: {e}")

    # Match images to captions
    records = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        key = os.path.splitext(basename)[0]
        rel_path = os.path.relpath(img_path, output_dir)

        data = key_to_data.get(key, {})
        caption = data.get("caption", "")
        if caption:
            records.append({
                "video": rel_path,
                "prompt": caption,
            })

    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)
    print(f"  Saved {len(df)} entries to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Download ContraStyles dataset")
    parser.add_argument("--output", type=str, default="./data/contrastyles_full",
                        help="Output directory for dataset")
    parser.add_argument("--threads", type=int, default=64,
                        help="Number of parallel download threads")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout for each download (seconds)")

    args = parser.parse_args()
    download_dataset(args.output, args.threads, args.timeout)


if __name__ == "__main__":
    main()
