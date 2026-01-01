#!/usr/bin/env python3
"""
Download full ContraStyles dataset (500k images) using img2dataset.

This downloads all images locally and preserves ALL original metadata columns
including: key, md5, tags, merged_tags.

After downloading:
1. Validates all images and removes corrupt files
2. Generates metadata.csv with only valid images

Estimated size: ~150-300GB depending on image sizes.

Usage:
    # Download dataset (includes validation + cleanup)
    python download_data.py --output ./data/contrastyles_full

    # Custom settings
    python download_data.py --output ./data --threads 32 --timeout 60

    # Regenerate metadata only (after download complete)
    python download_data.py --output ./data --regenerate-metadata

    # Regenerate with validation (filter corrupt, don't delete)
    python download_data.py --output ./data --regenerate-metadata --validate

    # Regenerate with validation + cleanup (delete corrupt files)
    python download_data.py --output ./data --regenerate-metadata --validate --clean
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

    print("\nStep 4: Validating images and removing corrupt files...")
    valid_keys = validate_and_clean_images(images_dir)

    print("\nStep 5: Creating unified metadata parquet...")
    create_unified_metadata(output_dir, images_dir, full_metadata_path)

    print("\nStep 6: Generating metadata.csv for training...")
    generate_metadata(output_dir, images_dir, metadata_path, valid_keys=valid_keys)

    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"  - Images: {images_dir}")
    print(f"  - Full metadata: {full_metadata_path}")
    print(f"  - Training metadata: {metadata_path}")


def validate_and_clean_images(images_dir: str) -> set:
    """Validate all downloaded images and remove corrupt ones.

    Returns:
        Set of valid image keys (filename without extension)
    """
    from PIL import Image

    # Find all image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))

    print(f"  Found {len(image_files)} image files to validate")

    valid_keys = set()
    corrupt_count = 0

    for i, img_path in enumerate(image_files):
        if i % 50000 == 0 and i > 0:
            print(f"  Validated {i}/{len(image_files)}... valid={len(valid_keys)}, corrupt={corrupt_count}")

        try:
            with Image.open(img_path) as img:
                img.load()  # Force full decode
                img.convert('RGB')  # Ensure can convert to RGB

            # Image is valid
            key = os.path.splitext(os.path.basename(img_path))[0]
            valid_keys.add(key)
        except Exception:
            # Remove corrupt image and its associated files
            corrupt_count += 1
            base_path = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.txt', '.json']:
                path = base_path + ext
                if os.path.exists(path):
                    os.remove(path)

    print(f"  Validation complete: {len(valid_keys)} valid, {corrupt_count} corrupt (removed)")
    return valid_keys


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


def generate_metadata(output_dir: str, images_dir: str, metadata_path: str,
                      valid_keys: set = None, validate_images: bool = False):
    """Generate metadata.csv from downloaded images.

    Args:
        output_dir: Base output directory
        images_dir: Directory containing downloaded images
        metadata_path: Path to save metadata.csv
        valid_keys: Pre-validated set of image keys (skips validation if provided)
        validate_images: If True and valid_keys not provided, verify each image can be loaded
    """
    from PIL import Image

    # Load metadata from shard parquets - only successful downloads
    shard_parquets = sorted(glob.glob(os.path.join(images_dir, "*.parquet")))
    print(f"  Found {len(shard_parquets)} shard parquets")

    # Combine all shards efficiently
    dfs = []
    for pq in shard_parquets:
        try:
            df = pd.read_parquet(pq)
            # Only keep successful downloads
            df = df[df['status'] == 'success'][['key', 'caption']]
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {pq}: {e}")

    if not dfs:
        print("  No shard parquets found!")
        return

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined)} successful downloads from parquets")

    # Build key->caption dict
    key_to_caption = dict(zip(combined['key'], combined['caption']))

    # If valid_keys provided, use them directly
    if valid_keys is not None:
        print(f"  Using {len(valid_keys)} pre-validated image keys")
        records = []
        missing = 0
        for key in valid_keys:
            caption = key_to_caption.get(key)
            if not caption:
                missing += 1
                continue
            # Find the image file for this key
            for ext in ['jpg', 'jpeg', 'png', 'webp']:
                # img2dataset uses shard folders like 00000, 00001, etc.
                shard = key[:5]
                img_path = os.path.join(images_dir, shard, f"{key}.{ext}")
                if os.path.exists(img_path):
                    rel_path = os.path.relpath(img_path, output_dir)
                    records.append({"video": rel_path, "prompt": caption})
                    break
        print(f"  Matched {len(records)} images with captions")
        if missing > 0:
            print(f"  Skipped {missing} images without captions in parquet")
    else:
        # Find all downloaded images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
        print(f"  Found {len(image_files)} downloaded image files")

        # Match images to captions
        records = []
        missing = 0
        corrupt = 0

        for i, img_path in enumerate(image_files):
            if i % 50000 == 0 and i > 0:
                print(f"  Processed {i}/{len(image_files)}... valid={len(records)}")

            basename = os.path.basename(img_path)
            key = os.path.splitext(basename)[0]

            caption = key_to_caption.get(key)
            if not caption:
                missing += 1
                continue

            # Optionally validate image integrity
            if validate_images:
                try:
                    with Image.open(img_path) as img:
                        img.load()
                        img.convert('RGB')
                except Exception:
                    corrupt += 1
                    continue

            rel_path = os.path.relpath(img_path, output_dir)
            records.append({"video": rel_path, "prompt": caption})

        if missing > 0:
            print(f"  Skipped {missing} images without captions")
        if corrupt > 0:
            print(f"  Skipped {corrupt} corrupt images")

    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)
    print(f"  Saved {len(df)} entries to {metadata_path}")


def regenerate_metadata_only(output_dir: str, validate: bool = False, clean: bool = False):
    """Regenerate metadata.csv without re-downloading images.

    Args:
        output_dir: Dataset directory
        validate: Validate image integrity
        clean: Remove corrupt images (requires validate=True)
    """
    images_dir = os.path.join(output_dir, "images")
    metadata_path = os.path.join(output_dir, "metadata.csv")

    print(f"Regenerating metadata for {output_dir}...")

    valid_keys = None
    if validate:
        print("\nValidating images...")
        if clean:
            valid_keys = validate_and_clean_images(images_dir)
        else:
            # Validate but don't remove corrupt files
            from PIL import Image
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                image_files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))

            print(f"  Found {len(image_files)} image files to validate")
            valid_keys = set()
            corrupt_count = 0

            for i, img_path in enumerate(image_files):
                if i % 50000 == 0 and i > 0:
                    print(f"  Validated {i}/{len(image_files)}... valid={len(valid_keys)}, corrupt={corrupt_count}")
                try:
                    with Image.open(img_path) as img:
                        img.load()
                        img.convert('RGB')
                    key = os.path.splitext(os.path.basename(img_path))[0]
                    valid_keys.add(key)
                except Exception:
                    corrupt_count += 1

            print(f"  Validation complete: {len(valid_keys)} valid, {corrupt_count} corrupt")

    generate_metadata(output_dir, images_dir, metadata_path, valid_keys=valid_keys, validate_images=False)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Download ContraStyles dataset")
    parser.add_argument("--output", type=str, default="./data/contrastyles_full",
                        help="Output directory for dataset")
    parser.add_argument("--threads", type=int, default=64,
                        help="Number of parallel download threads")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout for each download (seconds)")
    parser.add_argument("--regenerate-metadata", action="store_true",
                        help="Only regenerate metadata.csv without downloading")
    parser.add_argument("--validate", action="store_true",
                        help="Validate image integrity when regenerating metadata")
    parser.add_argument("--clean", action="store_true",
                        help="Remove corrupt images (use with --validate)")

    args = parser.parse_args()

    if args.regenerate_metadata:
        regenerate_metadata_only(args.output, validate=args.validate, clean=args.clean)
    else:
        download_dataset(args.output, args.threads, args.timeout)


if __name__ == "__main__":
    main()
