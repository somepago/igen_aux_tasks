#!/usr/bin/env python3
"""
Script to validate images, remove corrupt ones, and rebuild metadata.csv

Usage:
    python rebuild_metadata.py /path/to/data/contrastyles_full
    python rebuild_metadata.py  # Uses current directory
"""

import os
import sys
import json
import csv
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set BASE_DIR from command line argument or current directory
if len(sys.argv) > 1:
    BASE_DIR = Path(sys.argv[1]).resolve()
else:
    BASE_DIR = Path.cwd()

IMAGES_DIR = BASE_DIR / "images"
OUTPUT_CSV = BASE_DIR / "metadata.csv"
CORRUPT_LOG = BASE_DIR / "corrupt_images.log"


def validate_image(img_path: str) -> tuple[str, bool, str | None]:
    """
    Validate a single image file.
    Returns: (path, is_valid, caption_or_error)
    """
    img_path = Path(img_path)
    json_path = img_path.with_suffix('.json')

    try:
        # Try to open and verify the image
        with Image.open(img_path) as img:
            img.verify()

        # Re-open to actually load the image data (verify() doesn't load pixels)
        with Image.open(img_path) as img:
            img.load()

        # Get caption from JSON if exists
        caption = ""
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    caption = data.get('caption', '')
            except (json.JSONDecodeError, KeyError):
                pass

        return (str(img_path), True, caption)

    except Exception as e:
        return (str(img_path), False, str(e))


def get_all_image_paths():
    """Get all jpg image paths."""
    image_paths = []
    for subdir in sorted(IMAGES_DIR.iterdir()):
        if subdir.is_dir():
            for img_file in subdir.glob("*.jpg"):
                image_paths.append(str(img_file))
    return image_paths


def main():
    print(f"Data directory: {BASE_DIR}")

    if not IMAGES_DIR.exists():
        print(f"Error: Images directory not found: {IMAGES_DIR}")
        print("Usage: python rebuild_metadata.py /path/to/data/contrastyles_full")
        sys.exit(1)

    print("Finding all images...")
    image_paths = get_all_image_paths()
    print(f"Found {len(image_paths)} images")

    valid_images = []
    corrupt_images = []

    print("Validating images (this may take a while)...")

    # Use multiprocessing for faster validation
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(validate_image, path): path for path in image_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
            path, is_valid, result = future.result()
            if is_valid:
                valid_images.append((path, result))  # result is caption
            else:
                corrupt_images.append((path, result))  # result is error

    print(f"\nValid images: {len(valid_images)}")
    print(f"Corrupt images: {len(corrupt_images)}")

    # Log and remove corrupt images
    if corrupt_images:
        print(f"\nRemoving {len(corrupt_images)} corrupt images...")
        with open(CORRUPT_LOG, 'w') as f:
            for path, error in corrupt_images:
                f.write(f"{path}\t{error}\n")
                # Remove the corrupt image and its JSON
                try:
                    os.remove(path)
                    json_path = Path(path).with_suffix('.json')
                    if json_path.exists():
                        os.remove(json_path)
                except Exception as e:
                    print(f"Failed to remove {path}: {e}")
        print(f"Corrupt images log saved to {CORRUPT_LOG}")

    # Write new metadata.csv
    print(f"\nWriting metadata.csv with {len(valid_images)} entries...")

    # Sort valid images by path for consistency
    valid_images.sort(key=lambda x: x[0])

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'prompt'])  # Keep the original header format

        for img_path, caption in tqdm(valid_images, desc="Writing CSV"):
            # Convert absolute path to relative path from BASE_DIR
            rel_path = str(Path(img_path).relative_to(BASE_DIR))
            writer.writerow([rel_path, caption])

    print(f"\nDone! metadata.csv created with {len(valid_images)} entries")
    print(f"Removed {len(corrupt_images)} corrupt images")


if __name__ == "__main__":
    main()
