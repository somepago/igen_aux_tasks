#!/usr/bin/env python3
"""
Test script for the batched image captioning pipeline

Usage:
    python test_pipeline.py                    # Quick test (3 images, no HF push)
    python test_pipeline.py --full             # Full e2e test with HF push
    python test_pipeline.py --model 4b         # Test with smaller model
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image

sys.path.append('/home/duality/projects/imagecaption')

from batched_image_inference import (
    GemmaCaptioner, BatchConfig,
    ContraStylesProcessor, DatasetConfig,
    HuggingFaceUploader, process_images
)


def test_data_loading():
    """Test that dataset loading works"""
    print("=" * 50)
    print("TEST: Data Loading")
    print("=" * 50)

    config = DatasetConfig(max_images=5)
    processor = ContraStylesProcessor(config)
    image_paths = processor.get_image_paths()
    print(f"Found {len(image_paths)} test images")

    if image_paths:
        # Test metadata loading from shard parquet
        metadata = processor.get_metadata_for_image(image_paths[0])
        print(f"Sample image: {image_paths[0]}")
        print(f"Metadata keys: {list(metadata.keys())}")
        print(f"  - key: {metadata.get('key')}")
        print(f"  - url: {metadata.get('url', '')[:80]}...")
        print(f"  - caption: {metadata.get('caption', '')[:80]}...")
        return True
    return False


def test_model_loading(model_id: str = "google/gemma-3-27b-it"):
    """Test that the Gemma model loads correctly"""
    print("\n" + "=" * 50)
    print(f"TEST: Model Loading ({model_id})")
    print("=" * 50)

    config = BatchConfig(batch_size=1, model_id=model_id)
    captioner = GemmaCaptioner(config)
    print("Model loaded successfully")
    return captioner


def test_caption_generation(captioner: GemmaCaptioner, num_images: int = 3):
    """Test caption generation on a few images"""
    print("\n" + "=" * 50)
    print(f"TEST: Caption Generation ({num_images} images)")
    print("=" * 50)

    dataset_config = DatasetConfig(max_images=num_images)
    processor = ContraStylesProcessor(dataset_config)
    image_paths = processor.get_image_paths()

    results = []
    for image_path in image_paths:
        print(f"\nProcessing: {Path(image_path).name}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"  Image size: {image.size}")

        # Get metadata
        metadata = processor.get_metadata_for_image(image_path)
        print(f"  Original caption: {metadata.get('caption', '')[:60]}...")

        # Generate caption
        caption = captioner.generate_caption(image)
        print(f"  VLM caption: {caption[:100]}...")

        results.append({
            "key": metadata.get("key"),
            "url": metadata.get("url", ""),
            "original_caption": metadata.get("caption", ""),
            "vlm_caption": caption,
            "width": metadata.get("width"),
            "height": metadata.get("height"),
        })

    return results


def test_end_to_end(captioner: GemmaCaptioner, push_to_hf: bool = False, num_images: int = 5):
    """Full end-to-end test"""
    print("\n" + "=" * 50)
    print(f"TEST: End-to-End Pipeline ({num_images} images)")
    print("=" * 50)

    dataset_config = DatasetConfig(
        max_images=num_images,
        push_every_n_samples=100,  # Won't trigger for small test
        checkpoint_dir="./test_checkpoints"
    )

    processor = ContraStylesProcessor(dataset_config)
    image_paths = processor.get_image_paths()

    uploader = None
    if push_to_hf:
        print("Initializing HuggingFace uploader...")
        uploader = HuggingFaceUploader("ContraStylesRecap", "somepago")

    # Process images
    samples = process_images(
        image_paths,
        captioner,
        processor,
        dataset_config,
        uploader,
        already_processed=set(),
    )

    print(f"\nProcessed {len(samples)} images")

    # Save test results
    import pandas as pd
    output_path = Path("./test_checkpoints/test_results.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_parquet(output_path, index=False)
    print(f"Saved test results to {output_path}")

    # Show sample output
    print("\n--- Sample Output ---")
    for s in samples[:2]:
        print(f"\nKey: {s['key']}")
        print(f"URL: {s['url'][:60]}...")
        print(f"Original: {s['original_caption'][:60]}...")
        print(f"VLM: {s['vlm_caption'][:100]}...")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Test the image captioning pipeline")
    parser.add_argument("--full", action="store_true", help="Run full e2e test")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace (requires --full)")
    parser.add_argument("--model", type=str, default="27b", choices=["4b", "12b", "27b"],
                        help="Model size to use")
    parser.add_argument("--num-images", type=int, default=3, help="Number of images to test")
    parser.add_argument("--skip-model", action="store_true", help="Skip model loading (data test only)")

    args = parser.parse_args()

    model_map = {
        "4b": "google/gemma-3-4b-it",
        "12b": "google/gemma-3-12b-it",
        "27b": "google/gemma-3-27b-it",
    }
    model_id = model_map[args.model]

    print("=" * 50)
    print("Image Captioning Pipeline Test")
    print("=" * 50)

    try:
        # Test data loading first (no GPU needed)
        if not test_data_loading():
            print("Data loading test failed!")
            return 1

        if args.skip_model:
            print("\nSkipping model tests (--skip-model)")
            return 0

        # Load model
        captioner = test_model_loading(model_id)

        if args.full:
            # Full end-to-end test
            test_end_to_end(captioner, push_to_hf=args.push, num_images=args.num_images)
        else:
            # Quick caption test
            test_caption_generation(captioner, num_images=args.num_images)

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
