#!/usr/bin/env python3
"""
Pixel-wise comparison of two binary mask PNGs (e.g. AI vs human annotation).
Runs in tiles to keep memory sane on the big orthomosaics.
Outputs a confusion matrix and accuracy/precision/recall/F1.

AI discalaimer - script is AI-assisted and human edited by Joel Betteridge
"""
import argparse
import sys

import numpy as np
from PIL import Image as PILImage
from sklearn.metrics import confusion_matrix

from mask_utils import TILE_SIZE, get_red_mask_tile, iter_tiles

PILImage.MAX_IMAGE_PIXELS = None


def compare_masks(file1, file2):
    img1 = PILImage.open(file1).convert("RGB")
    img2 = PILImage.open(file2).convert("RGB")

    if img1.size != img2.size:
        raise ValueError(f"Image sizes don't match: {img1.size} vs {img2.size}")

    width, height = img1.size
    print(f"Comparing {width}x{height} images in {TILE_SIZE}px tiles...")

    cm_total = np.zeros((2, 2), dtype=np.int64)
    total_pixels = 0
    total_tiles = ((height - 1) // TILE_SIZE + 1) * ((width - 1) // TILE_SIZE + 1)
    tile_count = 0

    for row, col, box in iter_tiles(width, height):
        tile1 = np.array(img1.crop(box))
        tile2 = np.array(img2.crop(box))

        mask1 = get_red_mask_tile(tile1)
        mask2 = get_red_mask_tile(tile2)

        cm = confusion_matrix(mask1.flatten(), mask2.flatten(), labels=[0, 1])
        cm_total += cm
        total_pixels += mask1.size

        tile_count += 1
        if tile_count % 10 == 0 or tile_count == total_tiles:
            print(f"  tile {tile_count}/{total_tiles}")

    return cm_total, total_pixels


def print_results(cm, total_pixels):
    tn, fp = cm[0]
    fn, tp = cm[1]

    cover1 = (fp + tp) / total_pixels * 100
    cover2 = (fn + tp) / total_pixels * 100

    print("\nCoverage (% of image marked red):")
    print(f"  Mask 1: {cover1:.2f}%")
    print(f"  Mask 2: {cover2:.2f}%")

    print("\nConfusion matrix (mask1 vs mask2):")
    print(f"{'':20}{'pred background':>18}{'pred staghorn':>16}")
    print(f"{'true background':20}{tn:>18,}{fp:>16,}")
    print(f"{'true staghorn':20}{fn:>18,}{tp:>16,}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Pixel-wise comparison of two binary mask PNGs.")
    parser.add_argument("mask1", help="path to first mask PNG")
    parser.add_argument("mask2", help="path to second mask PNG")
    args = parser.parse_args()

    print(f"Mask 1: {args.mask1}")
    print(f"Mask 2: {args.mask2}\n")

    try:
        cm, total_pixels = compare_masks(args.mask1, args.mask2)
    except FileNotFoundError as e:
        print(f"Couldn't find one of the input files: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Problem with inputs: {e}")
        sys.exit(1)

    print_results(cm, total_pixels)


if __name__ == "__main__":
    main()
