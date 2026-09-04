#!/usr/bin/env python3
"""
Makes a spatial disagreement map between an AI mask and a human mask.
Green = both agree it's staghorn, red = AI only (false positive),
blue = human only (AI missed it), black = both agree it's background.

Processes in tiles so it doesn't fall over on the big orthomosaics.
"""
import argparse
import sys

import numpy as np
from PIL import Image as PILImage

from mask_utils import TILE_SIZE, get_red_mask_tile, iter_tiles

PILImage.MAX_IMAGE_PIXELS = None  # these images are huge on purpose, don't let PIL complain


def build_disagreement_map(file_ai, file_human):
    img_ai = PILImage.open(file_ai).convert("RGB")
    img_human = PILImage.open(file_human).convert("RGB")

    if img_ai.size != img_human.size:
        raise ValueError(f"Image sizes don't match: {img_ai.size} vs {img_human.size}")

    width, height = img_ai.size
    print(f"Building disagreement map for {width}x{height}...")

    disagreement_map = np.zeros((height, width, 3), dtype=np.uint8)

    total_tiles = ((height - 1) // TILE_SIZE + 1) * ((width - 1) // TILE_SIZE + 1)
    tile_count = 0

    for row, col, box in iter_tiles(width, height):
        tile_ai = np.array(img_ai.crop(box))
        tile_human = np.array(img_human.crop(box))

        mask_ai = get_red_mask_tile(tile_ai)
        mask_human = get_red_mask_tile(tile_human)

        tile_h, tile_w = mask_ai.shape
        tile_map = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        tp = (mask_ai == 1) & (mask_human == 1)
        fp = (mask_ai == 1) & (mask_human == 0)
        fn = (mask_ai == 0) & (mask_human == 1)

        tile_map[tp] = [0, 255, 0]
        tile_map[fp] = [255, 0, 0]
        tile_map[fn] = [0, 0, 255]

        disagreement_map[row:row + tile_h, col:col + tile_w] = tile_map

        tile_count += 1
        if tile_count % 10 == 0 or tile_count == total_tiles:
            print(f"  tile {tile_count}/{total_tiles}")

    return disagreement_map


def print_stats(disagreement_map):
    green = int(np.all(disagreement_map == [0, 255, 0], axis=2).sum())
    red = int(np.all(disagreement_map == [255, 0, 0], axis=2).sum())
    blue = int(np.all(disagreement_map == [0, 0, 255], axis=2).sum())
    black = int(np.all(disagreement_map == [0, 0, 0], axis=2).sum())
    total = disagreement_map.shape[0] * disagreement_map.shape[1]

    print("\nDisagreement stats")
    print(f"  True positives  (green): {green:>12,} ({100*green/total:.4f}%)")
    print(f"  False positives (red):   {red:>12,} ({100*red/total:.4f}%)")
    print(f"  False negatives (blue):  {blue:>12,} ({100*blue/total:.4f}%)")
    print(f"  True negatives  (black): {black:>12,} ({100*black/total:.4f}%)")
    print(f"  Total disagreement:      {red+blue:>12,} ({100*(red+blue)/total:.4f}%)\n")

    return {"tp": green, "fp": red, "fn": blue, "tn": black, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Compare AI vs human coral masks and output a disagreement map.")
    parser.add_argument("ai_mask", help="path to AI-generated mask PNG")
    parser.add_argument("human_mask", help="path to human-annotated mask PNG")
    parser.add_argument("-o", "--output", default="disagreement_map.png", help="output PNG path")
    args = parser.parse_args()

    print(f"AI mask:    {args.ai_mask}")
    print(f"Human mask: {args.human_mask}\n")

    try:
        disagreement_map = build_disagreement_map(args.ai_mask, args.human_mask)
    except FileNotFoundError as e:
        print(f"Couldn't find one of the input files: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Problem with inputs: {e}")
        sys.exit(1)

    print_stats(disagreement_map)

    PILImage.fromarray(disagreement_map).save(args.output)
    print(f"Saved to {args.output}")
    print("Green = agree (staghorn), Red = AI only, Blue = human only, Black = agree (background)")


if __name__ == "__main__":
    main()
