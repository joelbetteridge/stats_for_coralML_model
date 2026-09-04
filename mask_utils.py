"""
Shared helpers for the mask comparison scripts.
Both disagreement_map.py and mask_comparison_tiled.py use these.
"""
import numpy as np

TILE_SIZE = 2048  # seemed to be a good balance between speed and memory on my laptop


def get_red_mask_tile(tile_rgb):
    """
    Takes an (H, W, 3) RGB tile and returns a binary mask where
    pure red pixels (255, 0, 0) = 1, everything else = 0.

    TagLab exports annotations as pure red on top of the orthomosaic,
    so this is just picking out exactly that colour.
    """
    r, g, b = tile_rgb[:, :, 0], tile_rgb[:, :, 1], tile_rgb[:, :, 2]
    red_mask = (r == 255) & (g == 0) & (b == 0)
    return red_mask.astype(np.uint8)


def iter_tiles(width, height, tile_size=TILE_SIZE):
    """
    Yields (row, col, box) for tiling an image of given width/height.
    box is the (left, top, right, bottom) tuple PIL's crop() wants.
    """
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            box = (col, row, min(col + tile_size, width), min(row + tile_size, height))
            yield row, col, box
