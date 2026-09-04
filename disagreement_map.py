#!/usr/bin/env python3
"""
Create spatial disagreement maps between AI and Human masks.
Outputs PNG showing false positives, false negatives, and agreement.
Uses PIL to properly handle RGBA PNGs.
"""
import numpy as np
from PIL import Image as PILImage
import sys

TILE_SIZE = 2048

def get_red_mask_tile(tile_rgb):
    """Convert RGB tile to binary red mask. Expects (height, width, 3)."""
    r, g, b = tile_rgb[:,:,0], tile_rgb[:,:,1], tile_rgb[:,:,2]
    red_mask = (r == 255) & (g == 0) & (b == 0)
    return red_mask.astype(np.uint8)

def process_masks_to_disagreement(file_ai, file_human):
    """
    Create disagreement map using PIL:
    - Red: False Positives (AI only)
    - Blue: False Negatives (Human only)
    - Green: True Positives (both agree)
    - Black: True Negatives (both agree on background)
    """
    
    # Load images with PIL and convert to RGB
    PILImage.MAX_IMAGE_PIXELS = None
    img_ai = PILImage.open(file_ai).convert('RGB')
    img_human = PILImage.open(file_human).convert('RGB')
    
    height, width = img_ai.height, img_ai.width
    
    print(f"Creating disagreement map for {width}x{height} image...")
    print("Processing tiles...\n")
    
    # Create output array
    disagreement_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    tile_count = 0
    total_tiles = ((height - 1) // TILE_SIZE + 1) * ((width - 1) // TILE_SIZE + 1)
    
    for row in range(0, height, TILE_SIZE):
        for col in range(0, width, TILE_SIZE):
            # Define crop box
            box = (col, row, min(col + TILE_SIZE, width), min(row + TILE_SIZE, height))
            
            # Crop tiles
            tile_ai_rgb = np.array(img_ai.crop(box))
            tile_human_rgb = np.array(img_human.crop(box))
            
            # Get masks
            mask_ai = get_red_mask_tile(tile_ai_rgb)
            mask_human = get_red_mask_tile(tile_human_rgb)
            
            tile_h, tile_w = mask_ai.shape
            
            # Create color map for this tile
            tile_map = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            
            # True Positives (both agree on staghorn) = Green
            tp = (mask_ai == 1) & (mask_human == 1)
            tile_map[tp] = [0, 255, 0]
            
            # False Positives (AI only) = Red
            fp = (mask_ai == 1) & (mask_human == 0)
            tile_map[fp] = [255, 0, 0]
            
            # False Negatives (Human only) = Blue
            fn = (mask_ai == 0) & (mask_human == 1)
            tile_map[fn] = [0, 0, 255]
            
            # Place tile in output
            disagreement_map[row:row+tile_h, col:col+tile_w] = tile_map
            
            tile_count += 1
            if tile_count % 10 == 0:
                print(f"  {tile_count}/{total_tiles} tiles processed ({100*tile_count/total_tiles:.1f}%)")
    
    return disagreement_map

def create_statistics(disagreement_map):
    """Extract statistics from disagreement map."""
    # Use int64 to avoid overflow
    green = np.int64(np.all(disagreement_map == [0, 255, 0], axis=2).sum())  # TP
    red = np.int64(np.all(disagreement_map == [255, 0, 0], axis=2).sum())    # FP
    blue = np.int64(np.all(disagreement_map == [0, 0, 255], axis=2).sum())   # FN
    black = np.int64(np.all(disagreement_map == [0, 0, 0], axis=2).sum())    # TN
    
    total = np.int64(disagreement_map.shape[0]) * np.int64(disagreement_map.shape[1])
    
    print(f"\n{'='*60}")
    print(f"DISAGREEMENT STATISTICS")
    print(f"{'='*60}")
    print(f"True Positives  (Green - both agree staghorn): {green:12,d} ({100.0*green/total:.4f}%)")
    print(f"False Positives (Red - AI only):               {red:12,d} ({100.0*red/total:.4f}%)")
    print(f"False Negatives (Blue - Human only):           {blue:12,d} ({100.0*blue/total:.4f}%)")
    print(f"True Negatives  (Black - both background):     {black:12,d} ({100.0*black/total:.4f}%)")
    print(f"{'='*60}")
    print(f"Total Disagreement Pixels:                     {red+blue:12,d} ({100.0*(red+blue)/total:.4f}%)")
    print(f"Pixels where AI was wrong:                     {red:12,d} (false positives)")
    print(f"Pixels where AI missed staghorn:               {blue:12,d} (false negatives)")
    print(f"{'='*60}\n")
    
    return {
        'tp': green,
        'fp': red,
        'fn': blue,
        'tn': black,
        'total': total
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python disagreement_map.py <imageAI.png> <imageHuman.png>")
        sys.exit(1)
    
    file_ai, file_human = sys.argv[1], sys.argv[2]
    
    print(f"AI Mask:    {file_ai}")
    print(f"Human Mask: {file_human}\n")
    
    try:
        disagreement_map = process_masks_to_disagreement(file_ai, file_human)
        stats = create_statistics(disagreement_map)
        
        # Save output
        output_file = "disagreement_map.png"
        PILImage.fromarray(disagreement_map).save(output_file)
        print(f"Saved: {output_file}")
        print("\nColor guide:")
        print("  🟩 Green:  Both agree (staghorn correctly identified)")
        print("  🟥 Red:    AI only (false positive - fish, soft coral, etc)")
        print("  🟦 Blue:   Human only (false negative - staghorn AI missed)")
        print("  ⬛ Black:  Both agree (background)")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
