#!/usr/bin/env python3
"""
Tiled pixel-wise comparison of large binary mask PNGs using PIL.
Processes in chunks to avoid memory overload on huge images.
"""
import numpy as np
from PIL import Image as PILImage
from sklearn.metrics import confusion_matrix
import sys

TILE_SIZE = 2048  # Process in 2048x2048 chunks

def get_red_mask_tile(tile_rgb):
    """
    Convert RGB tile to binary red mask (1=red, 0=background).
    Expects data shape (height, width, 3) from PIL.
    """
    r, g, b = tile_rgb[:,:,0], tile_rgb[:,:,1], tile_rgb[:,:,2]
    # Detect pure red (255, 0, 0)
    red_mask = (r == 255) & (g == 0) & (b == 0)
    return red_mask.astype(np.uint8)

def process_masks_tiled(file1, file2):
    """
    Process two mask files in tiles, accumulating confusion matrix.
    Uses PIL to properly handle RGBA PNG files.
    """
    cm_total = np.zeros((2, 2), dtype=np.int64)
    total_pixels = 0
    tile_count = 0
    
    # Load images with PIL and convert to RGB
    PILImage.MAX_IMAGE_PIXELS = None
    img1 = PILImage.open(file1).convert('RGB')
    img2 = PILImage.open(file2).convert('RGB')
    
    height, width = img1.height, img1.width
    
    print(f"Processing {width}x{height} images in {TILE_SIZE}x{TILE_SIZE} tiles...")
    
    total_tiles = ((height - 1) // TILE_SIZE + 1) * ((width - 1) // TILE_SIZE + 1)
    
    # Iterate over tiles
    for row in range(0, height, TILE_SIZE):
        for col in range(0, width, TILE_SIZE):
            # Define crop box
            box = (col, row, min(col + TILE_SIZE, width), min(row + TILE_SIZE, height))
            
            # Crop tiles and convert to numpy arrays
            tile1_rgb = np.array(img1.crop(box))
            tile2_rgb = np.array(img2.crop(box))
            
            # Convert to binary masks
            mask1 = get_red_mask_tile(tile1_rgb)
            mask2 = get_red_mask_tile(tile2_rgb)
            
            # Accumulate confusion matrix
            cm = confusion_matrix(mask1.flatten(), mask2.flatten(), 
                                 labels=[0, 1])
            cm_total += cm
            tile_count += 1
            total_pixels += mask1.size
            
            if tile_count % 10 == 0:
                print(f"  Processed {tile_count} tiles ({100*tile_count/total_tiles:.1f}%)...")
    
    return cm_total, total_pixels

def print_results(cm, total_pixels):
    """Print confusion matrix and metrics."""
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    cover1 = (fp + tp) / total_pixels * 100
    cover2 = (fn + tp) / total_pixels * 100
    
    print(f"\n{'='*50}")
    print(f"Coverage (% of image marked as red):")
    print(f"  Mask 1: {cover1:.2f}%")
    print(f"  Mask 2: {cover2:.2f}%")
    print(f"\n{'='*50}")
    print(f"Confusion Matrix (Mask1 vs Mask2):")
    print(f"{'':20} Pred Background  Pred Staghorn")
    print(f"{'True Background':20} {tn:15,d}  {fp:15,d}")
    print(f"{'True Staghorn':20} {fn:15,d}  {tp:15,d}")
    print(f"{'='*50}")
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*50}\n")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    if len(sys.argv) < 3:
        print("Usage: python mask_comparison_tiled.py <mask1.png> <mask2.png>")
        sys.exit(1)
    
    file1, file2 = sys.argv[1], sys.argv[2]
    
    print(f"Mask 1: {file1}")
    print(f"Mask 2: {file2}\n")
    
    try:
        cm, total_pixels = process_masks_tiled(file1, file2)
        print_results(cm, total_pixels)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
