# Staghorn Coral AI–Human Mask Comparison Tools

**Quantifying segmentation accuracy and spatial disagreement between AI and manual annotations**

---

## Overview

This repository contains utility scripts for comparing AI-generated coral segmentation masks against human annotations. It's designed to work with the outputs from the [Staghorn Coral AI Detection Model](https://github.com/joelbetteridge/ML-coral-detection-model), primarily for evaluating model performance and identifying systematic errors.

The scripts process large orthomosaics (500+ megapixels) in tiles to avoid memory overflow, accumulate confusion matrices, and generate spatial disagreement maps showing exactly where the AI and human annotator disagree.

---

## What These Scripts Do

### `disagreement_map.py`

Creates a spatial visualization of disagreement:
- **Green:** AI and human agree (staghorn correctly identified)
- **Red:** AI only (false positive — fish, soft coral, etc.)
- **Blue:** Human only (false negative — AI missed staghorn)
- **Black:** Both agree (background)

Output is a PNG matching the input resolution, useful for identifying spatial patterns in errors and spotting regions that need retraining.

### `mask_comparison_tiled.py`

Generates pixel-level confusion matrix and accuracy metrics:
- Accuracy, precision, recall, F1 score
- Pixel coverage statistics
- Total disagreement count

Outputs formatted to stdout; pipe to a file if you need to log results.

---

## Usage

Both scripts require two input mask PNGs (typically one from AI output, one from human annotation). Masks must be:
- Same dimensions
- Pure red (255, 0, 0) for positive class, black (0, 0, 0) for background
- RGB or RGBA (internally converted to RGB)

```bash
# Generate disagreement map
python disagreement_map.py <ai_mask.png> <human_mask.png> -o output.png

# Print accuracy metrics
python mask_comparison_tiled.py <ai_mask.png> <human_mask.png>

# Both support --help
python disagreement_map.py --help
python mask_comparison_tiled.py --help
```

---

## Implementation Notes

### Tiled Processing

Both scripts process the image in 2048×2048 tiles (configurable in `mask_utils.py`) to keep memory usage constant regardless of orthomosaic size.

**Why PIL instead of rasterio?**  
These scripts work with plain PNG masks (not georeferenced GeoTIFFs). Rasterio's windowed reads only save memory on tiled/stripped image formats; PNG decoding is sequential regardless of library. PIL is simpler for this use case and rasterio ran into errors when I used it.

### Shared Utilities

`mask_utils.py` contains:
- `get_red_mask_tile()` — Extracts binary red mask from RGB tile
- `iter_tiles()` — Generator for tiling coordinates
- `TILE_SIZE` — Configurable tile size (2048px default)

Both scripts import these to avoid code duplication.

---

## Output Interpretation

### Disagreement Map

Useful for:
- Visual inspection of error patterns (e.g., are false positives clustered near sand/rubble?)
- Identifying regions that might need retraining data
- Spotting systematic biases (e.g., model underperforms on certain coral morphologies)

### Accuracy Metrics

Standard confusion matrix terminology:
- **True Positives (TP):** Pixels both agree are staghorn
- **False Positives (FP):** AI marked as staghorn, human didn't
- **False Negatives (FN):** Human marked as staghorn, AI missed
- **True Negatives (TN):** Both agree on background

**Precision** = TP / (TP + FP) — "Of what the AI called staghorn, how much was right?"  
**Recall** = TP / (TP + FN) — "Of the staghorn present, how much did AI find?"  
**F1** = Harmonic mean of precision and recall

---

## Workflow Integration

Typical use in the RRFB restoration pipeline:

1. Run AI segmentation on new orthomosaic in TagLab
2. Export AI mask as PNG
3. Perform targeted human review/annotation
4. Export human mask as PNG
5. Run `disagreement_map.py` to visualize errors
6. Run `mask_comparison_tiled.py` to quantify performance
7. Use disagreement map to guide retraining or identify edge cases

---

## Known Issues & Limitations

- **Size matching:** Script will fail if input masks are different dimensions (intentional and prevents silent misalignment)
- **Color strictness:** Looks for exactly pure red (255,0,0)
- **No temporal tracking:** Scripts compare two masks as a snapshot; they don't track individual coral colonies over time

---

## Requirements

numpy
Pillow
scikit-learn

---

## Author

**Joel Betteridge**  
Coral Reef Restoration Technician Intern  
Reef Renewal Foundation Bonaire  
University of York, BSc Ecology (Third Year)

Companion repository to [Staghorn Coral AI Detection Model](https://github.com/joelbetteridge/ML-coral-detection-model).