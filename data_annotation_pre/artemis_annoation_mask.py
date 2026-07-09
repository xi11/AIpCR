#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
in_dir  = Path("/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/maskPng")     # folder with digital masks (0,1,2,...)
out_dir = Path("/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/maskPng_color")   # output folder
out_dir.mkdir(parents=True, exist_ok=True)

# BGR color table (index = label id)
class_colors_artemis = [
    (0, 0, 0),       # 0: background
    (0, 0, 128),     # 1
    (0, 204, 255),   # 2
    (255, 255, 0),   # 3
    (255, 0, 255),   # 4
    (0, 128, 128),   # 5
]

color_lut = np.array(class_colors_artemis, dtype=np.uint8)  # (C,3)

# Supported image extensions
exts = {".png", ".tif", ".tiff", ".jpg"}

# -----------------------
# MAIN
# -----------------------
for img_path in sorted(in_dir.iterdir()):
    if img_path.suffix.lower() not in exts:
        continue

    # Read as single-channel (label image)
    label = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

    if label is None:
        print(f"[WARNING] Failed to read {img_path.name}, skipping.")
        continue

    # If accidentally saved as 3-channel, take the first channel
    if label.ndim == 3:
        label = label[:, :, 0]

    label = label.astype(np.int32)

    # Safety check
    max_label = label.max()
    if max_label >= len(color_lut):
        raise ValueError(
            f"{img_path.name}: label value {max_label} exceeds "
            f"defined colors ({len(color_lut)})."
        )

    # Map labels -> BGR colors
    colored = color_lut[label]   # (H,W,3), uint8, BGR

    # Write output
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), colored)

    print(f"[OK] {img_path.name} -> {out_path.name}")

print("Done.")
