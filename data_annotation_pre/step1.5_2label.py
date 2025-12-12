import os
from pathlib import Path
import numpy as np
from PIL import Image

src_path = Path("/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch2_patch512_mppRaw/mask_color")
dst_path = Path("/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch2_patch512_mppRaw/maskPng")
dst_path.mkdir(parents=True, exist_ok=True)

# RGB -> label mapping
rgb_to_label = {
    (0,   0,   0): 0,  # background
    (128, 0,   0): 1,
    (255, 0, 255): 2,
    (0,   128, 0): 3,
    (128, 0, 128): 4,
    (0, 0,   255): 5,
    (0, 0,   128): 6
}

def indexed_png_to_labels(img: Image.Image) -> np.ndarray:
    """Convert a 'P' mode indexed PNG to label mask via its palette."""
    idx = np.array(img)  # HxW indices
    pal = img.getpalette()[:256*3]  # flatten [r0,g0,b0, r1,g1,b1, ...]
    pal = np.array(pal, dtype=np.uint8).reshape(-1, 3)  # (<=256, 3)

    uniq_idx = np.unique(idx)
    # map each palette index used to a label
    idx_to_label = np.full(256, -1, dtype=np.int16)
    unknown = []
    for k in uniq_idx:
        rgb = tuple(pal[k])
        if rgb in rgb_to_label:
            idx_to_label[k] = rgb_to_label[rgb]
        else:
            unknown.append((k, rgb))

    if unknown:
        raise ValueError(f"Found unmapped palette colors: {unknown[:10]}... (total {len(unknown)})")

    return idx_to_label[idx].astype(np.uint8)

def rgb_png_to_labels(img: Image.Image) -> np.ndarray:
    """Fallback: match pixel colors in an RGB image to labels."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)  # HxWx3
    H, W, _ = arr.shape
    labels = np.full((H, W), 255, dtype=np.uint8)  # 255 = sentinel for unmapped

    for rgb, lab in rgb_to_label.items():
        mask = np.all(arr == rgb, axis=2)
        labels[mask] = lab

    if (labels == 255).any():
        # report a few unknown colors
        unknown_pixels = arr[labels == 255]
        # get unique unknown colors (up to 10 for the message)
        uniq = np.unique(unknown_pixels.reshape(-1, 3), axis=0)
        raise ValueError(f"Unmapped RGB colors present, e.g.: {uniq[:10].tolist()}")

    return labels

for file in sorted(src_path.glob("*.png")):
    with Image.open(file) as im:
        try:
            if im.mode == "P":  # indexed palette image
                label_mask = indexed_png_to_labels(im)
            else:
                label_mask = rgb_png_to_labels(im)
        except ValueError as e:
            print(f"[SKIP] {file.name}: {e}")
            continue

    # Save as single-channel uint8 PNG with labels 0..5
    out_path = dst_path / file.name
    Image.fromarray(label_mask, mode="L").save(out_path)
