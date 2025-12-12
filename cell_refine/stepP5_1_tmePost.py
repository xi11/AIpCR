import os
from pathlib import Path
import numpy as np

try:
    import imageio.v3 as iio  # imageio>=2.28
except Exception:
    import imageio as iio

from skimage.morphology import disk, binary_closing, remove_small_objects
from scipy.ndimage import binary_fill_holes

def postprocess_tumor_masks(
    src_path: str,
    dst_path: str,
    disk_radius: int = 15,
    area_min: int = 900,
    verbose: bool = True,
) -> None:

    src = Path(src_path)
    dst = Path(dst_path)
    dst.mkdir(parents=True, exist_ok=True)

    tumor_colors = [
        (128, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (128, 128, 0),
    ]

    for png in sorted(src.glob("*.png")):
        out_name = png.stem + "_Ss1.png"
        out_path = dst / out_name
        if out_path.exists():
            if verbose:
                print(f"[skip] {out_name} (already exists)")
            continue

        if verbose:
            print(png.name)

        img = iio.imread(png)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[..., :3]  # ignore alpha if present

        mask = np.zeros(img.shape[:2], dtype=bool)
        for c in tumor_colors:
            mask |= np.all(img == np.array(c, dtype=img.dtype), axis=2)

        mask = binary_fill_holes(mask)
        mask = binary_closing(mask, footprint=disk(disk_radius))
        mask = binary_fill_holes(mask)

        mask = remove_small_objects(mask, min_size=area_min)

        mask_u8 = mask.astype(np.uint8)
        masked_img = (mask_u8[..., None] * img).astype(np.uint8)

        iio.imwrite(out_path, masked_img)


if __name__ == "__main__":
    # python xxxx.py --src /path/to/src --dst /path/to/dst --radius 15 --minarea 900
    import argparse

    parser = argparse.ArgumentParser(description="Post-process tumor masks")
    parser.add_argument("--src", required=True, help="Source directory with input masks")
    parser.add_argument("--dst", required=True, help="Destination directory for outputs")
    parser.add_argument("--radius", type=int, default=15, help="Disk radius for closing")
    parser.add_argument("--minarea", type=int, default=900, help="Minimum component area to keep")
    parser.add_argument("--quiet", action="store_true", help="Suppress filename prints")

    args = parser.parse_args()
    postprocess_tumor_masks(
        src_path=args.src,
        dst_path=args.dst,
        disk_radius=args.radius,
        area_min=args.minarea,
        verbose=not args.quiet,
    )
