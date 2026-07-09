import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
tme_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/tms_tnbc_wenyi/tcga/mit-b3-finetuned-tmeTCGA-60-lr00001-s512-20x768/mask_ss1512"
binary_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/anthracosis/TbedEval/breast/tbed_ss1_multi"
out_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/tms_tnbc_wenyi/tcga/mit-b3-finetuned-tmeTCGA-60-lr00001-s512-20x768/mask_ss1512_tbedmanual"

os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Collect TME masks
# -----------------------------
tme_masks = sorted(glob(os.path.join(tme_dir, "*.png"))) + \
            sorted(glob(os.path.join(tme_dir, "*.tif"))) + \
            sorted(glob(os.path.join(tme_dir, "*.tiff")))

if len(tme_masks) == 0:
    raise RuntimeError("No TME masks found.")

# -----------------------------
# Processing loop
# -----------------------------
for tme_path in tqdm(tme_masks, desc="Filtering TME masks"):
    fname = os.path.basename(tme_path)[:-12]
    binary_path = os.path.join(binary_dir, fname+'_Tbed.png')

    if not os.path.exists(binary_path):
        print(f"[WARNING] Missing binary mask for {fname}, skipping.")
        continue

    # -------------------------
    # Load images
    # -------------------------
    tme = cv2.imread(tme_path, cv2.IMREAD_COLOR)
    binary = cv2.imread(binary_path, cv2.IMREAD_UNCHANGED)

    if tme is None or binary is None:
        print(f"[WARNING] Failed to read {fname}, skipping.")
        continue

    h, w = tme.shape[:2]

    # -------------------------
    # Normalize binary mask
    # -------------------------
    if binary.ndim == 3:
        # convert RGB binary to single channel
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

    # Resize binary mask to match TME mask
    binary_resized = cv2.resize(
        binary,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert to boolean mask
    binary_bool = binary_resized > 0

    # -------------------------
    # Apply mask to TME
    # -------------------------
    filtered_tme = np.zeros_like(tme)
    filtered_tme[binary_bool] = tme[binary_bool]

    # -------------------------
    # Save
    # -------------------------
    out_path = os.path.join(out_dir, fname+'.svs_Ss1.png_manual_tumorBed.png')
    cv2.imwrite(out_path, filtered_tme)

print("Done.")
