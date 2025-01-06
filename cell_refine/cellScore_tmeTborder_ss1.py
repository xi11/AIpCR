import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

#same with xxxTborder
# Define RGB values for regions in the segmentation mask
REGION_COLORS = {
    "tumor": (128, 0, 0),
    "stroma": (255, 204, 0),
}

# Paths
csv_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellPos_noround" #it doesn't matter using noround or round, eventually will be rounded
mask_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512_post_tumor15_900_tbedraw_orng"
output_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellScoreOth_tbedraw"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to determine the region based on pixel color
def get_region(color):
    for region, region_color in REGION_COLORS.items():
        if color == region_color:
            return region
    return None

files = sorted(glob(os.path.join(csv_dir, '*.csv')))
for csv_file in tqdm(files):
    file_name = os.path.basename(csv_file)[:-4]
    mask_path = os.path.join(mask_dir, file_name + ".svs_Ss1.png_Ss1.png")

    if not os.path.exists(mask_path):
        print(f"Mask not found for {file_name}, skipping...")
        continue

    # Read data
    cell_data = pd.read_csv(csv_file, usecols=['class2', 'x', 'y'])
    mask = np.array(Image.open(mask_path))

    # Initialize counters
    counts = {
        "t_tumor": 0,
        "l_tumor": 0,
        "f_tumor": 0,
        "o_tumor": 0,
        "t_stroma": 0,
        "l_stroma": 0,
        "f_stroma": 0,
        "o_stroma": 0,
    }

    for cell_type in cell_data["class2"].unique():
        subset = cell_data[cell_data["class2"] == cell_type]
        for _, row in subset.iterrows():
            x, y = int(np.round(row["x"]-0.0625)), int(np.round(row["y"]-0.0625))   #x, y locations starts from 1

            # Validate coordinates
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                continue

            pixel_color = tuple(mask[y, x, :])
            region = get_region(pixel_color)

            if region:
                counts[f"{cell_type}_{region}"] += 1

    # Save the results
    output_data = pd.DataFrame([{
        "file_name": file_name,
        **counts
    }])
    output_path = os.path.join(output_dir, file_name + "_tborder.csv")
    output_data.to_csv(output_path, index=False)

