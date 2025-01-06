import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

# Define RGB values for regions in the segmentation mask
REGION_COLORS = {
    "tumor": (128, 0, 0),
    "stroma": (255, 204, 0),
}

# Paths
csv_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellPos"
mask_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512_post_tumor15_900_tbed_orng"
output_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellScoreOth_tbed"

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
        "o_tumor": 0,
        "l_stroma": 0,
        "f_stroma": 0,
        "o_stroma": 0,
    }

    for _, row in cell_data.iterrows():
        x, y, cell_type = int(row["x"]), int(row["y"]), row["class2"]

        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            print(f"Invalid cell location ({x}, {y}) in {file_name}, skipping...")
            continue

        pixel_color = tuple(mask[y, x, :])
        region = get_region(pixel_color)

        if region == "tumor":
            if cell_type == "t":
                counts["t_tumor"] += 1
            elif cell_type == "l":
                counts["l_tumor"] += 1
            else:
                counts["o_tumor"] += 1
        elif region == "stroma":
            if cell_type == "l":
                counts["l_stroma"] += 1
            elif cell_type == "f":
                counts["f_stroma"] += 1
            else:
                counts["o_stroma"] += 1

    # Save the results
    output_data = pd.DataFrame([{
        "file_name": file_name,
        **counts
    }])
    output_path = os.path.join(output_dir, file_name + "_tbed.csv")
    output_data.to_csv(output_path, index=False)

