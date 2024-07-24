import numpy as np
import pandas as pd
import cv2
import os

def calculate_lymphocyte_density(image_path, csv_path, patch_size=256, stride=128):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) 

    # Calculate the amount of padding needed
    pad_height = (patch_size - (image.shape[0] % patch_size)) % patch_size
    pad_width = (patch_size - (image.shape[1] % patch_size)) % patch_size
    image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image_height, image_width, _ = image.shape
    print(image_height, image_width)
   
    df = pd.read_csv(csv_path)
    df['x'] = df['x'] - 1
    df['y'] = df['y'] - 1

    stroma_color = np.array([255, 255, 0])  # Yellow
    tumor_color = np.array([128, 0, 0])
    # Function to check if a patch contains stroma area
    def is_main_area(patch):
        stroma_mask = np.all(patch == stroma_color, axis=-1)
        tumor_mask = np.all(patch == tumor_color, axis=-1)
        main_mask = np.logical_or(stroma_mask, tumor_mask)
        return np.sum(main_mask) > 0

    # Function to calculate lymphocyte density in stroma area within a patch
    def lymphocyte_density_in_main(patch, patch_df, patch_x, patch_y):
        stroma_mask = np.all(patch == stroma_color, axis=-1)
        tumor_mask = np.all(patch == tumor_color, axis=-1)
        main_mask = np.logical_or(stroma_mask, tumor_mask)
        main_area = np.sum(main_mask) *16 *16 *0.44*0.44*10e-6
        if main_area == 0:
            return np.nan
        
        # Get lymphocytes within the patch
        lymphocytes_in_patch = patch_df[
            (patch_df['class2'] == 'l') &
            (patch_df['x'] >= patch_x) & (patch_df['x'] < patch_x + patch_size) &
            (patch_df['y'] >= patch_y) & (patch_df['y'] < patch_y + patch_size)
        ]

        fibroblast_in_patch = patch_df[
            (patch_df['class2'] == 'f') &
            (patch_df['x'] >= patch_x) & (patch_df['x'] < patch_x + patch_size) &
            (patch_df['y'] >= patch_y) & (patch_df['y'] < patch_y + patch_size)
        ]

        tumor_in_patch = patch_df[
            (patch_df['class2'] == 't') &
            (patch_df['x'] >= patch_x) & (patch_df['x'] < patch_x + patch_size) &
            (patch_df['y'] >= patch_y) & (patch_df['y'] < patch_y + patch_size)
        ]

        # Check if lymphocytes are within the stroma area
        lymphocytes_in_main = 0
        for _, lymphocyte in lymphocytes_in_patch.iterrows():
            if main_mask[lymphocyte['y'] - patch_y, lymphocyte['x'] - patch_x]:
                lymphocytes_in_main += 1

        fibroblast_in_main = 0
        for _, fibro in fibroblast_in_patch.iterrows():
            if main_mask[fibro['y'] - patch_y, fibro['x'] - patch_x]:
                fibroblast_in_main += 1

        tumor_in_main = 0
        for _, tum in tumor_in_patch.iterrows():
            if main_mask[tum['y'] - patch_y, tum['x'] - patch_x]:
                tumor_in_main += 1

        return lymphocytes_in_main / (lymphocytes_in_main + fibroblast_in_main + tumor_in_main + 10e-6)


    densities = []
    
    for y in range(0, image_height - patch_size + 1, stride):
        for x in range(0, image_width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            if is_main_area(patch):
                density = lymphocyte_density_in_main(patch, df, x, y)
                densities.append(density)
                    

    # Calculate average densities
    all_densities = np.array(densities)
    average_density = np.nanmean(all_densities) if densities else np.nan
    average_density_non_zero = np.nanmean(all_densities[all_densities > 0]) if np.any(all_densities > 0) else np.nan
    
    return average_density, average_density_non_zero



image_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512_post_tumor15_900'
csv_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellPos'

patch_list = [1024, 768, 512, 384, 256, 160, 128, 64]
for path_size in patch_list: 
    stride = path_size //2
    results = []
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.endswith('.png'):
            print(file_name)
            image_path = os.path.join(image_dir, file_name)
            csv_path = os.path.join(csv_dir, file_name.replace('.svs_Ss1.png_Ss1.png', '.csv'))
            average_density, average_density_non_zero = calculate_lymphocyte_density(image_path, csv_path, patch_size=path_size, stride=stride)
            results.append({
                'ID': file_name[:-20],
                'average_density': average_density,
                'average_density_non_zero': average_density_non_zero
            })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    output_csv = f"/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/localtilTumorratio_{path_size}_str{stride}pad.csv"
    results_df.to_csv(output_csv, index=False)

