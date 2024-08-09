import numpy as np
import pandas as pd
import cv2
import os

def calculate_lymphocyte_density(image_path, csv_path, patch_save_path, patch_size=256, stride=128):
    if not os.path.exists(patch_save_path):
        os.makedirs(patch_save_path)

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) 
    # Calculate the amount of padding needed
    pad_height = (patch_size - (image.shape[0] % patch_size)) % patch_size
    pad_width = (patch_size - (image.shape[1] % patch_size)) % patch_size
    image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0]) ##padding
    image_height, image_width, _ = image.shape
    print(image_height, image_width)
    df = pd.read_csv(csv_path)
    df['x'] = df['x'] - 1
    df['y'] = df['y'] - 1
    stroma_color = np.array([255, 255, 0])  # Yellow


    # Function to check if a patch contains stroma area
    def is_stroma_area(patch):
        stroma_mask = np.all(patch == stroma_color, axis=-1)
        return np.sum(stroma_mask) > 0


    # Function to calculate lymphocyte density in stroma area within a patch
    def lymphocyte_density_in_stroma(patch, patch_df, patch_x, patch_y):
        stroma_mask = np.all(patch == stroma_color, axis=-1)
        stroma_area = np.sum(stroma_mask) *16 *16 *0.44*0.44*10e-6
        if stroma_area == 0:
            return np.nan
        
        # Get lymphocytes within the patch
        lymphocytes_in_patch = patch_df[
            (patch_df['class2'] == 'l') &
            (patch_df['x'] >= patch_x) & (patch_df['x'] < patch_x + patch_size) &
            (patch_df['y'] >= patch_y) & (patch_df['y'] < patch_y + patch_size)
        ]

        # Check if lymphocytes are within the stroma area
        lymphocytes_in_stroma = 0
        for _, lymphocyte in lymphocytes_in_patch.iterrows():
            if stroma_mask[lymphocyte['y'] - patch_y, lymphocyte['x'] - patch_x]:
                lymphocytes_in_stroma += 1

        return lymphocytes_in_stroma / stroma_area


    densities = []
    for y in range(0, image_height - patch_size + 1, stride):
        for x in range(0, image_width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patch_name = os.path.join(patch_save_path, 'h_' + str(y) + '_w_' + str(x) + '.png')
            if not os.path.exists(patch_name):
                cv2.imwrite(patch_name, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            if is_stroma_area(patch):
                density = lymphocyte_density_in_stroma(patch, df, x, y)
                densities.append(density)
                    
    #Calculate average densities
    all_densities = np.array(densities)
    average_density = np.nanmean(all_densities) if densities else np.nan
    average_density_non_zero = np.nanmean(all_densities[all_densities > 0]) if np.any(all_densities > 0) else np.nan
    
    return average_density, average_density_non_zero



image_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512_post_tumor15_900'
csv_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellPos'
results = []
patch_size = 213 #The diameter of a core is around 1 mm
stride = 213

output_csv = f'/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/localTIL/localtil_{patch_size}_str{stride}pad.csv'
patch_dst_dir = f'/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/patch{patch_size}_str{stride}'
if not os.path.exists(patch_dst_dir):
    os.makedirs(patch_dst_dir)

for file_name in sorted(os.listdir(image_dir)):
    if file_name.endswith('.png'):
        print(file_name)
        image_path = os.path.join(image_dir, file_name)
        csv_path = os.path.join(csv_dir, file_name.replace('.svs_Ss1.png_Ss1.png', '.csv'))
        patch_save = os.path.join(patch_dst_dir, file_name[:-20])
        average_density, average_density_non_zero = calculate_lymphocyte_density(image_path, csv_path, patch_save, patch_size=patch_size, stride=stride)
        results.append({
            'ID': file_name[:-20],
            'average_density': average_density,
            'average_density_non_zero': average_density_non_zero
        })


results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

