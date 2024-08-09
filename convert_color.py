import os
from glob import glob
import cv2
import numpy as np

src_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/gi_spore/pilot/tme/mit-b3-finetuned-TCGAbcssWsss10xLuadMacroMuscle-40x896-20x512-10x256re/mask_cws512'
dst_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/gi_spore/pilot/tme/mit-b3-finetuned-TCGAbcssWsss10xLuadMacroMuscle-40x896-20x512-10x256re/mask_cws512_cyan'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

# for artemis ms
#yellow = [0, 255, 255]  # Original color (B, G, R)
#orange = [0, 204, 255]  # New color (B, G, R)
    
# for GI spore 
cyan = [255, 255, 0]  # Original color (B, G, R)
darkred = [0, 0, 128]   # New color (B, G, R)

folders = sorted(glob(os.path.join(src_path, '*.svs')))
for folder in folders:
    folder_name = os.path.basename(folder)
    print(folder_name)
    dst_folder = os.path.join(dst_path, folder_name)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = sorted(glob(os.path.join(folder, '*.png')))
    for file in files:
        file_name = os.path.basename(file)
        dst_file = os.path.join(dst_folder, file_name)
        if not os.path.exists(dst_file):
            image = cv2.imread(file)
            #Create a mask where the pixel matches the yellow color
            mask = np.all(image == cyan, axis=-1)
            image[mask] = darkred
            cv2.imwrite(dst_file, image)
