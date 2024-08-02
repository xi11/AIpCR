import os
from glob import glob
import cv2
import numpy as np

src_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/validation_new2/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_cws512'
dst_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/validation_new2/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_cws512_orng'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

yellow = [0, 255, 255]  # Original color (R, G, B)
orange = [0, 204, 255]  # New color (R, G, B)

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
            mask = np.all(image == yellow, axis=-1)
            image[mask] = orange
            cv2.imwrite(dst_file, image)
