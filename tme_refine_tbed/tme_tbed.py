import os
from glob import glob
import cv2
import numpy as np

tme_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/IMPRESS_TNBC/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512'
tbed_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/IMPRESS_TNBC/tbed1536_ss1/maskLuadLusc_tmeArtemis_tumor7dilate'
dst_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/IMPRESS_TNBC/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512_tumor7dilate_orng'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

# for artemis ms
yellow = [0, 255, 255]  # Original color (B, G, R)
orange = [0, 204, 255]  # New color (B, G, R)

files = sorted(glob(os.path.join(tme_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(dst_path, file_name)
    #if not os.path.exists(dst_file):
    tme_raw = cv2.imread(file)
    tbed = cv2.imread(os.path.join(tbed_path, file_name[:-12]+'_tme_tbed.png')) #20
    #Create a mask where the pixel matches the yellow color
    mask = np.all(tme_raw == yellow, axis=-1)
    tme_raw[mask] = orange

    height_diff = tme_raw.shape[0] - tbed.shape[0]
    width_diff = tme_raw.shape[1] - tbed.shape[1]
    tbed_padded = np.pad(tbed, ((0, height_diff), (0, width_diff), (0, 0)), mode='constant', constant_values=0)
    #tbed_padded = np.repeat(tbed_padded, axis=-1) 
    tme_tbed = np.where(tbed_padded == 255, tme_raw, 0)
    cv2.imwrite(dst_file, tme_tbed)
