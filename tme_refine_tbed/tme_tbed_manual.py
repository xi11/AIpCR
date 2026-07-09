import os
from glob import glob
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

tme_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512'
tbed_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/yutong-tnbc-pcr/discovery_pilot25/tumorBorder_manual/mask'
dst_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/yutong-tnbc-pcr/discovery_pilot25/tme_tborder_mask'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)


files = sorted(glob(os.path.join(tbed_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)[:-9]
    dst_file = os.path.join(dst_path, file_name + '_tme_tbed.png')

    tbed = Image.open(file).convert('RGB')
    tme = Image.open(os.path.join(tme_path, file_name + '.svs_Ss1.png')).convert('RGB')

    # I want to keep the tme mask only within the tumor border defined by tbed, so I will set the tme mask to 0 outside the tumor border
    # but they are not the same size, so I need to resize the tbed mask to the same size as tme mask first using nearest neighbor interpolation, and then apply the mask
    tbed_resized = tbed.resize((tme.width, tme.height), resample=Image.NEAREST)

    tbed_arr = np.array(tbed_resized)
    tme_arr = np.array(tme)

    mask = np.all(tbed_arr == [255, 255, 255], axis=-1)
    tme_tbed = np.zeros_like(tme_arr)
    tme_tbed[mask] = tme_arr[mask]

    Image.fromarray(tme_tbed).save(dst_file)



