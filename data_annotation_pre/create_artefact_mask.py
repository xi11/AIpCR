import os
import random
import pathlib
import numpy as np
import shutil
import cv2
from glob import glob

# to convert artefact masks to all zeros
src_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/cartilage/image'
dst_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/cartilage/mask'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

files = sorted(glob(os.path.join(src_path, '*.jpg')))
for file in files:
    file_name = os.path.basename(file)[:-4]
    dst_file = os.path.join(dst_path, file_name+'.png')
    img = cv2.imread(file)
    mask_gt = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    cv2.imwrite(dst_file, mask_gt)