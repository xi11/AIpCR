import os
import cv2
from glob import glob
import shutil
import pandas as pd


#This is step2, to remove hard-cropped HE images relevant to mask,
src_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/patch512/patch512img'
ref_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/patch512/patch512mask_hardcrop0.567'
dst_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/patch512/patch512img_hardcrop0.567'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
#patch_id = []
patches = sorted(glob(os.path.join(ref_path, '*-labelled.png')))
print(len(patches))
for patch in patches:
    patch_name = os.path.basename(patch)[:-13]
    print(patch_name)
    src_file = os.path.join(src_path, patch_name+'.png')
    dst_file = os.path.join(dst_path, patch_name+'.png')
    shutil.move(src_file, dst_file)

#patch_removal = pd.DataFrame({'patch_id': patch_id})
#patch_removal.to_csv(os.path.join('/Volumes/yuan_lab/TIER2/barrett/fully_supervised/results', 'internal_testROI1_remove02.csv'), index=False)

#Step3-to convert colored index image to digital masks, it's achieved with matlab script
#Step4-to include pen marker patches from TCGA