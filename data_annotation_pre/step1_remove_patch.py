import os
import cv2
from glob import glob
import shutil
import pandas as pd
'''
#Step0: using Qupath to generate patches, and move images and masks to two separate folders
src_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/raw'
dst_img = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/img'
dst_mask = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/mask_color'
if not os.path.exists(dst_img):
    os.mkdir(dst_img)

if not os.path.exists(dst_mask):
    os.mkdir(dst_mask)

folders = sorted(glob(os.path.join(src_path, '3974*')))
for folder in folders:
    files = sorted(glob(os.path.join(folder, '*.png')))
    for file in files:
        file_name = os.path.basename(file)
        file_name = file_name.replace(" ", "_")
        print(file_name)
        shutil.copy(file, os.path.join(dst_img, file_name))


##what if new annotations are added with qupath, how to leverage previous QC
src_path = '/Users/xiaoxipan/Documents/project/public_data/bcss/luad_annotation_xp/patch512_mpp44muscle/patch896x40mask'
dst_path = '/Users/xiaoxipan/Documents/project/public_data/bcss/luad_annotation_xp/patch512_mpp44muscle/patch896mask'
ref_path = '/Users/xiaoxipan/Documents/project/public_data/bcss/luad_annotation_xp/patch512_mpp44/patch896mask'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

patches = sorted(glob(os.path.join(ref_path, '*.png')))
print(len(patches))
for patch in patches:
    patch_name = os.path.basename(patch)[5:-4]
    src_file = os.path.join(src_path, patch_name + '-labelled.png')
    dst_file = os.path.join(dst_path, patch_name + '-labelled.png')
    shutil.move(src_file, dst_file)


#This is step1, to remove hard-cropped annotations, after which still need manually check
src_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/mask_color'
dst_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/mask_color_0.3'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
#patch_id = []
patches = sorted(glob(os.path.join(src_path, '*.png')))
print(len(patches))
for patch in patches:
    patch_name = os.path.basename(patch)
    img = cv2.imread(patch)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n, _ = img.shape
    area = cv2.countNonZero(gray_image)
    if area < 0.3 * m * n:
        print(patch_name)
        print(area)
        #patch_id.append(patch_name)
        dst_file = os.path.join(dst_path, patch_name)
        shutil.move(patch, dst_file)

#patch_removal = pd.DataFrame({'patch_id': patch_id})
#patch_removal.to_csv(os.path.join('/Volumes/yuan_lab/TIER2/barrett/fully_supervised/results', 'internal_testROI1_remove02.csv'), index=False)


### remove image to hard crop
src_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/img'
ref_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/mask_color_0.05'
dst_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/img_0.05'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
#patch_id = []
patches = sorted(glob(os.path.join(ref_path, '*.png')))
print(len(patches))
for patch in patches:
    patch_name = os.path.basename(patch)[:-13]
    src_file = os.path.join(src_path, patch_name + '.png')
    dst_file = os.path.join(dst_path, patch_name + '.png')
    shutil.move(src_file, dst_file)

'''
###rename to 'mask_'
src_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/ping-cell-therapy/Liver_annotation/batch1_patch512_mppRaw/mask_color'

#patch_id = []
patches = sorted(glob(os.path.join(src_path, '*.png')))
print(len(patches))
for patch in patches:
    patch_name = os.path.basename(patch)[:-13]
    dst_file = os.path.join(src_path, 'mask_'+patch_name + '.png')
    os.rename(patch, dst_file)


####covert to digital label with matlab