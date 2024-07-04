import os
from glob import glob
import pandas as pd
import shutil

###T6 data organization
src_path = '/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/maskPng'
dst_path = '/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/maskPng_test'
os.makedirs(dst_path, exist_ok=True)

prefixes = [1, 2, 15, 19, 21, 32, 40, 48, 52, 62, 67]
prefixes_str = [f"mask_{prefix}_HE" for prefix in prefixes]

for filename in os.listdir(src_path):
    if any(filename.startswith(prefix) for prefix in prefixes_str):
        shutil.move(os.path.join(src_path, filename), os.path.join(dst_path, filename))



'''
src_path = r'Z:\TIER2\barrett\til\MDA\1_cws_tiling'
files = sorted(glob(os.path.join(src_path, '*.svs')))
for file in files:
    file_name = os.path.basename(file)
    src_file = os.path.join(file, 'Ss1.jpg')
    #print(src_file)
    if not os.path.exists(src_file):
        print(file_name)



src_path = r'Z:\TIER2\anthracosis\prospect_multi\AIgrading\mask_cws'
dst_path = r'Z:\TIER2\anthracosis\prospect_multi\AIgrading\mask_ss1'
files = sorted(glob(os.path.join(src_path, '*.svs')))
for file in files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(dst_path, file_name + '_Ss1.png')
    if not os.path.exists(dst_file):
        print(file_name)

src_path = r'W:\data\artemis_lei\1_cws_tiling'
ref_path = r'W:\data\artemis_lei\tmesegK8div12v2noNorm\mask_cws'
files = sorted(glob(os.path.join(src_path, '*.svs')))
count = 0
for file in files:
    count = count+1
    file_name = os.path.basename(file)
    dst_file = os.path.join(ref_path, file_name)
    if not os.path.exists(dst_file):
        print(count)
        print(file_name)



src_path = r'T:\project\tcga_tnbc\public_train\patch768_tcga20x\maskPng'
dst_path = r'T:\project\tcga_tnbc\public_train\patch768_tcga20x_test\maskPng'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

files = sorted(glob(os.path.join(src_path, 'mask_TCGA-OL*')))
for file in files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(dst_path, file_name)
    shutil.move(file, dst_file)




src_path = r'Z:\TIER2\share_tls-st\HE\til_json\1_cws_tiling'
dst_path = r'Z:\TIER2\share_tls-st\HE\til_json\ss1_images'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
folders = sorted(glob(os.path.join(src_path, '*.tiff')))
for folder in folders:
    folder_name = os.path.basename(folder)[:-4]
    src_file = os.path.join(folder, 'Ss1.jpg')
    dst_file = os.path.join(dst_path, folder_name+'_Ss1.jpg')
    if not os.path.exists(dst_file):
        shutil.copy(src_file, dst_file)


src_path = r'Z:\TIER2\artemis_lei\discovery\til\1_cws_tiling'
dst_path = r'Z:\TIER2\artemis_lei\discovery\til\2_tissue_seg\pre_processed'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
folders = [f.path for f in os.scandir(src_path) if f.is_dir()]
for folder in folders:
    folder_name = os.path.basename(folder)
    #print(folder_name)
    dst_folder = os.path.join(dst_path, folder_name)
    if not os.path.exists(dst_folder):
        print(folder_name)

#540_HE_A1_Primary.svs
#601_HE_A2_Primary.svs


src_path = r'X:\project\tcga_tnbc'
dst_path = r'X:\project\tcga_tnbc_svs'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
folders = [f.path for f in os.scandir(src_path) if f.is_dir()]
for folder in folders:
    folder_name = os.path.basename(folder)
    print(folder_name)
    src_file = glob(os.path.join(folder, '*.svs'))[0]
    #dst_file = os.path.join(dst_path, file_name+'-labelled.png')
    shutil.move(src_file, dst_path)


ref_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/hard_crop/mask'
files = sorted(glob(os.path.join(ref_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)[:-4]
    src_file = os.path.join(ref_path, file_name+'-labelled.png')
    os.rename(file, src_file)

src_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/hard_crop/mask'
ref_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/hard_crop/image2'
dst_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/hard_crop/mask2'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
files = sorted(glob(os.path.join(ref_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)[:-4]
    src_file = os.path.join(src_path, file_name+'-labelled.png')
    dst_file = os.path.join(dst_path, file_name+'-labelled.png')
    shutil.move(src_file, dst_file)


src_path = '/Users/xiaoxipan/Documents/project/artemis/annotation/patch768raw'
dst_img = '/Users/xiaoxipan/Documents/project/artemis/annotation/patch768img'
dst_mask = '/Users/xiaoxipan/Documents/project/artemis/annotation/patch768mask'
if not os.path.exists(dst_img):
    os.mkdir(dst_img)

if not os.path.exists(dst_mask):
    os.mkdir(dst_mask)

folders = sorted(glob(os.path.join(src_path, '*HE*')))
for folder in folders:
    files = sorted(glob(os.path.join(folder, '*.png')))
    for file in files:
        file_name = os.path.basename(file)
        file_name = file_name .replace(" ", "_")
        print(file_name)
        shutil.move(file, os.path.join(dst_img, file_name))


ref_path = '/Users/xiaoxipan/Documents/project/anthracosis/data_square/annotated_raw'
src_path = '/Users/xiaoxipan/Documents/project/anthracosis/data_square/annotated_raw_artefact'
dst_path = '/Users/xiaoxipan/Documents/project/anthracosis/data_square/annotated_raw_artefact'


folders = sorted(glob(os.path.join(ref_path, '*.svs')))
for folder in folders:
    folder_name = os.path.basename(folder)
    slide_ID = folder_name[:-4]
    dst_folder = os.path.join(dst_path, folder_name)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        files = sorted(glob(os.path.join(src_path, slide_ID+'_*.png')))
        if len(files)>0:
            for file in files:
                file_name = os.path.basename(file)
                shutil.move(file, os.path.join(dst_folder, file_name))
        else:
            print(folder_name)



src_path = '/Volumes/yuan_lab/TIER2/anthracosis/never_smoker/1_cws_tiling'
dst_path = '/Users/xiaoxipan/Documents/project/anthracosis'
files = sorted(glob(os.path.join(src_path, '*.svs')))
slide_id = []
patient_id = []
for file in files:
    file_name = os.path.basename(file)[:-4]
    slide_id.append(file_name)
    patient_id.append(file_name.split('_')[0])
raw_slide = pd.DataFrame({'slide_id': slide_id, 'patient_id': patient_id})
raw_slide.to_csv(os.path.join(dst_path, 'never-smoker_slideID.csv'), index=False)



src_path = '/Volumes/yuan_lab/TIER1/anthracosis/Anthracosis-NEVER-SMOKER/folder1'
dst_path = '/Volumes/yuan_lab/TIER1/anthracosis/Anthracosis-NEVER-SMOKER/folder3'
src_files = sorted(glob(os.path.join(src_path, '*.svs')))
for file in src_files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(dst_path, file_name)
    if os.path.exists(dst_file):
        print(file_name)

'''
